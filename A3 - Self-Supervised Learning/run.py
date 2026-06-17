import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.simclr import SimCLR, NTXentLoss, CIFAR10SSL
from models.dino import (build_dino_model, DINOLoss, CIFAR10DINO,
                          dino_collate)
from models.mae import MAE, CIFAR10MAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
os.makedirs('saved', exist_ok=True)
os.makedirs('figures', exist_ok=True)

CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# ── Shared eval transform ─────────────────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
])


# ── Linear evaluation (shared) ────────────────────────────────────────────────
def linear_eval(encoder_fn, feat_dim, tag, epochs=10):
    """Freeze encoder_fn, train a linear head, return test accuracy."""
    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  download=False, transform=eval_tf)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=eval_tf)
    trl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    clf = nn.Linear(feat_dim, 10).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        clf.train()
        for imgs, labels in tqdm(trl, desc=f'[{tag}] linear eval {epoch+1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                h = encoder_fn(imgs)
            loss = F.cross_entropy(clf(h), labels)
            opt.zero_grad(); loss.backward(); opt.step()

    clf.eval()
    correct = total = 0
    embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            h = encoder_fn(imgs)
            correct += (clf(h).argmax(1) == labels).sum().item()
            total   += labels.size(0)
            embeddings.append(h.cpu()); all_labels.append(labels.cpu())

    acc = correct / total * 100
    print(f'[{tag}] Linear Eval Test Accuracy: {acc:.2f}%')
    return acc, torch.cat(embeddings), torch.cat(all_labels)


# ── SimCLR ────────────────────────────────────────────────────────────────────
def train_simclr(epochs=10, batch_size=256):
    loader = DataLoader(CIFAR10SSL(), batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    model  = SimCLR().to(device)
    loss_fn = NTXentLoss(temperature=0.5)
    opt    = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train(); ep = []
        for x_i, x_j, _ in tqdm(loader, desc=f'SimCLR {epoch+1}/{epochs}'):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j, _, _ = model(x_i, x_j)
            loss = loss_fn(z_i, z_j)
            opt.zero_grad(); loss.backward(); opt.step()
            ep.append(loss.item())
        losses.append(np.mean(ep))
        print(f'  Epoch {epoch+1:02d} | Loss: {losses[-1]:.4f}')
    epoch_time = (time.time() - t0) / epochs

    torch.save(model.state_dict(), 'saved/simclr.pt')

    plt.figure(figsize=(8,3))
    plt.plot(losses, marker='o'); plt.title('SimCLR Loss')
    plt.xlabel('Epoch'); plt.ylabel('NT-Xent'); plt.grid(True)
    plt.savefig('figures/simclr_loss.png', bbox_inches='tight'); plt.close()

    print(f'SimCLR time/epoch: {epoch_time:.1f}s')
    return model, epoch_time


def eval_simclr():
    model = SimCLR().to(device)
    model.load_state_dict(torch.load('saved/simclr.pt', map_location=device))
    for p in model.encoder.parameters(): p.requires_grad = False
    enc = lambda imgs: torch.flatten(model.encoder(imgs), 1)
    return linear_eval(enc, 512, 'SimCLR')


# ── DINO ──────────────────────────────────────────────────────────────────────
def train_dino(epochs=10, batch_size=56, n_local=4,
               use_centering=True, tag='dino'):
    dataset = CIFAR10DINO(n_local=n_local)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True, collate_fn=dino_collate)

    student_vit, student_head = build_dino_model()
    teacher_vit, teacher_head = build_dino_model()
    student_vit, student_head = student_vit.to(device), student_head.to(device)
    teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)

    teacher_vit.load_state_dict(student_vit.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_vit.parameters():  p.requires_grad = False
    for p in teacher_head.parameters(): p.requires_grad = False

    out_dim = 256
    loss_fn = DINOLoss(out_dim=out_dim, n_crops=2+n_local,
                       use_centering=use_centering).to(device)
    opt = torch.optim.AdamW(
        list(student_vit.parameters()) + list(student_head.parameters()),
        lr=5e-5, weight_decay=0.04
    )
    EMA_M = 0.996

    losses, center_norms = [], []
    t0 = time.time()
    for epoch in range(epochs):
        student_vit.train(); student_head.train(); ep = []
        for crops, _ in tqdm(loader, desc=f'DINO[{tag}] {epoch+1}/{epochs}'):
            crops = [c.to(device) for c in crops]
            student_out = [student_head(student_vit(c)) for c in crops]
            with torch.no_grad():
                teacher_out = [teacher_head(teacher_vit(crops[0])),
                               teacher_head(teacher_vit(crops[1]))]
            loss = loss_fn(student_out, teacher_out)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student_vit.parameters()) + list(student_head.parameters()), max_norm=3.0
            )
            opt.step()
            with torch.no_grad():
                for s, t in zip(student_vit.parameters(), teacher_vit.parameters()):
                    t.data = EMA_M * t.data + (1 - EMA_M) * s.data
                for s, t in zip(student_head.parameters(), teacher_head.parameters()):
                    t.data = EMA_M * t.data + (1 - EMA_M) * s.data
            ep.append(loss.item())
        losses.append(np.mean(ep))
        center_norms.append(loss_fn.center.norm().item())
        print(f'  Epoch {epoch+1:02d} | Loss: {losses[-1]:.4f} | Center norm: {center_norms[-1]:.4f}')
    epoch_time = (time.time() - t0) / epochs

    torch.save({'student_vit': student_vit.state_dict(),
                'student_head': student_head.state_dict()}, f'saved/{tag}.pt')

    plt.figure(figsize=(8,3))
    plt.plot(losses, marker='o', color='darkorange')
    plt.title(f'DINO Loss [{tag}]'); plt.xlabel('Epoch'); plt.ylabel('CE'); plt.grid(True)
    plt.savefig(f'figures/{tag}_loss.png', bbox_inches='tight'); plt.close()

    if use_centering:
        plt.figure(figsize=(8,3))
        plt.plot(center_norms, marker='s', color='steelblue')
        plt.title('DINO Center Norm'); plt.xlabel('Epoch'); plt.ylabel('||c||'); plt.grid(True)
        plt.savefig('figures/dino_center_norm.png', bbox_inches='tight'); plt.close()

    print(f'DINO[{tag}] time/epoch: {epoch_time:.1f}s')
    return student_vit, epoch_time


def eval_dino(tag='dino'):
    student_vit, _ = build_dino_model()
    ckpt = torch.load(f'saved/{tag}.pt', map_location=device)
    student_vit.load_state_dict(ckpt['student_vit'])
    student_vit = student_vit.to(device)
    for p in student_vit.parameters(): p.requires_grad = False
    enc = lambda imgs: student_vit(imgs)
    return linear_eval(enc, student_vit.embed_dim, tag)


# ── DINO Attention Maps ───────────────────────────────────────────────────────
def visualize_attention(tag='dino', n_images=5):
    student_vit, _ = build_dino_model()
    ckpt = torch.load(f'saved/{tag}.pt', map_location=device)
    student_vit.load_state_dict(ckpt['student_vit'])
    student_vit = student_vit.to(device).eval()

    attentions = {}
    def hook_fn(module, input, output):
        # output is a tuple (attn_output, attn_weights) or just attn_output
        # We need to access the raw attention weights directly
        attentions['last'] = output

    # Hook on the attention module directly
    handle = student_vit.blocks[-1].attn.register_forward_hook(hook_fn)

    raw_test = torchvision.datasets.CIFAR10('./data', train=False, transform=eval_tf)
    loader = DataLoader(raw_test, batch_size=1, shuffle=True)

    mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(3,1,1)

    patch_h = patch_w = 32 // 4  # 8x8
    n_patches = patch_h * patch_w  # 64

    # Get number of heads from model
    n_heads = student_vit.blocks[-1].attn.num_heads

    fig, axes = plt.subplots(n_images, n_heads + 1, figsize=(2*(n_heads+1), 3*n_images))
    sample_iter = iter(loader)

    from PIL import Image as PILImage

    for row in range(n_images):
        img_tensor, label = next(sample_iter)
        img_tensor = img_tensor.to(device)

        # Forward pass with grad disabled and attention saved
        with torch.no_grad():
            # Manually get attention weights by going through blocks
            x = student_vit.patch_embed(img_tensor)
            cls_token = student_vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = student_vit.pos_drop(x + student_vit.pos_embed)
            for i, block in enumerate(student_vit.blocks):
                if i < len(student_vit.blocks) - 1:
                    x = block(x)
                else:
                    # Last block: get attention weights manually
                    B, N, C = x.shape
                    qkv = block.attn.qkv(block.norm1(x))
                    qkv = qkv.reshape(B, N, 3, n_heads, C // n_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale = (C // n_heads) ** -0.5
                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = attn_weights.softmax(dim=-1)  # (B, n_heads, N, N)

        # CLS token attention to patches: row 0, cols 1:
        cls_attn = attn_weights[0, :, 0, 1:]  # (n_heads, 64)

        img_disp = torch.clamp(img_tensor[0].cpu() * std + mean, 0, 1).permute(1,2,0).numpy()
        axes[row][0].imshow(img_disp)
        axes[row][0].set_title(CIFAR_CLASSES[label.item()], fontsize=9)
        axes[row][0].axis('off')

        for h in range(n_heads):
            head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
            head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
            head_up = np.array(PILImage.fromarray((head_map*255).astype(np.uint8)).resize((32,32)))
            axes[row][h+1].imshow(img_disp, alpha=0.4)
            axes[row][h+1].imshow(head_up, cmap='hot', alpha=0.7, vmin=0, vmax=255)
            if row == 0: axes[row][h+1].set_title(f'Head {h+1}', fontsize=8)
            axes[row][h+1].axis('off')

    handle.remove()
    plt.suptitle(f'DINO Attention [{tag}]', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(f'figures/{tag}_attention.png', bbox_inches='tight', dpi=150)
    plt.close()
    # NEW
    print(f'Attention maps saved: figures/{tag}_attention.png')


# ── MAE ───────────────────────────────────────────────────────────────────────
def train_mae(mask_ratio=0.75, epochs=5, batch_size=256, tag=None):
    if tag is None:
        tag = f'mae_{int(mask_ratio*100)}'
    loader = DataLoader(CIFAR10MAE(), batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    model = MAE(mask_ratio=mask_ratio).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train(); ep = []
        for imgs, _ in tqdm(loader, desc=f'MAE[{tag}] {epoch+1}/{epochs}'):
            imgs = imgs.to(device)
            loss, _, _, _ = model(imgs)
            opt.zero_grad(); loss.backward(); opt.step()
            ep.append(loss.item())
        losses.append(np.mean(ep))
        print(f'  Epoch {epoch+1:02d} | Recon Loss: {losses[-1]:.4f}')
    epoch_time = (time.time() - t0) / epochs

    torch.save(model.state_dict(), f'saved/{tag}.pt')
    plt.figure(figsize=(8,3))
    plt.plot(losses, marker='o', color='green')
    plt.title(f'MAE Loss [mask={mask_ratio}]'); plt.xlabel('Epoch')
    plt.ylabel('MSE Recon Loss'); plt.grid(True)
    plt.savefig(f'figures/{tag}_loss.png', bbox_inches='tight'); plt.close()

    print(f'MAE[{tag}] time/epoch: {epoch_time:.1f}s')
    return model, losses[-1], epoch_time


def eval_mae(mask_ratio=0.75, tag=None):
    if tag is None:
        tag = f'mae_{int(mask_ratio*100)}'
    model = MAE(mask_ratio=mask_ratio).to(device)
    model.load_state_dict(torch.load(f'saved/{tag}.pt', map_location=device))
    for p in model.encoder.parameters(): p.requires_grad = False
    enc = lambda imgs: model.encoder(imgs)
    return linear_eval(enc, model.encoder.embed_dim, tag)


# ── t-SNE ─────────────────────────────────────────────────────────────────────
def plot_tsne(embeddings_dict):
    """embeddings_dict: {name: (embeddings_tensor, labels_tensor)}"""
    from sklearn.manifold import TSNE
    n = len(embeddings_dict)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (name, (emb, lbls)) in zip(axes, embeddings_dict.items()):
        idx = np.random.choice(len(emb), min(2000, len(emb)), replace=False)
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb[idx].numpy())
        for c in range(10):
            mask = lbls[idx].numpy() == c
            ax.scatter(proj[mask,0], proj[mask,1], c=[colors[c]],
                       label=CIFAR_CLASSES[c], alpha=0.6, s=10)
        ax.set_title(name, fontsize=12); ax.legend(fontsize=7, markerscale=2); ax.axis('off')
    plt.suptitle('t-SNE: SSL Representations on CIFAR-10', fontsize=13)
    plt.tight_layout()
    plt.savefig('figures/tsne_comparison.png', bbox_inches='tight'); plt.close()
    print('t-SNE saved → figures/tsne_comparison.png')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       choices=['simclr','dino','mae'], required=True)
    p.add_argument('--epochs',      type=int, default=None)
    p.add_argument('--train',       action='store_true')
    p.add_argument('--evaluate',    action='store_true')
    p.add_argument('--linear',      action='store_true')
    p.add_argument('--attention',   action='store_true')
    p.add_argument('--tsne',        action='store_true')
    p.add_argument('--weights',     type=str, default=None)
    p.add_argument('--no-centering',action='store_true')
    p.add_argument('--n-local',     type=int, default=4)
    p.add_argument('--mask-ratio',  type=float, default=0.75)
    p.add_argument('--tag',         type=str, default=None)
    p.add_argument('--dataset',     type=str, default='cifar10')  # kept for compat
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-tag
    if args.tag is None:
        if args.model == 'dino':
            tag = 'dino'
            if args.no_centering: tag = 'dino_no_center'
            if args.n_local == 0: tag = 'dino_no_local'
        elif args.model == 'mae':
            tag = f'mae_{int(args.mask_ratio*100)}'
        else:
            tag = 'simclr'
    else:
        tag = args.tag

    # ── Train ──
    if args.train:
        if args.model == 'simclr':
            ep = args.epochs or 10
            train_simclr(epochs=ep)

        elif args.model == 'dino':
            ep = args.epochs or 10
            train_dino(epochs=ep, n_local=args.n_local,
                       use_centering=not args.no_centering, tag=tag)

        elif args.model == 'mae':
            ep = args.epochs or 5
            train_mae(mask_ratio=args.mask_ratio, epochs=ep, tag=tag)

    # ── Evaluate ──
    if args.evaluate or args.linear:
        if args.model == 'simclr':
            acc, emb, lbl = eval_simclr()

        elif args.model == 'dino':
            acc, emb, lbl = eval_dino(tag=tag)
            if args.attention:
                visualize_attention(tag=tag)

        elif args.model == 'mae':
            acc, emb, lbl = eval_mae(mask_ratio=args.mask_ratio, tag=tag)

    if args.attention and args.model == 'dino':
        visualize_attention(tag=tag)


if __name__ == '__main__':
    main()