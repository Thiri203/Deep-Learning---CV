import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST, CelebA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
os.makedirs('saved', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, choices=['gan','cyclegan','ddpm'])
    parser.add_argument('--dataset',  type=str, choices=['mnist','celeba'])
    parser.add_argument('--epochs',   type=int, default=20)
    parser.add_argument('--schedule', type=str, choices=['linear','cosine'], default='linear')
    parser.add_argument('--weights',  type=str, default=None)
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--n',        type=int, default=64)
    parser.add_argument('--test-image', type=str, default=None)
    parser.add_argument('--mode-collapse-check', action='store_true')
    parser.add_argument('--celeba-subset', type=int, default=15000)  # time saver
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lambda-cyc', type=float, default=10.0)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--skip-ablation', action='store_true')
    return parser.parse_args()

#data loaders
def get_mnist_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

class FaceDataset(Dataset):
    def __init__(self, filenames, img_dir, transform):
        self.filenames = filenames
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.filenames[idx])).convert('RGB')
        return self.transform(img), 0

def get_celeba_loader(batch_size=64, subset_size=15000, num_workers=2):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    
    # load attr file directly to avoid torchvision path issues
    attr_path = './data/celeba/celeba/list_attr_celeba.txt'
    img_dir   = './data/celeba/celeba/img_align_celeba/'

    blonde_files, dark_files = [], []
    max_per_class = subset_size // 2
    with open(attr_path, 'r', encoding='utf-8') as f:
        next(f)  # image count
        header = next(f).split()
        blond_hair_idx = header.index('Blond_Hair') + 1  # +1 for filename column
        for line in f:
            parts = line.split()
            if not parts:
                continue

            filename = parts[0]
            is_blonde = int(parts[blond_hair_idx]) == 1
            if is_blonde and len(blonde_files) < max_per_class:
                blonde_files.append(filename)
            elif not is_blonde and len(dark_files) < max_per_class:
                dark_files.append(filename)

            if len(blonde_files) >= max_per_class and len(dark_files) >= max_per_class:
                break

    blonde_set = FaceDataset(blonde_files, img_dir, transform)
    dark_set   = FaceDataset(dark_files,   img_dir, transform)

    pin_memory = device.type == 'cuda'
    blonde_loader = DataLoader(
        blonde_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    dark_loader = DataLoader(
        dark_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return dark_loader, blonde_loader

#Gan Model
# ── Vanilla GAN ──────────────────────────────────────────────
class GANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),  nn.LeakyReLU(0.2),
            nn.Linear(256, 512),    nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),   nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class GANDiscriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512),     nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),      nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),        nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

#CycleGAN Model

# ── CycleGAN building blocks ──────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch)
        )
    def forward(self, x): return x + self.block(x)

class CycleGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_res=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True),
            # downsample
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True),
        ]
        for _ in range(n_res):
            layers.append(ResBlock(256))
        layers += [
            # upsample
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  3, 2, 1, 1), nn.InstanceNorm2d(64),  nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_ch, 7), nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class CycleDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64,  4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,  128,   4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256,   4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512,   4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1,     4, 1, 1)
        )
    def forward(self, x): return self.net(x)


#DDPM Model
# ── DDPM ─────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:,None].float() * emb[None,:]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class DDPMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:,:,None,None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.enc1 = DDPMResBlock(in_ch,      base_ch,   time_dim)
        self.enc2 = DDPMResBlock(base_ch,    base_ch*2, time_dim)
        self.down = nn.MaxPool2d(2)
        self.bot  = DDPMResBlock(base_ch*2,  base_ch*4, time_dim)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DDPMResBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)
        self.dec1 = DDPMResBlock(base_ch*2 + base_ch,   base_ch,   time_dim)
        self.out  = nn.Conv2d(base_ch, in_ch, 1)
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        b  = self.bot(self.down(e2), t_emb)
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        return self.out(d1)
    
# DDPM noise schedule
# ── Noise schedules ───────────────────────────────────────────
def linear_beta_schedule(T=1000):
    return torch.linspace(1e-4, 0.02, T)

def cosine_beta_schedule(T=1000, s=0.008):
    t = torch.linspace(0, T, T+1)
    ab = torch.cos(((t/T)+s)/(1+s)*torch.pi*0.5)**2
    ab = ab / ab[0]
    betas = 1 - (ab[1:] / ab[:-1])
    return torch.clamp(betas, 1e-4, 0.9999)

def get_ddpm_constants(schedule='linear', T=1000):
    betas = (linear_beta_schedule(T) if schedule=='linear'
             else cosine_beta_schedule(T)).to(device)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        'betas':             betas,
        'alphas':            alphas,
        'alpha_bar':         alpha_bar,
        'sqrt_ab':           torch.sqrt(alpha_bar),
        'sqrt_one_minus_ab': torch.sqrt(1 - alpha_bar),
        'sqrt_recip_a':      torch.sqrt(1.0 / alphas),
        'posterior_var':     betas * (1 - F.pad(alpha_bar[:-1],(1,0),value=1.0)) / (1 - alpha_bar)
    }

# Training and evaluation functions

# ── Train GAN ─────────────────────────────────────────────────
def train_gan(epochs=20, d_lr=2e-4):
    loader = get_mnist_loader()
    G = GANGenerator().to(device)
    D = GANDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4,  betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr,  betas=(0.5,0.999))
    criterion = nn.BCELoss()
    Z_DIM = 100
    fixed_z = torch.randn(64, Z_DIM).to(device)

    for epoch in range(epochs):
        t0 = time.time()
        for real_imgs, _ in loader:
            B = real_imgs.size(0)
            real_imgs = real_imgs.view(B,-1).to(device)
            real_lab  = torch.ones(B,1).to(device)
            fake_lab  = torch.zeros(B,1).to(device)

            z = torch.randn(B, Z_DIM).to(device)
            d_loss = criterion(D(real_imgs), real_lab) + criterion(D(G(z).detach()), fake_lab)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            z = torch.randn(B, Z_DIM).to(device)
            g_loss = criterion(D(G(z)), real_lab)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        print(f'[GAN] Epoch {epoch+1}/{epochs} | D:{d_loss.item():.3f} G:{g_loss.item():.3f} | {time.time()-t0:.1f}s')

    torch.save(G.state_dict(), 'saved/gan_mnist.pt')
    # save output grid
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z).view(-1,1,28,28).cpu()
    grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
    plt.figure(figsize=(8,8)); plt.imshow(grid.permute(1,2,0)); plt.axis('off')
    plt.title('GAN Generated MNIST')
    plt.savefig('outputs/gan_grid.png', bbox_inches='tight'); plt.close()
    print('Saved: outputs/gan_grid.png')
    return G

# ── Train CycleGAN ────────────────────────────────────────────
def train_cyclegan(
    epochs=20,
    lambda_cyc=10.0,
    subset_size=15000,
    batch_size=64,
    log_every=25,
    num_workers=2,
    use_amp=True,
):
    dark_loader, blonde_loader = get_celeba_loader(
        batch_size=batch_size,
        subset_size=subset_size,
        num_workers=num_workers,
    )

    G_d2b = CycleGenerator().to(device)  # dark  → blonde
    G_b2d = CycleGenerator().to(device)  # blonde → dark
    D_b   = CycleDiscriminator().to(device)
    D_d   = CycleDiscriminator().to(device)

    opt_G = torch.optim.Adam(list(G_d2b.parameters())+list(G_b2d.parameters()), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(list(D_b.parameters())+list(D_d.parameters()),     lr=2e-4, betas=(0.5,0.999))
    crit_gan = nn.MSELoss()
    crit_cyc = nn.L1Loss()
    amp_enabled = use_amp and device.type == 'cuda'
    scaler_G = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_D = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    tag = f'cyc{int(lambda_cyc)}'
    total_batches = min(len(dark_loader), len(blonde_loader))
    print(
        f'[CycleGAN] lambda_cyc={lambda_cyc:g} | epochs={epochs} | '
        f'subset={subset_size} | batch_size={batch_size} | '
        f'batches/epoch={total_batches} | amp={amp_enabled}'
    )

    for epoch in range(epochs):
        t0 = time.time()
        for batch_idx, ((real_d, _), (real_b, _)) in enumerate(zip(dark_loader, blonde_loader), start=1):
            real_d = real_d.to(device, non_blocking=True)
            real_b = real_b.to(device, non_blocking=True)

            # ── Generator step ──
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                fake_b = G_d2b(real_d)
                fake_d = G_b2d(real_b)
                rec_d  = G_b2d(fake_b)
                rec_b  = G_d2b(fake_d)

                pred_fake_b = D_b(fake_b)
                pred_fake_d = D_d(fake_d)
                loss_g = crit_gan(pred_fake_b, torch.ones_like(pred_fake_b)) + \
                         crit_gan(pred_fake_d, torch.ones_like(pred_fake_d))
                loss_cyc = crit_cyc(rec_d, real_d) + crit_cyc(rec_b, real_b)
                loss_G = loss_g + lambda_cyc * loss_cyc

            opt_G.zero_grad(set_to_none=True)
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # ── Discriminator step ──
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_real_b = D_b(real_b)
                pred_fake_b = D_b(fake_b.detach())
                pred_real_d = D_d(real_d)
                pred_fake_d = D_d(fake_d.detach())
                loss_D = crit_gan(pred_real_b, torch.ones_like(pred_real_b)) + \
                         crit_gan(pred_fake_b, torch.zeros_like(pred_fake_b)) + \
                         crit_gan(pred_real_d, torch.ones_like(pred_real_d)) + \
                         crit_gan(pred_fake_d, torch.zeros_like(pred_fake_d))

            opt_D.zero_grad(set_to_none=True)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            if log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - t0
                print(
                    f'[CycleGAN] Epoch {epoch+1}/{epochs} '
                    f'Batch {batch_idx}/{total_batches} | '
                    f'G:{loss_G.item():.3f} D:{loss_D.item():.3f} | {elapsed:.1f}s',
                    flush=True
                )

        print(f'[CycleGAN] Epoch {epoch+1}/{epochs} | G:{loss_G.item():.3f} D:{loss_D.item():.3f} | {time.time()-t0:.1f}s')
        torch.save(
            {'G_d2b': G_d2b.state_dict(), 'G_b2d': G_b2d.state_dict()},
            f'saved/cyclegan_celeba_{tag}_epoch{epoch+1}.pt'
        )

    torch.save({'G_d2b': G_d2b.state_dict(), 'G_b2d': G_b2d.state_dict()},
               f'saved/cyclegan_celeba_{tag}.pt')

    # save translation grid
    _save_cyclegan_grid(G_d2b, G_b2d, dark_loader, blonde_loader, tag)
    return G_d2b, G_b2d

def _save_cyclegan_grid(G_d2b, G_b2d, dark_loader, blonde_loader, tag):
    G_d2b.eval(); G_b2d.eval()
    real_d = next(iter(dark_loader))[0][:4].to(device)
    real_b = next(iter(blonde_loader))[0][:4].to(device)
    with torch.no_grad():
        fake_b = G_d2b(real_d)
        fake_d = G_b2d(real_b)
    imgs = torch.cat([real_d, fake_b, real_b, fake_d], dim=0).cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    plt.figure(figsize=(12,6)); plt.imshow(grid.permute(1,2,0)); plt.axis('off')
    plt.title(f'CycleGAN ({tag}): real dark | →blonde | real blonde | →dark')
    plt.savefig(f'outputs/cyclegan_grid_{tag}.png', bbox_inches='tight'); plt.close()
    print(f'Saved: outputs/cyclegan_grid_{tag}.png')
    G_d2b.train(); G_b2d.train()

# ── Train DDPM ────────────────────────────────────────────────
def train_ddpm(epochs=20, schedule='linear'):
    loader = get_mnist_loader()
    unet   = SimpleUNet().to(device)
    opt    = torch.optim.Adam(unet.parameters(), lr=2e-4)
    C      = get_ddpm_constants(schedule)
    T      = 1000

    for epoch in range(epochs):
        t0 = time.time()
        losses = []
        for x0, _ in loader:
            x0   = x0.to(device)
            B    = x0.size(0)
            t    = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt   = C['sqrt_ab'][t][:,None,None,None]*x0 + C['sqrt_one_minus_ab'][t][:,None,None,None]*noise
            loss = F.mse_loss(unet(xt, t), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f'[DDPM-{schedule}] Epoch {epoch+1}/{epochs} | Loss:{np.mean(losses):.4f} | {time.time()-t0:.1f}s')

    torch.save(unet.state_dict(), f'saved/ddpm_mnist_{schedule}.pt')
    print(f'Saved: saved/ddpm_mnist_{schedule}.pt')
    return unet

# ── Generate samples ─────────────────────────────────────────
# ── DDPM sampling ─────────────────────────────────────────────
@torch.no_grad()
def ddpm_generate(unet, C, n=64, save_path='outputs/ddpm_grid.png', trajectory=False):
    unet.eval()
    T   = 1000
    x   = torch.randn(n, 1, 28, 28).to(device)
    snapshots = []
    show_at = {999,800,600,400,200,100,50,0}

    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        pred    = unet(x, t_batch)
        coeff   = C['betas'][t] / C['sqrt_one_minus_ab'][t]
        mean    = C['sqrt_recip_a'][t] * (x - coeff * pred)
        x = mean if t==0 else mean + torch.sqrt(C['posterior_var'][t])*torch.randn_like(x)
        if trajectory and t in show_at:
            snapshots.append((t, x[:8].cpu().clone()))

    grid = torchvision.utils.make_grid(x.cpu(), nrow=8, normalize=True)
    plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f'Saved: {save_path}')

    if trajectory and snapshots:
        fig, axes = plt.subplots(8, len(snapshots), figsize=(len(snapshots)*1.5, 12))
        for col,(t,imgs) in enumerate(snapshots):
            for row in range(8):
                axes[row][col].imshow(imgs[row].squeeze(), cmap='gray')
                axes[row][col].axis('off')
                if row==0: axes[row][col].set_title(f't={t}', fontsize=8)
        plt.suptitle('Reverse Diffusion: Noise → Digit')
        plt.tight_layout()
        plt.savefig('outputs/ddpm_trajectory.png', bbox_inches='tight'); plt.close()
        print('Saved: outputs/ddpm_trajectory.png')

# ── Exercise 1: Mode collapse check ──────────────────────────
def check_mode_collapse(weights_path, collapsed=False):
    from torchvision.models import resnet18
    Z_DIM = 100
    G = GANGenerator().to(device)
    G.load_state_dict(torch.load(weights_path, map_location=device))
    G.eval()

    # load small MNIST classifier (resnet18 fine-tuned or use simple CNN)
    class MNISTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64*7*7, 128), nn.ReLU(),
                nn.Linear(128, 10)
            )
        def forward(self, x): return self.net(x)

    clf = MNISTClassifier().to(device)
    # train classifier quickly on real MNIST
    clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loader  = get_mnist_loader(batch_size=256)
    print('Training MNIST classifier...')
    for _ in range(3):  # 3 epochs is enough
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(clf(imgs), labels)
            clf_opt.zero_grad(); loss.backward(); clf_opt.step()
    clf.eval()

    # generate 1000 images and classify
    counts = torch.zeros(10)
    with torch.no_grad():
        for _ in range(16):  # 16 × 64 = 1024 ≈ 1000
            z    = torch.randn(64, Z_DIM).to(device)
            fake = G(z).view(-1,1,28,28)
            preds = clf(fake).argmax(dim=1).cpu()
            for p in preds: counts[p] += 1
    counts = counts[:1000]  # trim to 1000

    tag = 'collapsed' if collapsed else 'normal'
    print(f'\nMode collapse check ({tag}):')
    print('Digit  | ' + ' | '.join(str(i) for i in range(10)))
    print('Count  | ' + ' | '.join(f'{int(counts[i]):4d}' for i in range(10)))

    plt.figure(figsize=(8,4))
    plt.bar(range(10), counts.numpy(), color='steelblue')
    plt.xticks(range(10)); plt.xlabel('Digit'); plt.ylabel('Count (out of 1000)')
    plt.title(f'GAN Mode Coverage ({tag})')
    plt.savefig(f'outputs/mode_collapse_histogram_{tag}.png', bbox_inches='tight'); plt.close()
    print(f'Saved: outputs/mode_collapse_histogram_{tag}.png')

# ── Exercise 3: Test your own face ───────────────────────────
def test_own_face(image_path, weights_path):
    G_d2b = CycleGenerator().to(device)
    G_b2d = CycleGenerator().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    G_d2b.load_state_dict(ckpt['G_d2b'])
    G_b2d.load_state_dict(ckpt['G_b2d'])
    G_d2b.eval(); G_b2d.eval()

    transform = transforms.Compose([
        transforms.CenterCrop(min(Image.open(image_path).size)),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img   = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        dark_to_blonde = G_d2b(img)
        blonde_to_dark = G_b2d(img)

    def to_pil(t):
        t = (t.squeeze().cpu() * 0.5 + 0.5).clamp(0,1)
        return transforms.ToPILImage()(t)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Original', 'Dark → Blonde', 'Blonde → Dark']
    imgs   = [img.squeeze().cpu(), dark_to_blonde.squeeze().cpu(), blonde_to_dark.squeeze().cpu()]
    for ax, title, im in zip(axes, titles, imgs):
        im = (im * 0.5 + 0.5).clamp(0,1).permute(1,2,0).numpy()
        ax.imshow(im); ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/my_face_result.png', bbox_inches='tight'); plt.close()
    print('Saved: outputs/my_face_result.png')

# ── Exercise 4: Noise schedule comparison plot ───────────────
def plot_noise_schedules():
    T  = 1000
    C_lin = get_ddpm_constants('linear')
    C_cos = get_ddpm_constants('cosine')
    plt.figure(figsize=(8,4))
    plt.plot(C_lin['alpha_bar'].cpu(), label='Linear', color='steelblue')
    plt.plot(C_cos['alpha_bar'].cpu(), label='Cosine', color='orange')
    plt.xlabel('Timestep t'); plt.ylabel('ᾱ_t (signal retained)')
    plt.title('Noise Schedule Comparison'); plt.legend(); plt.grid(True)
    plt.savefig('outputs/noise_schedule_comparison.png', bbox_inches='tight'); plt.close()
    print('Saved: outputs/noise_schedule_comparison.png')


# Main function
# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()

    # ── GAN ──
    if args.model == 'gan':
        if args.train:
            train_gan(epochs=args.epochs)

        if args.mode_collapse_check:
            w = args.weights or 'saved/gan_mnist.pt'
            # normal check
            check_mode_collapse(w, collapsed=False)
            # collapsed check (3x D lr)
            print('\nNow training collapsed GAN (3x D lr)...')
            train_gan(epochs=args.epochs, d_lr=6e-4)
            # save collapsed weights separately
            # (train_gan always saves to gan_mnist.pt, rename it)
            os.rename('saved/gan_mnist.pt', 'saved/gan_mnist_collapsed.pt')
            check_mode_collapse('saved/gan_mnist_collapsed.pt', collapsed=True)

    # ── CycleGAN ──
    elif args.model == 'cyclegan':
        if args.train:
            G_d2b, G_b2d = train_cyclegan(
                epochs=args.epochs,
                lambda_cyc=args.lambda_cyc,
                subset_size=args.celeba_subset,
                batch_size=args.batch_size,
                log_every=args.log_every,
                num_workers=args.num_workers,
                use_amp=not args.no_amp
            )
            # Ex 2: also train with lambda_cyc=0 for ablation
            if args.lambda_cyc != 0 and not args.skip_ablation:
                print('\nTraining CycleGAN ablation (lambda_cyc=0)...')
                train_cyclegan(
                    epochs=10,
                    lambda_cyc=0.0,
                    subset_size=args.celeba_subset,
                    batch_size=args.batch_size,
                    log_every=args.log_every,
                    num_workers=args.num_workers,
                    use_amp=not args.no_amp
                )

        if args.test_image:
            w = args.weights or 'saved/cyclegan_celeba_cyc10.pt'
            test_own_face(args.test_image, w)

    # ── DDPM ──
    elif args.model == 'ddpm':
        if args.train:
            unet = train_ddpm(epochs=args.epochs, schedule=args.schedule)
            C    = get_ddpm_constants(args.schedule)
            ddpm_generate(unet, C,
                          save_path=f'outputs/ddpm_{args.schedule}_grid.png',
                          trajectory=True)
            plot_noise_schedules()

        if args.generate:
            unet = SimpleUNet().to(device)
            w    = args.weights or f'saved/ddpm_mnist_{args.schedule}.pt'
            unet.load_state_dict(torch.load(w, map_location=device))
            C    = get_ddpm_constants(args.schedule)
            ddpm_generate(unet, C, n=args.n,
                          save_path=f'outputs/ddpm_{args.schedule}_grid.png',
                          trajectory=True)
