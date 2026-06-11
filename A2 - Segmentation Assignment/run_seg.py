"""
run_seg.py — A2-02 Image Segmentation

Usage:
    # Train with skip connections (baseline)
    py run_seg.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

    # Train without skip connections (ablation)
    py run_seg.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

    # Evaluate
    py run_seg.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms, models
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# SECTION 1: Dataset
# ─────────────────────────────────────────────────────────────

IMG_SIZE = 128
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img  = self.img_tf(img)
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)
        return img, mask


def build_dataloader(batch_size=16):
    os.makedirs('./data', exist_ok=True)
    train_raw = OxfordIIITPet('./data', split='trainval',
                               target_types='segmentation', download=True)
    test_raw  = OxfordIIITPet('./data', split='test',
                               target_types='segmentation', download=True)
    train_loader = DataLoader(PetSegDataset(train_raw, IMG_SIZE),
                              batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(PetSegDataset(test_raw, IMG_SIZE),
                              batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")
    return train_loader, test_loader, test_raw


# ─────────────────────────────────────────────────────────────
# SECTION 2: Models
# ─────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNetResNet18(nn.Module):
    """U-Net with pretrained ResNet-18 encoder — WITH skip connections."""

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1,
                                   resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1   # 64ch
        self.enc2 = resnet.layer2   # 128ch
        self.enc3 = resnet.layer3   # 256ch
        self.enc4 = resnet.layer4   # 512ch

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        x = self.up4(x);  x = self._cat(x, s4);  x = self.dec4(x)
        x = self.up3(x);  x = self._cat(x, s3);  x = self.dec3(x)
        x = self.up2(x);  x = self._cat(x, s2);  x = self.dec2(x)
        x = self.up1(x);  x = self._cat(x, s1);  x = self.dec1(x)
        x = self.up0(x);  x = self.dec0(x)
# ensure output matches input resolution
        if x.shape[2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return self.output(x)


    def _cat(self, x, skip):
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        return torch.cat([skip, x], dim=1)


class UNetResNet18NoSkip(nn.Module):
    """Same ResNet-18 encoder — skip connections REMOVED."""

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1,
                                   resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        # No skip concat — decoder input channels are halved
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 64)
        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        # No skip concat — just upsample and decode
        x = self.up4(x);  x = self.dec4(x)
        x = self.up3(x);  x = self.dec3(x)
        x = self.up2(x);  x = self.dec2(x)
        x = self.up1(x);  x = self.dec1(x)
        x = self.up0(x);  x = self.dec0(x)

        return self.output(x)


# ─────────────────────────────────────────────────────────────
# SECTION 3: Train / Evaluate helpers
# ─────────────────────────────────────────────────────────────

def compute_miou(pred, target, n_classes=3):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


def train(model, train_loader, test_loader, device, epochs, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n{'='*60}")
    print(f"Training | Epochs: {epochs} | Save to: {save_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = []

        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss.append(loss.item())

        model.eval()
        ep_iou = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                ep_iou.append(compute_miou(model(imgs.to(device)), masks.to(device)))

        scheduler.step()
        elapsed = time.time() - t0
        miou = np.mean(ep_iou)
        print(f"Epoch {epoch:02d} | Loss: {np.mean(ep_loss):.4f} "
              f"| mIoU: {miou:.4f} | Time: {elapsed:.1f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), save_path)

    print(f"\nBest mIoU: {best_miou:.4f} | Saved → {save_path}")


def evaluate(model, test_loader, device):
    model.eval()
    ep_iou = []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc='Evaluating'):
            ep_iou.append(compute_miou(model(imgs.to(device)), masks.to(device)))
    miou = np.mean(ep_iou)
    print(f"\nVal mIoU: {miou:.4f}")
    return miou


# ─────────────────────────────────────────────────────────────
# SECTION 4: Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A2-02 Image Segmentation")
    parser.add_argument("--model",   required=True,
                        choices=["unet_resnet18", "unet_resnet18_no_skip"])
    parser.add_argument("--dataset", default="oxford_pet")
    parser.add_argument("--epochs",  default=20, type=int)
    parser.add_argument("--batch",   default=16, type=int)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    if args.model == "unet_resnet18":
        model = UNetResNet18(n_classes=3, pretrained=True).to(device)
        save_path = "unet_resnet18_pet.pt"
    else:
        model = UNetResNet18NoSkip(n_classes=3, pretrained=True).to(device)
        save_path = "unet_resnet18_no_skip_pet.pt"

    # Load weights if provided
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Weights loaded from {args.weights}")

    train_loader, test_loader, _ = build_dataloader(batch_size=args.batch)

    if args.train:
        train(model, train_loader, test_loader, device, args.epochs, save_path)

    if args.evaluate:
        evaluate(model, test_loader, device)

    if not args.train and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    main()