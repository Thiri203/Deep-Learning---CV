import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import timm


class CIFAR10MAE(Dataset):
    def __init__(self, root='./data', train=True):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train,
                                                     download=False, transform=tf)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                      img_size=32, patch_size=4, num_classes=0)
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        return self.vit(x)


class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=192, patch_size=4, n_patches=64):
        super().__init__()
        patch_dim = 3 * patch_size * patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, n_patches * patch_dim)
        )
        self.n_patches = n_patches
        self.patch_dim = patch_dim

    def forward(self, x):
        out = self.decoder(x)
        return out.view(out.shape[0], self.n_patches, self.patch_dim)


class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75, patch_size=4, image_size=32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        self.n_patches = n_patches
        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder(embed_dim=self.encoder.embed_dim,
                                   patch_size=patch_size, n_patches=n_patches)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, h * w, C * p * p)
        return x

    def forward(self, imgs):
        B = imgs.shape[0]
        patches = self.patchify(imgs)
        n_mask = int(self.n_patches * self.mask_ratio)
        noise = torch.rand(B, self.n_patches, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask_ids = ids_shuffle[:, :n_mask]
        features = self.encoder(imgs)
        pred = self.decoder(features)
        loss = 0
        for b in range(B):
            loss += F.mse_loss(pred[b][mask_ids[b]], patches[b][mask_ids[b]])
        loss = loss / B
        return loss, pred, patches, mask_ids