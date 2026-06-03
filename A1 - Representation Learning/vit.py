import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)           # (N, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (N, embed_dim, n_patches)
        return x.transpose(1, 2)   # (N, n_patches, embed_dim)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: LN -> MSA -> residual, LN -> MLP -> residual."""
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViTSmall(nn.Module):
    """
    Small Vision Transformer for CIFAR-10 (32x32 images).
    Uses 4x4 patches -> 64 patches per image.
    Reference: Dosovitskiy et al. (2020)
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=128, depth=6, n_heads=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout   = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.ln   = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        N = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(N, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.dropout(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.ln(x[:, 0])
        return self.head(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTSmall(num_classes=10).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    out = model(torch.randn(2, 3, 32, 32).to(device))
    print(f'Output shape: {out.shape}')
    print('ViTSmall OK!')