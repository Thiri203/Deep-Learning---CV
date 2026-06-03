import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 1x1 -> 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 1x1 -> 5x5 (implemented as two 3x3)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3x3 pool -> 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class AuxClassifier(nn.Module):
    """Side classifier attached mid-network to fight vanishing gradients."""
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1  = nn.Linear(128 * 4 * 4, 1024)
        self.fc2  = nn.Linear(1024, num_classes)
        self.drop = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) with proper backbone and two auxiliary classifiers.
    Reference: Szegedy et al. (2014)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Backbone before first inception (matches original paper for 224x224 input)
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),   # -> 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                    # -> 64x56x56
            nn.Conv2d(64, 64, kernel_size=1),                        # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),            # -> 192x56x56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                    # -> 192x28x28
        )

        # Inception blocks
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)   # -> 256x28x28
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)   # -> 480x28x28
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)    # -> 480x14x14

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64) # -> 512x14x14
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64) # -> 512x14x14
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64) # -> 512x14x14
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64) # -> 528x14x14
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128) # -> 832x14x14
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)   # -> 832x7x7

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128) # -> 832x7x7
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128) # -> 1024x7x7

        # Two auxiliary classifiers (attached after a4 and d4)
        self.aux1 = AuxClassifier(512, num_classes)
        self.aux2 = AuxClassifier(528, num_classes)

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)

        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)

        x = self.a4(x)
        aux1_out = self.aux1(x)   # auxiliary output 1

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        aux2_out = self.aux2(x)   # auxiliary output 2

        x = self.e4(x)
        x = self.maxpool2(x)

        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        # During training return all three outputs
        # During eval return only the main output
        if self.training:
            return x, aux1_out, aux2_out
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(num_classes=10).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    model.train()
    out, aux1, aux2 = model(torch.randn(2, 3, 224, 224).to(device))
    print(f'Main output: {out.shape}, Aux1: {aux1.shape}, Aux2: {aux2.shape}')

    model.eval()
    out = model(torch.randn(2, 3, 224, 224).to(device))
    print(f'Eval output: {out.shape}')
    print('GoogLeNet OK!')