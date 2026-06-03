import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet CNN for image classification.
    Includes Local Response Normalization (LRN) as described in the original paper.

    Architecture:
        - 5 convolutional layers with ReLU activations
        - LRN after conv1 and conv2 (as in the paper)
        - 3 max pooling layers
        - 3 fully connected layers with dropout
    
    Reference:
        Krizhevsky et al. (2012) - ImageNet Classification with Deep CNNs
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: input 3x224x224 -> output 64x55x55
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN after conv1
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: -> output 192x27x27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN after conv2
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: -> output 384x13x13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: -> output 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: -> output 256x13x13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pool so any input size works (outputs 6x6)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=10).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'AlexNet total parameters: {total_params:,}')
    
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f'Output shape: {out.shape}')  # should be (2, 10)
    print('AlexNet OK!')