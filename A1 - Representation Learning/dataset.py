import torchvision
import torchvision.transforms as transforms
import torch


def get_dataloaders(dataset='cifar10', batch_size=64, img_size=32):
    """
    Returns train, val, test dataloaders for the given dataset.
    img_size=32  -> for ResNet, ViTSmall (native CIFAR-10 size)
    img_size=224 -> for AlexNet, GoogLeNet, pretrained models
    """

    if img_size == 224:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform)
        test_data = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f'Dataset {dataset} not supported. Use cifar10.')

    # Split train into 80% train / 20% val
    train_data, val_data = torch.utils.data.random_split(train_data, [40000, 10000])

    # Use same test transform for val
    val_data.dataset.transform = test_transform

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train, val, test = get_dataloaders('cifar10', batch_size=64, img_size=32)
    print(f'Train batches: {len(train)}')
    print(f'Val batches:   {len(val)}')
    print(f'Test batches:  {len(test)}')
    print('dataset.py OK!')