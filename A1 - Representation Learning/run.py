import argparse
import torch
import torch.nn as nn
import torchvision

from alexnet import AlexNet
from googlenet import GoogLeNet
from resnet import ResNet18
from vit import ViTSmall
from dataset import get_dataloaders
from train import train_model, test_model


def get_model(model_name, device):
    if model_name == 'alexnet':
        model = AlexNet(num_classes=10)
        img_size = 224
        is_inception = False
    elif model_name == 'googlenet':
        model = GoogLeNet(num_classes=10)
        img_size = 224
        is_inception = True
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=10)
        img_size = 32
        is_inception = False
    elif model_name == 'vit_small':
        model = ViTSmall(num_classes=10)
        img_size = 32
        is_inception = False
    elif model_name == 'resnet18_pretrained':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(512, 10)
        img_size = 32
        is_inception = False
    elif model_name == 'vit_b16_pretrained':
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        model.heads = nn.Linear(768, 10)
        img_size = 224
        is_inception = False
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {model_name} | Parameters: {total_params:,} | img_size: {img_size}')
    return model, img_size, is_inception


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['alexnet','googlenet','resnet18','vit_small',
                                 'resnet18_pretrained','vit_b16_pretrained'])
    parser.add_argument('--dataset',    type=str,   default='cifar10')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--weights',    type=str,   default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, img_size, is_inception = get_model(args.model, device)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, img_size=img_size)

    criterion = nn.CrossEntropyLoss()

    if args.train:
        if args.model == 'resnet18_pretrained':
            print('\n--- Stage 1: FC head only (5 epochs) ---')
            for param in model.parameters():
                param.requires_grad = False
            model.fc.requires_grad_(True)
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
            train_model(model, train_loader, val_loader, criterion, optimizer,
                        num_epochs=5, weights_name=f'{args.model}_stage1')

            print('\n--- Stage 2: Fine-tune all layers ---')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            model, val_acc, loss_hist, epoch_times = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=args.epochs - 5, weights_name=args.model)

        elif args.model == 'vit_b16_pretrained':
            print('\n--- Stage 1: Head only (5 epochs) ---')
            for param in model.parameters():
                param.requires_grad = False
            model.heads.requires_grad_(True)
            optimizer = torch.optim.Adam(model.heads.parameters(), lr=args.lr)
            train_model(model, train_loader, val_loader, criterion, optimizer,
                        num_epochs=5, weights_name=f'{args.model}_stage1')

            print('\n--- Stage 2: Fine-tune all layers ---')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            model, val_acc, loss_hist, epoch_times = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=args.epochs - 5, weights_name=args.model)

        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=5e-4)
            model, val_acc, loss_hist, epoch_times = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                num_epochs=args.epochs, weights_name=args.model,
                is_inception=is_inception)

        avg_time = sum(epoch_times) / len(epoch_times)
        print(f'\nAverage time per epoch: {avg_time:.1f}s')

    if args.test:
        if args.weights:
            print(f'Loading weights from {args.weights}')
            model.load_state_dict(torch.load(args.weights, map_location=device))
        test_model(model, test_loader)


if __name__ == '__main__':
    main()
