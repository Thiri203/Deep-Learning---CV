import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import timm


class DINOAugmentation:
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_


class CIFAR10DINO(Dataset):
    def __init__(self, root='./data', train=True, n_local=4):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
        self.augment = DINOAugmentation(n_local=n_local)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label


def dino_collate(batch):
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
    return stacked, torch.tensor(labels)


class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_dino_model(out_dim=256):
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             img_size=32, patch_size=4, num_classes=0)
    embed_dim = vit.embed_dim
    head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
    return vit, head


class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, n_crops=6, teacher_temp=0.04,
                 student_temp=0.1, center_momentum=0.9, use_centering=True):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.n_crops = n_crops
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_out = [s / self.student_temp for s in student_out]

        # Save raw teacher output BEFORE temperature for center update
        teacher_raw = teacher_out  

        if self.use_centering:
            teacher_out = [(t - self.center) / self.teacher_temp for t in teacher_out]
        else:
            teacher_out = [t / self.teacher_temp for t in teacher_out]

        teacher_probs = [F.softmax(t, dim=-1).detach() for t in teacher_out]

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_logit in enumerate(student_out):
                if s_idx == t_idx:
                    continue
                loss = -(t_prob * F.log_softmax(s_logit, dim=-1)).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center with RAW teacher output (not temperature-scaled)
        self.update_center(torch.cat(teacher_raw))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)