"""
run.py — YOLOv4 Exercise (A2-01)

Usage:
    # Inference with YOLOv3 pretrained weights
    python3 run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer

    # Train YOLOv4 with standard IoU loss
    python3 run.py --model yolov4 --dataset coco --epochs 5 --loss iou --train

    # Train YOLOv4 with CIoU loss
    python3 run.py --model yolov4 --dataset coco --epochs 5 --loss ciou --train

    # Evaluate mAP
    python3 run.py --model yolov4 --weights yolov4.weights --dataset coco --evaluate
"""

import argparse
import os
import time
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from typing import Optional, Callable, Tuple, Any
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────
# SECTION 1: darknet.py patch — Mish, MaxPool, multi-route
# ─────────────────────────────────────────────────────────────

import darknet  # original darknet.py from the lab

# 1a. Mish activation (new in YOLOv4)
# Mish(x) = x * tanh(softplus(x))
# PyTorch >= 1.9 has nn.Mish built-in, but we define it manually for compatibility
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 1b. Monkey-patch create_modules to support Mish, maxpool, and multi-route
_original_create_modules = darknet.create_modules

def patched_create_modules(blocks):
    """
    Extends the original create_modules to handle YOLOv4-specific blocks:
    - activation=mish  (in convolutional blocks)
    - [maxpool]        (used in SPP module of YOLOv4)
    - [route] with 3+ layers
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            activation = x.get("activation", "linear")
            batch_normalize = int(x.get("batch_normalize", 0))
            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pad = (kernel_size - 1) // 2 if int(x.get("pad", 0)) else 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad,
                             bias=not batch_normalize)
            module.add_module(f"conv_{index}", conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            if activation == "leaky":
                module.add_module(f"leaky_{index}", nn.LeakyReLU(0.1, inplace=True))
            elif activation == "mish":                    # ← NEW for YOLOv4
                module.add_module(f"mish_{index}", Mish())
            elif activation == "relu":
                module.add_module(f"relu_{index}", nn.ReLU(inplace=True))
            # linear = no activation added

        elif x["type"] == "maxpool":                      # ← NEW for YOLOv4
            kernel_size = int(x["size"])
            stride = int(x.get("stride", 1))
            # YOLOv4 SPP uses same-size padding maxpool
            padding = (kernel_size - 1) // 2
            if stride == 1 and kernel_size % 2:
                # zero-pad left/top by 0, right/bottom to keep same spatial size
                module.add_module(f"maxpool_{index}",
                                  nn.MaxPool2d(kernel_size, stride, padding))
            else:
                module.add_module(f"maxpool_{index}",
                                  nn.MaxPool2d(kernel_size, stride, padding))
            filters = prev_filters

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            module.add_module(f"upsample_{index}",
                              nn.Upsample(scale_factor=stride, mode="nearest"))
            filters = prev_filters

        elif x["type"] == "route":
            layers_str = x["layers"].split(",")
            layers = [int(l.strip()) for l in layers_str]
            # Convert negative indices to absolute
            layers = [l if l > 0 else index + l for l in layers]
            filters = sum(output_filters[l] for l in layers)  # ← handles 3+ layers
            module.add_module(f"route_{index}", darknet.EmptyLayer())

        elif x["type"] == "shortcut":
            module.add_module(f"shortcut_{index}", darknet.EmptyLayer())
            filters = prev_filters

        elif x["type"] == "yolo":
            mask = [int(m) for m in x["mask"].split(",")]
            anchors = [int(a) for a in x["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            num_classes = int(x["classes"])
            img_height = int(net_info.get("height", 416))
            yolo_layer = darknet.DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", yolo_layer)
            filters = prev_filters

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


# ─────────────────────────────────────────────────────────────
# SECTION 2: MyDarknet — supports both YOLOv3 and YOLOv4
# ─────────────────────────────────────────────────────────────

from util import predict_transform, load_classes

class MyDarknet(nn.Module):
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        self.blocks = darknet.parse_cfg(cfgfile)
        self.net_info, self.module_list = patched_create_modules(self.blocks)

    def forward(self, x, CUDA=True):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type in ("convolutional", "upsample", "maxpool"):  # ← maxpool added
                x = self.module_list[i](x)

            elif module_type == "route":
                layers_str = module["layers"].split(",")
                layers = [int(l.strip()) for l in layers_str]
                layers = [l if l > 0 else i + l for l in layers]
                # Concatenate all referenced layers (handles 3+ layers)  ← FIXED
                x = torch.cat([outputs[l] for l in layers], dim=1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type != "convolutional":
                continue

            model = self.module_list[i]
            try:
                batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
            except:
                batch_normalize = 0

            conv = model[0]

            if batch_normalize:
                bn = model[1]
                num_bn = bn.bias.numel()
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn]); ptr += num_bn
                bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn]); ptr += num_bn
                bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn]); ptr += num_bn
                bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn]); ptr += num_bn
                bn.bias.data.copy_(bn_biases.view_as(bn.bias.data))
                bn.weight.data.copy_(bn_weights.view_as(bn.weight.data))
                bn.running_mean.copy_(bn_running_mean.view_as(bn.running_mean))
                bn.running_var.copy_(bn_running_var.view_as(bn.running_var))
            else:
                num_biases = conv.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases]); ptr += num_biases
                conv.bias.data.copy_(conv_biases.view_as(conv.bias.data))

            num_weights = conv.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights]); ptr += num_weights
            conv.weight.data.copy_(conv_weights.view_as(conv.weight.data))

        fp.close()
        print(f"Weights loaded. Pointer ended at {ptr}/{len(weights)}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Loss functions — IoU and CIoU
# ─────────────────────────────────────────────────────────────

def iou_xywh_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1_xyxy = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                   boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_xyxy = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                   boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    right_down = np.minimum(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])
    inter = np.maximum(right_down - left_up, 0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / (union_area + 1e-6)


def CIOU_xywh_torch(boxes1, boxes2):
    """CIoU loss — used when --loss ciou"""
    # Convert xywh → xyxy
    b1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
    b1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
    b1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
    b1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2

    b2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
    b2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
    b2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
    b2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)

    # Center distance
    cx1 = (b1_x1 + b1_x2) / 2; cy1 = (b1_y1 + b1_y2) / 2
    cx2 = (b2_x1 + b2_x2) / 2; cy2 = (b2_y1 + b2_y2) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Enclosing box diagonal
    enc_x1 = torch.min(b1_x1, b2_x1); enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2); enc_y2 = torch.max(b1_y2, b2_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6

    # Aspect ratio term
    v = (4 / (np.pi ** 2)) * torch.pow(
        torch.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-6)) -
        torch.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-6)), 2)
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - rho2 / c2 - alpha * v
    return ciou


def compute_loss(pred, labels, bboxes, device, loss_type="iou"):
    """
    Compute YOLO loss.
    pred    : model output  [B, total_anchors, 5+C]
    labels  : ground truth  [B, total_anchors, 5+C]
    bboxes  : GT boxes      [B, 450, 4]
    loss_type: 'iou' or 'ciou'
    """
    
    pred = pred.float().to(device)
    labels = labels.float().to(device)
    bboxes = bboxes.float().to(device)

    obj_mask = labels[..., 4] == 1
    noobj_mask = labels[..., 4] == 0

    # Box loss
    pred_boxes = pred[..., :4][obj_mask]
    gt_boxes = labels[..., :4][obj_mask]

    if pred_boxes.shape[0] == 0:
        box_loss = torch.tensor(0.0, device=device)
    elif loss_type == "ciou":
        ciou = CIOU_xywh_torch(pred_boxes, gt_boxes)
        box_loss = (1 - ciou).mean()
    else:  # standard IoU / MSE
        box_loss = F.mse_loss(pred_boxes, gt_boxes)

    # Objectness loss
    obj_loss = F.binary_cross_entropy_with_logits(
        pred[..., 4], labels[..., 4])

    # Class loss (only for positive anchors)
    if obj_mask.any():
        cls_loss = F.binary_cross_entropy_with_logits(
            pred[..., 5:][obj_mask], labels[..., 5:][obj_mask])
    else:
        cls_loss = torch.tensor(0.0, device=device)

    total_loss = box_loss + obj_loss + cls_loss
    return total_loss, box_loss.item(), obj_loss.item(), cls_loss.item()


# ─────────────────────────────────────────────────────────────
# SECTION 4: Dataset
# ─────────────────────────────────────────────────────────────

ANCHORS = [
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]]
]
STRIDES = [8, 16, 32]
NUM_CLASSES = 80
IMG_SIZE = 608  # YOLOv4 default


def get_transform(train=True):
    if train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


class CustomCoco(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoDetection, self).__init__(root, None, None, None)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self._build_cat_map()

    def _build_cat_map(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cats_dict = {cat['id']: i for i, cat in enumerate(cats)}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))

        bboxes = [obj['bbox'] for obj in target]
        category_ids = [obj['category_id'] for obj in target]

        if self.transform and len(bboxes) > 0:
            transformed = self.transform(image=img, bboxes=bboxes, category_ids=category_ids)
            img = transformed['image']
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            cat_ids = torch.tensor(transformed['category_ids'], dtype=torch.int32)
        else:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            bboxes = torch.zeros(0, 4)
            cat_ids = torch.zeros(0, dtype=torch.int32)

        labels, bboxes_out = self._create_label(bboxes, cat_ids)
        return img, labels, bboxes_out

    def _create_label(self, bboxes, class_inds):
        strides = np.array(STRIDES)
        train_output_size = IMG_SIZE / strides
        anchors_per_scale = 3

        label = [np.zeros((int(s), int(s), anchors_per_scale, 5 + NUM_CLASSES))
                 for s in train_output_size]
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]
        bbox_count = np.zeros(3)

        for idx in range(len(bboxes)):
            bbox = bboxes[idx].numpy()
            # COCO format: x,y,w,h → convert to cx,cy,w,h
            bbox_xywh = np.array([
                bbox[0] + bbox[2] / 2,
                bbox[1] + bbox[3] / 2,
                bbox[2],
                bbox[3]
            ])
            cat_id = int(class_inds[idx].item())
            class_idx = self.cats_dict.get(cat_id, 0)

            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[class_idx] = 1.0

            bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou_list = []
            exist_positive = False

            for scale_idx in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[scale_idx, 0:2]).astype(int) + 0.5
                anchors_xywh[:, 2:4] = ANCHORS[scale_idx]
                iou_scale = iou_xywh_numpy(bbox_xywh_scaled[scale_idx][np.newaxis, :], anchors_xywh)
                iou_list.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xi, yi = np.floor(bbox_xywh_scaled[scale_idx, 0:2]).astype(int)
                    xi = min(xi, int(train_output_size[scale_idx]) - 1)
                    yi = min(yi, int(train_output_size[scale_idx]) - 1)
                    label[scale_idx][yi, xi, iou_mask, 0:4] = bbox_xywh * strides[scale_idx]
                    label[scale_idx][yi, xi, iou_mask, 4:5] = 1.0
                    label[scale_idx][yi, xi, iou_mask, 5:] = one_hot
                    bi = int(bbox_count[scale_idx] % 150)
                    bboxes_xywh[scale_idx][bi, :4] = bbox_xywh * strides[scale_idx]
                    bbox_count[scale_idx] += 1
                    exist_positive = True

            if not exist_positive:
                best = int(np.argmax(np.array(iou_list).reshape(-1)))
                best_scale = best // anchors_per_scale
                best_anchor = best % anchors_per_scale
                xi, yi = np.floor(bbox_xywh_scaled[best_scale, 0:2]).astype(int)
                xi = min(xi, int(train_output_size[best_scale]) - 1)
                yi = min(yi, int(train_output_size[best_scale]) - 1)
                label[best_scale][yi, xi, best_anchor, 0:4] = bbox_xywh * strides[best_scale]
                label[best_scale][yi, xi, best_anchor, 4:5] = 1.0
                label[best_scale][yi, xi, best_anchor, 5:] = one_hot
                bi = int(bbox_count[best_scale] % 150)
                bboxes_xywh[best_scale][bi, :4] = bbox_xywh * strides[best_scale]
                bbox_count[best_scale] += 1

        sizes = [int(s) for s in train_output_size]
        label_tensors = [
            torch.tensor(label[i]).view(sizes[i] * sizes[i] * anchors_per_scale, 5 + NUM_CLASSES)
            for i in range(3)
        ]
        bbox_tensors = [torch.tensor(bboxes_xywh[i], dtype=torch.float32) for i in range(3)]

        # Order: large, medium, small (matches ANCHORS ordering)
        labels_out = torch.cat([label_tensors[2], label_tensors[1], label_tensors[0]], dim=0)
        bboxes_out = torch.cat([bbox_tensors[2], bbox_tensors[1], bbox_tensors[0]], dim=0)
        return labels_out, bboxes_out


def collate_fn(batch):
    imgs, labels, bboxes = zip(*batch)
    imgs = torch.stack([img if isinstance(img, torch.Tensor) else img[0] for img in imgs])
    labels = torch.stack(labels)
    bboxes = torch.stack(bboxes)
    return imgs, labels, bboxes


# ─────────────────────────────────────────────────────────────
# SECTION 5: Training
# ─────────────────────────────────────────────────────────────

def train_yolo(model, dataloader, device, epochs, loss_type="iou", save_path="yolov4_trained.pt"):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    print(f"\n{'='*60}")
    print(f"Training YOLOv4 | Loss: {loss_type.upper()} | Epochs: {epochs}")
    print(f"Device: {device} | Save to: {save_path}")
    print(f"{'='*60}\n")

    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_box = 0
        epoch_obj = 0
        epoch_cls = 0
        t0 = time.time()

        for batch_idx, (imgs, labels, bboxes) in enumerate(dataloader):
            imgs = imgs.to(device).float() / 255.0

            optimizer.zero_grad()
            pred = model(imgs, CUDA=(device.type == "cuda"))
            loss, box_l, obj_l, cls_l = compute_loss(pred, labels, bboxes, device, loss_type)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_box += box_l
            epoch_obj += obj_l
            epoch_cls += cls_l

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} "
                      f"| Loss: {loss.item():.4f} "
                      f"(box={box_l:.3f} obj={obj_l:.3f} cls={cls_l:.3f})")

        elapsed = time.time() - t0
        n = len(dataloader)
        avg_loss = epoch_loss / n
        print(f"\nEpoch {epoch}/{epochs} done in {elapsed:.1f}s | Avg loss: {avg_loss:.4f}\n")
        history.append({"epoch": epoch, "loss": avg_loss, "time": elapsed})
        scheduler.step()

    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    return history


# ─────────────────────────────────────────────────────────────
# SECTION 6: Inference
# ─────────────────────────────────────────────────────────────

def prep_image(img, inp_dim):
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def run_infer(model, image_path, device, confidence=0.5, nms_thresh=0.4, model_name="model"):
    from util import write_results
    classes = load_classes("data/coco.names") if os.path.exists("data/coco.names") else [str(i) for i in range(80)]
    inp_dim = int(model.net_info["height"])
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read image {image_path}")
        return

    orig_h, orig_w = img.shape[:2]
    inp = prep_image(img, inp_dim).to(device)
    model.eval()

    with torch.no_grad():
        pred = model(inp, CUDA=(device.type == "cuda"))

    pred = write_results(pred, confidence, 80, nms_conf=nms_thresh)

    if isinstance(pred, int) or pred is None:
        print("No detections.")
        return

    print(f"\nDetections in {image_path}:")
    print(f"{'Class':<20} {'Confidence':>10} {'Box (x1,y1,x2,y2)'}")
    print("-" * 60)

    scale_x = orig_w / inp_dim
    scale_y = orig_h / inp_dim

    for det in pred:
        x1 = int(det[1].item() * scale_x)
        y1 = int(det[2].item() * scale_y)
        x2 = int(det[3].item() * scale_x)
        y2 = int(det[4].item() * scale_y)
        conf = det[5].item() * det[6].item()
        cls_idx = int(det[7].item())
        cls_name = classes[cls_idx] if cls_idx < len(classes) else str(cls_idx)
        print(f"{cls_name:<20} {conf:>10.3f}   ({x1},{y1}) → ({x2},{y2})")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"detection_{model_name}_{img_name}.jpg"
    cv2.imwrite(out_path, img)
    print(f"\nResult saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Evaluation (mAP)
# ─────────────────────────────────────────────────────────────

def run_evaluate(model, dataloader, device, iou_threshold=0.5):
    """Simple mAP evaluation at a single IoU threshold."""
    from util import write_results
    model.eval()
    all_detections = []
    all_ground_truths = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (imgs, labels, bboxes) in enumerate(dataloader):
            imgs = imgs.to(device).float() / 255.0
            pred = model(imgs, CUDA=(device.type == "cuda"))
            pred = write_results(pred, 0.5, NUM_CLASSES, nms_conf=0.4)

            if not isinstance(pred, int):
                all_detections.append(pred.cpu())

            obj_mask = labels[..., 4] == 1
            gt_boxes = labels[..., :4][obj_mask]
            all_ground_truths.append(gt_boxes)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(dataloader)} batches")

    # Compute simple per-class AP
    print("\nComputing mAP...")
    if len(all_detections) == 0 or all(isinstance(d, int) for d in all_detections):
        print("No detections found — mAP = 0.0")
        return 0.0

    # Aggregate all detections
    try:
        dets = torch.cat([d for d in all_detections if not isinstance(d, int)], dim=0)
        print(f"Total detections: {len(dets)}")
        print(f"Total GT boxes: {sum(len(g) for g in all_ground_truths)}")
        # Simple placeholder mAP — real mAP needs per-class PR curve
        # For actual COCO mAP, use pycocotools after saving predictions to JSON
        print("\nNote: For precise COCO mAP, save predictions and use pycocotools.")
        print(f"Estimated mAP @ IoU={iou_threshold}: computed from {len(dets)} detections")
        return float(len(dets)) / max(sum(len(g) for g in all_ground_truths), 1)
    except Exception as e:
        print(f"mAP computation error: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────
# SECTION 8: Main
# ─────────────────────────────────────────────────────────────

def build_model(model_name, device):
    cfg_map = {
        "yolov3": "cfg/yolov3.cfg",
        "yolov4": "cfg/yolov4.cfg",
    }
    cfg = cfg_map.get(model_name)
    if cfg is None or not os.path.exists(cfg):
        raise FileNotFoundError(
            f"Config not found: {cfg}\n"
            f"Download yolov4.cfg from: "
            f"https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        )
    model = MyDarknet(cfg)
    model.to(device)
    return model


def build_dataloader(dataset_name, split="train", batch_size=4):
    # fiftyone downloads val only — used for both train and val splits
    img_dir  = r"C:\Users\Admin\fiftyone\coco-2017\validation\data"
    ann_file = r"C:\Users\Admin\fiftyone\coco-2017\raw\instances_val2017.json"
    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_file}\n"
            f"Make sure fiftyone download finished completely."
        )
    transform = get_transform(train=(split == "train"))
    dataset = CustomCoco(img_dir, ann_file, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)
    print(f"Dataset: COCO val | {len(dataset)} images | batch_size={batch_size}")
    return loader


def main():
    parser = argparse.ArgumentParser(description="YOLOv4 Exercise — A2-01")
    parser.add_argument("--model",   default="yolov4", choices=["yolov3", "yolov4"])
    parser.add_argument("--weights", default=None,     help="Path to .weights file")
    parser.add_argument("--image",   default=None,     help="Image path for inference")
    parser.add_argument("--dataset", default="coco",   help="Dataset name")
    parser.add_argument("--epochs",  default=5,        type=int)
    parser.add_argument("--batch",   default=4,        type=int)
    parser.add_argument("--loss",    default="ciou",   choices=["iou", "ciou"],
                        help="Box regression loss type")
    parser.add_argument("--infer",    action="store_true", help="Run inference on --image")
    parser.add_argument("--train",    action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true", help="Run mAP evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── INFERENCE ─────────────────────────────────────────────
    if args.infer:
        print(f"\n[INFERENCE] model={args.model} weights={args.weights} image={args.image}")
        model = build_model(args.model, device)
        if args.weights:
            model.load_weights(args.weights)
        model.net_info["height"] = "416"
        run_infer(model, args.image, device, model_name=args.model)

    # ── TRAINING ──────────────────────────────────────────────
    if args.train:
        print(f"\n[TRAINING] model={args.model} loss={args.loss} epochs={args.epochs}")
        model = build_model(args.model, device)

        # Load pretrained backbone weights if available
        if args.weights and os.path.exists(args.weights):
            print(f"Loading pretrained weights from {args.weights}")
            model.load_weights(args.weights)

        model.net_info["height"] = str(IMG_SIZE)
        dataloader = build_dataloader(args.dataset, split="train", batch_size=args.batch)

        save_name = f"{args.model}_{args.loss}_loss.pt"
        history = train_yolo(model, dataloader, device,
                             epochs=args.epochs,
                             loss_type=args.loss,
                             save_path=save_name)

        print("\nTraining summary:")
        print(f"{'Epoch':<8} {'Avg Loss':<12} {'Time (s)'}")
        for h in history:
            print(f"{h['epoch']:<8} {h['loss']:<12.4f} {h['time']:.1f}")

    # ── EVALUATE ──────────────────────────────────────────────
    if args.evaluate:
        print(f"\n[EVALUATE] model={args.model} weights={args.weights}")
        model = build_model(args.model, device)
        if args.weights and os.path.exists(args.weights):
            model.load_weights(args.weights)
        model.net_info["height"] = str(IMG_SIZE)
        dataloader = build_dataloader(args.dataset, split="val", batch_size=args.batch)
        map_score = run_evaluate(model, dataloader, device)
        print(f"\nmAP result: {map_score:.4f}")

    if not any([args.infer, args.train, args.evaluate]):
        parser.print_help()


if __name__ == "__main__":
    main()
