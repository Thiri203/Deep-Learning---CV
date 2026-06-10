#!/bin/bash
# A2-02 Image Segmentation — run all
# Usage: bash run_all_seg.sh

set -e

echo "================================================"
echo " A2-02 U-Net Segmentation — Full Pipeline"
echo "================================================"

# ── STEP 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/4] Installing dependencies..."
py -m pip install torch torchvision tqdm

# ── STEP 2: Train baseline (with skip connections) ───────────
echo ""
echo "[2/4] Training unet_resnet18 (with skip connections)..."
py run_seg.py --model unet_resnet18 --dataset oxford_pet --epochs 20 --train

# ── STEP 3: Train ablation (without skip connections) ────────
echo ""
echo "[3/4] Training unet_resnet18_no_skip (no skip connections)..."
py run_seg.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# ── STEP 4: Evaluate both ────────────────────────────────────
echo ""
echo "[4/4] Evaluating baseline..."
py run_seg.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate

echo ""
echo "Evaluating no-skip model..."
py run_seg.py --model unet_resnet18_no_skip --weights unet_resnet18_no_skip_pet.pt --dataset oxford_pet --evaluate

echo ""
echo "================================================"
echo " Done! Fill in README.md with mIoU results."
echo "================================================"