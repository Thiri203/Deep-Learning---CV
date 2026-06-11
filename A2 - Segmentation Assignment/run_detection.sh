#!/bin/bash
# A2-01 Object Detection — Full run script
# Run from: D:/Deep-Learning---CV/A2 - Segmentation Assignment
# Usage: bash run_all.sh

set -e  # stop on first error

echo "================================================"
echo " A2-01 YOLOv4 — Full Pipeline"
echo "================================================"

# ── STEP 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/6] Installing dependencies..."
py -m pip install torch torchvision albumentations opencv-python-headless fiftyone tqdm pycocotools

# ── STEP 2: YOLOv3 inference ─────────────────────────────────
echo ""
echo "[2/6] YOLOv3 inference on dog-cycle-car.png..."
py run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer
echo "Result saved: detection_yolov3_dog-cycle-car.jpg"

# ── STEP 3: YOLOv4 inference ─────────────────────────────────
echo ""
echo "[3/6] YOLOv4 inference on dog-cycle-car.png..."
py run.py --model yolov4 --weights yolov4.weights --image dog-cycle-car.png --infer
echo "Result saved: detection_yolov4_dog-cycle-car.jpg"

# ── STEP 4: Train YOLOv4 with IoU loss ───────────────────────
echo ""
echo "[4/6] Training YOLOv4 with IoU loss (5 epochs)..."
echo "This will take a while. Saved to: yolov4_iou_loss.pt"
py run.py --model yolov4 --weights yolov4.weights --dataset coco --epochs 5 --loss iou --train

# ── STEP 5: Train YOLOv4 with CIoU loss ──────────────────────
echo ""
echo "[5/6] Training YOLOv4 with CIoU loss (5 epochs)..."
echo "This will take a while. Saved to: yolov4_ciou_loss.pt"
py run.py --model yolov4 --weights yolov4.weights --dataset coco --epochs 5 --loss ciou --train

# ── STEP 6: Evaluate both models ─────────────────────────────
echo ""
echo "[6/6] Evaluating IoU model..."
py run.py --model yolov4 --weights yolov4_iou_loss.pt --dataset coco --evaluate

echo ""
echo "Evaluating CIoU model..."
py run.py --model yolov4 --weights yolov4_ciou_loss.pt --dataset coco --evaluate

echo ""
echo "================================================"
echo " All done! Fill in README.md with the mAP results."
echo "================================================"
