#!/bin/bash
set -e

echo "=== Installing dependencies ==="
py -m pip install timm scikit-learn

echo "=== Creating folders ==="
mkdir -p models saved figures
type nul > models/__init__.py 2>/dev/null || touch models/__init__.py

echo "=== [1/9] SimCLR Train ==="
py run.py --model simclr --epochs 10 --train

echo "=== [2/9] SimCLR Linear Eval ==="
py run.py --model simclr --evaluate --linear

echo "=== [3/9] DINO Train (default) ==="
py run.py --model dino --epochs 10 --train

echo "=== [4/9] DINO Linear Eval + Attention ==="
py run.py --model dino --evaluate --linear --attention

echo "=== [5/9] DINO Ablation: No Centering ==="
py run.py --model dino --epochs 5 --no-centering --train
py run.py --model dino --no-centering --evaluate --linear --attention

echo "=== [6/9] DINO Ablation: No Local Crops ==="
py run.py --model dino --epochs 5 --n-local 0 --train
py run.py --model dino --n-local 0 --evaluate --linear --attention

echo "=== [7/9] MAE mask=0.25 ==="
py run.py --model mae --mask-ratio 0.25 --epochs 5 --train
py run.py --model mae --mask-ratio 0.25 --evaluate --linear

echo "=== [8/9] MAE mask=0.50 ==="
py run.py --model mae --mask-ratio 0.50 --epochs 5 --train
py run.py --model mae --mask-ratio 0.50 --evaluate --linear

echo "=== [9/9] MAE mask=0.75 ==="
py run.py --model mae --mask-ratio 0.75 --epochs 5 --train
py run.py --model mae --mask-ratio 0.75 --evaluate --linear

echo ""
echo "=== ALL DONE. Check figures/ for plots and saved/ for checkpoints. ==="