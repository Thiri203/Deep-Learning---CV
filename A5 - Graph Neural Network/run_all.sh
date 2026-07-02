#!/bin/bash
set -e

CELEBA_SUBSET="${CELEBA_SUBSET:-5000}"
CYCLE_BATCH_SIZE="${CYCLE_BATCH_SIZE:-32}"
CYCLE_WORKERS="${CYCLE_WORKERS:-2}"
CYCLE_LOG_EVERY="${CYCLE_LOG_EVERY:-10}"

run_if_missing() {
  local output="$1"
  local label="$2"
  shift 2

  if [ -f "$output" ]; then
    echo "===== Skipping: $label ($output exists) ====="
  else
    echo "===== $label ====="
    "$@"
  fi
}

echo "===== Assignment run started ====="

run_if_missing "saved/gan_mnist.pt" \
  "Step 1: Train GAN on MNIST" \
  uv run run.py --model gan --dataset mnist --epochs 20 --train

run_if_missing "outputs/mode_collapse_histogram_collapsed.png" \
  "Step 2: Mode Collapse Check" \
  uv run run.py --model gan --weights saved/gan_mnist.pt --mode-collapse-check

run_if_missing "saved/ddpm_mnist_linear.pt" \
  "Step 3: Train DDPM linear" \
  uv run run.py --model ddpm --dataset mnist --epochs 20 --schedule linear --train

run_if_missing "saved/ddpm_mnist_cosine.pt" \
  "Step 4: Train DDPM cosine" \
  uv run run.py --model ddpm --dataset mnist --epochs 20 --schedule cosine --train

run_if_missing "outputs/ddpm_linear_grid.png" \
  "Step 5: Generate DDPM linear samples" \
  uv run run.py --model ddpm --schedule linear --weights saved/ddpm_mnist_linear.pt --generate --n 64

run_if_missing "saved/cyclegan_celeba_cyc10.pt" \
  "Step 6: Train CycleGAN on CelebA, lambda_cyc=10" \
  uv run run.py --model cyclegan --dataset celeba --epochs 20 \
    --celeba-subset "$CELEBA_SUBSET" \
    --batch-size "$CYCLE_BATCH_SIZE" \
    --num-workers "$CYCLE_WORKERS" \
    --log-every "$CYCLE_LOG_EVERY" \
    --skip-ablation \
    --train

run_if_missing "saved/cyclegan_celeba_cyc0.pt" \
  "Step 7: Train CycleGAN ablation, lambda_cyc=0" \
  uv run run.py --model cyclegan --dataset celeba --epochs 10 \
    --celeba-subset "$CELEBA_SUBSET" \
    --batch-size "$CYCLE_BATCH_SIZE" \
    --num-workers "$CYCLE_WORKERS" \
    --lambda-cyc 0 \
    --log-every "$CYCLE_LOG_EVERY" \
    --skip-ablation \
    --train

run_if_missing "outputs/my_face_result.png" \
  "Step 8: Test your own face" \
  uv run run.py --model cyclegan --weights saved/cyclegan_celeba_cyc10.pt --test-image my_face.jpg

echo "===== All done! Check outputs/ and saved/ ====="
