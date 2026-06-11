# A2-02: Image Segmentation — U-Net Skip Connections Ablation

## Commands Used

```bash
# Train baseline (with skip connections)
py run_seg.py --model unet_resnet18 --dataset oxford_pet --epochs 20 --train

# Train ablation (without skip connections)
py run_seg.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# Evaluate baseline
py run_seg.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate

# Evaluate no-skip model
py run_seg.py --model unet_resnet18_no_skip --weights unet_resnet18_no_skip_pet.pt --dataset oxford_pet --evaluate
```

## Results

| Model | Encoder | Skip Connections | Val mIoU | Time/epoch |
|---|---|---|---|---|
| `unet_resnet18` | ResNet-18 (ImageNet) | ✅ Yes | 0.7616 | ~20s |
| `unet_resnet18_no_skip` | ResNet-18 (ImageNet) | ❌ No | 0.6850 | ~23s |

## Discussion

Skip connections improved mIoU by 0.0766 (0.7616 vs 0.6850), a meaningful gap given both models share the identical pretrained ResNet-18 encoder. Without skip connections, the decoder must reconstruct fine spatial detail — exact object boundaries and edges — solely from the bottleneck representation, which has already compressed spatial information through 5 downsampling stages. Skip connections bypass this bottleneck by directly concatenating high-resolution encoder feature maps into the decoder at each scale, allowing the model to recover precise pixel-level boundaries that are otherwise lost. The no-skip model also converged more slowly, starting at mIoU 0.4493 vs 0.6673 in epoch 1, showing that skip connections provide crucial spatial grounding from the very start of training. U-Net is preferred over Mask R-CNN when the task requires dense pixel-level segmentation without needing per-instance separation — such as medical imaging or satellite imagery — since it is simpler, faster to train, and performs well with limited labeled data.

## Q c: Why do skip connections help segmentation more than classification?

Classification only needs to answer "what is in this image" — spatial information is irrelevant, so compressing everything into a bottleneck vector is fine. Segmentation must answer "which exact pixels belong to each class" — requiring both semantic understanding (what) and precise spatial localization (where). Skip connections carry high-resolution feature maps from early encoder stages directly to the decoder, preserving fine edge and boundary detail that is destroyed by repeated downsampling. Without them, the decoder has no way to recover exact pixel positions from a heavily compressed bottleneck.

## Q d: Which skip level hurts most when removed?

The first skip connection (stage 1, 64ch, highest resolution) hurts the most when removed. It carries the finest spatial detail — exact edges, textures, and boundaries at near-input resolution. Later skip connections (stage 4, 512ch) carry mostly semantic information that is already encoded in the bottleneck anyway. Removing the earliest skip means the decoder can never recover pixel-precise boundaries regardless of how well it learns semantics.