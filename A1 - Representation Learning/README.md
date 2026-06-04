# A1: Representation Learning

## Training Commands

```bash
python run.py --model alexnet --dataset cifar10 --epochs 10 --batch_size 64 --train
python run.py --model googlenet --dataset cifar10 --epochs 25 --batch_size 64 --train
python run.py --model resnet18 --dataset cifar10 --epochs 20 --batch_size 64 --train
python run.py --model vit_small --dataset cifar10 --epochs 20 --batch_size 64 --train
python run.py --model resnet18_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
python run.py --model vit_b16_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
```

## Results

| Model | # Params | Test Accuracy | Time/epoch | Architecture Type |
|-------|----------|---------------|------------|-------------------|
| AlexNet + LRN (scratch) | 57,044,810 | 52.45% | 69.5s | CNN |
| GoogLeNet + 2 Aux (scratch) | 10,635,774 | 84.24% | 136.6s | CNN + Inception |
| ResNet-18 (scratch) | 11,173,962 | 78.47% | 37.5s | CNN + Skip connections |
| ResNet-18 (pretrained) | 11,181,642 | 79.93% | 28.8s | CNN + Skip connections |
| ViT-Small (scratch) | 1,205,898 | 59.75% | 24.8s | Transformer |
| ViT-B/16 (pretrained) | 85,806,346 | 95.08% | 344s | Transformer |

## Discussion

**Which model performed best and why?**
ViT-B/16 pretrained achieved the highest test accuracy at 95.08%. This is because it was pretrained on ImageNet with 1.2M images, giving it rich feature representations that transfer well to CIFAR-10. The two-stage fine-tuning strategy — first training only the classification head, then fine-tuning all layers — prevented catastrophic forgetting of pretrained features while adapting to the 10-class task.

**CNN vs Transformer architectures:**
CNNs have built-in inductive biases such as locality and translation invariance, making them efficient on small datasets. This is why GoogLeNet from scratch (84.24%) and ResNet-18 from scratch (78.47%) both outperform ViT-Small from scratch (59.75%) — ViT has no such biases and needs much more data to learn them. However, when pretrained on large data, ViT-B/16 dominates all CNN models significantly.

**Parameter efficiency:**
GoogLeNet achieves 84.24% with only 10.6M parameters compared to AlexNet's 57M for just 52.45%. The Inception module's multi-scale parallel convolutions learn richer features far more efficiently than simply stacking larger convolutions. This shows that architecture design matters much more than raw parameter count.

**Skip connections:**
ResNet-18 from scratch reaches 78.47% with 11M parameters in just 37.5s per epoch — faster and more accurate than AlexNet despite being much smaller. Skip connections solve the vanishing gradient problem by providing a direct path for gradients to flow through, allowing the network to train effectively even with many layers.

**Pretrained CNN vs Pretrained Transformer on small dataset:**
ResNet-18 pretrained only reached 79.93% while ViT-B/16 pretrained reached 95.08%. This suggests that Transformer-based models, when pretrained on sufficient data, learn more generalizable and transferable representations than CNNs. The global attention mechanism in ViT captures long-range dependencies that are harder to learn with local convolutions alone.