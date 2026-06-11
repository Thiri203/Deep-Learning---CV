# A2-01: Object Detection — YOLOv4

## Commands Used

```bash
# YOLOv3 inference
py run_detection.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer

# YOLOv4 inference
py run_detection.py --model yolov4 --weights yolov4.weights --image dog-cycle-car.png --infer

# Train YOLOv4 with IoU loss
py run_detection.py --model yolov4 --weights yolov4.weights --dataset coco --epochs 5 --loss iou --batch 8 --train

# Train YOLOv4 with CIoU loss
py run_detection.py --model yolov4 --weights yolov4.weights --dataset coco --epochs 5 --loss ciou --batch 8 --train

# Evaluate
py run_detection.py --model yolov4 --weights yolov4_iou_loss.pt --dataset coco --evaluate
py run_detection.py --model yolov4 --weights yolov4_ciou_loss.pt --dataset coco --evaluate
```

## Results

| Model | Dataset | Avg Loss (epoch 5) | Time/epoch | Notes |
|---|---|---|---|---|
| YOLOv3 (pretrained) | COCO | N/A | — | Inference only. Top detections: dog 1.00, bicycle 0.99, truck 0.88 |
| YOLOv4 (IoU loss) | COCO (500 subset) | 15.9780 | ~1093s | Trained 5 epochs on personal PC (RTX 4060) |
| YOLOv4 (CIoU loss) | COCO (500 subset) | 3.0251 | ~1096s | Trained 5 epochs on personal PC (RTX 4060) |

> Note: Formal mAP evaluation requires pycocotools per-class PR curve computation.
> Training used a 500-image subset of COCO val2017 due to hardware constraints on a personal PC.

## Discussion

CIoU loss converged to a significantly lower final loss (3.03) compared to standard IoU loss (15.98) over the same 5 epochs, demonstrating that CIoU's additional geometric terms — center distance and aspect ratio consistency — provide stronger gradient signal during training. Standard IoU loss only penalizes overlap area, which gives zero gradient when boxes do not overlap at all, slowing early training. CIoU addresses this by incorporating the normalized center-point distance and an aspect ratio consistency term, guiding the predicted box toward the ground truth more precisely even when overlap is low. Both runs used identical pretrained YOLOv4 weights, dataset, and hyperparameters, so the loss difference is directly attributable to the loss function. In practice, YOLOv4 and later YOLO versions adopt CIoU as the default box regression loss for this reason.

## Q3: Why is YOLOv3 Faster than Faster R-CNN?

Faster R-CNN is a two-stage detector: it first runs a Region Proposal Network (RPN) to generate ~300 candidate boxes, then classifies each proposal through a separate detection head — two sequential forward passes share the backbone but the RPN and detection head still process proposals one stage at a time. YOLOv3 eliminates the proposal stage entirely by dividing the image into a grid and predicting bounding boxes and class probabilities simultaneously in a single forward pass across three scales (13×13, 26×26, 52×52). This single-shot architecture means there is no region proposal bottleneck, no RoI pooling, and no sequential dependency between stages — the entire detection happens in one network pass, which is what makes YOLO significantly faster while still achieving competitive accuracy through multi-scale prediction.