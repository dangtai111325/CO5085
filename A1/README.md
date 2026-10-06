# A1 - CNN vs Transformer on a larger image dataset

Proposed domain: traffic-sign recognition with **GTSRB**. The implementation deliberately exposes the model layers and the training loop. The custom ResNet-18 is parameter-name compatible with torchvision for ImageNet weight transfer; the custom ViT-Tiny is a DeiT-style implementation and can transfer matching pretrained weights from timm.

Main matrix: scratch, pretrained head-only, partial fine-tuning, and full fine-tuning for both CNN and Transformer families.

Smoke tests do not download data or pretrained weights:

```bash
python -m A1.run --smoke --family resnet18
python -m A1.run --smoke --family vit_tiny
```
