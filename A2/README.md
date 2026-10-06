# A2 - Object Detection: Faster R-CNN + FPN

Proposed domain: BDD100K road-scene object detection. The paper analysis is centered on *Feature Pyramid Networks for Object Detection* (CVPR 2017), with Faster R-CNN as the detector framework.

The code does **not** call the one-line `fasterrcnn_resnet50_fpn` constructor. It explicitly assembles ResNet stages, a custom FPN, anchors, RoIAlign, and Faster R-CNN, while keeping optimized low-level detection operators from torchvision.

Smoke test (synthetic images; no dataset/pretrained download):

```bash
python -m A2.run --smoke --variant fpn
```

Full BDD100K example:

```bash
python -m A2.run --variant fpn \
  --images-dir data/bdd100k/images/100k/train \
  --labels-json data/bdd100k/labels/det_20/det_train.json
```
