import torch

from A2.src.box_ops import box_iou, encode_boxes, decode_boxes
from A2.src.model import build_detector


def test_box_codec_roundtrip():
    proposals = torch.tensor([[10., 10., 30., 40.], [3., 5., 20., 25.]])
    targets = torch.tensor([[12., 8., 33., 44.], [4., 6., 18., 29.]])
    deltas = encode_boxes(targets, proposals)
    decoded = decode_boxes(deltas, proposals).reshape(-1, 4)
    assert torch.allclose(decoded, targets, atol=1e-4)
    assert box_iou(targets, targets).diag().min() > 0.999


def test_custom_detector_train_and_eval_smoke():
    model = build_detector(
        'fpn', pretrained_backbone=False, image_size=(96, 96),
        rpn_pre_nms_top_n=60, rpn_post_nms_top_n=30,
    )
    images = [torch.rand(3, 96, 96)]
    targets = [{'boxes': torch.tensor([[15., 12., 60., 70.]]), 'labels': torch.tensor([1], dtype=torch.int64)}]
    model.train()
    losses = model(images, targets)
    assert {'loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg'} <= set(losses)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    model.eval()
    with torch.no_grad():
        pred = model(images)
    assert len(pred) == 1
    assert {'boxes','scores','labels'} <= set(pred[0])
