import torch
from A2.src.fpn import FeaturePyramidNetwork


def test_a2_fpn_preserves_spatial_pyramid():
    feats = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    fpn = FeaturePyramidNetwork([64, 128, 256, 512], out_channels=128)
    out = fpn(feats)
    assert list(out) == ["0", "1", "2", "3"]
    assert [v.shape for v in out.values()] == [
        (2, 128, 64, 64), (2, 128, 32, 32), (2, 128, 16, 16), (2, 128, 8, 8)
    ]
