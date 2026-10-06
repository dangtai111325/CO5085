import torch
from E1.src.models import build_e1_models


def test_e1_all_models_output_shape():
    x = torch.randn(3, 1, 28, 28)
    models = build_e1_models()
    assert set(models) == {"softmax", "mlp", "cnn_2block", "cnn_3block_bn"}
    for model in models.values():
        y = model(x)
        assert y.shape == (3, 10)
        assert torch.isfinite(y).all()
