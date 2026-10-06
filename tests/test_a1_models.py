import torch
from A1.src.resnet import ResNet18, set_resnet_trainability
from A1.src.vit import VisionTransformerTiny, set_vit_trainability


def test_a1_custom_models_forward():
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        cnn = ResNet18(num_classes=43)
        assert cnn(x).shape == (1, 43)
        vit = VisionTransformerTiny(num_classes=43, depth=2)
        assert vit(x).shape == (1, 43)


def test_a1_freeze_modes_leave_trainable_parameters():
    cnn = ResNet18(num_classes=43)
    for mode in ("scratch", "head", "partial", "full"):
        set_resnet_trainability(cnn, mode)
        assert any(p.requires_grad for p in cnn.parameters())
    vit = VisionTransformerTiny(num_classes=43)
    for mode in ("scratch", "head", "partial", "full"):
        set_vit_trainability(vit, mode)
        assert any(p.requires_grad for p in vit.parameters())
