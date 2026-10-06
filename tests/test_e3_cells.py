import torch
from E3.src.cells import ManualLSTMCell, ManualGRUCell
from E3.src.sequence import image_to_sequence


def test_e3_manual_cells_shapes():
    x = torch.randn(4, 28)
    h = torch.zeros(4, 32)
    c = torch.zeros(4, 32)
    h2, c2 = ManualLSTMCell(28, 32)(x, (h, c))
    assert h2.shape == c2.shape == (4, 32)
    g2 = ManualGRUCell(28, 32)(x, h)
    assert g2.shape == (4, 32)


def test_e3_sequence_representations():
    x = torch.randn(2, 1, 28, 28)
    assert image_to_sequence(x, "rows").shape == (2, 28, 28)
    assert image_to_sequence(x, "columns").shape == (2, 28, 28)
    assert image_to_sequence(x, "patch4").shape == (2, 49, 16)
