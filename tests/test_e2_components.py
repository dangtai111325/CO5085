import torch
from E2.src.attention import ManualMultiHeadSelfAttention, TorchMultiHeadSelfAttention
from E2.src.tokenizers import RowTokenizer, PatchTokenizer, CNNStemTokenizer


def test_e2_tokenizer_shapes():
    x = torch.randn(2, 1, 28, 28)
    assert RowTokenizer(64)(x).shape == (2, 28, 64)
    assert PatchTokenizer(4, 64)(x).shape == (2, 49, 64)
    assert CNNStemTokenizer(64)(x).shape == (2, 49, 64)


def test_e2_attention_shapes_and_probabilities():
    x = torch.randn(2, 17, 64)
    for attn in (ManualMultiHeadSelfAttention(64, 4), TorchMultiHeadSelfAttention(64, 4)):
        y = attn(x)
        assert y.shape == x.shape
        weights = attn.last_attention
        assert weights is not None
        assert weights.shape == (2, 4, 17, 17)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
