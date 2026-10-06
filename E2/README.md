# E2 - Multi-Head Self-Attention and tokenization

The main implementation writes MSA from primitive tensor operations; `nn.MultiheadAttention` is kept only as the required reference implementation. Three tokenizers are included: rows, 4x4 patches, and CNN-stem spatial tokens.

```bash
python -m E2.run --smoke --tokenizer patch4 --attention manual
```
