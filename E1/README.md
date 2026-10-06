# E1 - Softmax, MLP, CNN on Fashion-MNIST

Required core: Softmax classifier, MLP, CNN, explicit PyTorch training loop, validation/test evaluation, parameter/capacity comparison, learning curves, and error analysis. The repository includes four variants so the report can compare more than the minimum.

Quick structural test:

```bash
python -m E1.run --smoke --model cnn_2block
```

Full run example:

```bash
python -m E1.run --model cnn_3block_bn --epochs 20
```
