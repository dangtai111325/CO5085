# Dataset layout

Do **not** commit datasets to GitHub.

## E1-E3 - Fashion-MNIST

Downloaded automatically by `torchvision.datasets.FashionMNIST` into `data/` when `SMOKE=False`.

## A1 - proposed GTSRB

Downloaded automatically by `torchvision.datasets.GTSRB` into `data/` when the A1 loader is run in full mode.

## A2 - proposed BDD100K detection

The repository parses the standard frame-level detection JSON. Expected layout:

```text
data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/*.jpg
│       └── val/*.jpg
└── labels/
    └── .../det_train.json
```

BDD100K is large and may require an official download/login or a legally permitted mirror. The code intentionally does not fabricate or silently download a different dataset. After download, point `A2.run` to the actual image directory and JSON path.

For development without data, every assignment has a `--smoke` path using synthetic tensors solely to verify code structure; synthetic outputs must never be reported as experimental results.
