from __future__ import annotations

import platform
import sys

import torch
import torchvision


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    else:
        print("GPU: not visible to current PyTorch build")


if __name__ == "__main__":
    main()
