from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "E1.run", "--smoke", "--model", "cnn_2block"],
    [sys.executable, "-m", "E2.run", "--smoke", "--tokenizer", "patch4", "--attention", "manual"],
    [sys.executable, "-m", "E3.run", "--smoke", "--kind", "manual_gru", "--sequence", "rows"],
    [sys.executable, "-m", "A1.run", "--smoke", "--family", "resnet18"],
    [sys.executable, "-m", "A1.run", "--smoke", "--family", "vit_tiny"],
    [sys.executable, "-m", "A2.run", "--smoke", "--variant", "fpn"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("\n==>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
