from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "requirements.txt",
    "docs/index.html",
    "E1/E1.ipynb",
    "E2/E2.ipynb",
    "E3/E3.ipynb",
    "A1/notebooks/01_data_eda.ipynb",
    "A2/notebooks/10_paper_reproduction.ipynb",
    "report/source_latex/main.tex",
    "assets/logos/hcmut_logo.png",
]


def main() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    notebooks = list(ROOT.rglob("*.ipynb"))
    for nb in notebooks:
        with nb.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "cells" not in payload:
            raise SystemExit(f"Invalid notebook: {nb}")

    pdf = ROOT / "report/CO5085_report.pdf"
    if pdf.exists():
        if pdf.stat().st_size < 10_000:
            raise SystemExit("Report PDF is unexpectedly small")
        pdf_status = f"; report={pdf.stat().st_size} bytes"
    else:
        pdf_status = "; report PDF pending automated build"

    print(f"Repository structure OK; {len(notebooks)} notebooks{pdf_status}")


if __name__ == "__main__":
    main()
