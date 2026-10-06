from pathlib import Path
from html.parser import HTMLParser
import json

ROOT = Path(__file__).resolve().parents[1]


class Parser(HTMLParser):
    pass


def test_all_notebooks_are_valid_json():
    notebooks = list(ROOT.rglob("*.ipynb"))
    assert len(notebooks) >= 20
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload.get("cells"), list)


def test_site_html_parses_and_report_exists():
    Parser().feed((ROOT / "docs/index.html").read_text(encoding="utf-8"))
    pdf = ROOT / "report/CO5085_report.pdf"
    assert pdf.exists() and pdf.stat().st_size > 10_000
