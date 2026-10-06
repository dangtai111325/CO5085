#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/report/source_latex"
BUILD="$ROOT/report/.build"
OUT="$ROOT/report/CO5085_report.pdf"
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$SRC"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$BUILD" main.tex
cp "$BUILD/main.pdf" "$OUT"
test -s "$OUT"
echo "Built: $OUT"
