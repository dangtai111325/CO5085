# LaTeX source

`main.tex` is the single entry point. Team metadata is centralized in `metadata.tex`.

Sections:

1. `00_overview.tex` - assignment mapping and scope
2. `01_shared_setup.tex` - environment, code rules, reproducibility
3. `02_e1.tex` - Softmax/MLP/CNN
4. `03_e2.tex` - manual MSA and tokenization
5. `04_e3.tex` - LSTM/GRU image sequences
6. `05_a1.tex` - CNN vs Transformer
7. `06_a2.tex` - detection + FPN paper analysis
8. `07_conclusion.tex` - cross-assignment conclusions
9. `08_appendix.tex` - commands/artifact checklist

Figures are referenced from the repository-level `assets/` directory. Use `\safeimage{e1/foo.png}{Caption}` so a missing figure renders as a visible red placeholder instead of breaking the build.
