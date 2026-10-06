# Unified CO5085 report

- LaTeX source: `report/source_latex/`
- Final PDF: `report/CO5085_report.pdf`
- Figures/images: `assets/`
- Selected CSV/JSON artifacts: `report/results/`

The PDF intentionally combines E1-E3, A1 and A2 into one master report. Missing experimental content is rendered as red placeholders; those fields must only be replaced with measured results from the repository runs.

Build locally:

```bash
bash scripts/build_report.sh
```

The style follows the supplied HCMUT reference report: institutional cover page, serif body text, running header, table of contents, list of figures/tables, numbered sections, compact `booktabs` tables, and figure captions below figures.
