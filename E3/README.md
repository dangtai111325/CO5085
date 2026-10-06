# E3 - LSTM / GRU over image sequences

Three sequence views are prepared: rows, columns, and 4x4 patches. Manual LSTM and GRU cells are included in addition to the PyTorch reference modules explicitly permitted by the brief. Final analysis should compare recurrent models against E1 MLP/CNN baselines.

```bash
python -m E3.run --smoke --kind manual_gru --sequence rows
```
