# Figure code

Each script redraws paper figures from the tabulated metrics. Only matplotlib (and numpy) are required.

| Script | Paper figures |
|--------|----------------|
| `fig01_workflow_flowchart.py` | Figure 1 (experimental workflow) |
| `fig06_07_08.py` | Equity comparison, Sharpe ranking, drawdown |
| `fig09_10.py` | Win-rate vs equity; risk-return scatter |
| `fig12_13_14.py` | Gross vs fee; identical-selection; metric heatmap |

## How to run

```bash
python figures/fig01_workflow_flowchart.py
python figures/fig06_07_08.py
python figures/fig09_10.py
python figures/fig12_13_14.py
```

PNG (300 dpi) and PDF files are written next to the script, or into `paper_figures/` if that path exists. Header comments in each file list the exact command and the colour conventions (navy ink, muted fills, no 3-D).

Do not change the numeric arrays unless you have re-run the evaluation block; those arrays are the paper's Table 5.
