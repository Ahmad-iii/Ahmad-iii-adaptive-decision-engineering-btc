# Adaptive Post-Model Decision Engineering for Deep Learning Ensemble Trading on Bitcoin

Replication package for the paper of the same title.

**Authors:** M. Ahmad Khan, Sawera Qureshi  
**Affiliation:** Department of Computer Science, Abdul Wali Khan University Mardan, Garden Campus  
**Paper:** `[DOI / journal link — add after acceptance]`  
**Contact:** `[email]`

---

## What this repository is for

The paper holds a five-model deep learning ensemble **frozen** and varies only the post-model decision layer — position sizing, trade filtering, and risk management — across fourteen configurations (baseline + Versions 4 to 16). Every performance difference reported is therefore attributable to the decision layer rather than to the predictive models.

**Read this before you run anything:** the strong gross results do **not** survive transaction costs at this turnover, at either 0.05% or 0.02% per round-trip. Section 5.1 of the paper explains why the failure is structural rather than marginal. See `DISCLAIMER.md`. Do not deploy any configuration here with live capital.

The 80% consensus gate is **not a separate script**. It lives inside the Version 8–11, 13 and 16 notebooks: a trade is taken only when at least four of the five frozen models agree. That is why those six versions select exactly the same 24,900 candles and record exactly the same 52.04% win rate.

---

## Repository layout

```
Adaptive-Decision-Engineering-BTC/
├── README.md                          <- this file (GitHub renders it)
├── README.txt                         <- same content, plain text
├── LICENSE                            <- MIT for code
├── DISCLAIMER.md                      <- cost failure; not live-trading code
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md                      <- symbol, interval, split, integrity
│   └── fetch_binance_ohlcv.py         <- rebuild the Binance 15m dataset
│
├── models/
│   └── README.md                      <- where to place the five .pt files
│
├── notebooks/
│   ├── README.md                      <- how to run one configuration
│   ├── backup.ipynb                   <- baseline (majority vote, fixed $1,000)
│   ├── main_v4_improved.ipynb
│   ├── main_v5_improved.ipynb
│   ├── ...
│   └── main_v16_improved.ipynb        <- all fourteen decision-layer versions
│
├── cost_sensitivity/
│   ├── README.md                      <- set TRANSACTION_FEE_RATE
│   ├── V9_fee_adjusted_0.05pct.py
│   └── V16_fee_adjusted_0.05pct.py
│
├── evaluation/
│   ├── README.md
│   └── standardized_eval_block.py     <- the ten metrics, identical for all runs
│
├── figures/
│   ├── README.md                      <- which script draws which paper figure
│   ├── fig01_workflow_flowchart.py
│   ├── fig06_07_08.py
│   ├── fig09_10.py
│   └── fig12_13_14.py
│
└── results/
    ├── README.md
    └── all_versions_metrics.csv       <- Table 5 as machine-readable data
```

The candlestick CSV is **not** in this repository. Rebuild it with `data/fetch_binance_ohlcv.py`. The five frozen checkpoints are also **not** redistributed here by default (GitHub file-size limits). Place them in `models/` as described in `models/README.md`, or download them from the Google Drive link in the paper.

---

## Reproducing the paper

### 1. Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Rebuild the dataset

```bash
python data/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 15m
```

Produces `BTCUSDT-15m-data.csv`, roughly 276,000 candles covering August 2017 to July 2025. No API key is required; the klines endpoint is public. Put the CSV next to the notebook you intend to run, or in the working directory of the cost-sensitivity scripts.

### 3. Place the five frozen checkpoints

Copy these files into `models/` (or `ensemble_models/`, matching the path inside each notebook):

```
eq_3778_ep_25.pt
eq_3768_ep_80.pt
eq_3590_ep_122.pt
eq_3301_ep_78.pt
eq_3296_ep_60.pt
```

**No training is required or permitted.** The notebooks load these weights and never update them.

### 4. Reproduce any single configuration

Open the matching notebook, point it at the CSV and the checkpoints, and run all cells. Runtime is a few minutes on CPU.

| Paper row | Notebook |
|-----------|----------|
| Baseline  | `notebooks/backup.ipynb` |
| V4        | `notebooks/main_v4_improved.ipynb` |
| V5        | `notebooks/main_v5_improved.ipynb` |
| V6        | `notebooks/main_v6_improved.ipynb` |
| V7        | `notebooks/main_v7_improved.ipynb` |
| V8        | `notebooks/main_v8_improved.ipynb` |
| V9        | `notebooks/main_v9_improved.ipynb` |
| V10       | `notebooks/main_v10_improved.ipynb` |
| V11       | `notebooks/main_v11_improved.ipynb` |
| V12       | `notebooks/main_v12_improved.ipynb` |
| V13       | `notebooks/main_v13_improved.ipynb` |
| V14       | `notebooks/main_v14_improved.ipynb` |
| V15       | `notebooks/main_v15_improved.ipynb` |
| V16       | `notebooks/main_v16_improved.ipynb` |

### 5. Reproduce the metrics exactly

After a notebook finishes, paste `evaluation/standardized_eval_block.py` into a new final cell and run it. All fourteen rows in the paper's results table were scored with this identical block. It expects `equities`, `correct_pred`, `trades_taken`, and `total_steps` (with fallbacks documented in the file).

### 6. Reproduce the cost sensitivity

```bash
python cost_sensitivity/V9_fee_adjusted_0.05pct.py
python cost_sensitivity/V16_fee_adjusted_0.05pct.py
```

Inside each script, change **one line**:

```python
TRANSACTION_FEE_RATE = 0.0005   # 0.05% taker-equivalent; use 0.0002 for 0.02%
```

### 7. Redraw paper figures

```bash
python figures/fig01_workflow_flowchart.py
python figures/fig06_07_08.py
python figures/fig09_10.py
python figures/fig12_13_14.py
```

---

## Reproducibility notes, stated honestly

**Seeds.** `random`, `numpy` and `torch` are seeded in the notebooks. The decision layer is deterministic, so repeated runs of any version reproduce the reported figures.

**The checkpoints are not clean out-of-sample.** They were retained during training whenever test-block backtest equity exceeded $3,000, using the same 35,903-candle block later used to compare decision layers. Absolute equity for the predictive layer is therefore optimistically biased. The decision-layer *comparison* remains internally valid because this bias is identical across all fourteen configurations.

**Sharpe is per-candle with a zero risk-free rate,** not annualised. Absolute values are small by construction; only the ordering across configurations is meaningful.

**No slippage anywhere.** Even the cost-adjusted runs charge only a fee on notional.

**Execution model.** A trade is a same-candle open-to-close round-trip. The simulator has no insolvency halt, so negative terminal balances in the cost runs are artefacts, not liquidation values.

---

## Attribution

The predictive layer is **inherited, not original**. The data pipeline, the eight engineered features, the scaler protocol, the 96-step sequences, the three network architectures, the majority-vote ensemble, and the fixed-position baseline follow publicly available instructional material, from which the five checkpoints also originate. This is stated in Section 3.3 of the paper.

Original contribution in this repository: the decision layers for Versions 4 to 16, the standardised evaluation module, the cost-sensitivity scripts, the dataset acquisition script, and the figure code.

---

## Licence

Code: MIT. Candlestick data belongs to Binance and is subject to their terms; the acquisition script is provided instead of a redistributed dataset.

## Citation

```bibtex
@article{khan2026decision,
  title   = {Adaptive Post-Model Decision Engineering for Deep Learning
             Ensemble Trading on Bitcoin},
  author  = {Khan, M. Ahmad and Qureshi, Sawera},
  journal = {[journal]},
  year    = {2026}
}
```
