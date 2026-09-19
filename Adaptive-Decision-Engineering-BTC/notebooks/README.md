# Decision-layer notebooks

Each notebook is one frozen-ensemble configuration. The predictive models are loaded, never trained.

The **80% consensus gate** (trade only on 4-of-5 or 5-of-5 agreement) is implemented inside Versions 8, 9, 10, 11, 13 and 16. There is no separate consensus script. That shared gate is why those six notebooks select the same 24,900 candles and the same 52.04% directional win rate; they differ only in how much capital they commit to those trades.

## Before you run

1. Rebuild `BTCUSDT-15m-data.csv` with `data/fetch_binance_ohlcv.py`.
2. Place the five `.pt` checkpoints where the notebook expects them (`./ensemble_models/` relative to the notebook, or edit the path list).
3. Run all cells top to bottom.
4. Paste `../evaluation/standardized_eval_block.py` into a new cell at the end to print the ten paper metrics.

## Versions

- `backup.ipynb` — baseline: simple majority vote, fixed USD 1,000 per candle.
- `main_v4_improved.ipynb` — graduated vote weights 1–5, losing-streak protection.
- `main_v5_improved.ipynb` — proportional (percentage-of-equity) sizing.
- `main_v6_improved.ipynb` — milestone tier sizing, soft circuit breaker.
- `main_v7_improved.ipynb` — external SMA trend filter (failed).
- `main_v8_improved.ipynb` — 80% consensus gate.
- `main_v9_improved.ipynb` — Dynamic Velocity Scoring (gross champion on equity).
- `main_v10_improved.ipynb` — tightened risk thresholds on the V9 stack.
- `main_v11_improved.ipynb` — extreme conviction sizing (highest equity, worst drawdown).
- `main_v12_improved.ipynb` — permissive signal affirmer.
- `main_v13_improved.ipynb` — affirmer with restored participation.
- `main_v14_improved.ipynb` — logistic-regression affirmer (highest win rate, low equity).
- `main_v15_improved.ipynb` — online Bayesian affirmer (lowest drawdown by not trading).
- `main_v16_improved.ipynb` — adaptive window, loss dampening, PnL-regime (best Sharpe / Calmar).

Comments inside each notebook mark the decision-layer block. Do not change the model-loading cells if you want to reproduce the paper.
