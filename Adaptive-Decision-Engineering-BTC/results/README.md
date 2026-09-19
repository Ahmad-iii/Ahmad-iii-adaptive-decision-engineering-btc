# Machine-readable results

`all_versions_metrics.csv` is the paper's full performance table (baseline + V4–V16), gross of transaction costs.

Columns match the standardised evaluation module: final equity, peak equity, total return, win rate, trades taken, participation, max drawdown, Sharpe (per-candle), profit factor, Calmar, annualised return.

Sharpe is **not** annualised. Win rate is 52.04% and trade count is 24,900 for V8, V9, V10, V11, V13 and V16 because they share the 80% consensus gate.

Rebuild this file by running `evaluation/standardized_eval_block.py` after each notebook; do not hand-edit the numbers.
