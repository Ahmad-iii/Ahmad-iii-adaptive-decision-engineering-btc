# Standardised evaluation module

`standardized_eval_block.py` is the scoring code that produced every number in the paper's results table.

It does **not** change the strategy. It only reads `equities`, `correct_pred`, `trades_taken`, and `total_steps` (with fallbacks if a notebook uses older names) and prints ten metrics:

1. Final equity
2. Peak equity
3. Total return (%)
4. Win rate (%)
5. Trade count and participation
6. Maximum drawdown (%)
7. Sharpe ratio (per-candle, risk-free rate = 0; **not** multiplied by sqrt(35,040))
8. Profit factor
9. Calmar ratio
10. Annualised return (linear: total return / (N / (96 × 365)))

## How to use

1. Run a decision-layer notebook to completion.
2. Create one new cell at the bottom.
3. Paste the entire contents of `standardized_eval_block.py`.
4. Run that cell.

Comments at the top of the file repeat these steps. Do not mix this block with a different Sharpe or drawdown implementation if you want to match the paper.
