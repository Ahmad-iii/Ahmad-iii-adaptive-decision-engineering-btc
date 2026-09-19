# Dataset acquisition

Rebuild the paper's BTC/USDT 15-minute series from Binance. The candlestick data is **not** redistributed here; it remains Binance's property.

## What the paper used

| Item | Value |
|------|--------|
| Symbol | BTCUSDT (spot, USDT-quoted) |
| Interval | 15 minutes |
| Coverage | August 2017 to July 2025 |
| Rows | approximately 276,000 |
| Train split | first 240,000 candles |
| Test split | remaining 35,903 candles, chronological, no shuffle |

## How to run

From the repository root (after `pip install -r requirements.txt`):

```bash
python data/fetch_binance_ohlcv.py --symbol BTCUSDT --interval 15m
```

No API key is required. The script pages the public `klines` endpoint (1,000 rows per request), drops the unclosed final candle, indexes by `open_time`, and prints an integrity report (row count, monotonic index, missing-candle gaps, OHLC bound violations).

Place the resulting `BTCUSDT-15m-data.csv` in the working directory of the notebook or cost-sensitivity script you intend to run.

## Lookahead convention

A row stamped `t` contains the close of the interval that ends at `t + 14:59`. Feature windows ending at `t` are used to predict the candle stamped `t+1`, never the candle stamped `t`. Do not mix `open_time` and `close_time` indexing.

See the header comments in `fetch_binance_ohlcv.py` for the full rationale.
