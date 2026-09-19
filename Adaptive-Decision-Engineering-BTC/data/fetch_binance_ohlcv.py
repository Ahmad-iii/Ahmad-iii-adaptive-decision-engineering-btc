"""
================================================================================
 fetch_binance_ohlcv.py
 Reproducible acquisition of Binance historical OHLCV candlestick data
================================================================================

Companion code for:
    "Adaptive Post-Model Decision Engineering for Deep Learning Ensemble
     Trading on Bitcoin"
    M. Ahmad Khan and Sawera Qureshi
    Department of Computer Science, Abdul Wali Khan University Mardan

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
Papers in this area routinely say "we used Binance BTC/USDT 15-minute data"
and stop there. That sentence is not reproducible. Two researchers following it
can end up with different datasets, because the result depends on which
timestamp is used as the index, whether partially-formed candles are included,
how pagination gaps are handled, and which endpoint is queried. This script
documents the exact procedure used for the dataset in the paper so that the
dataset can be rebuilt rather than requested.

The dataset used in the paper:
    symbol      BTCUSDT   (spot, USDT-quoted)
    interval    15m
    coverage    August 2017 to July 2025
    candles     approximately 276,000
    split       first 240,000 candles train, final 35,903 candles test,
                split chronologically with no shuffling

--------------------------------------------------------------------------------
HOW BINANCE SERVES THIS DATA
--------------------------------------------------------------------------------
Binance exposes candles through a "klines" REST endpoint. Three properties of
that endpoint drive the design of this script.

1. It is hard-capped at 1000 candles per request. Eight years of 15-minute
   candles is roughly 276,000 rows, so about 280 sequential requests are
   required. `python-binance` wraps this loop in
   `get_historical_klines()`, which pages automatically.

2. Each kline is a 12-element list, not a dict. The ordering is fixed and
   undocumented in the response itself, so the column names must be supplied
   by the caller. Getting this ordering wrong silently produces a dataset in
   which, for example, `volume` holds close prices. The correct order is
   encoded in KLINE_COLUMNS below.

3. Each kline carries BOTH an `open_time` and a `close_time`, and the choice
   between them as the DataFrame index is a genuine methodological decision
   rather than a formatting preference. See the next section.

--------------------------------------------------------------------------------
THE ONE DECISION THAT CAUSES LOOKAHEAD BIAS
--------------------------------------------------------------------------------
A 15-minute candle stamped 12:00 covers the interval 12:00:00 to 12:14:59.
Its high, low, close and volume are unknown until 12:14:59 has passed.

If you index by `close_time`, the row labelled 12:14:59 contains information
that only became available at 12:14:59 -- which is correct and safe.

If you index by `open_time`, the row labelled 12:00 contains the close price
of 12:14:59. This is fine in itself, but it becomes a lookahead bug the moment
you build a feature window: a window that ends at row 12:00 appears to end at
12:00 while actually containing information from 12:14:59.

This script indexes by `open_time`, matching the dataset used in the paper,
and therefore compensates in the modelling stage: a feature window whose final
row is stamped t is only ever used to predict the candle stamped t+1, never
the candle stamped t. If you index by `close_time` instead, that offset is
already built in. Pick one convention, write it down, and never mix them.

The second trap is the final row. A request issued at 12:07 returns a candle
stamped 12:00 whose close, high, low and volume are still forming. Training on
it teaches the model from a value that will change. This script drops the last
row unconditionally (DROP_UNCLOSED_FINAL_CANDLE).

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    pip install python-binance pandas numpy

API KEYS: the klines endpoint is PUBLIC. Historical candles can be downloaded
with empty credentials, and this script defaults to doing exactly that. Supply
keys only if you hit aggressive rate limiting on a shared IP address. Never
commit real keys; read them from environment variables as shown below.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    # Rebuild the paper's dataset
    python fetch_binance_ohlcv.py

    # A different market or interval
    python fetch_binance_ohlcv.py --symbol ETHUSDT --interval 1h

    # Resume an interrupted download (re-run the identical command; the script
    # reads the existing CSV and fetches only what is missing)
    python fetch_binance_ohlcv.py

Expect roughly 3 to 8 minutes for eight years of 15-minute candles, depending
on your connection and current Binance rate limits. Output is a CSV of about
25 to 30 MB.

--------------------------------------------------------------------------------
LICENCE / CITATION
--------------------------------------------------------------------------------
Released for academic reuse. If this acquisition procedure is useful, a
citation of the paper above is appreciated. The data itself belongs to Binance
and is subject to their terms of use.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Run:  pip install python-binance pandas numpy")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Reproducibility. Nothing in this script is stochastic, but the seed is set so
# that this file can be concatenated with downstream modelling code without
# changing their behaviour.
np.random.seed(123)

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

# Binance spot trading for BTC/USDT began in August 2017. Requesting an earlier
# start is harmless: the exchange simply returns data from the first candle it
# has. A fixed literal is used rather than "earliest available" so that the
# command is self-documenting.
DEFAULT_START = "1 Aug 2017"

# Drop the final row, which is an in-progress candle whose OHLCV values are not
# yet final. Leave this True unless you are deliberately handling live data.
DROP_UNCLOSED_FINAL_CANDLE = True

# Index convention. "open_time" reproduces the paper's dataset; "close_time" is
# the safer default for new work. See the module docstring.
INDEX_ON = "open_time"

# Minutes per candle. Needed only to report progress and to detect gaps.
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}

# Fixed positional schema of a Binance kline. Do not reorder.
KLINE_COLUMNS = [
    "open_time",     # 0  candle open,  ms epoch
    "open",          # 1
    "high",          # 2
    "low",           # 3
    "close",         # 4
    "volume",        # 5  base-asset volume (BTC)
    "close_time",    # 6  candle close, ms epoch
    "quote_av",      # 7  quote-asset volume (USDT)
    "trades",        # 8  number of individual trades
    "tb_base_av",    # 9  taker buy base volume
    "tb_quote_av",   # 10 taker buy quote volume
    "ignore",        # 11 unused by Binance
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_av", "trades", "tb_base_av", "tb_quote_av",
]

# The eight features used in the paper are derived from these five columns.
# Everything else is retained so the CSV can support other feature sets.
ESSENTIAL_COLUMNS = ["open", "high", "low", "close", "volume"]


# ==============================================================================
# CLIENT
# ==============================================================================

def build_client() -> Client:
    """
    Create a Binance REST client.

    Credentials are read from the environment, never hard-coded:
        BINANCE_API_KEY, BINANCE_API_SECRET
    Both are optional. The klines endpoint is public, so an unauthenticated
    client downloads full history without restriction.
    """
    key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_API_SECRET", "")
    if key and secret:
        print("Using authenticated client (credentials found in environment).")
    else:
        print("Using unauthenticated client (klines is a public endpoint).")
    return Client(api_key=key, api_secret=secret)


# ==============================================================================
# RESUME LOGIC
# ==============================================================================

def determine_fetch_window(client: Client,
                           symbol: str,
                           interval: str,
                           existing: pd.DataFrame,
                           default_start: str) -> tuple[datetime, datetime]:
    """
    Decide which time range still needs downloading.

    A full eight-year download is slow enough that an interrupted run should not
    have to start over. If a CSV already exists, resume from its final timestamp
    instead of from `default_start`. Re-fetching one candle of overlap is
    deliberate: it guarantees no gap at the seam, and the duplicate is removed
    later by `_deduplicate`.

    Returns (start, end) as timezone-naive UTC datetimes.
    """
    if not existing.empty:
        start = existing.index[-1].to_pydatetime()
        print(f"Existing file ends at {start}. Resuming from there.")
    else:
        start = pd.to_datetime(default_start).to_pydatetime()
        print(f"No existing file. Starting from {start:%d %b %Y}.")

    # Anchor the end of the window to the exchange's own clock rather than the
    # local machine's, so that a skewed system clock cannot truncate the data.
    latest = client.get_klines(symbol=symbol, interval=interval, limit=1)[-1]
    end = pd.to_datetime(latest[6], unit="ms").to_pydatetime()

    span_minutes = (end - start).total_seconds() / 60.0
    per_candle = INTERVAL_MINUTES.get(interval)
    if per_candle:
        print(f"Requesting ~{math.ceil(span_minutes / per_candle):,} "
              f"{interval} candles for {symbol}.")

    return start, end


# ==============================================================================
# DOWNLOAD
# ==============================================================================

def download_klines(client: Client,
                    symbol: str,
                    interval: str,
                    start: datetime,
                    end: datetime,
                    max_retries: int = 5) -> pd.DataFrame:
    """
    Page through the klines endpoint and return a tidy, typed DataFrame.

    `get_historical_klines` handles the 1000-candle cap internally, issuing as
    many sequential requests as the range requires. Transient failures (rate
    limits, brief network faults) are retried with exponential backoff; a long
    download should not be lost to one dropped packet.
    """
    fmt = "%d %b %Y %H:%M:%S"
    delay = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            print("Downloading (this pages automatically, please wait) ...")
            raw = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start.strftime(fmt),
                end_str=end.strftime(fmt),
            )
            break
        except (BinanceAPIException, ConnectionError, TimeoutError) as exc:
            if attempt == max_retries:
                raise
            print(f"  attempt {attempt} failed ({exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    else:  # pragma: no cover
        raise RuntimeError("Download failed after all retries.")

    if not raw:
        print("Endpoint returned no rows; nothing new to add.")
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)

    # Binance returns every numeric field as a JSON string. Convert explicitly:
    # errors="coerce" turns any malformed value into NaN so that it surfaces in
    # the integrity report below rather than propagating silently as a string.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    df = df.set_index(INDEX_ON).drop(columns=["ignore"])
    df.index.name = "timestamp"

    print(f"Received {len(df):,} rows.")
    return df


# ==============================================================================
# CLEANING AND INTEGRITY
# ==============================================================================

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timestamps, keeping the last occurrence, then sort.

    Duplicates arise legitimately from the one-candle resume overlap and
    occasionally from Binance returning a boundary candle in two consecutive
    pages. `keep="last"` is correct because a later copy of the same candle is
    the more settled one.
    """
    before = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if before != len(df):
        print(f"Removed {before - len(df):,} duplicate timestamps.")
    return df


def report_integrity(df: pd.DataFrame, interval: str) -> None:
    """
    Print the checks a reviewer will reasonably ask about.

    None of these are fatal by themselves, but all of them should be known
    before the data is used. Real exchange history does contain gaps -- Binance
    has had maintenance windows and outages -- and an honest paper states that
    rather than pretending the series is perfectly regular.
    """
    print("\n" + "=" * 62)
    print("  DATASET INTEGRITY REPORT")
    print("=" * 62)
    print(f"  Rows                  : {len(df):,}")
    print(f"  First candle          : {df.index[0]}")
    print(f"  Last candle           : {df.index[-1]}")
    print(f"  Index convention      : {INDEX_ON}")

    missing = df[ESSENTIAL_COLUMNS].isna().sum().sum()
    print(f"  Missing OHLCV values  : {missing:,}")

    # A monotonically increasing index is required for any chronological split.
    print(f"  Monotonic index       : {df.index.is_monotonic_increasing}")

    # Gap detection: any interval larger than one candle means missing candles.
    per_candle = INTERVAL_MINUTES.get(interval)
    if per_candle:
        deltas = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(minutes=per_candle)
        gaps = deltas[deltas > expected]
        print(f"  Gaps (missing candles): {len(gaps):,}")
        if len(gaps):
            worst = gaps.nlargest(3)
            for ts, d in worst.items():
                skipped = int(d / expected) - 1
                print(f"      {ts}  ->  {skipped:,} candle(s) missing")

    # OHLC sanity: high must bound open/close, low must be bounded by them.
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    print(f"  Invalid high          : {bad_high:,}")
    print(f"  Invalid low           : {bad_low:,}")
    print(f"  Non-positive prices   : {nonpos:,}")
    print(f"  Zero-volume candles   : {int((df['volume'] == 0).sum()):,}")
    print("=" * 62)

    # Reproduce the paper's split so the numbers can be checked directly.
    if len(df) >= 275_903:
        test_n = 35_903
        print(f"\n  Paper's chronological split:")
        print(f"    train : rows 0 to {len(df) - test_n - 1:,}")
        print(f"    test  : final {test_n:,} rows "
              f"({df.index[-test_n]}  ->  {df.index[-1]})")
        print("    No shuffling. The scaler is fitted on train only.\n")


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def fetch(symbol: str = SYMBOL,
          interval: str = INTERVAL,
          start: str = DEFAULT_START,
          out_path: str | None = None,
          save: bool = True) -> pd.DataFrame:
    """
    Download (or extend) the candle history for one symbol and interval.

    Returns the full cleaned DataFrame. Writes a CSV when `save` is True.
    """
    out_path = out_path or f"{symbol}-{interval}-data.csv"
    client = build_client()

    if os.path.isfile(out_path):
        existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(existing):,} existing rows from {out_path}.")
    else:
        existing = pd.DataFrame()

    win_start, win_end = determine_fetch_window(
        client, symbol, interval, existing, start
    )

    fresh = download_klines(client, symbol, interval, win_start, win_end)

    if fresh.empty and existing.empty:
        raise RuntimeError("No data retrieved. Check the symbol and interval.")

    combined = fresh if existing.empty else pd.concat([existing, fresh], axis=0)
    combined = _deduplicate(combined)

    if DROP_UNCLOSED_FINAL_CANDLE and len(combined) > 1:
        dropped = combined.index[-1]
        combined = combined.iloc[:-1]
        print(f"Dropped final in-progress candle ({dropped}).")

    report_integrity(combined, interval)

    if save:
        combined.to_csv(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved {len(combined):,} rows to {out_path} ({size_mb:.1f} MB).")

    return combined


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download Binance historical OHLCV candles reproducibly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default=SYMBOL, help="e.g. BTCUSDT, ETHUSDT")
    p.add_argument("--interval", default=INTERVAL,
                   choices=sorted(INTERVAL_MINUTES), help="candle interval")
    p.add_argument("--start", default=DEFAULT_START,
                   help='first candle, e.g. "1 Aug 2017"')
    p.add_argument("--out", default=None, help="output CSV path")
    p.add_argument("--no-save", action="store_true", help="do not write a CSV")
    args = p.parse_args()

    started = time.time()
    fetch(symbol=args.symbol, interval=args.interval, start=args.start,
          out_path=args.out, save=not args.no_save)
    print(f"Completed in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    main()
