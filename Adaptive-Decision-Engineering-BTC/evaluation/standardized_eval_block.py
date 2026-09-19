# ================================================================
# UNIVERSAL STANDARDIZED EVALUATION BLOCK v2.0
# ================================================================
# HOW TO USE:
#   1. Run your entire notebook normally (all cells top to bottom)
#   2. AFTER the LAST cell finishes (the one with the chart),
#      create ONE NEW empty cell at the very bottom
#   3. Copy-paste this ENTIRE block into that new cell
#   4. Run it — it will print all 10 metrics
#
# This block auto-detects variable names from ALL versions:
#   - backup.ipynb (baseline)
#   - main_v4_improved.ipynb through main_v15_improved.ipynb
#
# It does NOT modify any strategy logic. It only READS.
# ================================================================

import numpy as np

# ── AUTO-DETECT VARIABLES ──────────────────────────────────────
# Different versions use different variable names.
# This block handles ALL patterns safely.

# 1. equities — always exists in all versions
#    (already defined above in every notebook)

# 2. Total candles evaluated
try:
    _total_candles = total_steps          # v5+ define this
except NameError:
    try:
        _total_candles = len(X_test) - 1  # baseline/v4 don't define total_steps
    except NameError:
        _total_candles = len(equities)    # absolute fallback

# 3. Trades taken
try:
    _trades_taken = trades_taken          # v5+ define this
except NameError:
    _trades_taken = _total_candles        # baseline/v4: every candle = a trade

# 4. Correct predictions
try:
    _correct_pred = correct_pred          # all versions use this name
except NameError:
    _correct_pred = 0

# ── METRIC CALCULATIONS ───────────────────────────────────────

# M1: Final Equity
_final_equity = equities[-1]

# M2: Peak Equity
_peak_equity = max(equities)

# M3: Total Return (%)
_starting_capital = 1000.0
_total_return_pct = ((_final_equity - _starting_capital) / _starting_capital) * 100

# M4: Win Rate (%)
if _trades_taken > 0:
    _win_rate = (_correct_pred / _trades_taken) * 100
else:
    _win_rate = 0.0

# M5: Trade Count and Ratio
_trade_ratio = (_trades_taken / _total_candles) * 100 if _total_candles > 0 else 0.0

# M6: Max Drawdown (%)
_equity_arr = np.array(equities, dtype=np.float64)
_running_peak = np.maximum.accumulate(_equity_arr)
_drawdown_arr = (_running_peak - _equity_arr) / _running_peak
_max_drawdown_pct = float(np.max(_drawdown_arr)) * 100

_max_dd_idx = int(np.argmax(_drawdown_arr))
_peak_idx = int(np.argmax(_equity_arr[:_max_dd_idx + 1])) if _max_dd_idx > 0 else 0
_peak_value = float(_equity_arr[_peak_idx])
_trough_value = float(_equity_arr[_max_dd_idx])

# M7: Sharpe Ratio (per-candle, risk-free = 0)
_returns = []
for _i in range(1, len(equities)):
    _ret = (equities[_i] - equities[_i-1]) / equities[_i-1] if equities[_i-1] != 0 else 0
    _returns.append(_ret)
_returns_arr = np.array(_returns)
_mean_return = float(np.mean(_returns_arr))
_std_return = float(np.std(_returns_arr, ddof=1)) if len(_returns_arr) > 1 else 0.0001
_sharpe_ratio = _mean_return / _std_return if _std_return != 0 else 0.0

# M8: Profit Factor (sum of gains / sum of losses)
_gains = _returns_arr[_returns_arr > 0]
_losses = _returns_arr[_returns_arr < 0]
_total_gains = float(np.sum(_gains)) if len(_gains) > 0 else 0.0
_total_losses = float(np.abs(np.sum(_losses))) if len(_losses) > 0 else 0.0001
_profit_factor = _total_gains / _total_losses

# M9: Annualized Return (%/yr)
_test_duration_years = _total_candles / (96 * 365) if _total_candles > 0 else 1.0
_annualized_return = _total_return_pct / _test_duration_years if _test_duration_years > 0 else 0.0

# M10: Calmar Ratio (annualized return / max drawdown)
_calmar_ratio = _annualized_return / _max_drawdown_pct if _max_drawdown_pct > 0 else 0.0

# ── PRINT REPORT ──────────────────────────────────────────────

print()
print("=" * 62)
print("       STANDARDIZED EVALUATION REPORT v2.0")
print("=" * 62)
print(f"  Final Equity          : ${_final_equity:,.2f}")
print(f"  Peak Equity           : ${_peak_equity:,.2f}")
print(f"  Total Return          : {_total_return_pct:+.2f}%")
print(f"  Win Rate              : {_win_rate:.2f}%")
print(f"  Trade Count           : {_trades_taken:,} / {_total_candles:,} ({_trade_ratio:.1f}%)")
print(f"  Max Drawdown          : {_max_drawdown_pct:.2f}%")
print(f"    (Peak ${_peak_value:,.0f} -> Trough ${_trough_value:,.0f})")
print(f"  Sharpe Ratio          : {_sharpe_ratio:.4f}")
print(f"  Profit Factor         : {_profit_factor:.3f}")
print(f"  Calmar Ratio          : {_calmar_ratio:.2f}")
print(f"  Annualized Return     : {_annualized_return:+.1f}%/yr")
print("=" * 62)

# ── COPY-PASTE LINE FOR RESULTS FILE ─────────────────────────
# Replace __VERSION__ with your version name (e.g. v9)
print()
print("COPY THIS LINE into standardized_results.txt:")
print(f"__VERSION__ | ${_final_equity:,.0f} | {_win_rate:.2f}% | {_trades_taken} ({_trade_ratio:.1f}%) | {_max_drawdown_pct:.2f}% | {_sharpe_ratio:.4f} | {_profit_factor:.3f} | {_calmar_ratio:.2f} | {_total_return_pct:+.1f}%")
