# V9 fee-adjusted backtest — 0.05% transaction fee (0.0005) per taken trade.
# Original strategy logic is unchanged. Models are loaded, not retrained.
# Historical GROSS results in v9 vs v16.txt remain the baseline; this script
# computes fee-adjusted results only.
#
# Fee convention: one taken trade = one 15m round-trip (open -> close).
# Fee is charged once on notional pos_size at execution (not subtracted from
# final equity, and not charged twice for entry+exit).

import os
from datetime import datetime
import pickle
import random
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

os.makedirs("models", exist_ok=True)
os.makedirs("ensemble_models", exist_ok=True)

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed = 0
set_all_seeds(seed)

df = pd.read_csv(f"BTCUSDT-15m-data.csv")
print(df)

print(df.isnull())
print(f"Counts how many missing values there are in each column: {df.isnull().sum()}")
print(f"Total missing values: {df.isnull().sum().sum()}")

train_end_idx = 240_000

df_train = df.iloc[:train_end_idx].copy()
df_test = df.iloc[train_end_idx:].copy()

df_test.reset_index(drop=True, inplace=True)

print(f"Length of df_train: {len(df_train)}")
print(f"Length of df_test: {len(df_test)}")

def build_features(opens, highs, lows, closes, volumes, train_scalers=None):

    # Build features
    feature1 = (closes - opens) / opens
    feature2 = (highs - opens) / opens
    feature3 = (highs - closes) / closes
    feature4 = (lows - opens) / opens
    feature5 = (lows - closes) / closes
    feature6 = (highs - lows) / opens
    feature7 = (highs - lows) / closes
    feature8 = volumes

    # Stack features
    features = [
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
    ]
    num_features = len(features)
    scaled_fts = []

    if train_scalers is None:
        train_scalers = [MaxAbsScaler() for _ in range(num_features)]
        is_train = True
    else:
        is_train = False

    for i in range(num_features):
        # Get train scaler
        scaler = train_scalers[i]
        if is_train:
            # Scaler -> Fit -> Train dataset
            scaled_ft = scaler.fit_transform(features[i].reshape(-1, 1))
            # Save the train scaler in a list
        elif not is_train:
            # Scaler -> NOT Fit -> Test dataset
            scaled_ft = scaler.transform(features[i].reshape(-1, 1))

        # Update list of SCALED features
        scaled_fts.append(scaled_ft.flatten())

    # Stack features
    scaled_features = np.stack(scaled_fts, axis=-1)

    # Get num. features
    num_features = scaled_features.shape[-1]
    return scaled_features, num_features, train_scalers

def preprocess_data(seq_len, df, train_scalers=None):
    m = len(df)

    opens = np.array(df['open'].values)
    highs = np.array(df['high'].values)
    lows = np.array(df['low'].values)
    closes = np.array(df['close'].values)
    volumes = np.array(df['volume'].values)

    # Build features
    features, num_features, train_scalers = build_features(opens, highs, lows, closes, volumes, train_scalers)

    # Calculate number of samples
    num_samples = m - seq_len

    # Create storage for inputs & targets
    X = np.zeros([num_samples, seq_len, num_features], dtype=np.float32)
    Y = np.zeros([num_samples, num_features], dtype=np.float32)

    # Create samples (X, Y)
    for i in range(num_samples):
        X[i] = features[i : i+seq_len]
        Y[i] = features[i+seq_len : i+seq_len+1]

    # Fix alignment
    opens = np.array(opens[seq_len:])
    closes = np.array(closes[seq_len:])

    return X, Y, num_features, opens, closes, train_scalers

# Sequence Length
seq_len = 96

# Preprocess data
X_train, Y_train, num_features, _, _, train_scalers = preprocess_data(seq_len, df_train)
X_test, Y_test, num_features, backtest_opens, backtest_closes, _ = preprocess_data(seq_len, df_test, train_scalers)

m_train = X_train.shape[0]
m_test = X_test.shape[0]
print(f"m_train: {m_train}")
print(f"m_test: {m_test}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("my device: ", device)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device, dtype=torch.float32)
Y_train = torch.from_numpy(Y_train.astype(np.float32)).to(device, dtype=torch.float32)

X_test = torch.from_numpy(X_test.astype(np.float32)).to(device, dtype=torch.float32)
Y_test = torch.from_numpy(Y_test.astype(np.float32)).to(device, dtype=torch.float32)

Y_pred_test = torch.zeros([m_test, num_features], device=device, dtype=torch.float32)

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1 block #1
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()

        # Conv1 block #2
        self.conv2 = nn.Conv1d(
            in_channels=32, 
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.act2 = nn.GELU()

        # Output layer
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, num_features, seq_len)

        # Conv block #1
        x = self.conv1(x)
        x = self.act1(x)

        # Conv block #2
        x = self.conv2(x)
        x = self.act2(x)

        x = x.permute(0, 2, 1) # (batch, seq_len, num_features)
        x = x[:, -1, :] # get last hidden

        # Output layer
        x = self.fc_out(x)
        return x

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(64, num_features)

    def forward(self, x):
        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :] 

        # Output layer
        x = self.fc_out(x)
        return x

class Model_3(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1D
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.act1 = nn.GELU()

        # LSTM
        self.lstm = nn.LSTM(64, 32, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(32, num_features)

    def forward(self, x):
        # Conv1D
        x = x.permute(0, 2, 1) # (batch, num_features, seq_len)
        x = self.conv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, num_features)

        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]

        # Output layer
        x = self.fc_out(x)
        return x

MODELS_INFOS = [
    # Model 1
    {"paths":
     [
        "/content/ensemble_models/eq_3778_ep_25.pt",
        "/content/ensemble_models/eq_3768_ep_80.pt",
      ],
     "architecture": Model_1},

    # Model 2
    {"paths":
     [
        "/content/ensemble_models/eq_3590_ep_122.pt",
      ],
     "architecture": Model_2},

    # Model 3
    {"paths":
     [
        "/content/ensemble_models/eq_3301_ep_78.pt",
        "/content/ensemble_models/eq_3296_ep_60.pt", 
      ],
     "architecture": Model_3},
]

all_actions = [] # List to hold all actions from each model

for model_info in MODELS_INFOS:
    model_paths = model_info["paths"]
    architecture = model_info["architecture"]

    for model_path in model_paths:
        # Initialize model
        model = architecture()

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)

        with torch.no_grad():
            model.eval()

            # # Forward prop
            # Y_pred_test = model(X_test) 

            # Forward propagation
            while True:
                try:
                    # split the test set into chunks
                    chunks = 16
                    m = m_test // chunks
                    
                    for j in range(chunks+1):
                        Y_pred_test[m*j:m*j+m] = model(X_test[m*j:m*j+m])
                    
                    break

                except Exception as err:
                    print(f"Chunks ({chunks}) err: {err}")
                    chunks += 2

        Y_pred_test_cpu = Y_pred_test.detach().cpu().numpy()
        actions = Y_pred_test_cpu[:, 0] > 0
        all_actions.append(actions)

# ===================== IMPROVED BACKTEST v9 =====================
# Base: v6 (Equity $4,679 | Sharpe 0.0339 | Drawdown 8.56%)
#
# FUNDAMENTALLY NEW: Dynamic Velocity Scoring
# ─────────────────────────────────────────────────────────────────
# Every 50 trades, measure the actual win rate of the ensemble.
# If the models are HOT (recent win rate > 50%) → BET BIGGER.
# If the models are COLD (recent win rate < 50%) → BET SMALLER.
#
# This is "double what's working, reduce what's losing" applied
# automatically and continuously in real-time. It's not a manual
# rule — it's a LIVE feedback loop on model performance.
#
# velocity_multiplier = 1.0 + (recent_win_rate - 0.50) * VELOCITY_SCALE
#   Win rate 70% → multiplier ≈ 1.80  (bet 80% more)
#   Win rate 60% → multiplier ≈ 1.40  (bet 40% more)
#   Win rate 50% → multiplier ≈ 1.00  (neutral, base size)
#   Win rate 40% → multiplier ≈ 0.60  (bet 40% less)
#   Win rate 30% → multiplier ≈ 0.20  (bet 80% less)
#
# ALSO ADDED: Properly fixed consensus multipliers from v8
#   80% threshold to avoid skippping too many 4/5 trades
# ─────────────────────────────────────────────────────────────────
# Kept from v6:
#   Weighted Ensemble Voting       (graduated weights)
#   Equity Milestone Tier Sizing   ($750/$1000/$1300/$1600)
#   Soft Circuit Breaker           (half-size at 10% below peak)
#   Losing Streak Protection       (half-size after 2 losses)
# =============================================================================

equities            = []
equity              = 1000
correct_pred        = 0
actions             = []
trades_taken        = 0
skipped_consensus   = 0

# v4/v6: Losing streak tracker
consecutive_losses  = 0
MAX_LOSSES_REDUCE   = 2

# v4/v6: Weighted votes
n_models    = len(all_actions)
raw_weights = [i + 1 for i in range(n_models)]
total_w     = sum(raw_weights)
weights     = [w / total_w for w in raw_weights]

# v6: Soft circuit breaker
peak_equity            = equity
SOFT_CIRCUIT_THRESHOLD = 0.10

# v6: Equity milestone tier sizing
def get_tier_size(eq):
    if   eq >= 5000: return 1600
    elif eq >= 3000: return 1300
    elif eq >= 1500: return 1000
    else:            return 750

# v8: Consensus threshold (keeps the proven win-rate improvement)
CONSENSUS_THRESHOLD = 0.80

# ── v9: Dynamic Velocity Scoring ─────────────────────────────────────────────
VELOCITY_WINDOW = 50        # measure win rate over last N TAKEN trades
VELOCITY_SCALE  = 4.0       # sensitivity: 1% change in win rate → 4% position change
VELOCITY_MIN    = 0.30      # floor: never go below 30% of base size (cold protection)
VELOCITY_MAX    = 2.00      # ceiling: never go above 200% of base size (safety cap)

recent_outcomes = []        # rolling window of 1=win, 0=loss for taken trades

# ── Transaction fee (ONLY intended change vs original) ────────────────────────
# Each taken trade is one round-trip on a single 15m candle: enter at open, exit at close.
# Charge 0.05% once on notional position size. Do not charge entry and exit separately.
TRANSACTION_FEE_RATE = 0.0005
total_fees = 0.0

# =============================================================================
# MAIN BACKTEST LOOP
# =============================================================================
for i in range(len(X_test) - 1):
    curr_open  = backtest_opens[i]
    curr_close = backtest_closes[i]

    # ── Soft Circuit Breaker ──────────────────────────────────────────────────
    if equity > peak_equity:
        peak_equity = equity
    drawdown_from_peak = (peak_equity - equity) / peak_equity
    in_drawdown_zone   = drawdown_from_peak >= SOFT_CIRCUIT_THRESHOLD

    # ── Weighted direction vote ───────────────────────────────────────────────
    combined_actions = [acts[i] for acts in all_actions]

    weighted_long  = sum(weights[j] for j, a in enumerate(combined_actions) if a == 1)
    weighted_short = sum(weights[j] for j, a in enumerate(combined_actions) if a == 0)
    curr_action    = 1 if weighted_long > weighted_short else 0

    long_count  = combined_actions.count(1)
    short_count = combined_actions.count(0)
    agreement   = max(long_count, short_count) / n_models

    # ── Consensus Gate (from v8 — proven to improve win rate) ─────────────────
    if agreement < CONSENSUS_THRESHOLD:
        equities.append(equity)
        actions.append(-1)
        skipped_consensus += 1
        continue

    # ── v9: Dynamic Velocity Multiplier ──────────────────────────────────────
    if len(recent_outcomes) >= VELOCITY_WINDOW:
        recent_win_rate      = sum(recent_outcomes[-VELOCITY_WINDOW:]) / VELOCITY_WINDOW
        velocity_multiplier  = 1.0 + (recent_win_rate - 0.50) * VELOCITY_SCALE
        velocity_multiplier  = max(VELOCITY_MIN, min(VELOCITY_MAX, velocity_multiplier))
    else:
        velocity_multiplier  = 1.0     # neutral until we have enough trades

    # ── Confidence Multiplier (by agreement level) ────────────────────────────
    # Fixed from v8: 4/5 stays at 1.0x (not penalised), 5/5 gets full 1.5x
    if   agreement >= 1.00: confidence_multiplier = 1.5
    elif agreement >= 0.85: confidence_multiplier = 1.0
    else:                   confidence_multiplier = 1.0   # 80-84% = same as strong

    # ── Final Position Size ───────────────────────────────────────────────────
    pos_size = get_tier_size(equity) * confidence_multiplier * velocity_multiplier

    if consecutive_losses >= MAX_LOSSES_REDUCE:
        pos_size *= 0.5         # losing streak halving

    if in_drawdown_zone:
        pos_size *= 0.5         # circuit breaker halving

    # ── Execute Trade ─────────────────────────────────────────────────────────
    pct_change = (curr_close - curr_open) / curr_open
    pnl        = abs(pos_size * pct_change)

    if curr_action == 1:        # Long
        if pct_change > 0:
            equity += pnl;  correct_pred += 1;  consecutive_losses = 0
            recent_outcomes.append(1)           # record win
        elif pct_change < 0:
            equity -= pnl;  consecutive_losses += 1
            recent_outcomes.append(0)           # record loss
        else:
            recent_outcomes.append(0)
    else:                       # Short
        if pct_change < 0:
            equity += pnl;  correct_pred += 1;  consecutive_losses = 0
            recent_outcomes.append(1)
        elif pct_change > 0:
            equity -= pnl;  consecutive_losses += 1
            recent_outcomes.append(0)
        else:
            recent_outcomes.append(0)

    # Apply 0.05% fee once per taken trade (round-trip), after gross PnL.
    # Fee-adjusted equity feeds subsequent tier sizing / circuit breaker.
    fee = pos_size * TRANSACTION_FEE_RATE
    equity -= fee
    total_fees += fee
    equities.append(equity)
    actions.append(curr_action)
    trades_taken += 1

# ── Summary ───────────────────────────────────────────────────────────────────
total_steps = len(X_test) - 1
accuracy    = correct_pred / max(trades_taken, 1) * 100

# Final velocity state
if len(recent_outcomes) >= VELOCITY_WINDOW:
    final_velocity_wr  = sum(recent_outcomes[-VELOCITY_WINDOW:]) / VELOCITY_WINDOW
    final_velocity_mul = 1.0 + (final_velocity_wr - 0.50) * VELOCITY_SCALE
    final_velocity_mul = max(VELOCITY_MIN, min(VELOCITY_MAX, final_velocity_mul))
else:
    final_velocity_wr, final_velocity_mul = 0.0, 1.0

print(f"Total candles evaluated   : {total_steps}")
print(f"Trades taken              : {trades_taken}  ({trades_taken/total_steps*100:.1f}%)")
print(f"Skipped (low consensus)   : {skipped_consensus} ({skipped_consensus/total_steps*100:.1f}%)")
print(f"Win rate on taken trades  : {accuracy:.2f}%")
print(f"Peak equity reached       : ${peak_equity:.0f}")
print(f"Transaction fee rate      : {TRANSACTION_FEE_RATE*100:.2f}% per taken trade")
print(f"Total fees paid           : ${total_fees:.2f}")
print(f"Final equity (fee-adj.)   : ${equity:.2f}")
print(f"Final velocity win rate   : {final_velocity_wr*100:.1f}%")
print(f"Final velocity multiplier : {final_velocity_mul:.2f}x")

# Calculate max drawdown
def calculate_max_drawdown(equities):
    # Convert to numpy array for vectorized operations
    equity = np.array(equities)

    # Calculate cumulative maximum (peak equity)
    peak = np.maximum.accumulate(equity)

    # Drawdown: (Peak - Current) / Peak
    drawdown = (peak - equity) / peak

    # Index of maximum drawdown
    max_dd_idx = np.argmax(drawdown)

    # Find the peak before the maximum drawdown
    peak_idx = np.argmax(equity[:max_dd_idx + 1])

    return {
        "max_drawdown": drawdown[max_dd_idx],
        "peak_index": peak_idx,
        "trough_index": max_dd_idx,
        "peak_value": equity[peak_idx],
        "trough_value": equity[max_dd_idx]
    }

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(equities):
    # Step 1: Compute 15-minute returns
    returns = []
    for i in range(1, len(equities)):
        ret = (equities[i] - equities[i-1]) / equities[i-1]
        returns.append(ret)

    # Step 2: Mean return
    mean_return = np.mean(returns)

    # Step 3: Standard deviation
    std_dev = np.std(returns, ddof=1)

    # Step 4: Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = mean_return / std_dev

    sharpe_ratio = round(sharpe_ratio, 4)
    return sharpe_ratio


def plot_equity(backtest_opens, equities, result, sharpe_ratio):
    # Chart title
    chart_title = f"Equity: {equities[-1]:.0f}, Sharpe Ratio {sharpe_ratio:.4f}, Drawdown {result['max_drawdown']:.2%}"
    print(f"Result: {chart_title}")
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    scaler_earned = MinMaxScaler(feature_range=(min(equities)-0.1, max(equities)+0.1))
    plt.plot(scaler_earned.fit_transform(np.array(backtest_opens).reshape(-1,1)), color='green', label='BTC Price')
    plt.plot(equities, label="Equity Curve", color="blue")
    # Highlight peak and trough of max drawdown
    plt.scatter(
        [result["peak_index"], result["trough_index"]],
        [result["peak_value"], result["trough_value"]],
        color="red",
        zorder=5,
        label="Max Drawdown"
    )
    # Annotate max drawdown
    plt.annotate(
        f"Max Drawdown: {result['max_drawdown']:.2%}",
        xy=(result["trough_index"], result["trough_value"]),
        xytext=(result["trough_index"] + 10, result["trough_value"] * 0.9),
        arrowprops=dict(facecolor="red", shrink=0.05),
        )
    plt.title(chart_title)
    plt.xlabel("Time Step")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(chart_title.replace('%', '')+'.png', dpi=300)

# Calculate sharpe ratio
sharpe_ratio = calculate_sharpe_ratio(equities)

# Calculate Max DD
result = calculate_max_drawdown(equities)

# Generate the plot
plot_equity(backtest_opens, equities, result, sharpe_ratio)

# ── Existing standardized metrics (same block used for historical v9 vs v16) ──
_starting_capital = 1000.0
_final_equity = equities[-1]
_peak_equity = max(equities)
_total_return_pct = ((_final_equity - _starting_capital) / _starting_capital) * 100
_win_rate = (correct_pred / trades_taken) * 100 if trades_taken > 0 else 0.0
_trade_ratio = (trades_taken / total_steps) * 100 if total_steps > 0 else 0.0
_equity_arr = np.array(equities, dtype=np.float64)
_running_peak = np.maximum.accumulate(_equity_arr)
_drawdown_arr = (_running_peak - _equity_arr) / _running_peak
_max_drawdown_pct = float(np.max(_drawdown_arr)) * 100
_max_dd_idx = int(np.argmax(_drawdown_arr))
_peak_idx = int(np.argmax(_equity_arr[:_max_dd_idx + 1])) if _max_dd_idx > 0 else 0
_peak_value = float(_equity_arr[_peak_idx])
_trough_value = float(_equity_arr[_max_dd_idx])
_returns = []
for _i in range(1, len(equities)):
    _ret = (equities[_i] - equities[_i-1]) / equities[_i-1] if equities[_i-1] != 0 else 0
    _returns.append(_ret)
_returns_arr = np.array(_returns)
_mean_return = float(np.mean(_returns_arr))
_std_return = float(np.std(_returns_arr, ddof=1)) if len(_returns_arr) > 1 else 0.0001
_sharpe_ratio = _mean_return / _std_return if _std_return != 0 else 0.0
_gains = _returns_arr[_returns_arr > 0]
_losses = _returns_arr[_returns_arr < 0]
_total_gains = float(np.sum(_gains)) if len(_gains) > 0 else 0.0
_total_losses = float(np.abs(np.sum(_losses))) if len(_losses) > 0 else 0.0001
_profit_factor = _total_gains / _total_losses
_test_duration_years = total_steps / (96 * 365) if total_steps > 0 else 1.0
_annualized_return = _total_return_pct / _test_duration_years if _test_duration_years > 0 else 0.0
_calmar_ratio = _annualized_return / _max_drawdown_pct if _max_drawdown_pct > 0 else 0.0

print()
print("=" * 62)
print("       STANDARDIZED EVALUATION REPORT (FEE-ADJUSTED)")
print("=" * 62)
print(f"  Final Equity          : ${_final_equity:,.2f}")
print(f"  Peak Equity           : ${_peak_equity:,.2f}")
print(f"  Total Return          : {_total_return_pct:+.2f}%")
print(f"  Win Rate              : {_win_rate:.2f}%")
print(f"  Trade Count           : {trades_taken:,} / {total_steps:,} ({_trade_ratio:.1f}%)")
print(f"  Max Drawdown          : {_max_drawdown_pct:.2f}%")
print(f"    (Peak ${_peak_value:,.0f} -> Trough ${_trough_value:,.0f})")
print(f"  Sharpe Ratio          : {_sharpe_ratio:.4f}")
print(f"  Profit Factor         : {_profit_factor:.3f}")
print(f"  Calmar Ratio          : {_calmar_ratio:.2f}")
print(f"  Annualized Return     : {_annualized_return:+.1f}%/yr")
print("=" * 62)
