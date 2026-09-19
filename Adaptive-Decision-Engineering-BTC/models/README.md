# Frozen model checkpoints

The paper never retrains. Every configuration loads the same five weights:

| File | Architecture |
|------|----------------|
| `eq_3778_ep_25.pt` | Conv1D A (2 × Conv1D, 32 then 64 filters, kernel 3, GELU) |
| `eq_3768_ep_80.pt` | Conv1D B (same architecture, different run) |
| `eq_3590_ep_122.pt` | LSTM (hidden size 64) |
| `eq_3301_ep_78.pt` | Hybrid CNN-LSTM A |
| `eq_3296_ep_60.pt` | Hybrid CNN-LSTM B |

## How to add them

1. Download the five `.pt` files from the Google Drive/GitHub link in the paper's Data and Code Availability section (or from the Colab `ensemble_models/` folder used during the original runs).
2. Copy them into **this** `models/` folder.
3. Notebooks also look for `./ensemble_models/`. Either:
   - copy the same five files into `notebooks/ensemble_models/`, or
   - edit the path list in the notebook to `../models/eq_....pt`.

Do **not** train replacements if you want to reproduce the paper's numbers.

## Selection caveat (stated in the paper)

These checkpoints were written to disk when a directional backtest on the **same** 35,903-candle test block exceeded USD 3,000 equity. They are therefore not a clean out-of-sample predictive layer. The decision-layer comparison is still valid because every version uses these identical weights.
