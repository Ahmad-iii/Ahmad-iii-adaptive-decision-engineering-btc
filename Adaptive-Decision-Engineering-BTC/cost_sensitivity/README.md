# Cost-sensitivity scripts

These are the Version 9 and Version 16 decision layers with **one accounting change**: a fee charged once on traded notional for every taken round-trip.

The rest of the logic (frozen checkpoints, 80% consensus, velocity, milestone tiers, circuit breaker, V16 regime rules) is unchanged.

## How to run

```bash
# from the repository root, with BTCUSDT-15m-data.csv in this folder
# or in the current working directory
python cost_sensitivity/V9_fee_adjusted_0.05pct.py
python cost_sensitivity/V16_fee_adjusted_0.05pct.py
```

## How to switch the cost rate

Open the script and edit the single line:

```python
TRANSACTION_FEE_RATE = 0.0005   # paper's 0.05% (taker-equivalent) run
# TRANSACTION_FEE_RATE = 0.0002 # paper's 0.02% (maker-equivalent) run
```

Skipped candles (consensus not met) are not charged. The fee is applied after gross PnL and then the **same** equity variable is read by the sizing rules, which is why costs prevent the milestone ratchet from ever engaging.

Paper numbers for a USD 1,000 start:

| Version | Rate | Final equity | Total costs |
|---------|------|--------------|-------------|
| V9 | 0% | +4,822.42 | 0 |
| V9 | 0.02% | −61.58 | 2,408.09 |
| V9 | 0.05% | −3,577.36 | 5,939.86 |
| V16 | 0% | +4,734.36 | 0 |
| V16 | 0.02% | +71.56 | 2,303.83 |
| V16 | 0.05% | −3,234.01 | 5,577.23 |
