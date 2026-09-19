# Disclaimer

This repository is **research code**. It is not investment advice and it is not a live-trading system.

The paper's headline gross results (Version 9 at USD 4,822 and Version 16 at USD 4,734 from a USD 1,000 start) were produced **with zero transaction costs**. The same two configurations were then re-run at 0.02% and 0.05% per simulated round-trip. **Neither rate is survivable** at this turnover (24,900 trades on 35,903 test candles). Version 9 finishes at USD -61.58 / USD -3,577.36; Version 16 finishes at USD +71.56 / USD -3,234.01.

The simulator has no insolvency halt, so large negative balances are artefacts, not liquidation values. Slippage is not modelled.

Do not deploy any configuration in this repository with live capital.
