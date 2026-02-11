# Execution Diagnostics Blocked (L2 Missing)

Execution-quality conclusions are disabled because required L2 orderbook + tick-trade coverage is incomplete for selected episodes.

## L2 Coverage

```text
                       episode  l2_orderbook_available  tick_trades_available  l2_ready                                                 l2_root
         bybit_usdc_depeg_2023                   False                  False     False          data/processed/orderbook/bybit_usdc_depeg_2023
           okx_usdc_depeg_2023                   False                  False     False            data/processed/orderbook/okx_usdc_depeg_2023
        march_vol_2024_binance                   False                  False     False         data/processed/orderbook/march_vol_2024_binance
       yen_unwind_2024_binance                   False                  False     False        data/processed/orderbook/yen_unwind_2024_binance
yen_followthrough_2024_binance                   False                  False     False data/processed/orderbook/yen_followthrough_2024_binance
```

No bar-proxy fallback was produced because `--allow-bar-proxy-without-l2` was not set.
This fail-closed behavior prevents unsupported venue/quote liquidity rankings.

## Artifacts

- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
