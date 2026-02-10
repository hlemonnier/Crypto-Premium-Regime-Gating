# Executive Summary

This report consolidates final episode performance and on-chain diagnostics.

## Included Episodes

- `bybit_usdc_depeg_2023`
- `okx_usdc_depeg_2023`
- `march_vol_2024_binance`
- `yen_unwind_2024_binance`
- `yen_followthrough_2024_binance`

## Performance Snapshot (Gated)

```text
                       episode variant    sharpe   pnl_net  max_drawdown  turnover  flip_rate  active_ratio  hit_rate
         bybit_usdc_depeg_2023   gated  0.000000  0.000000      0.000000       0.0   0.007292      0.000000  0.000000
           okx_usdc_depeg_2023   gated 55.664006  0.004909     -0.002644      18.0   0.014931      0.035417  0.539216
        march_vol_2024_binance   gated 21.951045  0.000697     -0.001071      20.0   0.010069      0.107986  0.501608
       yen_unwind_2024_binance   gated  7.063795  0.000423     -0.002042      54.0   0.025694      0.532986  0.495765
yen_followthrough_2024_binance   gated  5.860367 -0.001800     -0.002405      52.0   0.024653      0.261806  0.502653
```

- Mean Sharpe delta (gated - naive): `-24.5141`

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
         bybit_usdc_depeg_2023                 1.0                     0.031995                 1436                  1517
        march_vol_2024_binance                 1.0                     0.000422                 1436                  1436
           okx_usdc_depeg_2023                 1.0                     0.005698                  956                   962
yen_followthrough_2024_binance                 1.0                     0.000606                    0                     0
       yen_unwind_2024_binance                 1.0                     0.000688                    0                     0
```

## Generated Artifacts

- `reports/final/final_episode_metrics_long.csv`
- `reports/final/final_episode_metrics_wide.csv`
- `reports/final/figures/sharpe_naive_vs_gated.png`
- `reports/final/figures/pnl_naive_vs_gated.png`
- `reports/final/figures/fliprate_naive_vs_gated.png`
- `reports/final/final_onchain_snapshot.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
