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
         bybit_usdc_depeg_2023   gated  0.000000  0.000000      0.000000       0.0   0.004861      0.000000  0.000000
           okx_usdc_depeg_2023   gated 57.961992  0.013657     -0.004786      40.0   0.014583      0.066319  0.036458
        march_vol_2024_binance   gated 14.106649 -0.000322     -0.002110      74.0   0.028125      0.364583  0.185417
       yen_unwind_2024_binance   gated 12.040133  0.001009     -0.001981      64.0   0.025347      0.465278  0.232986
yen_followthrough_2024_binance   gated  9.023636 -0.001739     -0.002162      60.0   0.023611      0.268403  0.135417
```

- Mean Sharpe delta (gated - naive): `-62.1795`

## On-Chain Validation Snapshot

```text
              episode  onchain_data_ratio  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
bybit_usdc_depeg_2023                 1.0                     0.031995                 1436                  1517
  okx_usdc_depeg_2023                 1.0                     0.005698                  956                   962
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
