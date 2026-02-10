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
                       episode variant     sharpe   pnl_net  max_drawdown  turnover  flip_rate  active_ratio  hit_rate
         bybit_usdc_depeg_2023   gated 114.260760  0.002156     -0.001634       8.0   0.009028      0.009722  0.500000
           okx_usdc_depeg_2023   gated  25.950138  0.013228     -0.008564      38.0   0.017361      0.137847  0.523929
        march_vol_2024_binance   gated  16.002749  0.001014     -0.001095      28.0   0.013194      0.211111  0.521382
       yen_unwind_2024_binance   gated  16.644719  0.026293     -0.003966      47.0   0.018403      0.618403  0.494104
yen_followthrough_2024_binance   gated  10.131636 -0.000617     -0.001211      66.0   0.027778      0.488889  0.501420
```

- Mean Sharpe delta (gated - naive): `-5.1606`

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_usdc_minus_1_abs_mean  onchain_usdt_minus_1_abs_mean  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
         bybit_usdc_depeg_2023                 1.0                       0.001220                       0.001550                     0.031995                 1436                  1517
        march_vol_2024_binance                 1.0                       0.000693                       0.000799                     0.000422                 1436                  1436
           okx_usdc_depeg_2023                 1.0                       0.000820                       0.001067                     0.005698                  956                   962
yen_followthrough_2024_binance                 1.0                       0.002500                       0.003093                     0.000606                    0                     0
       yen_unwind_2024_binance                 1.0                       0.000000                       0.001000                     0.000688                    0                     0
```

## Proxy Coverage Notes

```text
                       episode  proxy_component_count
         bybit_usdc_depeg_2023                      2
        march_vol_2024_binance                      3
           okx_usdc_depeg_2023                      1
yen_followthrough_2024_binance                      3
       yen_unwind_2024_binance                      3
```

Interpretation: debiased premium is strongest when proxy_component_count > 0. When coverage is missing, treat the episode primarily as depeg safety/on-chain validation.

## Generated Artifacts

- `reports/final/final_episode_metrics_long.csv`
- `reports/final/final_episode_metrics_wide.csv`
- `reports/final/figures/sharpe_naive_vs_gated.png`
- `reports/final/figures/pnl_naive_vs_gated.png`
- `reports/final/figures/fliprate_naive_vs_gated.png`
- `reports/final/final_onchain_snapshot.csv`
- `reports/final/final_proxy_coverage.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
