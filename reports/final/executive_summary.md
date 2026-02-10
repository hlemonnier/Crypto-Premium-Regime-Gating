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
         bybit_usdc_depeg_2023   gated  0.000000  0.000000      0.000000       0.0   0.005208      0.000000  0.000000
           okx_usdc_depeg_2023   gated 42.764729  0.004972     -0.003079      16.0   0.010764      0.044097  0.535433
        march_vol_2024_binance   gated 23.550434  0.001240     -0.001071      26.0   0.012153      0.151389  0.511468
       yen_unwind_2024_binance   gated  5.514261 -0.000239     -0.002042      50.0   0.023958      0.504861  0.495186
yen_followthrough_2024_binance   gated  5.814444 -0.001743     -0.002397      56.0   0.024306      0.336111  0.501033
```

- Mean Sharpe delta (gated - naive): `-27.0932`

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
