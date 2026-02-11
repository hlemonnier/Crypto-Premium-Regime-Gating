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
                       episode variant    sharpe   pnl_net  max_drawdown  turnover  flip_rate  active_ratio  hit_rate  n_bars  n_active_bars  horizon_days  sharpe_full_annualized  sharpe_active  sharpe_active_annualized  annualization_factor
         bybit_usdc_depeg_2023   gated  0.000000  0.000000      0.000000       0.0   0.005208      0.000000  0.000000    2880              0      1.999306                0.000000       0.000000                  0.000000            725.230998
           okx_usdc_depeg_2023   gated -0.006167 -0.000850     -0.001472       2.0   0.003125      0.001736  0.400000    2880              5      1.999306               -4.472204      -0.131812                -95.594453            725.230998
        march_vol_2024_binance   gated -0.008120 -0.001158     -0.001573      28.0   0.012500      0.111111  0.503125    2880            320      1.999306               -5.889201       0.005144                  3.730856            725.230998
       yen_unwind_2024_binance   gated -0.003271 -0.000791     -0.002133      40.0   0.017014      0.205208  0.492386    2880            591      1.999306               -2.372238       0.011086                  8.040071            725.230998
yen_followthrough_2024_binance   gated  0.000271  0.000045     -0.000809      20.0   0.008681      0.182986  0.504744    2880            527      1.999306                0.196519       0.014857                 10.774897            725.230998
```

Metric convention: `sharpe` is full-series and non-annualized. Annualized Sharpe columns are exported for reference only.

- Mean Sharpe delta (gated - naive, full-series non-annualized): `-0.0107`
- Median Sharpe delta (gated - naive, full-series non-annualized): `0.0003`
- Episodes with Sharpe improvement: `3/5`
- Episodes with Sharpe degradation: `2/5`
- Mean PnL delta (gated - naive): `-0.049356`
- Median PnL delta (gated - naive): `0.000045`
- Episodes with PnL improvement: `3/5`
- Episodes with PnL degradation: `2/5`

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_usdc_minus_1_abs_mean  onchain_usdt_minus_1_abs_mean  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
         bybit_usdc_depeg_2023                 1.0                       0.001220                       0.001550                     0.031995                 1440                  1517
        march_vol_2024_binance                 1.0                       0.000693                       0.000799                     0.000422                    0                     0
           okx_usdc_depeg_2023                 1.0                       0.000820                       0.001067                     0.005698                  960                   966
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

## Execution Quality Snapshot

```text
  venue  n_root_episode_pairs  usdc_preferred_count  usdt_preferred_count  mean_delta_usdc_minus_usdt_bps  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance                    12                    11                     1                       -1.147746                        3.0                        3.5                          0.0                     0.002874
  bybit                     3                     1                     2                        0.770769                        2.0                        2.5                          0.0                     0.000000
    okx                     2                     2                     0                       -4.030721                        1.0                        4.0                          0.0                     0.000000
```

Interpretation: lower `mean_delta_usdc_minus_usdt_bps` indicates lower large-size impact in USDC quotes versus USDT for the same root/venue.

## Generated Artifacts

- `reports/final/final_episode_metrics_long.csv`
- `reports/final/final_episode_metrics_wide.csv`
- `reports/final/figures/sharpe_naive_vs_gated.png`
- `reports/final/figures/pnl_naive_vs_gated.png`
- `reports/final/figures/fliprate_naive_vs_gated.png`
- `reports/final/final_onchain_snapshot.csv`
- `reports/final/final_proxy_coverage.csv`
- `reports/final/execution_quality_report.md`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
