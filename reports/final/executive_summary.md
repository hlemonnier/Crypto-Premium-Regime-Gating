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
       yen_unwind_2024_binance   gated -0.008586 -0.002226     -0.002599      46.0   0.021181      0.237153  0.493411    2880            683      1.999306               -6.226889       0.000592                  0.429375            725.230998
yen_followthrough_2024_binance   gated  0.000271  0.000045     -0.000809      20.0   0.008681      0.182986  0.504744    2880            527      1.999306                0.196519       0.014857                 10.774897            725.230998
```

Metric convention: `sharpe` is full-series and non-annualized. Annualized Sharpe columns are exported for reference only.

- Mean Sharpe delta (gated - naive, full-series non-annualized): `-0.0112`
- Median Sharpe delta (gated - naive, full-series non-annualized): `0.0003`
- Episodes with Sharpe improvement: `3/5`
- Episodes with Sharpe degradation: `2/5`
- Conclusion (Sharpe): gated improvement is **not** demonstrated on average.
- Mean PnL delta (gated - naive): `-0.044091`
- Median PnL delta (gated - naive): `0.000045`
- Episodes with PnL improvement: `3/5`
- Episodes with PnL degradation: `2/5`
- Conclusion (PnL): gated improvement is **not** demonstrated on average.

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_usdc_minus_1_abs_mean  onchain_usdt_minus_1_abs_mean  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
         bybit_usdc_depeg_2023                 1.0                       0.001220                       0.001550                     0.031990                 1440                  1517
        march_vol_2024_binance                 1.0                       0.000693                       0.000799                     0.000422                    0                     0
           okx_usdc_depeg_2023                 1.0                       0.000820                       0.001067                     0.005698                  960                   966
yen_followthrough_2024_binance                 1.0                       0.002500                       0.003093                     0.000606                    0                     0
       yen_unwind_2024_binance                 1.0                       0.000000                       0.001000                     0.000687                    0                     0
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

## PnL Localization Diagnostics

```text
                       episode variant   net_pnl  top1_abs_pnl_share  top3_abs_pnl_share  top5_abs_pnl_share  best_window_bars  best_window_share_of_net_pnl  best_window_share_of_positive_pnl  localized_positive_pnl_flag         best_window_start           best_window_end
         bybit_usdc_depeg_2023   gated  0.000000            0.000000            0.000000            0.000000                10                           NaN                                NaN                        False 2023-03-10 00:00:00+00:00 2023-03-10 00:09:00+00:00
         bybit_usdc_depeg_2023   naive -0.038098            0.017617            0.042178            0.058012                10                           NaN                           0.040633                        False 2023-03-11 07:13:00+00:00 2023-03-11 07:22:00+00:00
        march_vol_2024_binance   gated -0.001158            0.014462            0.036776            0.056691                10                           NaN                           0.031282                        False 2024-03-13 11:51:00+00:00 2024-03-13 12:00:00+00:00
        march_vol_2024_binance   naive -0.002353            0.020285            0.055161            0.078983                10                           NaN                           0.042771                        False 2024-03-13 14:46:00+00:00 2024-03-13 14:55:00+00:00
           okx_usdc_depeg_2023   gated -0.000850            0.272061            0.771708            0.981524                10                           NaN                           0.272558                        False 2023-03-10 04:19:00+00:00 2023-03-10 04:28:00+00:00
           okx_usdc_depeg_2023   naive  0.254991            0.078744            0.135040            0.167344                10                      0.378932                           0.158192                        False 2023-03-11 07:47:00+00:00 2023-03-11 07:56:00+00:00
yen_followthrough_2024_binance   gated  0.000045            0.009685            0.028555            0.043948                10                     12.347062                           0.020072                        False 2024-08-07 05:25:00+00:00 2024-08-07 05:34:00+00:00
yen_followthrough_2024_binance   naive  0.000000            0.000000            0.000000            0.000000                10                           NaN                                NaN                        False 2024-08-07 00:00:00+00:00 2024-08-07 00:09:00+00:00
       yen_unwind_2024_binance   gated -0.002226            0.012031            0.025904            0.038212                10                           NaN                           0.016339                        False 2024-08-05 03:01:00+00:00 2024-08-05 03:10:00+00:00
       yen_unwind_2024_binance   naive  0.001724            0.128880            0.331680            0.452121                10                      0.678682                           0.240795                         True 2024-08-05 13:06:00+00:00 2024-08-05 13:15:00+00:00
```

- Naive positive-PnL episodes with >50% of net PnL explained by one `10`-bar window: `1/2`.
Interpretation: when `localized_positive_pnl_flag` is true, performance is structurally fragile and should not be treated as robust signal quality.

## Execution Proxy Snapshot (Bar-Level)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                    12                 -0.882780                   -0.686499                    -0.917649                      -0.686171              -0.091662                -0.075198                     3                        3.0                        3.0                          0.0                     0.002874
  bybit        spot                     3                  0.770769                    0.929259                     0.873654                       1.037368               1.009124                -0.138869                     0                        2.0                        2.5                          0.0                     0.000000
    okx derivatives                     2                 -4.030721                   -4.030721                    -2.057024                      -2.057024               0.383998                 0.383998                     0                        1.0                        4.0                          0.0                     0.000000
```

Interpretation: compare raw, excess, and normalized deltas jointly. A negative delta means lower proxy impact in USDC quotes versus USDT for the same root/venue.
Scope note: this section is a bar-level proxy and does not validate order-book microstructure items from the Mike brief.
Comparability note: venue comparisons are only defensible within the same `market_type` (`spot` vs `derivatives`).
Decision guardrail: do not conclude 'better liquidity' without L2 order-book replay (book-walk), and normalization of tick/lot/fees/funding/contract specs.

## Generated Artifacts

- `reports/final/final_episode_metrics_long.csv`
- `reports/final/final_episode_metrics_wide.csv`
- `reports/final/figures/sharpe_naive_vs_gated.png`
- `reports/final/figures/pnl_naive_vs_gated.png`
- `reports/final/figures/fliprate_naive_vs_gated.png`
- `reports/final/final_onchain_snapshot.csv`
- `reports/final/final_proxy_coverage.csv`
- `reports/final/final_pnl_localization.csv`
- `reports/final/execution_quality_report.md`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
