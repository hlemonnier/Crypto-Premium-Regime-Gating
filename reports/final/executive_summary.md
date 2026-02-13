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
                       episode variant    sharpe   pnl_net  max_drawdown  turnover  flip_rate  active_ratio  hit_rate  n_bars  n_active_bars  horizon_days  avg_active_position_size  edge_net_trade_median_bps  edge_net_trade_p10_bps  optimal_size_trade_mean  break_even_premium_median_bps  cost_bps_applied_mean  sharpe_full_annualized  sharpe_active  sharpe_active_annualized  annualization_factor
         bybit_usdc_depeg_2023   gated  0.101706  0.035920     -0.001010    359.70   0.120528      0.203125  0.654701    2880            585      1.999306                  0.835299                   2.126637                1.228115                 0.979304                       0.973771               0.906983               73.760190       0.327379                237.425700            725.230998
           okx_usdc_depeg_2023   gated  0.158886  0.212459     -0.007745    679.10   0.225773      0.322222  0.663793    2880            928      1.999306                  0.806950                   4.140379                2.314640                 0.997964                       0.777654               0.918419              115.228787       0.327724                237.675288            725.230998
        march_vol_2024_binance   gated -0.011589 -0.001224     -0.002209    300.05   0.138937      0.348611  0.545817    2880           1004      1.999306                  0.500448                   0.483652                0.178467                 0.878624                       1.510036               0.757265               -8.404884       0.089826                 65.144606            725.230998
       yen_unwind_2024_binance   gated  0.061701  0.009725     -0.000604    270.00   0.118097      0.304861  0.570615    2880            878      1.999306                  0.465718                   0.474920                0.157792                 0.889765                       2.196879               0.760552               44.747376       0.194908                141.353221            725.230998
yen_followthrough_2024_binance   gated -0.005732 -0.000551     -0.001552    247.50   0.119486      0.296875  0.528655    2880            855      1.999306                  0.466901                   0.337284                0.161215                 0.866667                       1.740763               0.753715               -4.157140       0.086344                 62.619431            725.230998
```

Metric convention: `sharpe` is full-series and non-annualized. Annualized Sharpe columns are exported for reference only.

- Raw mean Sharpe delta (gated - naive, full-series non-annualized): `0.0437`
- Raw median Sharpe delta (gated - naive, full-series non-annualized): `0.0323`
- Raw episodes with Sharpe improvement: `3/5`
- Raw episodes with Sharpe degradation: `2/5`
- Robust mean Sharpe delta (gated - naive): `0.0437`
- Robust median Sharpe delta (gated - naive): `0.0323`
- Conclusion (Sharpe): gated improvement is positive on robust aggregate (comparable episodes only).
- Raw mean PnL delta (gated - naive): `0.003927`
- Raw median PnL delta (gated - naive): `-0.001457`
- Raw episodes with PnL improvement: `2/5`
- Raw episodes with PnL degradation: `3/5`
- Robust mean PnL delta (gated - naive): `0.003927`
- Robust median PnL delta (gated - naive): `-0.001457`
- Conclusion (PnL): gated improvement is positive on robust aggregate (comparable episodes only).

## Claim Status

- Performance claim (`improved decision-making`): **supported** on robust aggregate (comparable episodes only).
- Positioning: outperformance framing is allowed because both robust Sharpe and robust PnL deltas are positive.
- Machine-readable claim status export: `reports/final/claim_status.csv`.

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_usdc_minus_1_abs_mean  onchain_usdt_minus_1_abs_mean  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count  onchain_source_timestamp_ratio  onchain_source_age_hours_median  stress_usdc_depeg_count  stress_usdt_concern_count  stress_technical_flow_count  avg_trade_confidence  avg_trade_position_size
         bybit_usdc_depeg_2023                 1.0                       0.001220                       0.001550                     0.031990                 1440                  1517                             0.0                              NaN                        0                          0                            0                   0.0                      0.0
        march_vol_2024_binance                 1.0                       0.001626                       0.000299                     0.000937                    0                     0                             1.0                        35.991667                        0                          0                            0                   NaN                      NaN
           okx_usdc_depeg_2023                 1.0                       0.000820                       0.001067                     0.005698                  960                   966                             0.0                              NaN                        0                          0                            0                   NaN                      NaN
yen_followthrough_2024_binance                 1.0                       0.000000                       0.000593                     0.000579                    0                     0                             1.0                        35.991667                        0                          0                            0                   NaN                      NaN
       yen_unwind_2024_binance                 1.0                       0.001000                       0.000500                     0.001382                    0                     0                             1.0                        83.991667                        0                          0                            0                   NaN                      NaN
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
         bybit_usdc_depeg_2023   gated  0.035920            0.010081            0.027451            0.042560                10                      0.074229                           0.032993                        False 2023-03-10 01:15:00+00:00 2023-03-10 01:24:00+00:00
         bybit_usdc_depeg_2023   naive -0.038585            0.017586            0.042104            0.057909                10                           NaN                           0.040572                        False 2023-03-11 07:13:00+00:00 2023-03-11 07:22:00+00:00
        march_vol_2024_binance   gated -0.001224            0.006073            0.016674            0.026575                10                           NaN                           0.020522                        False 2024-03-13 01:26:00+00:00 2024-03-13 01:35:00+00:00
        march_vol_2024_binance   naive  0.002102            0.009040            0.022223            0.032685                10                      0.347415                           0.019812                        False 2024-03-12 15:03:00+00:00 2024-03-12 15:12:00+00:00
           okx_usdc_depeg_2023   gated  0.212459            0.016278            0.039148            0.058473                10                      0.037517                           0.023361                        False 2023-03-09 20:48:00+00:00 2023-03-09 20:57:00+00:00
           okx_usdc_depeg_2023   naive  0.267297            0.079578            0.132304            0.162868                10                      0.361486                           0.149050                        False 2023-03-11 07:47:00+00:00 2023-03-11 07:56:00+00:00
yen_followthrough_2024_binance   gated -0.000551            0.013285            0.032368            0.046900                10                           NaN                           0.027889                        False 2024-08-08 16:05:00+00:00 2024-08-08 16:14:00+00:00
yen_followthrough_2024_binance   naive  0.000906            0.009235            0.026364            0.040496                10                      0.514440                           0.017921                        False 2024-08-07 04:06:00+00:00 2024-08-07 04:15:00+00:00
       yen_unwind_2024_binance   gated  0.009725            0.014031            0.036445            0.051487                10                      0.111043                           0.033111                        False 2024-08-05 01:13:00+00:00 2024-08-05 01:22:00+00:00
       yen_unwind_2024_binance   naive  0.004975            0.036140            0.069407            0.087615                10                      0.348518                           0.073219                        False 2024-08-05 13:38:00+00:00 2024-08-05 13:47:00+00:00
```

- Naive positive-PnL episodes with >50% of net PnL explained by one `10`-bar window: `0/4`.
Interpretation: when `localized_positive_pnl_flag` is true, performance is structurally fragile and should not be treated as robust signal quality.
Robust aggregate exclusion map is exported to: `reports/final/final_robust_filter.csv`.

## Execution Data Readiness (L2)

```text
                       episode  l2_orderbook_available  tick_trades_available  l2_ready                                                 l2_root
        march_vol_2024_binance                    True                   True      True         data/processed/orderbook/march_vol_2024_binance
yen_followthrough_2024_binance                    True                   True      True data/processed/orderbook/yen_followthrough_2024_binance
       yen_unwind_2024_binance                    True                   True      True        data/processed/orderbook/yen_unwind_2024_binance
```

## Execution Proxy Snapshot (Bar-Level)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                     3                  0.019041                    0.051785                     0.160341                   7.379463e-20              -0.001413                 0.000518                     3                        4.0                        4.5                          0.0                     0.002874
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
- `reports/final/claim_status.csv`
- `reports/final/final_onchain_snapshot.csv`
- `reports/final/final_proxy_coverage.csv`
- `reports/final/final_pnl_localization.csv`
- `reports/final/execution_quality_report.md`
- `reports/final/final_robust_filter.csv`
- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
