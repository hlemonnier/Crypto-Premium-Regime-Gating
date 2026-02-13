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
         bybit_usdc_depeg_2023   gated  0.000000  0.000000      0.000000       0.0   0.001042      0.000000  0.000000    2880              0      1.999306                  0.000000                        NaN                     NaN                 0.000000                            NaN               1.089498                0.000000       0.000000                  0.000000            725.230998
           okx_usdc_depeg_2023   gated  0.000639  0.000120     -0.001374       7.0   0.014936      0.104167  0.453333    2880            300      1.999306                  0.304333                   0.903115                0.812433                 0.990909                       1.286634               1.087081                0.463411       0.005684                  4.122184            725.230998
        march_vol_2024_binance   gated -0.017507 -0.001347     -0.001495      18.2   0.011115      0.377083  0.482505    2880           1086      1.999306                  0.308517                   0.045755                0.022627                 0.793162                       1.660772               1.001455              -12.696265      -0.025194                -18.271812            725.230998
       yen_unwind_2024_binance   gated -0.002057 -0.000225     -0.001107      19.0   0.011115      0.342014  0.497462    2880            985      1.999306                  0.342741                   0.275138                0.043380                 0.803636                       1.813076               1.003464               -1.491509      -0.001013                 -0.734810            725.230998
yen_followthrough_2024_binance   gated -0.014557 -0.000596     -0.000778       7.8   0.003473      0.234722  0.483728    2880            676      1.999306                  0.270414                   0.016630                0.001741                 0.742857                       1.734707               1.001958              -10.557343      -0.018220                -13.213467            725.230998
```

Metric convention: `sharpe` is full-series and non-annualized. Annualized Sharpe columns are exported for reference only.

- Raw mean Sharpe delta (gated - naive, full-series non-annualized): `-0.0233`
- Raw median Sharpe delta (gated - naive, full-series non-annualized): `-0.0288`
- Raw episodes with Sharpe improvement: `1/5`
- Raw episodes with Sharpe degradation: `4/5`
- Robust Sharpe aggregate exclusions: `['bybit_usdc_depeg_2023']`
- Non-comparable episodes (gated inactive/low activity <= 0.01 active ratio): `['bybit_usdc_depeg_2023']`
- Robust mean Sharpe delta (gated - naive): `-0.0317`
- Robust median Sharpe delta (gated - naive): `-0.0295`
- Conclusion (Sharpe): gated improvement is **not** demonstrated on robust aggregate (comparable episodes only).
- Raw mean PnL delta (gated - naive): `-0.046144`
- Raw median PnL delta (gated - naive): `-0.003324`
- Raw episodes with PnL improvement: `1/5`
- Raw episodes with PnL degradation: `4/5`
- Robust PnL aggregate exclusions: `['bybit_usdc_depeg_2023']`
- Robust mean PnL delta (gated - naive): `-0.067332`
- Robust median PnL delta (gated - naive): `-0.004162`
- Conclusion (PnL): gated improvement is **not** demonstrated on robust aggregate (comparable episodes only).

## Claim Status

- Performance claim (`improved decision-making`): **not supported** by current robust aggregate (comparable episodes only).
- Positioning: present this build as a safety/risk-control gating framework under calibration, not as a proven outperformance strategy.
- Promotion rule: only switch to outperformance messaging when both robust mean Sharpe delta and robust mean PnL delta are strictly positive.
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
         bybit_usdc_depeg_2023   gated  0.000000            0.000000            0.000000            0.000000                10                           NaN                                NaN                        False 2023-03-10 00:00:00+00:00 2023-03-10 00:09:00+00:00
         bybit_usdc_depeg_2023   naive -0.038610            0.017585            0.042103            0.057908                10                           NaN                           0.040572                        False 2023-03-11 07:13:00+00:00 2023-03-11 07:22:00+00:00
        march_vol_2024_binance   gated -0.001347            0.009835            0.026219            0.040992                10                           NaN                           0.022531                        False 2024-03-12 04:42:00+00:00 2024-03-12 04:51:00+00:00
        march_vol_2024_binance   naive  0.001977            0.009035            0.022211            0.032667                10                      0.356736                           0.019156                        False 2024-03-12 15:03:00+00:00 2024-03-12 15:12:00+00:00
           okx_usdc_depeg_2023   gated  0.000120            0.037946            0.106031            0.151309                10                      9.510835                           0.062948                        False 2023-03-11 14:31:00+00:00 2023-03-11 14:40:00+00:00
           okx_usdc_depeg_2023   naive  0.259672            0.079570            0.132290            0.162851                10                      0.372101                           0.149919                        False 2023-03-11 07:47:00+00:00 2023-03-11 07:56:00+00:00
yen_followthrough_2024_binance   gated -0.000596            0.008994            0.025567            0.040737                10                           NaN                           0.018543                        False 2024-08-08 09:22:00+00:00 2024-08-08 09:31:00+00:00
yen_followthrough_2024_binance   naive  0.000856            0.009235            0.026364            0.040496                10                      0.515284                           0.016976                        False 2024-08-07 04:06:00+00:00 2024-08-07 04:15:00+00:00
       yen_unwind_2024_binance   gated -0.000225            0.010138            0.028781            0.041374                10                           NaN                           0.025976                        False 2024-08-05 13:38:00+00:00 2024-08-05 13:47:00+00:00
       yen_unwind_2024_binance   naive  0.004775            0.036140            0.069407            0.087615                10                      0.357880                           0.072469                        False 2024-08-05 13:38:00+00:00 2024-08-05 13:47:00+00:00
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
