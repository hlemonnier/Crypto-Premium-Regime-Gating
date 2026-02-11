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
        march_vol_2024_binance   gated -0.021731 -0.003375     -0.003753      20.0   0.010764      0.075000  0.481481    2880            216      1.999306              -15.759992      -0.056251                -40.794909            725.230998
       yen_unwind_2024_binance   gated  0.000728  0.000127     -0.001336      32.0   0.017014      0.109375  0.492063    2880            315      1.999306                0.528098       0.030224                 21.919403            725.230998
yen_followthrough_2024_binance   gated  0.003361  0.000474     -0.000985      12.0   0.011111      0.106597  0.491857    2880            307      1.999306                2.437394       0.023442                 17.000705            725.230998
```

Metric convention: `sharpe` is full-series and non-annualized. Annualized Sharpe columns are exported for reference only.

- Raw mean Sharpe delta (gated - naive, full-series non-annualized): `-0.0149`
- Raw median Sharpe delta (gated - naive, full-series non-annualized): `-0.0037`
- Raw episodes with Sharpe improvement: `1/5`
- Raw episodes with Sharpe degradation: `4/5`
- Robust Sharpe aggregate excludes localized naive episodes: `['yen_unwind_2024_binance']`
- Robust mean Sharpe delta (gated - naive): `-0.0128`
- Robust median Sharpe delta (gated - naive): `-0.0029`
- Conclusion (Sharpe): gated improvement is **not** demonstrated on robust aggregate.
- Raw mean PnL delta (gated - naive): `-0.044300`
- Raw median PnL delta (gated - naive): `-0.001657`
- Raw episodes with PnL improvement: `2/5`
- Raw episodes with PnL degradation: `3/5`
- Robust PnL aggregate excludes localized naive episodes: `['yen_unwind_2024_binance']`
- Robust mean PnL delta (gated - naive): `-0.054769`
- Robust median PnL delta (gated - naive): `-0.000666`
- Conclusion (PnL): gated improvement is **not** demonstrated on robust aggregate.

## On-Chain Validation Snapshot

```text
                       episode  onchain_data_ratio  onchain_usdc_minus_1_abs_mean  onchain_usdt_minus_1_abs_mean  onchain_divergence_abs_mean  onchain_depeg_count  combined_depeg_count
         bybit_usdc_depeg_2023                 1.0                       0.001220                       0.001550                     0.031990                 1440                  1517
        march_vol_2024_binance                 1.0                       0.001626                       0.000299                     0.000937                    0                     0
           okx_usdc_depeg_2023                 1.0                       0.000820                       0.001067                     0.005698                  960                   966
yen_followthrough_2024_binance                 1.0                       0.000000                       0.000593                     0.000579                    0                     0
       yen_unwind_2024_binance                 1.0                       0.001000                       0.000500                     0.001382                    0                     0
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
        march_vol_2024_binance   gated -0.003375            0.031280            0.071822            0.110171                10                           NaN                           0.046895                        False 2024-03-13 17:23:00+00:00 2024-03-13 17:32:00+00:00
        march_vol_2024_binance   naive -0.001718            0.018745            0.051219            0.080387                10                           NaN                           0.038691                        False 2024-03-12 14:06:00+00:00 2024-03-12 14:15:00+00:00
           okx_usdc_depeg_2023   gated -0.000850            0.272061            0.771708            0.981524                10                           NaN                           0.272558                        False 2023-03-10 04:19:00+00:00 2023-03-10 04:28:00+00:00
           okx_usdc_depeg_2023   naive  0.254991            0.078744            0.135040            0.167344                10                      0.378932                           0.158192                        False 2023-03-11 07:47:00+00:00 2023-03-11 07:56:00+00:00
yen_followthrough_2024_binance   gated  0.000474            0.015136            0.041416            0.063594                10                      1.201721                           0.031453                        False 2024-08-07 11:00:00+00:00 2024-08-07 11:09:00+00:00
yen_followthrough_2024_binance   naive  0.000149            0.544933            0.811093            0.961622                10                      2.731983                           0.802926                        False 2024-08-07 04:01:00+00:00 2024-08-07 04:10:00+00:00
       yen_unwind_2024_binance   gated  0.000127            0.016134            0.044825            0.067918                10                      5.277989                           0.029516                        False 2024-08-05 11:18:00+00:00 2024-08-05 11:27:00+00:00
       yen_unwind_2024_binance   naive  0.002550            0.214457            0.373820            0.472342                10                      0.670059                           0.352587                         True 2024-08-05 13:38:00+00:00 2024-08-05 13:47:00+00:00
```

- Naive positive-PnL episodes with >50% of net PnL explained by one `10`-bar window: `1/3`.
Interpretation: when `localized_positive_pnl_flag` is true, performance is structurally fragile and should not be treated as robust signal quality.
Robust aggregate exclusion map is exported to: `reports/final/final_robust_filter.csv`.

## Execution Data Readiness (L2)

```text
                       episode  l2_orderbook_available  tick_trades_available  l2_ready                                                 l2_root
         bybit_usdc_depeg_2023                   False                  False     False          data/processed/orderbook/bybit_usdc_depeg_2023
        march_vol_2024_binance                   False                  False     False         data/processed/orderbook/march_vol_2024_binance
           okx_usdc_depeg_2023                   False                  False     False            data/processed/orderbook/okx_usdc_depeg_2023
yen_followthrough_2024_binance                   False                  False     False data/processed/orderbook/yen_followthrough_2024_binance
       yen_unwind_2024_binance                   False                  False     False        data/processed/orderbook/yen_unwind_2024_binance
```
Execution-quality conclusions are withheld for episodes without complete L2 orderbook + tick-trade coverage.

## Execution Proxy Snapshot (Bar-Level)

No proxy table is reported because fail-closed mode blocked execution diagnostics without full L2 readiness.

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
- `reports/final/final_robust_filter.csv`
- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
- `reports/final/calibration_details.csv`
- `reports/final/calibration_aggregate.csv`

Calibration outputs are available and can be referenced directly in the deck.
