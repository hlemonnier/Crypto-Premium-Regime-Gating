# Calibration Report

Comparison: baseline Notice defaults vs tuned config on 2024 episodes.

Metric naming: `*_sharpe_full_raw` denotes full-series, non-annualized Sharpe.

## Tuned Parameters

- `strategy.entry_k`: `1.0`
- `strategy.t_widen_quantile`: `0.97`
- `strategy.chi_widen_quantile`: `0.99`
- `regimes.stress_quantile`: `0.95`
- `regimes.recovery_quantile`: `0.6`

## Episode Details

```text
                       episode                                                              matrix_path  baseline_gated_sharpe_full_raw  tuned_gated_sharpe_full_raw  delta_gated_sharpe_full_raw  baseline_gated_pnl_net  tuned_gated_pnl_net  delta_gated_pnl_net  baseline_gated_active_ratio  tuned_gated_active_ratio
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv                       -0.005924                    -0.018867                    -0.012943               -0.000261            -0.001459            -0.001197                     0.059722                  0.377083
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv                        0.000000                    -0.013725                    -0.013725                0.000000            -0.000562            -0.000562                     0.000000                  0.234722
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv                       -0.000754                    -0.002575                    -0.001821               -0.000059            -0.000291            -0.000231                     0.304514                  0.342014
```

## Aggregate Stats

```text
                                    mean    median       min       max
baseline_gated_sharpe_full_raw -0.002226 -0.000754 -0.005924  0.000000
tuned_gated_sharpe_full_raw    -0.011722 -0.013725 -0.018867 -0.002575
delta_gated_sharpe_full_raw    -0.009496 -0.012943 -0.013725 -0.001821
baseline_gated_pnl_net         -0.000107 -0.000059 -0.000261  0.000000
tuned_gated_pnl_net            -0.000770 -0.000562 -0.001459 -0.000291
delta_gated_pnl_net            -0.000663 -0.000562 -0.001197 -0.000231
baseline_gated_active_ratio     0.121412  0.059722  0.000000  0.304514
tuned_gated_active_ratio        0.317940  0.342014  0.234722  0.377083
```
