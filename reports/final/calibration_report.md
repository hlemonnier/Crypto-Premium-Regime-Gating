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
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv                       -0.021048                    -0.008120                     0.012928               -0.000541            -0.001158            -0.000617                     0.005208                  0.111111
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv                       -0.010655                     0.000271                     0.010926               -0.000270             0.000045             0.000315                     0.004514                  0.182986
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv                       -0.010388                    -0.008586                     0.001802               -0.000693            -0.002226            -0.001532                     0.011806                  0.237153
```

## Aggregate Stats

```text
                                    mean    median       min       max
baseline_gated_sharpe_full_raw -0.014031 -0.010655 -0.021048 -0.010388
tuned_gated_sharpe_full_raw    -0.005479 -0.008120 -0.008586  0.000271
delta_gated_sharpe_full_raw     0.008552  0.010926  0.001802  0.012928
baseline_gated_pnl_net         -0.000502 -0.000541 -0.000693 -0.000270
tuned_gated_pnl_net            -0.001113 -0.001158 -0.002226  0.000045
delta_gated_pnl_net            -0.000611 -0.000617 -0.001532  0.000315
baseline_gated_active_ratio     0.007176  0.005208  0.004514  0.011806
tuned_gated_active_ratio        0.177083  0.182986  0.111111  0.237153
```
