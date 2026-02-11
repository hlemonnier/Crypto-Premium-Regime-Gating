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
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv                        0.000202                    -0.021731                    -0.021933                0.000013            -0.003375            -0.003388                     0.011806                  0.075000
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv                       -0.007757                     0.003361                     0.011118               -0.000511             0.000474             0.000984                     0.021181                  0.106597
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv                       -0.001394                     0.000728                     0.002122               -0.000026             0.000127             0.000152                     0.001736                  0.109375
```

## Aggregate Stats

```text
                                    mean    median       min       max
baseline_gated_sharpe_full_raw -0.002983 -0.001394 -0.007757  0.000202
tuned_gated_sharpe_full_raw    -0.005881  0.000728 -0.021731  0.003361
delta_gated_sharpe_full_raw    -0.002897  0.002122 -0.021933  0.011118
baseline_gated_pnl_net         -0.000174 -0.000026 -0.000511  0.000013
tuned_gated_pnl_net            -0.000925  0.000127 -0.003375  0.000474
delta_gated_pnl_net            -0.000750  0.000152 -0.003388  0.000984
baseline_gated_active_ratio     0.011574  0.011806  0.001736  0.021181
tuned_gated_active_ratio        0.096991  0.106597  0.075000  0.109375
```
