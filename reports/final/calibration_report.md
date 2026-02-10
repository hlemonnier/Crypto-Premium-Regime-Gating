# Calibration Report

Comparison: baseline Notice defaults vs tuned config on 2024 episodes.

## Tuned Parameters

- `strategy.entry_k`: `0.5`
- `strategy.t_widen_quantile`: `0.99`
- `strategy.chi_widen_quantile`: `0.99`
- `regimes.stress_quantile`: `0.9`
- `regimes.recovery_quantile`: `0.8`

## Episode Details

```text
                       episode                                                              matrix_path  baseline_gated_sharpe  tuned_gated_sharpe  delta_gated_sharpe  baseline_gated_pnl_net  tuned_gated_pnl_net  delta_gated_pnl_net  baseline_gated_active_ratio  tuned_gated_active_ratio
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv             -12.766980           33.125863           45.892843               -0.000210             0.001224             0.001434                     0.001736                  0.103472
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv               0.000000            9.023636            9.023636                0.000000            -0.001739            -0.001739                     0.000000                  0.268403
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv            -239.414734           12.040133          251.454867               -0.001173             0.001009             0.002182                     0.004514                  0.465278
```

## Aggregate Stats

```text
                                   mean     median         min         max
baseline_gated_sharpe        -84.060571 -12.766980 -239.414734    0.000000
tuned_gated_sharpe            18.063211  12.040133    9.023636   33.125863
delta_gated_sharpe           102.123782  45.892843    9.023636  251.454867
baseline_gated_pnl_net        -0.000461  -0.000210   -0.001173    0.000000
tuned_gated_pnl_net            0.000165   0.001009   -0.001739    0.001224
delta_gated_pnl_net            0.000626   0.001434   -0.001739    0.002182
baseline_gated_active_ratio    0.002083   0.001736    0.000000    0.004514
tuned_gated_active_ratio       0.279051   0.268403    0.103472    0.465278
```
