# Calibration Report

Comparison: baseline Notice defaults vs tuned config on 2024 episodes.

Metric naming: `*_sharpe_full_raw` denotes full-series, non-annualized Sharpe.

## Tuned Parameters

- `strategy.entry_k`: `0.28`
- `strategy.t_widen_quantile`: `0.9`
- `strategy.chi_widen_quantile`: `0.99`
- `regimes.stress_quantile`: `0.9`
- `regimes.recovery_quantile`: `0.8`

## Episode Details

```text
                       episode                                                              matrix_path  baseline_gated_sharpe_full_raw  tuned_gated_sharpe_full_raw  delta_gated_sharpe_full_raw  baseline_gated_pnl_net  tuned_gated_pnl_net  delta_gated_pnl_net  baseline_gated_active_ratio  tuned_gated_active_ratio
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv                       -0.021048                     0.004241                     0.025289               -0.000541             0.001014             0.001555                     0.005208                  0.211111
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv                       -0.010655                    -0.002233                     0.008423               -0.000270            -0.000617            -0.000347                     0.004514                  0.488889
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv                       -0.008408                     0.016534                     0.024942               -0.000522             0.026293             0.026815                     0.008333                  0.618403
```

## Aggregate Stats

```text
                                    mean    median       min       max
baseline_gated_sharpe_full_raw -0.013370 -0.010655 -0.021048 -0.008408
tuned_gated_sharpe_full_raw     0.006181  0.004241 -0.002233  0.016534
delta_gated_sharpe_full_raw     0.019551  0.024942  0.008423  0.025289
baseline_gated_pnl_net         -0.000444 -0.000522 -0.000541 -0.000270
tuned_gated_pnl_net             0.008896  0.001014 -0.000617  0.026293
delta_gated_pnl_net             0.009341  0.001555 -0.000347  0.026815
baseline_gated_active_ratio     0.006019  0.005208  0.004514  0.008333
tuned_gated_active_ratio        0.439468  0.488889  0.211111  0.618403
```
