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
         bybit_usdc_depeg_2023          data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv               0.000000            0.000000            0.000000                0.000000             0.000000             0.000000                     0.000000                  0.000000
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv            -102.064655           23.550434          125.615090               -0.000541             0.001240             0.001781                     0.005208                  0.151389
           okx_usdc_depeg_2023            data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv               0.000000           42.764729           42.764729                0.000000             0.004972             0.004972                     0.000000                  0.044097
              smoke_2024_08_05               data/processed/episodes/smoke_2024_08_05/prices_matrix.csv             -42.343225           10.607601           52.950826               -0.000510             0.001039             0.001549                     0.010417                  0.436111
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv             -31.346834            5.814444           37.161278               -0.000270            -0.001743            -0.001473                     0.004514                  0.336111
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv             -15.853918            5.514261           21.368179               -0.000522            -0.000239             0.000283                     0.008333                  0.504861
```

## Aggregate Stats

```text
                                  mean     median         min         max
baseline_gated_sharpe       -31.934772 -23.600376 -102.064655    0.000000
tuned_gated_sharpe           14.708578   8.211023    0.000000   42.764729
delta_gated_sharpe           46.643350  39.963004    0.000000  125.615090
baseline_gated_pnl_net       -0.000307  -0.000390   -0.000541    0.000000
tuned_gated_pnl_net           0.000878   0.000519   -0.001743    0.004972
delta_gated_pnl_net           0.001185   0.000916   -0.001473    0.004972
baseline_gated_active_ratio   0.004745   0.004861    0.000000    0.010417
tuned_gated_active_ratio      0.245428   0.243750    0.000000    0.504861
```

## Skipped Episodes

```text
             episode                                                    matrix_path                                                                                                                                                                                                                                                             reason
     usdc_depeg_2023      data/processed/episodes/usdc_depeg_2023/prices_matrix.csv No compatible USDC/USDT target pair found in price matrix. Expected a matched pair like BTCUSDC/BTCUSDT for the same market suffix. Configured target=('BTCUSDC-PERP', 'BTCUSDT-PERP'), available columns=[BNBUSDT-PERP, BTCUSDT-PERP, ETHUSDT-PERP, SOLUSDT-PERP]
usdc_depeg_2023_spot data/processed/episodes/usdc_depeg_2023_spot/prices_matrix.csv No compatible USDC/USDT target pair found in price matrix. Expected a matched pair like BTCUSDC/BTCUSDT for the same market suffix. Configured target=('BTCUSDC-PERP', 'BTCUSDT-PERP'), available columns=[BNBUSDT-PERP, BTCUSDT-PERP, ETHUSDT-PERP, SOLUSDT-PERP]
```
