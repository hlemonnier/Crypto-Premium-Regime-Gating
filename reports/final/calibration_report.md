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
        march_vol_2024_binance         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv             -40.669653           21.951045           62.620698               -0.000274             0.000697             0.000971                     0.003472                  0.107986
           okx_usdc_depeg_2023            data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv               0.000000           55.664006           55.664006                0.000000             0.004909             0.004909                     0.000000                  0.035417
              smoke_2024_08_05               data/processed/episodes/smoke_2024_08_05/prices_matrix.csv             -54.738023           13.188629           67.926652               -0.000590             0.001930             0.002519                     0.011806                  0.511111
yen_followthrough_2024_binance data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv               0.000000            5.860367            5.860367                0.000000            -0.001800            -0.001800                     0.000000                  0.261806
       yen_unwind_2024_binance        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv             -15.853918            7.063795           22.917714               -0.000522             0.000423             0.000945                     0.008333                  0.532986
```

## Aggregate Stats

```text
                                  mean     median        min        max
baseline_gated_sharpe       -18.543599  -7.926959 -54.738023   0.000000
tuned_gated_sharpe           17.287974  10.126212   0.000000  55.664006
delta_gated_sharpe           35.831573  39.290860   0.000000  67.926652
baseline_gated_pnl_net       -0.000231  -0.000137  -0.000590   0.000000
tuned_gated_pnl_net           0.001026   0.000560  -0.001800   0.004909
delta_gated_pnl_net           0.001257   0.000958  -0.001800   0.004909
baseline_gated_active_ratio   0.003935   0.001736   0.000000   0.011806
tuned_gated_active_ratio      0.241551   0.184896   0.000000   0.532986
```

## Skipped Episodes

```text
             episode                                                    matrix_path                                                                                                                                                                                                                                                             reason
     usdc_depeg_2023      data/processed/episodes/usdc_depeg_2023/prices_matrix.csv No compatible USDC/USDT target pair found in price matrix. Expected a matched pair like BTCUSDC/BTCUSDT for the same market suffix. Configured target=('BTCUSDC-PERP', 'BTCUSDT-PERP'), available columns=[BNBUSDT-PERP, BTCUSDT-PERP, ETHUSDT-PERP, SOLUSDT-PERP]
usdc_depeg_2023_spot data/processed/episodes/usdc_depeg_2023_spot/prices_matrix.csv No compatible USDC/USDT target pair found in price matrix. Expected a matched pair like BTCUSDC/BTCUSDT for the same market suffix. Configured target=('BTCUSDC-PERP', 'BTCUSDT-PERP'), available columns=[BNBUSDT-PERP, BTCUSDT-PERP, ETHUSDT-PERP, SOLUSDT-PERP]
```
