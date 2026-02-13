# Execution Diagnostics (Order-Book Snapshot)

This report computes slippage-vs-size from L2 snapshot book-walk metrics.

## Scope

- `march_vol_2024_binance`
- `yen_unwind_2024_binance`
- `yen_followthrough_2024_binance`

## L2 Coverage

```text
                       episode  l2_orderbook_available  tick_trades_available  l2_ready                                                 l2_root
        march_vol_2024_binance                    True                   True      True         data/processed/orderbook/march_vol_2024_binance
       yen_unwind_2024_binance                    True                   True      True        data/processed/orderbook/yen_unwind_2024_binance
yen_followthrough_2024_binance                    True                   True      True data/processed/orderbook/yen_followthrough_2024_binance
```

Snapshot coverage note: rows with `slippage_method=orderbook_snapshot_bookwalk` use true trade-to-orderbook matching; remaining rows are explicit bar-proxy fallback.

## Method Notes

- Trade source: Binance tick trades (`*-trades-*.zip`) with aggressor side from `is_buyer_maker`.
- Snapshot source: Binance `bookDepth` snapshots (`*-bookDepth-*.zip`) on `±1..±5%` depth levels.
- Matching: each trade is aligned to the latest snapshot at or before trade time within `300` seconds.
- Book-walk slippage (`book_walk_bps`): piecewise interpolation over cumulative notional levels (1% to 5%) on the consumed side (asks for buy taker, bids for sell taker).
- DNL (`dnl`): `trade_notional / side_notional_at_1pct`.
- Queue proxy (`queue_load`): same-second side taker notional divided by side 1% notional.
- Large-size bucket: top 10% `dnl` per symbol.
- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.
- Comparability guardrail: cross-venue tables are segmented by `market_type` (`spot` vs `derivatives`).

## Slippage vs Size

```text
                       episode   venue market_type root quote       symbol                   size_bin  dnl_bin_low  dnl_bin_high   n_obs   dnl_median  book_walk_mean_bps  book_walk_median_bps  queue_load_mean
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP    (-0.00099774, 1.28e-05]    -0.000998      0.000013   67864 1.160378e-05            0.001058              0.001160         0.004140
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.28e-05, 1.79e-05]     0.000013      0.000018   67860 1.563586e-05            0.001545              0.001564         0.004749
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.79e-05, 2.44e-05]     0.000018      0.000024   67862 1.965637e-05            0.002020              0.001966         0.004067
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (2.44e-05, 4.83e-05]     0.000024      0.000048   67862 3.456469e-05            0.003496              0.003456         0.006465
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (4.83e-05, 8.67e-05]     0.000048      0.000087   67862 6.520341e-05            0.006625              0.006520         0.007901
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (8.67e-05, 0.000129]     0.000087      0.000129   67861 1.100626e-04            0.010914              0.011006         0.007881
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (0.000129, 0.00018]     0.000129      0.000180   67862 1.515189e-04            0.015263              0.015152         0.007965
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (0.00018, 0.000333]     0.000180      0.000333   67862 2.414935e-04            0.024700              0.024149         0.008990
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (0.000333, 0.000615]     0.000333      0.000615   67862 4.377120e-04            0.045168              0.043771         0.008450
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP          (0.000615, 0.112]     0.000615      0.112000   67862 8.792881e-04            0.140268              0.087929         0.014997
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP (-0.00099999354, 1.14e-06]    -0.001000      0.000001 1092085 9.651943e-07            0.000091              0.000097         0.015227
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.14e-06, 1.44e-06]     0.000001      0.000001 1092083 1.296435e-06            0.000129              0.000130         0.013604
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.44e-06, 1.86e-06]     0.000001      0.000002 1092088 1.595833e-06            0.000161              0.000160         0.018674
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.86e-06, 3.25e-06]     0.000002      0.000003 1092080 2.358444e-06            0.000242              0.000236         0.021073
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (3.25e-06, 5.94e-06]     0.000003      0.000006 1092085 4.535345e-06            0.000455              0.000454         0.016260
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (5.94e-06, 9.97e-06]     0.000006      0.000010 1092083 7.593912e-06            0.000771              0.000759         0.015968
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (9.97e-06, 1.9e-05]     0.000010      0.000019 1092083 1.366774e-05            0.001394              0.001367         0.018335
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (1.9e-05, 4.19e-05]     0.000019      0.000042 1092084 2.701574e-05            0.002822              0.002702         0.018959
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (4.19e-05, 0.000104]     0.000042      0.000104 1092084 6.460371e-05            0.006723              0.006460         0.018378
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (0.000104, 146.795]     0.000104    146.795000 1092084 2.164751e-04            0.199072              0.021648         1.170957
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP     (-0.0009968, 1.15e-05]    -0.000997      0.000012  126346 9.703052e-06            0.000923              0.000970         0.005129
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.15e-05, 1.59e-05]     0.000012      0.000016  126346 1.448134e-05            0.001433              0.001448         0.004844
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.59e-05, 2.93e-05]     0.000016      0.000029  126346 2.031055e-05            0.002110              0.002031         0.006723
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (2.93e-05, 4.94e-05]     0.000029      0.000049  126346 3.979477e-05            0.003972              0.003979         0.007689
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (4.94e-05, 5.8e-05]     0.000049      0.000058  126346 5.506770e-05            0.005464              0.005507         0.006558
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (5.8e-05, 6.42e-05]     0.000058      0.000064  126346 6.079584e-05            0.006088              0.006080         0.005177
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (6.42e-05, 9.04e-05]     0.000064      0.000090  126346 7.149933e-05            0.007350              0.007150         0.007383
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (9.04e-05, 0.000174]     0.000090      0.000174  126346 1.230480e-04            0.012590              0.012305         0.012008
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (0.000174, 0.000371]     0.000174      0.000371  126346 2.373249e-04            0.025067              0.023732         0.010677
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP          (0.000371, 0.139]     0.000371      0.139000  126346 5.938039e-04            0.087912              0.059380         0.018251
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP (-0.00099999662, 1.18e-06]    -0.001000      0.000001 1057687 8.901164e-07            0.000086              0.000089         0.011719
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.18e-06, 1.53e-06]     0.000001      0.000002 1057686 1.373733e-06            0.000137              0.000137         0.014881
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.53e-06, 1.96e-06]     0.000002      0.000002 1057688 1.697675e-06            0.000171              0.000170         0.014760
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.96e-06, 3.32e-06]     0.000002      0.000003 1057684 2.459728e-06            0.000252              0.000246         0.018718
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (3.32e-06, 6.93e-06]     0.000003      0.000007 1057687 4.786792e-06            0.000490              0.000479         0.019463
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (6.93e-06, 1.41e-05]     0.000007      0.000014 1057686 9.876316e-06            0.001009              0.000988         0.018564
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.41e-05, 2.86e-05]     0.000014      0.000029 1057700 1.987374e-05            0.002036              0.001987         0.020785
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (2.86e-05, 6.21e-05]     0.000029      0.000062 1057672 4.162260e-05            0.004290              0.004162         0.020394
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (6.21e-05, 0.00014]     0.000062      0.000140 1057686 9.207644e-05            0.009498              0.009208         0.019469
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP           (0.00014, 0.169]     0.000140      0.169000 1057687 2.497223e-04            0.036127              0.024972         0.024779
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP    (-0.00099789, 1.44e-05]    -0.000998      0.000014  298272 1.095547e-05            0.001093              0.001096         0.007159
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.44e-05, 1.81e-05]     0.000014      0.000018  298270 1.568060e-05            0.001586              0.001568         0.006750
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.81e-05, 3.42e-05]     0.000018      0.000034  298271 2.402579e-05            0.002481              0.002403         0.011922
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (3.42e-05, 5.72e-05]     0.000034      0.000057  298271 4.578063e-05            0.004589              0.004578         0.011370
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (5.72e-05, 6.81e-05]     0.000057      0.000068  298271 6.260295e-05            0.006267              0.006260         0.007807
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (6.81e-05, 9.8e-05]     0.000068      0.000098  298271 7.839990e-05            0.008021              0.007840         0.011225
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP        (9.8e-05, 0.000164]     0.000098      0.000164  298272 1.243973e-04            0.012662              0.012440         0.014652
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (0.000164, 0.000284]     0.000164      0.000284  298270 2.139732e-04            0.021736              0.021397         0.015655
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (0.000284, 0.000508]     0.000284      0.000508  298271 3.931104e-04            0.039302              0.039311         0.016596
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP          (0.000508, 0.201]     0.000508      0.201000  298271 7.551046e-04            0.116603              0.075510         0.029470
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP (-0.00099999629, 1.62e-06]    -0.001000      0.000002 1990183 1.326084e-06            0.000122              0.000133         0.014081
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.62e-06, 2.07e-06]     0.000002      0.000002 1990183 1.839553e-06            0.000184              0.000184         0.017104
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (2.07e-06, 2.97e-06]     0.000002      0.000003 1990183 2.418283e-06            0.000245              0.000242         0.022828
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (2.97e-06, 5.4e-06]     0.000003      0.000005 1990182 3.823993e-06            0.000395              0.000382         0.028895
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (5.4e-06, 1.12e-05]     0.000005      0.000011 1990184 7.916477e-06            0.000803              0.000792         0.027298
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.12e-05, 2.23e-05]     0.000011      0.000022 1990182 1.591742e-05            0.001620              0.001592         0.026193
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (2.23e-05, 4.39e-05]     0.000022      0.000044 1990182 3.104509e-05            0.003176              0.003105         0.029249
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (4.39e-05, 9.25e-05]     0.000044      0.000092 1990183 6.344357e-05            0.006502              0.006344         0.028021
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (9.25e-05, 0.000209]     0.000092      0.000209 1990183 1.369872e-04            0.014130              0.013699         0.029669
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP          (0.000209, 0.234]     0.000209      0.234000 1990183 3.579931e-04            0.052461              0.035799         0.038524
```

## Cross-Quote Comparison (USDC vs USDT)

```text
                       episode   venue market_type root  impact_large_mean_bps_usdc  impact_large_mean_bps_usdt  impact_large_delta_usdc_minus_usdt_bps  impact_large_mean_excess_bps_usdc  impact_large_mean_excess_bps_usdt  impact_large_delta_excess_usdc_minus_usdt_bps  impact_large_mean_norm_usdc  impact_large_mean_norm_usdt  impact_large_delta_norm_usdc_minus_usdt  impact_all_mean_bps_usdc  impact_all_mean_bps_usdt preferred_quote_on_large_norm  dnl_large_mean_usdc  dnl_large_mean_usdt  dnl_large_delta_usdc_minus_usdt  queue_load_large_mean_usdc  queue_load_large_mean_usdt  queue_load_large_delta_usdc_minus_usdt  snapshot_match_ratio_usdc  snapshot_match_ratio_usdt        slippage_method_usdc        slippage_method_usdt
        march_vol_2024_binance binance derivatives  BTC                    0.140268                    0.199072                               -0.058804                      -8.496980e-20                      -4.810226e-01                                   4.810226e-01                     0.001403                     0.006801                                -0.005398                  0.025106                  0.021186                 indeterminate             0.001403             0.006801                        -0.005398                    0.014997                    1.170957                               -1.155960                   0.991908                   0.990038 orderbook_snapshot_bookwalk orderbook_snapshot_bookwalk
yen_followthrough_2024_binance binance derivatives  BTC                    0.087912                    0.036127                                0.051785                       7.079159e-20                      -3.003042e-21                                   7.379463e-20                     0.000879                     0.000361                                 0.000518                  0.015291                  0.005409                 indeterminate             0.000879             0.000361                         0.000518                    0.018251                    0.024779                               -0.006528                   0.999958                   0.999976 orderbook_snapshot_bookwalk orderbook_snapshot_bookwalk
       yen_unwind_2024_binance binance derivatives  BTC                    0.116603                    0.052461                                0.064142                      -1.112006e-20                      -3.824757e-21                                  -7.295302e-21                     0.001166                     0.000525                                 0.000641                  0.021434                  0.007964                 indeterminate             0.001166             0.000525                         0.000641                    0.029470                    0.038524                               -0.009055                   0.999963                   0.999979 orderbook_snapshot_bookwalk orderbook_snapshot_bookwalk
```

- `derivatives` (normalized delta, tolerance=0.050): USDC lower-impact `0/3`, USDT lower-impact `0/3`, indeterminate `3/3`.

Interpretation guardrail: these are descriptive diagnostics, not a full fee/funding-adjusted TCA ranking.

## Resilience Summary

```text
                       episode   venue market_type root quote       symbol  shock_threshold_bps  baseline_abs_ret_bps  n_shocks  n_recovered  unrecovered_ratio  recovery_median_bars  recovery_p90_bars
        march_vol_2024_binance binance derivatives  BNB  USDC BNBUSDC-PERP            48.940232              6.134675        29           29           0.000000                   3.0                9.0
        march_vol_2024_binance binance derivatives  BNB  USDT BNBUSDT-PERP            44.814765              6.438420        29           28           0.034483                   2.5                9.0
        march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP            29.291923              3.308346        29           29           0.000000                   3.0                9.2
        march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP            30.636759              3.432224        29           29           0.000000                   5.0               23.2
        march_vol_2024_binance binance derivatives  ETH  USDC ETHUSDC-PERP            32.174082              3.901184        29           29           0.000000                   5.0               14.4
        march_vol_2024_binance binance derivatives  ETH  USDT ETHUSDT-PERP            32.937643              4.255524        29           29           0.000000                   5.0               14.4
        march_vol_2024_binance binance derivatives  SOL  USDC SOLUSDC-PERP            50.441938              8.063681        29           29           0.000000                   6.0               11.4
        march_vol_2024_binance binance derivatives  SOL  USDT SOLUSDT-PERP            52.174259              8.255715        29           29           0.000000                   4.0               11.0
yen_followthrough_2024_binance binance derivatives  BNB  USDC BNBUSDC-PERP            38.512817              5.836652        29           29           0.000000                   3.0               11.4
yen_followthrough_2024_binance binance derivatives  BNB  USDT BNBUSDT-PERP            36.137496              5.856705        29           29           0.000000                   4.0               17.2
yen_followthrough_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP            30.146115              5.106934        29           29           0.000000                   3.0                7.0
yen_followthrough_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP            29.658742              5.130883        29           29           0.000000                   3.0               10.2
yen_followthrough_2024_binance binance derivatives  ETH  USDC ETHUSDC-PERP            49.304137              7.531066        29           29           0.000000                   4.0                9.2
yen_followthrough_2024_binance binance derivatives  ETH  USDT ETHUSDT-PERP            49.952924              7.721190        29           29           0.000000                   3.0                7.0
yen_followthrough_2024_binance binance derivatives  SOL  USDC SOLUSDC-PERP            57.834938             10.293786        29           29           0.000000                   3.0               10.0
yen_followthrough_2024_binance binance derivatives  SOL  USDT SOLUSDT-PERP            57.619574             10.512261        29           29           0.000000                   3.0               10.0
       yen_unwind_2024_binance binance derivatives  BNB  USDC BNBUSDC-PERP            81.770320             11.277097        29           29           0.000000                   5.0               11.2
       yen_unwind_2024_binance binance derivatives  BNB  USDT BNBUSDT-PERP            77.359485             10.967652        29           29           0.000000                   6.0               11.0
       yen_unwind_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP            64.388581              8.654788        29           29           0.000000                   4.0               10.0
       yen_unwind_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP            62.036388              8.838933        29           29           0.000000                   5.0               11.0
       yen_unwind_2024_binance binance derivatives  ETH  USDC ETHUSDC-PERP            94.911311             11.925691        29           29           0.000000                   6.0               11.2
       yen_unwind_2024_binance binance derivatives  ETH  USDT ETHUSDT-PERP            95.104944             11.983906        29           29           0.000000                   6.0               12.0
       yen_unwind_2024_binance binance derivatives  SOL  USDC SOLUSDC-PERP           136.064472             17.393240        29           29           0.000000                   6.0               15.2
       yen_unwind_2024_binance binance derivatives  SOL  USDT SOLUSDT-PERP           134.784974             17.620047        29           29           0.000000                   8.0               15.2
```

## Venue Summary (Within Market Type)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                     3                  0.019041                    0.051785                     0.160341                   7.379463e-20              -0.001413                 0.000518                     3                        4.0                        4.5                          0.0                     0.002874
```

## Artifacts

- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_slippage_vs_size.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
