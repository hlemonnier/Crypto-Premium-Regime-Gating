# Execution Quality Diagnostics

This report extends stablecoin analysis with execution-quality proxies.

## Scope

- `bybit_usdc_depeg_2023`
- `okx_usdc_depeg_2023`
- `march_vol_2024_binance`
- `yen_unwind_2024_binance`
- `yen_followthrough_2024_binance`

## Method Notes

- Data source: `prices_resampled.csv` (price + volume bars).
- Slippage proxy: next-bar absolute return (bps) conditioned on relative size (`volume / rolling median volume`).
- Large-size bucket: top 10% relative-size bars per symbol.
- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.
- Limitation: this is **not** order-book snapshot slippage/depth; it is a trade-bar proxy given available data.

## Cross-Quote Comparison (USDC vs USDT)

```text
                       episode   venue root  impact_large_mean_bps_usdc  impact_large_mean_bps_usdt  impact_large_delta_usdc_minus_usdt_bps  impact_all_mean_bps_usdc  impact_all_mean_bps_usdt preferred_quote_on_large
         bybit_usdc_depeg_2023   bybit  BTC                   11.149728                    8.167494                                2.982234                  8.930263                  6.128325                     USDT
         bybit_usdc_depeg_2023   bybit  ETH                   11.288408                   10.359149                                0.929259                 10.101611                  7.548126                     USDT
         bybit_usdc_depeg_2023   bybit  SOL                   13.572481                   15.171667                               -1.599186                 14.973906                 13.630203                     USDC
        march_vol_2024_binance binance  BNB                   14.592461                   15.305733                               -0.713271                 10.010382                 10.114262                     USDC
        march_vol_2024_binance binance  BTC                   10.247856                   10.653646                               -0.405789                  5.923007                  6.021184                     USDC
        march_vol_2024_binance binance  ETH                    9.710517                   10.772549                               -1.062032                  6.279006                  6.422590                     USDC
        march_vol_2024_binance binance  SOL                   14.695266                   18.394902                               -3.699636                 11.631011                 11.671989                     USDC
           okx_usdc_depeg_2023     okx  BTC                    6.426770                   11.026529                               -4.599759                  5.634176                  6.762237                     USDC
           okx_usdc_depeg_2023     okx  ETH                    8.943809                   12.405492                               -3.461683                  6.724613                  7.988521                     USDC
yen_followthrough_2024_binance binance  BNB                   10.244869                   11.642265                               -1.397396                  8.183964                  8.115851                     USDC
yen_followthrough_2024_binance binance  BTC                   10.452958                   10.787928                               -0.334970                  7.667911                  7.711967                     USDC
yen_followthrough_2024_binance binance  ETH                   14.396067                   15.036634                               -0.640568                 10.359642                 10.491481                     USDC
yen_followthrough_2024_binance binance  SOL                   18.837158                   18.719335                                0.117824                 13.767216                 13.815289                     USDT
       yen_unwind_2024_binance binance  BNB                   21.452744                   22.825023                               -1.372279                 16.163334                 16.028285                     USDC
       yen_unwind_2024_binance binance  BTC                   18.301742                   18.627876                               -0.326134                 14.193102                 14.141153                     USDC
       yen_unwind_2024_binance binance  ETH                   25.835947                   27.984854                               -2.148907                 17.535075                 17.681031                     USDC
       yen_unwind_2024_binance binance  SOL                   37.924605                   39.714394                               -1.789788                 25.993425                 26.032203                     USDC
```

- USDC preferred on large-size proxy impact: `14/17`
- USDT preferred on large-size proxy impact: `3/17`

## Resilience Summary

```text
                       episode   venue root quote         symbol  shock_threshold_bps  baseline_abs_ret_bps  n_shocks  n_recovered  unrecovered_ratio  recovery_median_bars  recovery_p90_bars
         bybit_usdc_depeg_2023   bybit  BNB  USDT   BNBUSDT-SPOT            28.156058              3.650710        29           29           0.000000                   2.0                8.0
         bybit_usdc_depeg_2023   bybit  BTC  USDC   BTCUSDC-SPOT            54.668886              5.748309        29           29           0.000000                   3.0                8.4
         bybit_usdc_depeg_2023   bybit  BTC  USDT   BTCUSDT-SPOT            29.408682              4.308558        29           29           0.000000                   2.0                6.6
         bybit_usdc_depeg_2023   bybit  ETH  USDC   ETHUSDC-SPOT            62.062262              6.495326        29           29           0.000000                   2.0                8.4
         bybit_usdc_depeg_2023   bybit  ETH  USDT   ETHUSDT-SPOT            37.353090              5.345199        29           29           0.000000                   3.0                9.2
         bybit_usdc_depeg_2023   bybit  SOL  USDC   SOLUSDC-SPOT            83.483407              5.971932        29           29           0.000000                   2.0                7.2
         bybit_usdc_depeg_2023   bybit  SOL  USDT   SOLUSDT-SPOT            57.419955             11.248595        29           29           0.000000                   4.0                7.6
        march_vol_2024_binance binance  BNB  USDC   BNBUSDC-PERP            49.556395              6.821220        29           29           0.000000                   3.0                9.0
        march_vol_2024_binance binance  BNB  USDT   BNBUSDT-PERP            49.867023              6.874193        29           28           0.034483                   4.0               11.0
        march_vol_2024_binance binance  BTC  USDC   BTCUSDC-PERP            32.099120              3.982542        29           29           0.000000                   4.0               13.0
        march_vol_2024_binance binance  BTC  USDT   BTCUSDT-PERP            32.282711              4.094006        29           29           0.000000                   3.0               10.8
        march_vol_2024_binance binance  ETH  USDC   ETHUSDC-PERP            34.354210              4.289859        29           29           0.000000                   6.0               14.2
        march_vol_2024_binance binance  ETH  USDT   ETHUSDT-PERP            34.903859              4.484962        29           29           0.000000                   5.0               28.4
        march_vol_2024_binance binance  SOL  USDC   SOLUSDC-PERP            52.515715              8.518867        29           29           0.000000                   3.0               11.2
        march_vol_2024_binance binance  SOL  USDT   SOLUSDT-PERP            52.919949              8.363655        29           29           0.000000                   4.0               11.0
           okx_usdc_depeg_2023     okx  BTC  USDC BTCUSDC-230331            40.750571              3.124080        29           29           0.000000                   1.0                4.2
           okx_usdc_depeg_2023     okx  BTC  USDT BTCUSDT-230331            32.363505              4.554257        29           29           0.000000                   4.0               10.8
           okx_usdc_depeg_2023     okx  ETH  USDC ETHUSDC-230331            58.910478              3.556264        29           29           0.000000                   1.0                7.4
           okx_usdc_depeg_2023     okx  ETH  USDT ETHUSDT-230331            41.762796              5.418052        29           29           0.000000                   4.0               15.2
yen_followthrough_2024_binance binance  BNB  USDC   BNBUSDC-PERP            37.274400              6.106999        29           29           0.000000                   3.0               11.4
yen_followthrough_2024_binance binance  BNB  USDT   BNBUSDT-PERP            36.002197              5.877048        29           29           0.000000                   3.0               13.2
yen_followthrough_2024_binance binance  BTC  USDC   BTCUSDC-PERP            32.358028              5.606322        29           29           0.000000                   3.0                8.2
yen_followthrough_2024_binance binance  BTC  USDT   BTCUSDT-PERP            32.315146              5.815847        29           29           0.000000                   3.0                8.2
yen_followthrough_2024_binance binance  ETH  USDC   ETHUSDC-PERP            47.539040              7.606461        29           29           0.000000                   3.0                7.2
yen_followthrough_2024_binance binance  ETH  USDT   ETHUSDT-PERP            47.520719              7.648719        29           29           0.000000                   3.0                7.2
yen_followthrough_2024_binance binance  SOL  USDC   SOLUSDC-PERP            54.817208             10.612451        29           29           0.000000                   3.0               10.2
yen_followthrough_2024_binance binance  SOL  USDT   SOLUSDT-PERP            54.236691             10.635401        29           29           0.000000                   3.0                9.2
       yen_unwind_2024_binance binance  BNB  USDC   BNBUSDC-PERP            88.059921             11.394158        29           29           0.000000                   7.0               15.2
       yen_unwind_2024_binance binance  BNB  USDT   BNBUSDT-PERP            85.407313             11.069457        29           29           0.000000                   6.0               15.0
       yen_unwind_2024_binance binance  BTC  USDC   BTCUSDC-PERP            73.261649             10.001644        29           29           0.000000                   3.0               15.0
       yen_unwind_2024_binance binance  BTC  USDT   BTCUSDT-PERP            73.679166              9.926791        29           29           0.000000                   3.0               12.4
       yen_unwind_2024_binance binance  ETH  USDC   ETHUSDC-PERP            98.967948             11.864573        29           29           0.000000                   5.0                9.0
       yen_unwind_2024_binance binance  ETH  USDT   ETHUSDT-PERP            96.926886             11.785032        29           29           0.000000                   5.0                9.2
       yen_unwind_2024_binance binance  SOL  USDC   SOLUSDC-PERP           140.425789             18.207556        29           29           0.000000                   6.0               14.6
       yen_unwind_2024_binance binance  SOL  USDT   SOLUSDT-PERP           142.426324             18.127766        29           29           0.000000                   5.0               10.8
```

## Venue Summary

```text
  venue  n_root_episode_pairs  usdc_preferred_count  usdt_preferred_count  mean_delta_usdc_minus_usdt_bps  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance                    12                    11                     1                       -1.147746                        3.0                        3.5                          0.0                     0.002874
  bybit                     3                     1                     2                        0.770769                        2.0                        2.5                          0.0                     0.000000
    okx                     2                     2                     0                       -4.030721                        1.0                        4.0                          0.0                     0.000000
```

## Artifacts

- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
