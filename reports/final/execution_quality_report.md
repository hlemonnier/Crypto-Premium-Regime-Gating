# Execution Proxy Diagnostics (Bar-Level)

This report extends stablecoin analysis with bar-level execution proxies.

## Scope

- `bybit_usdc_depeg_2023`
- `okx_usdc_depeg_2023`
- `march_vol_2024_binance`
- `yen_unwind_2024_binance`
- `yen_followthrough_2024_binance`

## Method Notes

- Data source: `prices_resampled.csv` (price + volume bars).
- Slippage proxy: next-bar absolute return (bps) conditioned on relative size (`volume / rolling median volume`).
- Volatility control: report large-size deltas in raw bps, excess bps (`next_bar_abs_ret - local_median_abs_ret`), and normalized units (`next_bar_abs_ret / local_median_abs_ret`).
- Normalization floor: local volatility denominator floored at `1.000` bps.
- Large-size bucket: top 10% relative-size bars per symbol.
- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.
- Limitation: this is **not** order-book snapshot slippage/depth (book-walk, DNL, queueing); it is a trade-bar proxy given available data.
- Comparability guardrail: cross-venue tables are segmented by `market_type` (`spot` vs `derivatives`); avoid mixing them for venue ranking.

## Cross-Quote Comparison (USDC vs USDT)

```text
                       episode   venue market_type root  impact_large_mean_bps_usdc  impact_large_mean_bps_usdt  impact_large_delta_usdc_minus_usdt_bps  impact_large_mean_excess_bps_usdc  impact_large_mean_excess_bps_usdt  impact_large_delta_excess_usdc_minus_usdt_bps  impact_large_mean_norm_usdc  impact_large_mean_norm_usdt  impact_large_delta_norm_usdc_minus_usdt  impact_all_mean_bps_usdc  impact_all_mean_bps_usdt preferred_quote_on_large_norm
         bybit_usdc_depeg_2023   bybit        spot  BTC                   11.149728                    8.167494                                2.982234                           4.857518                           3.820150                                       1.037368                     1.785197                     1.924065                                -0.138869                  8.930263                  6.128325                          USDC
         bybit_usdc_depeg_2023   bybit        spot  ETH                   11.288408                   10.359149                                0.929259                           4.454376                           5.209996                                      -0.755620                     1.822462                     1.978723                                -0.156261                 10.101611                  7.548126                          USDC
         bybit_usdc_depeg_2023   bybit        spot  SOL                   13.572481                   15.171667                               -1.599186                           7.840160                           5.500945                                       2.339215                     4.990443                     1.667942                                 3.322501                 14.973906                 13.630203                          USDT
        march_vol_2024_binance binance derivatives  BNB                   14.592461                   15.305733                               -0.713271                           7.204430                           7.940295                                      -0.735864                     2.030614                     2.230672                                -0.200058                 10.010382                 10.114262                          USDC
        march_vol_2024_binance binance derivatives  BTC                   10.247856                   10.653646                               -0.405789                           5.918620                           6.026772                                      -0.108152                     2.353401                     2.295106                                 0.058295                  5.923007                  6.021184                          USDT
        march_vol_2024_binance binance derivatives  ETH                    9.710517                   10.772549                               -1.062032                           5.328314                           6.078467                                      -0.750153                     2.307516                     2.344147                                -0.036631                  6.279006                  6.422590                 indeterminate
        march_vol_2024_binance binance derivatives  SOL                   14.695266                   18.394902                               -3.699636                           6.115601                           9.831558                                      -3.715958                     1.795926                     2.203271                                -0.407345                 11.631011                 11.671989                          USDC
           okx_usdc_depeg_2023     okx derivatives  BTC                    6.426770                   11.026529                               -4.599759                           3.189991                           6.080331                                      -2.890340                     2.113397                     2.508437                                -0.395039                  5.634176                  6.762237                          USDC
           okx_usdc_depeg_2023     okx derivatives  ETH                    8.943809                   12.405492                               -3.461683                           5.623214                           6.846922                                      -1.223709                     3.598815                     2.435780                                 1.163035                  6.724613                  7.988521                          USDT
yen_followthrough_2024_binance binance derivatives  BNB                   10.244869                   11.642265                               -1.397396                           4.080108                           5.691251                                      -1.611143                     1.790893                     2.105955                                -0.315062                  8.183964                  8.115851                          USDC
yen_followthrough_2024_binance binance derivatives  BTC                   10.452958                   10.787928                               -0.334970                           4.816423                           4.922274                                      -0.105851                     2.080749                     1.978612                                 0.102137                  7.667911                  7.711967                          USDT
yen_followthrough_2024_binance binance derivatives  ETH                   14.396067                   15.036634                               -0.640568                           6.571335                           7.133030                                      -0.561695                     1.963327                     2.037653                                -0.074326                 10.359642                 10.491481                          USDC
yen_followthrough_2024_binance binance derivatives  SOL                   18.837158                   18.719335                                0.117824                           7.645854                           7.437214                                       0.208640                     1.818197                     1.774348                                 0.043849                 13.767216                 13.815289                 indeterminate
       yen_unwind_2024_binance binance derivatives  BNB                   19.065587                   20.610297                               -1.544711                           7.689097                           9.154946                                      -1.465849                     1.587018                     1.685819                                -0.098801                 15.913869                 15.804233                          USDC
       yen_unwind_2024_binance binance derivatives  BTC                   17.282338                   17.942064                               -0.659726                           6.801886                           7.438364                                      -0.636478                     1.637797                     1.713866                                -0.076069                 14.001024                 14.057262                          USDC
       yen_unwind_2024_binance binance derivatives  ETH                   23.880399                   26.194817                               -2.314418                          10.392777                          12.664624                                      -2.271847                     1.707618                     1.810360                                -0.102742                 17.268385                 17.419364                          USDC
       yen_unwind_2024_binance binance derivatives  SOL                   34.798391                   32.737058                                2.061333                          13.871752                          13.129188                                       0.742563                     1.630647                     1.623838                                 0.006809                 25.550200                 25.048396                 indeterminate
```

- `derivatives` (normalized delta, tolerance=0.050): USDC lower-proxy-impact `8/14`, USDT lower-proxy-impact `3/14`, indeterminate `3/14`.

- `spot` (normalized delta, tolerance=0.050): USDC lower-proxy-impact `2/3`, USDT lower-proxy-impact `1/3`, indeterminate `0/3`.

Interpretation guardrail: these are descriptive bar-level proxy gaps only; they are insufficient to rank venues for execution quality.

## Resilience Summary

```text
                       episode   venue market_type root quote         symbol  shock_threshold_bps  baseline_abs_ret_bps  n_shocks  n_recovered  unrecovered_ratio  recovery_median_bars  recovery_p90_bars
         bybit_usdc_depeg_2023   bybit        spot  BNB  USDT   BNBUSDT-SPOT            28.156058              3.650710        29           29           0.000000                   2.0                8.0
         bybit_usdc_depeg_2023   bybit        spot  BTC  USDC   BTCUSDC-SPOT            54.668886              5.748309        29           29           0.000000                   3.0                8.4
         bybit_usdc_depeg_2023   bybit        spot  BTC  USDT   BTCUSDT-SPOT            29.408682              4.308558        29           29           0.000000                   2.0                6.6
         bybit_usdc_depeg_2023   bybit        spot  ETH  USDC   ETHUSDC-SPOT            62.062262              6.495326        29           29           0.000000                   2.0                8.4
         bybit_usdc_depeg_2023   bybit        spot  ETH  USDT   ETHUSDT-SPOT            37.353090              5.345199        29           29           0.000000                   3.0                9.2
         bybit_usdc_depeg_2023   bybit        spot  SOL  USDC   SOLUSDC-SPOT            83.483407              5.971932        29           29           0.000000                   2.0                7.2
         bybit_usdc_depeg_2023   bybit        spot  SOL  USDT   SOLUSDT-SPOT            57.419955             11.248595        29           29           0.000000                   4.0                7.6
        march_vol_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            49.556395              6.821220        29           29           0.000000                   3.0                9.0
        march_vol_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            49.867023              6.874193        29           28           0.034483                   4.0               11.0
        march_vol_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            32.099120              3.982542        29           29           0.000000                   4.0               13.0
        march_vol_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            32.282711              4.094006        29           29           0.000000                   3.0               10.8
        march_vol_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            34.354210              4.289859        29           29           0.000000                   6.0               14.2
        march_vol_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            34.903859              4.484962        29           29           0.000000                   5.0               28.4
        march_vol_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP            52.515715              8.518867        29           29           0.000000                   3.0               11.2
        march_vol_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP            52.919949              8.363655        29           29           0.000000                   4.0               11.0
           okx_usdc_depeg_2023     okx derivatives  BTC  USDC BTCUSDC-230331            40.750571              3.124080        29           29           0.000000                   1.0                4.2
           okx_usdc_depeg_2023     okx derivatives  BTC  USDT BTCUSDT-230331            32.363505              4.554257        29           29           0.000000                   4.0               10.8
           okx_usdc_depeg_2023     okx derivatives  ETH  USDC ETHUSDC-230331            58.910478              3.556264        29           29           0.000000                   1.0                7.4
           okx_usdc_depeg_2023     okx derivatives  ETH  USDT ETHUSDT-230331            41.762796              5.418052        29           29           0.000000                   4.0               15.2
yen_followthrough_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            37.274400              6.106999        29           29           0.000000                   3.0               11.4
yen_followthrough_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            36.002197              5.877048        29           29           0.000000                   3.0               13.2
yen_followthrough_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            32.358028              5.606322        29           29           0.000000                   3.0                8.2
yen_followthrough_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            32.315146              5.815847        29           29           0.000000                   3.0                8.2
yen_followthrough_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            47.539040              7.606461        29           29           0.000000                   3.0                7.2
yen_followthrough_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            47.520719              7.648719        29           29           0.000000                   3.0                7.2
yen_followthrough_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP            54.817208             10.612451        29           29           0.000000                   3.0               10.2
yen_followthrough_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP            54.236691             10.635401        29           29           0.000000                   3.0                9.2
       yen_unwind_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            86.241091             11.365351        29           29           0.000000                   4.0               12.8
       yen_unwind_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            85.407313             11.063680        29           29           0.000000                   3.0               14.2
       yen_unwind_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            73.261649              9.995611        29           29           0.000000                   3.0               13.4
       yen_unwind_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            73.679166              9.918586        29           29           0.000000                   3.0                9.6
       yen_unwind_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            97.147553             11.832088        29           29           0.000000                   5.0                8.2
       yen_unwind_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            96.926886             11.768724        29           29           0.000000                   5.0                9.2
       yen_unwind_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP           140.425789             18.191775        29           29           0.000000                   4.0               14.4
       yen_unwind_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP           138.559312             18.022965        29           29           0.000000                   3.0               14.4
```

## Venue Summary (Within Market Type)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                    12                 -0.882780                   -0.686499                    -0.917649                      -0.686171              -0.091662                -0.075198                     3                        3.0                        3.0                          0.0                     0.002874
  bybit        spot                     3                  0.770769                    0.929259                     0.873654                       1.037368               1.009124                -0.138869                     0                        2.0                        2.5                          0.0                     0.000000
    okx derivatives                     2                 -4.030721                   -4.030721                    -2.057024                      -2.057024               0.383998                 0.383998                     0                        1.0                        4.0                          0.0                     0.000000
```

## Artifacts

- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
