# Execution Proxy Diagnostics (Bar-Level)

This report extends stablecoin analysis with bar-level execution proxies.

## Scope

- `bybit_usdc_depeg_2023`
- `okx_usdc_depeg_2023`
- `march_vol_2024_binance`
- `yen_unwind_2024_binance`
- `yen_followthrough_2024_binance`

## L2 Coverage

```text
                       episode  l2_orderbook_available  tick_trades_available  l2_ready                                                 l2_root
         bybit_usdc_depeg_2023                   False                  False     False          data/processed/orderbook/bybit_usdc_depeg_2023
           okx_usdc_depeg_2023                   False                   True     False            data/processed/orderbook/okx_usdc_depeg_2023
        march_vol_2024_binance                    True                   True      True         data/processed/orderbook/march_vol_2024_binance
       yen_unwind_2024_binance                    True                   True      True        data/processed/orderbook/yen_unwind_2024_binance
yen_followthrough_2024_binance                    True                   True      True data/processed/orderbook/yen_followthrough_2024_binance
```

Note: L2 coverage is provided for transparency. Current diagnostics below remain bar-level proxies.

## Method Notes

- Data source: `prices_resampled.csv` bars, with per-minute tick-trade aggregation overlays when local `data/processed/orderbook/<episode>/...` trade files are available.
- When Binance `bookDepth` snapshots are available, relative size uses `trade_notional / depth_notional_1pct` (fallback to rolling-volume scale otherwise).
- Slippage proxy: next-bar absolute return (bps) conditioned on `rel_size` (depth-scaled when available, otherwise volume-scaled).
- Volatility control: report large-size deltas in raw bps, excess bps (`next_bar_abs_ret - local_median_abs_ret`), and normalized units (`next_bar_abs_ret / local_median_abs_ret`).
- Normalization floor: local volatility denominator floored at `1.000` bps.
- Large-size bucket: top 10% relative-size bars per symbol.
- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.
- Limitation: this is **not** order-book snapshot slippage/depth (book-walk, DNL, queueing); it is a trade-bar proxy given available data.
- Comparability guardrail: cross-venue tables are segmented by `market_type` (`spot` vs `derivatives`); avoid mixing them for venue ranking.

## Cross-Quote Comparison (USDC vs USDT)

```text
                       episode   venue market_type root  impact_large_mean_bps_usdc  impact_large_mean_bps_usdt  impact_large_delta_usdc_minus_usdt_bps  impact_large_mean_excess_bps_usdc  impact_large_mean_excess_bps_usdt  impact_large_delta_excess_usdc_minus_usdt_bps  impact_large_mean_norm_usdc  impact_large_mean_norm_usdt  impact_large_delta_norm_usdc_minus_usdt  impact_all_mean_bps_usdc  impact_all_mean_bps_usdt preferred_quote_on_large_norm
         bybit_usdc_depeg_2023   bybit        spot  BTC                    7.833880                    7.052589                                0.781291                           4.314941                           4.483034                                      -0.168093                     2.342636                     2.664074                                -0.321438                  4.773018                  4.601015                          USDC
         bybit_usdc_depeg_2023   bybit        spot  ETH                   13.203403                   10.359149                                2.844254                           6.312213                           5.209996                                       1.102216                     2.029677                     1.978723                                 0.050954                 10.291343                  7.548126                          USDT
         bybit_usdc_depeg_2023   bybit        spot  SOL                   14.612650                   15.171667                               -0.559017                           8.906199                           5.500945                                       3.405253                     5.014398                     1.667942                                 3.346456                 15.282052                 13.630203                          USDT
        march_vol_2024_binance binance derivatives  BTC                   11.329447                   12.235063                               -0.905617                           5.191599                           5.712573                                      -0.520973                     2.141550                     2.142812                                -0.001262                  4.967224                  5.149921                 indeterminate
           okx_usdc_depeg_2023     okx derivatives  BTC                    7.451211                   11.044616                               -3.593405                           3.485311                           5.828338                                      -2.343028                     2.266490                     2.280759                                -0.014269                  6.850341                  6.967686                 indeterminate
           okx_usdc_depeg_2023     okx derivatives  ETH                    9.460290                   12.527317                               -3.067027                           5.870248                           6.765885                                      -0.895637                     3.724251                     2.310525                                 1.413726                  8.466428                  8.190720                          USDT
yen_followthrough_2024_binance binance derivatives  BTC                   11.463536                   12.395515                               -0.931979                           4.747915                           5.179331                                      -0.431416                     1.869876                     1.924903                                -0.055026                  6.988409                  7.058466                          USDC
       yen_unwind_2024_binance binance derivatives  BTC                   27.370991                   28.449762                               -1.078771                           9.190229                          10.025472                                      -0.835244                     1.557822                     1.605140                                -0.047318                 12.538414                 12.844881                 indeterminate
```

- `derivatives` (normalized delta, tolerance=0.050): USDC lower-proxy-impact `1/5`, USDT lower-proxy-impact `1/5`, indeterminate `3/5`.

- `spot` (normalized delta, tolerance=0.050): USDC lower-proxy-impact `1/3`, USDT lower-proxy-impact `2/3`, indeterminate `0/3`.

Interpretation guardrail: these are descriptive bar-level proxy gaps only; they are insufficient to rank venues for execution quality.

## Resilience Summary

```text
                       episode   venue market_type root quote         symbol  shock_threshold_bps  baseline_abs_ret_bps  n_shocks  n_recovered  unrecovered_ratio  recovery_median_bars  recovery_p90_bars
         bybit_usdc_depeg_2023   bybit        spot  BNB  USDT   BNBUSDT-SPOT            28.156058              3.650710        29           29           0.000000                   2.0                8.0
         bybit_usdc_depeg_2023   bybit        spot  BTC  USDC   BTCUSDC-SPOT            29.627699              2.913099       445          445           0.000000                   6.0               17.6
         bybit_usdc_depeg_2023   bybit        spot  BTC  USDT   BTCUSDT-SPOT            28.418095              2.759341       447          446           0.002237                   5.0               17.0
         bybit_usdc_depeg_2023   bybit        spot  ETH  USDC   ETHUSDC-SPOT            62.091681              6.497451        29           29           0.000000                   3.0                8.4
         bybit_usdc_depeg_2023   bybit        spot  ETH  USDT   ETHUSDT-SPOT            37.353090              5.345199        29           29           0.000000                   3.0                9.2
         bybit_usdc_depeg_2023   bybit        spot  SOL  USDC   SOLUSDC-SPOT            83.486022              5.979073        29           29           0.000000                   3.0                8.2
         bybit_usdc_depeg_2023   bybit        spot  SOL  USDT   SOLUSDT-SPOT            57.419955             11.248595        29           29           0.000000                   4.0                7.6
        march_vol_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            48.940232              6.134675        29           29           0.000000                   3.0                9.0
        march_vol_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            44.814765              6.438420        29           28           0.034483                   2.5                9.0
        march_vol_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            29.291923              3.308346        29           29           0.000000                   3.0                9.2
        march_vol_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            30.636759              3.432224        29           29           0.000000                   5.0               23.2
        march_vol_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            32.174082              3.901184        29           29           0.000000                   5.0               14.4
        march_vol_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            32.937643              4.255524        29           29           0.000000                   5.0               14.4
        march_vol_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP            50.441938              8.063681        29           29           0.000000                   6.0               11.4
        march_vol_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP            52.174259              8.255715        29           29           0.000000                   4.0               11.0
           okx_usdc_depeg_2023     okx derivatives  BTC  USDC BTCUSDC-230331            41.312696              3.449598        27           27           0.000000                   2.0                4.4
           okx_usdc_depeg_2023     okx derivatives  BTC  USDT BTCUSDT-230331            32.472730              4.839730        28           28           0.000000                   4.0                8.6
           okx_usdc_depeg_2023     okx derivatives  ETH  USDC ETHUSDC-230331            60.582987              4.140609        26           26           0.000000                   2.0                6.5
           okx_usdc_depeg_2023     okx derivatives  ETH  USDT ETHUSDT-230331            42.080464              5.712658        28           28           0.000000                   4.0               15.3
yen_followthrough_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            38.512817              5.836652        29           29           0.000000                   3.0               11.4
yen_followthrough_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            36.137496              5.856705        29           29           0.000000                   4.0               17.2
yen_followthrough_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            30.146115              5.106934        29           29           0.000000                   3.0                7.0
yen_followthrough_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            29.658742              5.130883        29           29           0.000000                   3.0               10.2
yen_followthrough_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            49.304137              7.531066        29           29           0.000000                   4.0                9.2
yen_followthrough_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            49.952924              7.721190        29           29           0.000000                   3.0                7.0
yen_followthrough_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP            57.834938             10.293786        29           29           0.000000                   3.0               10.0
yen_followthrough_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP            57.619574             10.512261        29           29           0.000000                   3.0               10.0
       yen_unwind_2024_binance binance derivatives  BNB  USDC   BNBUSDC-PERP            81.770320             11.277097        29           29           0.000000                   5.0               11.2
       yen_unwind_2024_binance binance derivatives  BNB  USDT   BNBUSDT-PERP            77.359485             10.967652        29           29           0.000000                   6.0               11.0
       yen_unwind_2024_binance binance derivatives  BTC  USDC   BTCUSDC-PERP            64.388581              8.654788        29           29           0.000000                   4.0               10.0
       yen_unwind_2024_binance binance derivatives  BTC  USDT   BTCUSDT-PERP            62.036388              8.838933        29           29           0.000000                   5.0               11.0
       yen_unwind_2024_binance binance derivatives  ETH  USDC   ETHUSDC-PERP            94.911311             11.925691        29           29           0.000000                   6.0               11.2
       yen_unwind_2024_binance binance derivatives  ETH  USDT   ETHUSDT-PERP            95.104944             11.983906        29           29           0.000000                   6.0               12.0
       yen_unwind_2024_binance binance derivatives  SOL  USDC   SOLUSDC-PERP           136.064472             17.393240        29           29           0.000000                   6.0               15.2
       yen_unwind_2024_binance binance derivatives  SOL  USDT   SOLUSDT-PERP           134.784974             17.620047        29           29           0.000000                   8.0               15.2
```

## Venue Summary (Within Market Type)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                     3                 -0.972122                   -0.931979                    -0.595878                      -0.520973              -0.034535                -0.047318                     2                        4.0                        4.5                          0.0                     0.002874
  bybit        spot                     3                  1.022176                    0.781291                     1.446459                       1.102216               1.025324                 0.050954                     0                        3.0                        3.5                          0.0                     0.000559
    okx derivatives                     2                 -3.330216                   -3.330216                    -1.619332                      -1.619332               0.699729                 0.699729                     1                        2.0                        4.0                          0.0                     0.000000
```

## Artifacts

- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
