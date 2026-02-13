# Execution Diagnostics (Order-Book Snapshot)

This report computes slippage-vs-size from L2 snapshot book-walk metrics.

## Scope

- `march_vol_2024_binance`

## L2 Coverage

```text
               episode  l2_orderbook_available  tick_trades_available  l2_ready                                         l2_root
march_vol_2024_binance                    True                   True      True data/processed/orderbook/march_vol_2024_binance
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
               episode   venue market_type root quote       symbol                   size_bin  dnl_bin_low  dnl_bin_high   n_obs  dnl_median  book_walk_mean_bps  book_walk_median_bps  queue_load_mean
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP    (-0.00099774, 1.67e-05]    -0.000998      0.000017  113103    0.000012            0.001214              0.001240         0.004601
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (0.000161, 0.000402]     0.000161      0.000402  113100    0.000241            0.025548              0.024150         0.008531
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP          (0.000402, 0.112]     0.000402      0.112000  113103    0.000691            0.103940              0.069074         0.012475
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (1.67e-05, 3.04e-05]     0.000017      0.000030  113103    0.000020            0.002099              0.001966         0.004467
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (3.04e-05, 8.67e-05]     0.000030      0.000087  113104    0.000054            0.005534              0.005404         0.007325
march_vol_2024_binance binance derivatives  BTC  USDC BTCUSDC-PERP       (8.67e-05, 0.000161]     0.000087      0.000161  113106    0.000123            0.012301              0.012333         0.007965
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP (-0.00099999354, 1.34e-06]    -0.001000      0.000001 1820208    0.000001            0.000105              0.000109         0.015039
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.34e-06, 2.13e-06]     0.000001      0.000002 1820074    0.000002            0.000164              0.000160         0.018516
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (1.52e-05, 5.56e-05]     0.000015      0.000056 1820141    0.000027            0.003003              0.002702         0.018631
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (2.13e-06, 5.94e-06]     0.000002      0.000006 1820139    0.000004            0.000378              0.000366         0.017348
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP        (5.56e-05, 146.795]     0.000056    146.795000 1820139    0.000128            0.122508              0.012798         0.710132
march_vol_2024_binance binance derivatives  BTC  USDT BTCUSDT-PERP       (5.94e-06, 1.52e-05]     0.000006      0.000015 1820138    0.000009            0.000958              0.000905         0.016795
```

## Cross-Quote Comparison (USDC vs USDT)

```text
               episode   venue market_type root  impact_large_mean_bps_usdc  impact_large_mean_bps_usdt  impact_large_delta_usdc_minus_usdt_bps  impact_large_mean_excess_bps_usdc  impact_large_mean_excess_bps_usdt  impact_large_delta_excess_usdc_minus_usdt_bps  impact_large_mean_norm_usdc  impact_large_mean_norm_usdt  impact_large_delta_norm_usdc_minus_usdt  impact_all_mean_bps_usdc  impact_all_mean_bps_usdt preferred_quote_on_large_norm  dnl_large_mean_usdc  dnl_large_mean_usdt  dnl_large_delta_usdc_minus_usdt  queue_load_large_mean_usdc  queue_load_large_mean_usdt  queue_load_large_delta_usdc_minus_usdt  snapshot_match_ratio_usdc  snapshot_match_ratio_usdt        slippage_method_usdc        slippage_method_usdt
march_vol_2024_binance binance derivatives  BTC                    0.140268                    0.199072                               -0.058804                      -8.496980e-20                          -0.481023                                       0.481023                     0.001403                     0.006801                                -0.005398                  0.025106                  0.021186                 indeterminate             0.001403             0.006801                        -0.005398                    0.014997                    1.170957                                -1.15596                   0.991908                   0.990038 orderbook_snapshot_bookwalk orderbook_snapshot_bookwalk
```

- `derivatives` (normalized delta, tolerance=0.050): USDC lower-impact `0/1`, USDT lower-impact `0/1`, indeterminate `1/1`.

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
```

## Venue Summary (Within Market Type)

```text
  venue market_type  n_root_episode_pairs  mean_delta_large_raw_bps  median_delta_large_raw_bps  mean_delta_large_excess_bps  median_delta_large_excess_bps  mean_delta_large_norm  median_delta_large_norm  n_indeterminate_norm  median_recovery_bars_usdc  median_recovery_bars_usdt  mean_unrecovered_ratio_usdc  mean_unrecovered_ratio_usdt
binance derivatives                     1                 -0.058804                   -0.058804                     0.481023                       0.481023              -0.005398                -0.005398                     1                        4.0                        4.5                          0.0                     0.008621
```

## Artifacts

- `reports/final/execution_l2_coverage.csv`
- `reports/final/execution_slippage_proxy.csv`
- `reports/final/execution_slippage_vs_size.csv`
- `reports/final/execution_cross_quote_comparison.csv`
- `reports/final/execution_resilience.csv`
- `reports/final/execution_venue_comparison.csv`
