# Robustness Summary

Strict verdict rule:
- `PASS` iff base scenario has `Sharpe > 0` and `PnL net > 0`
- and at least 3/4 single stress scenarios pass (`fees_x2`, `spread_x2`, `latency_1bar`, `liquidity_half`).

## Run Statistics

- splits: `5`
- base ablation rows: `80`
- stress rows: `480`
- verdict rows: `80`

## Verdict Rates

- overall verdict pass rate: `0.050`
- reference variant (`premium_debiased__gating_on__statmech_on__hawkes_on`) pass rate: `0.000`

## Walk-Forward Splits

```text
 split_id  train_episode_count                                                                                                                                                                                                                                                                                                           train_episode_ids                                                          test_episode_id               train_start                 train_end                test_start                  test_end  selected_entry_k  selected_t_widen_quantile  selected_chi_widen_quantile  selected_stress_quantile  selected_recovery_quantile  train_score  train_mean_sharpe_full_raw  train_mean_pnl_net                                reference_variant_id  reference_base_sharpe  reference_base_pnl_net
        1                    1                                                                                                                                                                                                                                                               data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv          data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv 2023-03-09 16:00:00+00:00 2023-03-11 15:59:00+00:00 2023-03-10 00:00:00+00:00 2023-03-11 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6     0.204802                    0.013099            0.007668 premium_debiased__gating_on__statmech_on__hawkes_on              -0.003161               -0.000181
        2                    2                                                                                                                                                                                               data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv 2023-03-09 16:00:00+00:00 2023-03-11 23:59:00+00:00 2024-03-12 00:00:00+00:00 2024-03-13 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6     0.102319                    0.007041            0.003872 premium_debiased__gating_on__statmech_on__hawkes_on              -0.002636               -0.000147
        3                    3                                                                                                                              data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/march_vol_2024_binance/prices_matrix.csv               data/processed/episodes/smoke_2024_08_05/prices_matrix.csv 2023-03-09 16:00:00+00:00 2024-03-13 23:59:00+00:00 2024-08-05 00:00:00+00:00 2024-08-05 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6     0.046049                   -0.001875            0.002052 premium_debiased__gating_on__statmech_on__hawkes_on              -0.005464               -0.000314
        4                    4                                                                   data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/march_vol_2024_binance/prices_matrix.csv|data/processed/episodes/smoke_2024_08_05/prices_matrix.csv        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv 2023-03-09 16:00:00+00:00 2024-08-05 23:59:00+00:00 2024-08-05 00:00:00+00:00 2024-08-06 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6     0.026110                   -0.003740            0.001316 premium_debiased__gating_on__statmech_on__hawkes_on              -0.008242               -0.000669
        5                    5 data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/march_vol_2024_binance/prices_matrix.csv|data/processed/episodes/smoke_2024_08_05/prices_matrix.csv|data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv 2023-03-09 16:00:00+00:00 2024-08-06 23:59:00+00:00 2024-08-07 00:00:00+00:00 2024-08-08 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6     0.018283                   -0.003491            0.000980 premium_debiased__gating_on__statmech_on__hawkes_on              -0.005299               -0.000133
```

## Reference Variant Verdict

```text
 split_id                                                             test_episode                                          variant_id  base_sharpe  base_pnl_net  base_pass  pass_fees_x2  pass_spread_x2  pass_latency_1bar  pass_liquidity_half  single_pass_count  singles_majority_pass  combined_worst_sharpe  combined_worst_pnl_net  combined_worst_pass  verdict_pass
        1          data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.003161     -0.000181      False         False           False              False                False                  0                  False              -0.013503               -0.000870                False         False
        2         data/processed/episodes/march_vol_2024_binance/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.002636     -0.000147      False         False           False              False                False                  0                  False              -0.008775               -0.000498                False         False
        3               data/processed/episodes/smoke_2024_08_05/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.005464     -0.000314      False         False           False              False                False                  0                  False              -0.016922               -0.000990                False         False
        4        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.008242     -0.000669      False         False           False              False                False                  0                  False              -0.014555               -0.001197                False         False
        5 data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.005299     -0.000133      False         False           False              False                False                  0                  False              -0.012872               -0.000335                False         False
```
