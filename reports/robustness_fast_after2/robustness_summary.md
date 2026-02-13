# Robustness Summary

Strict verdict rule:
- `PASS` iff base scenario has `Sharpe > 0` and `PnL net > 0`
- and at least 3/4 single stress scenarios pass (`fees_x2`, `spread_x2`, `latency_1bar`, `liquidity_half`).

## Run Statistics

- splits: `2`
- base ablation rows: `32`
- stress rows: `192`
- verdict rows: `32`

## Verdict Rates

- overall verdict pass rate: `0.000`
- reference variant (`premium_debiased__gating_on__statmech_on__hawkes_on`) pass rate: `0.000`

## Walk-Forward Splits

```text
 split_id  train_episode_count                                                                                                                                                                                train_episode_ids                                                          test_episode_id               train_start                 train_end                test_start                  test_end  selected_entry_k  selected_t_widen_quantile  selected_chi_widen_quantile  selected_stress_quantile  selected_recovery_quantile  selected_train_robust_score  train_score  train_mean_sharpe_full_raw  train_mean_pnl_net                                reference_variant_id  reference_base_sharpe  reference_base_pnl_net
        1                    2                                                                   data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/march_vol_2024_binance/prices_matrix.csv        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv 2023-03-09 16:00:00+00:00 2024-03-13 23:59:00+00:00 2024-08-05 00:00:00+00:00 2024-08-06 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6                     0.030338     0.051421                    0.001463            0.002068 premium_debiased__gating_on__statmech_on__hawkes_on              -0.003950               -0.000294
        2                    3 data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv|data/processed/episodes/march_vol_2024_binance/prices_matrix.csv|data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv 2023-03-09 16:00:00+00:00 2024-08-06 23:59:00+00:00 2024-08-07 00:00:00+00:00 2024-08-08 23:59:00+00:00               0.5                       0.97                         0.97                      0.99                         0.6                    -0.028184     0.034297                    0.001040            0.001387 premium_debiased__gating_on__statmech_on__hawkes_on              -0.003371               -0.000089
```

## Reference Variant Verdict

```text
 split_id                                                             test_episode                                          variant_id  base_sharpe  base_pnl_net  base_pass  pass_fees_x2  pass_spread_x2  pass_latency_1bar  pass_liquidity_half  single_pass_count  singles_majority_pass  combined_worst_sharpe  combined_worst_pnl_net  combined_worst_pass  verdict_pass
        1        data/processed/episodes/yen_unwind_2024_binance/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.003950     -0.000294      False         False           False              False                False                  0                  False              -0.007940               -0.000599                False         False
        2 data/processed/episodes/yen_followthrough_2024_binance/prices_matrix.csv premium_debiased__gating_on__statmech_on__hawkes_on    -0.003371     -0.000089      False         False           False              False                False                  0                  False              -0.010478               -0.000285                False         False
```
