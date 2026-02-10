# Crypto Premium Regime Gating (Hiring Project)

This repository implements the v5 framework described in `AGENT.md` and the two notes:

- `Notice.pdf` (baseline: premium + stat-mech + regime gating)
- `Notice + Hawkes.pdf` (optional Hawkes contagion overlay)

## Scope selected from the hiring brief

This project combines:

- Stable coin analysis:
  - fair-value proxy of `USDT/USDC` spread via cross-asset replication
  - depeg detection flag (`delta_log`, consecutive windows)
  - transmission into synthetic premium (`BTCUSDC` vs `BTCUSDT`)
- Premium analysis (robust estimation):
  - naive premium failure handling
  - robust smoothing + outlier events
  - regime detection (transient vs stress)
- Statistical mechanics features:
  - entropy `H_t`, temperature `T_t`, susceptibility `chi_t`
- Optional advanced method:
  - Hawkes branching ratio `n(t)` for stress contagion gating

## Data sources from the email

- Binance: [data.binance.vision](https://data.binance.vision/?prefix=data/)
- GateIO historical quotes: [gate.com developer docs](https://www.gate.com/developer/historical_quotes)
- Bybit historical data: [bybit history data](https://www.bybit.com/derivatives/en/history-data)
- OKX direct URLs (login bypass pattern):
  - trades: `https://www.okx.com/cdn/okex/traderecords/trades/monthly/YYYYMM/allfuture-trades-YYYY-MM-DD.zip`
  - funding: `https://www.okx.com/cdn/okex/traderecords/swaprate/monthly/YYYYMM/allswaprate-swaprate-YYYY-MM-DD.zip`

Helpers for OKX URL construction/download are implemented in `src/data_ingest.py`.

Quick example:

```python
from datetime import date
from src.data_ingest import download_okx_range

download_okx_range(
    start=date(2024, 8, 5),
    end=date(2024, 8, 6),
    kind="trades",
    output_dir="data/raw/okx/trades",
)
```

## Repository structure

```
src/
  data_ingest.py      # load/clean/resample market data
  binance_data.py     # Binance episode loader (spot/futures)
  bybit_data.py       # Bybit kline loader (spot/linear)
  okx_data.py         # OKX trade-archive loader (futures contracts)
  premium.py          # p_naive, stablecoin proxy, debiased p, depeg flag
  robust_filter.py    # p_smooth, sigma_hat, z_t, events
  statmech.py         # H_t, T_t, chi_t
  regimes.py          # stress/transient baseline change-point regime model
  hawkes.py           # optional rolling Hawkes fit + n(t)
  strategy.py         # Trade/Widen/Risk-off decision logic
  backtest.py         # naive vs gated backtest + metrics
  plots.py            # Figure 1/2/3 exports
  pipeline.py         # end-to-end runner
  tune_gating.py      # parameter tuning for regime/strategy gating
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected input

Main pipeline input is a price matrix (parquet recommended, csv supported):

- path: `data/processed/prices_matrix.parquet` (default, configurable)
- index: `timestamp_utc` (UTC DatetimeIndex)
- columns: symbols (e.g., `BTCUSDC-PERP`, `BTCUSDT-PERP`, `ETHUSDC-PERP`, ...)
- values: price

If you start from raw files, use functions from `src/data_ingest.py` to normalize and build this matrix.

CLI option:

```bash
python -m src.data_ingest \
  --inputs "data/raw/**/*.csv" \
  --output-dir data/processed \
  --resample-rule 1min
```

## Run

```bash
python -m src.pipeline --config configs/config.yaml
```

or override the matrix path:

```bash
python -m src.pipeline --config configs/config.yaml --price-matrix data/processed/my_prices.parquet
```

If parquet engine is unavailable locally, ingestion/export falls back to csv and you can pass a `.csv` path.

## One-command episode run (Binance)

Download daily Binance futures klines, build the matrix, and run the full pipeline:

```bash
python -m src.binance_data \
  --start 2024-08-05 \
  --end 2024-08-06 \
  --market futures \
  --episode-name yen_unwind_2024_binance \
  --run-pipeline \
  --skip-existing
```

For older windows (for example March 2023), Binance public data may not contain USDC quote pairs; the CLI prints an explicit availability error in that case.

This writes:

- processed data: `data/processed/episodes/yen_unwind_2024_binance/`
- per-episode reports: `reports/episodes/yen_unwind_2024_binance/`

## March 2023 USDC coverage loaders (Bybit / OKX)

Bybit spot (USDC pairs available in public API):

```bash
python -m src.bybit_data \
  --start 2023-03-10 \
  --end 2023-03-11 \
  --category spot \
  --episode-name bybit_usdc_depeg_2023 \
  --run-pipeline
```

OKX futures trade archive (auto-selects matched USDC/USDT contracts by liquidity):

```bash
python -m src.okx_data \
  --start 2023-03-10 \
  --end 2023-03-11 \
  --episode-name okx_usdc_depeg_2023 \
  --run-pipeline \
  --skip-existing
```

Example auto-selected contracts for this window:
- `BTC-USDC-230331` vs `BTC-USDT-230331`
- `ETH-USDC-230331` vs `ETH-USDT-230331`

Both commands write episode-specific outputs under `reports/episodes/<episode_name>/`.

## Gating parameter tuning (2024 episodes)

Grid-search regime + strategy gating parameters and apply the best combo:

```bash
python -m src.tune_gating \
  --episodes "data/processed/episodes/*2024_binance/prices_matrix.csv" \
  --apply
```

Latest tuned defaults now set in `configs/config.yaml`:

- `strategy.entry_k: 0.5`
- `strategy.t_widen_quantile: 0.99`
- `strategy.chi_widen_quantile: 0.99`
- `regimes.stress_quantile: 0.9`
- `regimes.recovery_quantile: 0.8`

## Outputs

- `reports/tables/metrics.csv`
- `reports/tables/trade_log_gated.csv`
- `reports/tables/trade_log_naive.csv`
- `reports/tables/signal_frame.parquet`
- `reports/figures/figure_1_timeline.png`
- `reports/figures/figure_2_panel.png`
- `reports/figures/figure_3_phase_space.png`

## Decision policy implemented

Priority order:

1. `depeg_flag == True` => `Risk-off`
2. `regime == stress` => `Risk-off`
3. Hawkes enabled:
   - `n(t) > 0.85` => `Risk-off`
   - `n(t) > 0.70` => `Widen`
4. Else transient mode:
   - `Trade` only if `|m_t| > k * T_t * sigma_hat` (unit-consistent implementation)
   - `Widen` when high `T_t` or `chi_t`

## Fixed episodes to evaluate

- LUNA/UST: 2022-05-09 to 2022-05-13
- FTX: 2022-11-06 to 2022-11-11
- USDC depeg: 2023-03-10 to 2023-03-11
- Yen carry unwind: 2024-08-05 to 2024-08-06

## Notes

- This repository is intentionally modular: the notebook is a report surface, while calculations stay in `src/`.
- Current implementation is a robust baseline and is designed for quick iteration under the 1-week timeline.
