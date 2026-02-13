# Crypto Premium Regime Gating (Hiring Project)

This repository implements the premium regime gating framework described in `AGENTS.md`.

## Quickstart (what the reviewer needs)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -q
python -m src.pipeline --config configs/config.yaml
```

Default runnable input already configured in `configs/config.yaml`:

- `data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv`

Main outputs after pipeline run:

- `reports/tables/`
- `reports/figures/`

Quick safety check after running:

- `reports/tables/safety_diagnostics.csv` should show `depeg_bars > 0`, `riskoff_bars > 0`, and `depeg_without_riskoff_bars = 0` on the default bundled depeg episode.
- `reports/tables/scope_audit.csv` provides machine-auditable pass/fail checks mapped to requested baseline scope.

## Repository map

Top-level:

- `README.md`: runbook
- `AGENTS.md`: project spec
- `AGENT.md`: coding guide used during implementation
- `requirements.txt`: pinned dependencies
- `configs/config.yaml`: default runtime config
- `src/`: implementation
- `tests/`: test suite
- `scripts/`: utility scripts
- `notebooks/01_report.ipynb`: notebook surface
- `data/`: local raw/processed inputs
- `reports/`: generated outputs

Runtime folders:

- `data/raw/`: downloaded raw exchange files
- `data/processed/episodes/<episode>/`: episode matrices (`prices_matrix.*`)
- `data/processed/onchain/`: on-chain cache
- `data/processed/orderbook/<episode>/`: execution/L2 datasets
- `reports/tables/`: latest pipeline tables
- `reports/figures/`: latest pipeline figures
- `reports/final/`: final consolidated deliverables
- `reports/episodes/<episode>/`: per-episode outputs from loader CLIs

### Source files (`src/`)

- `src/__init__.py`: package marker
- `src/pipeline.py`: end-to-end orchestrator
- `src/data_ingest.py`: raw normalization/resampling/UTC alignment
- `src/premium.py`: `p_naive`, proxy, debiased premium `p`, depeg market flag
- `src/robust_filter.py`: robust smoothing/scale and event detection
- `src/statmech.py`: `H_t`, `T_t`, `chi_t`
- `src/regimes.py`: transient/stress regime frame
- `src/thresholds.py`: causal quantile threshold utilities
- `src/hawkes.py`: optional rolling Hawkes + branching ratio quality checks
- `src/strategy.py`: `Trade` / `Widen` / `Risk-off` decisions + sizing
- `src/backtest.py`: naive vs gated backtest + metrics
- `src/plots.py`: figures export
- `src/binance_data.py`: Binance downloader + episode builder
- `src/bybit_data.py`: Bybit downloader + episode builder
- `src/okx_data.py`: OKX downloader + episode builder
- `src/onchain.py`: DefiLlama validation feed integration
- `src/execution_data.py`: execution/L2 dataset bootstrap
- `src/execution_quality.py`: slippage/resilience diagnostics
- `src/ablation_core.py`: shared core utilities for ablations
- `src/ablation_report.py`: ablation report CLI
- `src/robustness_report.py`: walk-forward OOS robustness CLI
- `src/tune_gating.py`: gating parameter tuning CLI
- `src/calibration_report.py`: tuned vs baseline calibration report CLI
- `src/presentation_pack.py`: final consolidated report pack CLI

### Tests and scripts

- `tests/test_regressions.py`: component + integration regressions
- `tests/test_robustness_report.py`: robustness report tests
- `scripts/package_submission.sh`: create clean recruiter zip
- `scripts/clean_local_artifacts.sh`: remove local temporary artifacts

## Command index (essential)

Run pipeline:

```bash
python -m src.pipeline --config configs/config.yaml
```

Build matrix from raw files:

```bash
python -m src.data_ingest --inputs "data/raw/**/*.csv" --output-dir data/processed --resample-rule 1min
```

Episode loaders (download + optional pipeline run):

```bash
python -m src.binance_data --start 2024-08-05 --end 2024-08-06 --market futures --futures-price-source mark --episode-name yen_unwind_2024_binance --run-pipeline --skip-existing
python -m src.bybit_data --start 2023-03-10 --end 2023-03-11 --category spot --episode-name bybit_usdc_depeg_2023 --run-pipeline
python -m src.okx_data --start 2023-03-10 --end 2023-03-11 --episode-name okx_usdc_depeg_2023 --run-pipeline --skip-existing
```

Optional report/analysis CLIs:

```bash
python -m src.ablation_report --output-dir reports/tables
python -m src.robustness_report --output-dir reports/robustness
python -m src.tune_gating --episodes "data/processed/episodes/*2024_binance/prices_matrix.csv" --apply
python -m src.calibration_report --episodes "data/processed/episodes/*2024_binance/prices_matrix.csv" --output-dir reports/final
python -m src.presentation_pack --output-dir reports/final --reports-root reports
python -m src.execution_data --skip-existing --include-agg-trades
python -m src.execution_quality --output-dir reports/final
```

Robustness defaults:

- walk-forward minimum train episodes: `2` (`--min-train-episodes`)
- smoke episodes are excluded by default; pass `--include-smoke` to include them
- compatibility skips are reported in structured CSV files with `reason_code` + `reason` fields.

Strategy gating default:

- `strategy.strict_notice_gating: true` enforces a hard precondition before `Trade`: `|m_t| > entry_k * T_t * sigma_hat`.

Packaging helpers:

```bash
./scripts/clean_local_artifacts.sh
./scripts/package_submission.sh
```

## Input and output contract (short)

Expected matrix format:

- index: `timestamp_utc` (UTC DatetimeIndex)
- columns: symbols (`BTCUSDC-PERP`, `BTCUSDT-PERP`, ...)
- values: price
- format: `.parquet` or `.csv`

Pipeline output files (default):

- `reports/tables/metrics.csv`
- `reports/tables/trade_log_gated.csv`
- `reports/tables/trade_log_naive.csv`
- `reports/tables/signal_frame.parquet` (fallback: `reports/tables/signal_frame.csv`)
- `reports/tables/stablecoin_proxy_components.parquet` (fallback: `reports/tables/stablecoin_proxy_components.csv`)
- `reports/tables/safety_diagnostics.csv`
- `reports/tables/scope_audit.csv`
- `reports/tables/edge_net_size_curve.csv`
- `reports/tables/break_even_premium_curve.csv`
- `reports/tables/edge_net_summary.csv`
- `reports/figures/figure_1_timeline.png`
- `reports/figures/figure_2_panel.png`
- `reports/figures/figure_3_phase_space.png`
- `reports/figures/figure_4_edge_net.png`

## Where to read deeper details

- Full quantitative policy, defaults, and definitions: `AGENTS.md`
