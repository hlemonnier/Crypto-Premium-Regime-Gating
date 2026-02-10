from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from src.data_ingest import IngestConfig, build_processed_prices, save_processed
from src.pipeline import export_outputs, load_config, run_pipeline


DEFAULT_SYMBOLS = [
    "BTCUSDC",
    "BTCUSDT",
    "ETHUSDC",
    "ETHUSDT",
    "SOLUSDC",
    "SOLUSDT",
    "BNBUSDC",
    "BNBUSDT",
]

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def build_binance_daily_kline_url(
    symbol: str,
    interval: str,
    day: date,
    *,
    market: str,
) -> str:
    day_str = day.strftime("%Y-%m-%d")
    if market == "futures":
        market_path = "futures/um"
    elif market == "spot":
        market_path = "spot"
    else:
        raise ValueError("market must be 'futures' or 'spot'")
    return (
        f"https://data.binance.vision/data/{market_path}/daily/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{day_str}.zip"
    )


def download_to_path(url: str, destination: Path, *, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def load_binance_zip(path: Path, symbol: str) -> pd.DataFrame:
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError(f"Empty zip archive: {path}")
        with archive.open(names[0]) as handle:
            df = pd.read_csv(handle, header=None)

    # Some files can include headers in row 1. Detect and normalize.
    if df.shape[1] >= 5 and str(df.iloc[0, 0]).lower().startswith("open"):
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] < len(BINANCE_KLINE_COLUMNS):
        raise ValueError(f"Unexpected Binance kline layout in {path.name}")
    df = df.iloc[:, : len(BINANCE_KLINE_COLUMNS)]
    df.columns = BINANCE_KLINE_COLUMNS

    open_time = pd.to_numeric(df["open_time"], errors="coerce")
    out = pd.DataFrame()
    out["timestamp_utc"] = pd.to_datetime(open_time, unit="ms", utc=True, errors="coerce")
    out["price"] = pd.to_numeric(df["close"], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    out["symbol"] = f"{symbol}-PERP"
    out["venue"] = "binance"
    out = out.dropna(subset=["timestamp_utc", "price"])
    return out


def daterange(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def collect_binance_klines(
    *,
    symbols: list[str],
    start: date,
    end: date,
    interval: str,
    market: str,
    raw_dir: Path,
    skip_existing: bool,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    failures: list[str] = []
    for symbol in symbols:
        for day in daterange(start, end):
            url = build_binance_daily_kline_url(symbol, interval, day, market=market)
            filename = Path(url).name
            target_path = raw_dir / market / symbol / interval / filename

            try:
                if not (skip_existing and target_path.exists()):
                    download_to_path(url, target_path)
                tables.append(load_binance_zip(target_path, symbol))
            except Exception:
                failures.append(url)

    if not tables:
        raise RuntimeError("No Binance klines were loaded. Check dates/symbols and connectivity.")

    if failures:
        print(f"Warning: failed downloads {len(failures)}")
        for sample in failures[:10]:
            print(f"- {sample}")

    return pd.concat(tables, ignore_index=True)


def find_matrix_path(processed_dir: Path) -> Path:
    parquet_path = processed_dir / "prices_matrix.parquet"
    csv_path = processed_dir / "prices_matrix.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No prices_matrix found in {processed_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binance futures klines and build processed matrix for episodes."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Binance symbols (without -PERP).")
    parser.add_argument(
        "--market",
        choices=["futures", "spot"],
        default="futures",
        help="Binance dataset family",
    )
    parser.add_argument("--interval", default="1m", help="Binance kline interval")
    parser.add_argument("--raw-dir", default="data/raw/binance/klines", help="Raw zip destination")
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Processed episode root")
    parser.add_argument("--episode-name", default=None, help="Optional custom episode name")
    parser.add_argument("--resample-rule", default="1min", help="Target resample rule")
    parser.add_argument("--ffill-limit", type=int, default=2, help="Forward-fill limit")
    parser.add_argument("--glitch-sigma-threshold", type=float, default=20.0, help="Glitch filter threshold")
    parser.add_argument("--skip-existing", action="store_true", help="Skip downloading files already present")
    parser.add_argument("--run-pipeline", action="store_true", help="Run full premium pipeline after ingest")
    parser.add_argument("--config", default="configs/config.yaml", help="Pipeline config path")
    parser.add_argument("--reports-root", default="reports/episodes", help="Per-episode reports root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    episode = args.episode_name or f"{start.isoformat()}_{end.isoformat()}"

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_root) / episode
    reports_root = Path(args.reports_root) / episode

    print(f"Collecting Binance data for {episode}...")
    raw_table = collect_binance_klines(
        symbols=args.symbols,
        start=start,
        end=end,
        interval=args.interval,
        market=args.market,
        raw_dir=raw_dir,
        skip_existing=args.skip_existing,
    )

    ingest_cfg = IngestConfig(
        resample_rule=args.resample_rule,
        ffill_limit=args.ffill_limit,
        glitch_sigma_threshold=args.glitch_sigma_threshold,
    )
    resampled, matrix = build_processed_prices(raw_table, ingest_cfg)
    save_processed(resampled, matrix, processed_dir)
    matrix_path = find_matrix_path(processed_dir)

    print("Ingestion completed.")
    print(f"- raw rows: {raw_table.shape[0]}")
    print(f"- resampled rows: {resampled.shape[0]}")
    print(f"- matrix shape: {matrix.shape}")
    print(f"- matrix path: {matrix_path}")

    if not args.run_pipeline:
        return

    config = load_config(args.config)
    run_cfg = deepcopy(config)
    run_cfg.setdefault("data", {})
    run_cfg["data"]["resample_rule"] = args.resample_rule
    run_cfg["data"]["price_matrix_path"] = str(matrix_path)
    run_cfg.setdefault("outputs", {})
    run_cfg["outputs"]["tables_dir"] = str(reports_root / "tables")
    run_cfg["outputs"]["figures_dir"] = str(reports_root / "figures")

    premium_cfg = run_cfg.get("premium", {})
    target_usdc = premium_cfg.get("target_usdc_symbol", "BTCUSDC-PERP")
    target_usdt = premium_cfg.get("target_usdt_symbol", "BTCUSDT-PERP")
    missing_targets = [s for s in (target_usdc, target_usdt) if s not in matrix.columns]
    if missing_targets:
        available = ", ".join(sorted(matrix.columns))
        raise SystemExit(
            "Target symbols unavailable for this episode on selected Binance dataset. "
            f"Missing: {missing_targets}. Available: [{available}]. "
            "Try a different episode/source or adjust config premium target symbols."
        )

    print("Running premium pipeline...")
    results = run_pipeline(run_cfg, matrix)
    exported = export_outputs(results, run_cfg)
    print("Pipeline completed.")
    print("Metrics:")
    print(results["metrics"])
    print("Artifacts:")
    for key, path in exported.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
