from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.data_ingest import IngestConfig, build_processed_prices, save_processed
from src.pipeline import export_outputs, load_config, run_pipeline
from src.premium import infer_cross_asset_pairs

DEFAULT_SPOT_SYMBOLS = [
    "BTCUSDC",
    "BTCUSDT",
    "ETHUSDC",
    "ETHUSDT",
    "SOLUSDC",
    "SOLUSDT",
    "BNBUSDC",
    "BNBUSDT",
]

DEFAULT_LINEAR_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def bybit_interval_to_ms(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * 60_000
    mapping = {
        "D": 24 * 60 * 60 * 1000,
        "W": 7 * 24 * 60 * 60 * 1000,
    }
    if interval in mapping:
        return mapping[interval]
    raise ValueError(f"Unsupported Bybit interval: {interval}")


def fetch_bybit_kline(
    *,
    session: requests.Session,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> list[list[str]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": str(start_ms),
        "end": str(end_ms),
        "limit": str(limit),
    }
    resp = session.get("https://api.bybit.com/v5/market/kline", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit kline error for {symbol}: {payload.get('retCode')} {payload.get('retMsg')}"
        )
    return payload.get("result", {}).get("list", []) or []


def collect_symbol_klines(
    *,
    session: requests.Session,
    category: str,
    symbol: str,
    interval: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    start_ms = int(datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_exclusive_ms = int(
        (datetime.combine(end, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp()
        * 1000
    )
    interval_ms = bybit_interval_to_ms(interval)
    chunk_span_ms = interval_ms * 1000

    rows: list[list[str]] = []
    cursor = start_ms
    while cursor < end_exclusive_ms:
        chunk_end = min(cursor + chunk_span_ms - interval_ms, end_exclusive_ms - interval_ms)
        chunk = fetch_bybit_kline(
            session=session,
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=chunk_end,
            limit=1000,
        )
        rows.extend(chunk)
        cursor = chunk_end + interval_ms

    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "price", "volume", "symbol", "venue"])

    frame = pd.DataFrame(rows, columns=["start_ms", "open", "high", "low", "close", "volume", "turnover"])
    frame["start_ms"] = pd.to_numeric(frame["start_ms"], errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(frame["start_ms"], unit="ms", utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    suffix = "PERP" if category in {"linear", "inverse"} else "SPOT"
    frame["symbol"] = f"{symbol}-{suffix}"
    frame["venue"] = "bybit"
    out = frame[["timestamp_utc", "price", "volume", "symbol", "venue"]].dropna(
        subset=["timestamp_utc", "price"]
    )
    out = out.drop_duplicates(subset=["timestamp_utc", "symbol"], keep="last")
    return out


def collect_bybit_klines(
    *,
    category: str,
    symbols: Iterable[str],
    interval: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, list[str]]:
    tables: list[pd.DataFrame] = []
    failed: list[str] = []
    with requests.Session() as session:
        for symbol in symbols:
            try:
                table = collect_symbol_klines(
                    session=session,
                    category=category,
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                )
                if table.empty:
                    failed.append(symbol)
                    continue
                tables.append(table)
            except Exception:
                failed.append(symbol)
    if not tables:
        raise RuntimeError("No Bybit data loaded. Check category/symbol/date selection.")
    return pd.concat(tables, ignore_index=True), failed


def find_matrix_path(processed_dir: Path) -> Path:
    parquet_path = processed_dir / "prices_matrix.parquet"
    csv_path = processed_dir / "prices_matrix.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No prices_matrix found in {processed_dir}")


def _build_proxy_pairs(columns: list[str], target_pair: tuple[str, str]) -> list[list[str]]:
    inferred = infer_cross_asset_pairs(columns, exclude=target_pair)
    return [[usdc, usdt] for usdc, usdt in inferred]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Bybit klines (spot/linear), build matrix, and optionally run the pipeline."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--category",
        choices=["spot", "linear", "inverse"],
        default="spot",
        help="Bybit kline category.",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Bybit symbols without suffix.")
    parser.add_argument("--interval", default="1", help="Bybit kline interval.")
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Processed episode root.")
    parser.add_argument("--episode-name", default=None, help="Optional custom episode name.")
    parser.add_argument("--resample-rule", default="1min", help="Target resample rule.")
    parser.add_argument("--ffill-limit", type=int, default=2, help="Forward-fill limit.")
    parser.add_argument("--glitch-sigma-threshold", type=float, default=20.0, help="Glitch filter threshold.")
    parser.add_argument("--run-pipeline", action="store_true", help="Run full premium pipeline after ingest.")
    parser.add_argument("--config", default="configs/config.yaml", help="Pipeline config path.")
    parser.add_argument("--reports-root", default="reports/episodes", help="Per-episode reports root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    episode = args.episode_name or f"bybit_{start.isoformat()}_{end.isoformat()}_{args.category}"

    symbols = args.symbols
    if symbols is None:
        symbols = DEFAULT_SPOT_SYMBOLS if args.category == "spot" else DEFAULT_LINEAR_SYMBOLS

    print(f"Collecting Bybit data for {episode}...")
    raw_table, failed = collect_bybit_klines(
        category=args.category,
        symbols=symbols,
        interval=args.interval,
        start=start,
        end=end,
    )
    if failed:
        print(f"Warning: unavailable symbols ({len(failed)}): {failed}")

    ingest_cfg = IngestConfig(
        resample_rule=args.resample_rule,
        ffill_limit=args.ffill_limit,
        glitch_sigma_threshold=args.glitch_sigma_threshold,
    )
    processed_dir = Path(args.processed_root) / episode
    reports_root = Path(args.reports_root) / episode
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

    suffix = "PERP" if args.category in {"linear", "inverse"} else "SPOT"
    target_usdc_symbol = f"BTCUSDC-{suffix}"
    target_usdt_symbol = f"BTCUSDT-{suffix}"
    missing_targets = [s for s in (target_usdc_symbol, target_usdt_symbol) if s not in matrix.columns]
    if missing_targets:
        available = ", ".join(sorted(matrix.columns))
        raise SystemExit(
            f"Target symbols unavailable for this Bybit episode. Missing={missing_targets}. "
            f"Available=[{available}]"
        )

    config = load_config(args.config)
    run_cfg = deepcopy(config)
    run_cfg.setdefault("data", {})
    run_cfg["data"]["resample_rule"] = args.resample_rule
    run_cfg["data"]["price_matrix_path"] = str(matrix_path)
    run_cfg.setdefault("outputs", {})
    run_cfg["outputs"]["tables_dir"] = str(reports_root / "tables")
    run_cfg["outputs"]["figures_dir"] = str(reports_root / "figures")
    run_cfg.setdefault("premium", {})
    run_cfg["premium"]["target_usdc_symbol"] = target_usdc_symbol
    run_cfg["premium"]["target_usdt_symbol"] = target_usdt_symbol
    run_cfg["premium"]["proxy_pairs"] = _build_proxy_pairs(
        list(matrix.columns), (target_usdc_symbol, target_usdt_symbol)
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
