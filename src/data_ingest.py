from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import glob
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

REQUIRED_COLUMNS = ("timestamp_utc", "symbol", "price")
TIMESTAMP_CANDIDATES = ("timestamp_utc", "timestamp", "time", "datetime", "ts")
PRICE_CANDIDATES = ("price", "mark_price", "last_price", "close")
SYMBOL_CANDIDATES = ("symbol", "instId", "instrument", "ticker")
VENUE_CANDIDATES = ("venue", "exchange")
VOLUME_CANDIDATES = ("volume", "qty", "quantity", "size")


@dataclass(frozen=True)
class IngestConfig:
    resample_rule: str = "1min"
    ffill_limit: int = 2
    glitch_sigma_threshold: float = 20.0
    keep_duplicate: str = "last"


def _pick_existing(candidates: Sequence[str], columns: Sequence[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def read_market_file(
    path: str | Path,
    *,
    column_map: Mapping[str, str] | None = None,
    symbol_override: str | None = None,
    venue_override: str | None = None,
) -> pd.DataFrame:
    file_path = Path(path)
    suffixes = [s.lower() for s in file_path.suffixes]
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if ".parquet" in suffixes:
        df = pd.read_parquet(file_path)
    elif ".csv" in suffixes or ".gz" in suffixes:
        df = pd.read_csv(file_path)
    elif ".zip" in suffixes:
        with ZipFile(file_path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError(f"Zip file is empty: {file_path}")
            with archive.open(names[0]) as handle:
                df = pd.read_csv(handle)
    else:
        raise ValueError(f"Unsupported file format: {file_path.name}")

    if column_map:
        df = df.rename(columns=column_map)

    timestamp_col = _pick_existing(TIMESTAMP_CANDIDATES, df.columns)
    price_col = _pick_existing(PRICE_CANDIDATES, df.columns)
    symbol_col = _pick_existing(SYMBOL_CANDIDATES, df.columns)
    venue_col = _pick_existing(VENUE_CANDIDATES, df.columns)
    volume_col = _pick_existing(VOLUME_CANDIDATES, df.columns)

    if timestamp_col is None or price_col is None:
        raise ValueError(
            f"Could not infer timestamp/price columns in {file_path.name}. "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["timestamp_utc"] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    out["price"] = pd.to_numeric(df[price_col], errors="coerce")

    if symbol_override is not None:
        out["symbol"] = symbol_override
    elif symbol_col is not None:
        out["symbol"] = df[symbol_col].astype(str)
    else:
        raise ValueError(
            "Missing symbol. Provide symbol_override or include a symbol column."
        )

    if venue_override is not None:
        out["venue"] = venue_override
    elif venue_col is not None:
        out["venue"] = df[venue_col].astype(str)

    if volume_col is not None:
        out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")

    return out


def load_market_files(
    files: Iterable[str | Path],
    *,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    tables = [read_market_file(path, column_map=column_map) for path in files]
    if not tables:
        raise ValueError("No input files were provided.")
    return pd.concat(tables, ignore_index=True)


def _remove_glitches(frame: pd.DataFrame, sigma_threshold: float) -> pd.DataFrame:
    if frame.empty:
        return frame

    local = frame.sort_values("timestamp_utc").copy()
    log_ret = np.log(local["price"]).diff()
    if log_ret.notna().sum() < 20:
        return local

    median = np.nanmedian(log_ret)
    mad = np.nanmedian(np.abs(log_ret - median))
    sigma = mad / 0.6745 if mad > 0 else np.nanstd(log_ret)

    if not np.isfinite(sigma) or sigma <= 1e-12:
        return local

    centered_abs = (log_ret - median).abs()
    keep = centered_abs.le(sigma_threshold * sigma) | log_ret.isna()
    return local.loc[keep]


def clean_market_table(raw: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Input table misses required columns: {missing}")

    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)

    df = df.dropna(subset=["timestamp_utc", "symbol", "price"])
    df = df.sort_values(["symbol", "timestamp_utc"])
    df = df.drop_duplicates(
        subset=["symbol", "timestamp_utc"], keep=cfg.keep_duplicate
    )
    cleaned_groups: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=False):
        cleaned_groups.append(_remove_glitches(group, cfg.glitch_sigma_threshold))
    df = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df.iloc[0:0]
    return df


def resample_market_table(clean: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for symbol, group in clean.groupby("symbol", sort=True):
        g = group.sort_values("timestamp_utc").set_index("timestamp_utc")
        resampled = pd.DataFrame(
            {"price": g["price"].resample(cfg.resample_rule).last().ffill(limit=cfg.ffill_limit)}
        )
        if "volume" in g.columns:
            resampled["volume"] = g["volume"].resample(cfg.resample_rule).sum(min_count=1)
        if "venue" in g.columns:
            resampled["venue"] = g["venue"].resample(cfg.resample_rule).last().ffill(
                limit=cfg.ffill_limit
            )

        resampled["symbol"] = symbol
        tables.append(resampled.reset_index())

    if not tables:
        raise ValueError("No symbol available after cleaning.")

    out = pd.concat(tables, ignore_index=True)
    out = out.sort_values(["timestamp_utc", "symbol"]).reset_index(drop=True)
    return out


def to_price_matrix(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Cannot build matrix from empty table.")
    matrix = table.pivot(index="timestamp_utc", columns="symbol", values="price")
    return matrix.sort_index()


def build_processed_prices(raw: pd.DataFrame, cfg: IngestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = clean_market_table(raw, cfg)
    resampled = resample_market_table(clean, cfg)
    matrix = to_price_matrix(resampled)
    return resampled, matrix


def save_processed(resampled: pd.DataFrame, matrix: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_table(resampled, out / "prices_resampled.parquet", index=False)
    _save_table(matrix, out / "prices_matrix.parquet", index=True)


def _save_table(frame: pd.DataFrame, preferred_path: Path, *, index: bool) -> Path:
    try:
        frame.to_parquet(preferred_path, index=index)
        return preferred_path
    except Exception:
        fallback = preferred_path.with_suffix(".csv")
        frame.to_csv(fallback, index=index)
        return fallback


def build_okx_trades_url(day: date) -> str:
    yyyymm = day.strftime("%Y%m")
    day_str = day.strftime("%Y-%m-%d")
    return (
        "https://www.okx.com/cdn/okex/traderecords/trades/monthly/"
        f"{yyyymm}/allfuture-trades-{day_str}.zip"
    )


def build_okx_funding_url(day: date) -> str:
    yyyymm = day.strftime("%Y%m")
    day_str = day.strftime("%Y-%m-%d")
    return (
        "https://www.okx.com/cdn/okex/traderecords/swaprate/monthly/"
        f"{yyyymm}/allswaprate-swaprate-{day_str}.zip"
    )


def download_file(url: str, destination: str | Path, timeout: int = 60) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return dest


def download_okx_range(
    start: date,
    end: date,
    *,
    kind: str,
    output_dir: str | Path,
) -> list[Path]:
    if end < start:
        raise ValueError("end date must be >= start date")
    if kind not in {"trades", "funding"}:
        raise ValueError("kind must be one of {'trades', 'funding'}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    current = start
    while current <= end:
        url = build_okx_trades_url(current) if kind == "trades" else build_okx_funding_url(current)
        filename = Path(url).name
        path = out / filename
        download_file(url, path)
        saved.append(path)
        current += timedelta(days=1)
    return saved


def _parse_column_map(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    mapping: dict[str, str] = {}
    chunks = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError(
                "column-map must follow old:new,old2:new2 format."
            )
        old, new = chunk.split(":", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def _expand_inputs(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        else:
            files.append(pattern)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and resample raw market files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files or glob patterns.")
    parser.add_argument("--output-dir", default="data/processed", help="Destination directory.")
    parser.add_argument("--resample-rule", default="1min", help="Pandas resample rule.")
    parser.add_argument("--ffill-limit", type=int, default=2, help="Max forward-fill intervals.")
    parser.add_argument(
        "--glitch-sigma-threshold",
        type=float,
        default=20.0,
        help="Outlier returns threshold in sigma units.",
    )
    parser.add_argument(
        "--column-map",
        default=None,
        help="Optional rename map old:new,old2:new2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = _expand_inputs(args.inputs)
    column_map = _parse_column_map(args.column_map)
    cfg = IngestConfig(
        resample_rule=args.resample_rule,
        ffill_limit=args.ffill_limit,
        glitch_sigma_threshold=args.glitch_sigma_threshold,
    )
    raw = load_market_files(files, column_map=column_map)
    resampled, matrix = build_processed_prices(raw, cfg)
    save_processed(resampled, matrix, args.output_dir)
    print("Ingestion completed.")
    print(f"- rows_raw: {raw.shape[0]}")
    print(f"- rows_resampled: {resampled.shape[0]}")
    print(f"- matrix_shape: {matrix.shape}")
    print(f"- output_dir: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
