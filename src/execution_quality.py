from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.data_ingest import parse_timestamp_utc


# L2-ready execution scope (recommended default for execution-quality conclusions).
DEFAULT_EPISODES = [
    "march_vol_2024_binance",
    "yen_unwind_2024_binance",
    "yen_followthrough_2024_binance",
]

BOOK_LEVEL_PCTS = (1, 2, 3, 4, 5)

# Broader premium/regime study scope (not all episodes have public historical L2).
PREMIUM_REGIME_EPISODES = [
    "bybit_usdc_depeg_2023",
    "okx_usdc_depeg_2023",
    "march_vol_2024_binance",
    "yen_unwind_2024_binance",
    "yen_followthrough_2024_binance",
]

SLIPPAGE_COLUMNS = [
    "episode",
    "venue",
    "market_type",
    "root",
    "quote",
    "symbol",
    "n_obs",
    "median_volume",
    "median_rel_size",
    "q90_rel_size",
    "baseline_vol_median_bps",
    "impact_all_mean_bps",
    "impact_all_median_bps",
    "impact_large_mean_bps",
    "impact_large_median_bps",
    "impact_all_mean_excess_bps",
    "impact_all_median_excess_bps",
    "impact_large_mean_excess_bps",
    "impact_large_median_excess_bps",
    "impact_all_mean_norm",
    "impact_all_median_norm",
    "impact_large_mean_norm",
    "impact_large_median_norm",
    "large_count",
    "slippage_method",
    "snapshot_match_ratio",
    "insufficient_depth_ratio",
    "book_walk_all_mean_bps",
    "book_walk_all_median_bps",
    "book_walk_large_mean_bps",
    "book_walk_large_median_bps",
    "dnl_all_mean",
    "dnl_all_median",
    "dnl_large_mean",
    "dnl_large_median",
    "queue_load_all_mean",
    "queue_load_all_median",
    "queue_load_large_mean",
    "queue_load_large_median",
]

COMPARISON_COLUMNS = [
    "episode",
    "venue",
    "market_type",
    "root",
    "impact_large_mean_bps_usdc",
    "impact_large_mean_bps_usdt",
    "impact_large_delta_usdc_minus_usdt_bps",
    "impact_large_mean_excess_bps_usdc",
    "impact_large_mean_excess_bps_usdt",
    "impact_large_delta_excess_usdc_minus_usdt_bps",
    "impact_large_mean_norm_usdc",
    "impact_large_mean_norm_usdt",
    "impact_large_delta_norm_usdc_minus_usdt",
    "impact_all_mean_bps_usdc",
    "impact_all_mean_bps_usdt",
    "preferred_quote_on_large_norm",
    "dnl_large_mean_usdc",
    "dnl_large_mean_usdt",
    "dnl_large_delta_usdc_minus_usdt",
    "queue_load_large_mean_usdc",
    "queue_load_large_mean_usdt",
    "queue_load_large_delta_usdc_minus_usdt",
    "snapshot_match_ratio_usdc",
    "snapshot_match_ratio_usdt",
    "slippage_method_usdc",
    "slippage_method_usdt",
]

RESILIENCE_COLUMNS = [
    "episode",
    "venue",
    "market_type",
    "root",
    "quote",
    "symbol",
    "shock_threshold_bps",
    "baseline_abs_ret_bps",
    "n_shocks",
    "n_recovered",
    "unrecovered_ratio",
    "recovery_median_bars",
    "recovery_p90_bars",
]

VENUE_COLUMNS = [
    "venue",
    "market_type",
    "n_root_episode_pairs",
    "mean_delta_large_raw_bps",
    "median_delta_large_raw_bps",
    "mean_delta_large_excess_bps",
    "median_delta_large_excess_bps",
    "mean_delta_large_norm",
    "median_delta_large_norm",
    "n_indeterminate_norm",
    "median_recovery_bars_usdc",
    "median_recovery_bars_usdt",
    "mean_unrecovered_ratio_usdc",
    "mean_unrecovered_ratio_usdt",
]

L2_COVERAGE_COLUMNS = [
    "episode",
    "l2_orderbook_available",
    "tick_trades_available",
    "l2_ready",
    "l2_root",
]

SLIPPAGE_CURVE_COLUMNS = [
    "episode",
    "venue",
    "market_type",
    "root",
    "quote",
    "symbol",
    "size_bin",
    "dnl_bin_low",
    "dnl_bin_high",
    "n_obs",
    "dnl_median",
    "book_walk_mean_bps",
    "book_walk_median_bps",
    "queue_load_mean",
]


def _empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _has_any_path(root: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(root.rglob(pattern)):
            return True
    return False


def build_l2_coverage(episodes: list[str], l2_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_root = l2_root / episode
        has_book = _has_any_path(
            episode_root,
            [
                "*orderbook*level*2*.parquet",
                "*orderbook*level*2*.csv",
                "*orderbook*level*2*.zip",
                "*orderbook*level*2*.csv.gz",
                "*orderbook*level*2*.tar.gz",
                "*orderbook*l2*.parquet",
                "*orderbook*l2*.csv",
                "*orderbook*l2*.zip",
                "*orderbook*l2*.csv.gz",
                "*orderbook*l2*.tar.gz",
                "*book*l2*.parquet",
                "*book*l2*.csv",
                "*book*l2*.zip",
                "*book*l2*.csv.gz",
                "*book*l2*.tar.gz",
                "*L2orderbook*.tar.gz",
                "orderbook*.parquet",
                "orderbook*.csv",
                "orderbook*.zip",
                "orderbook*.csv.gz",
                "orderbook*.tar.gz",
                "depth*.parquet",
                "depth*.csv",
                "depth*.zip",
                "depth*.csv.gz",
                "depth*.tar.gz",
                "*bookDepth*.zip",
                "*bookDepth*.csv",
                "*bookDepth*.csv.gz",
                "*bookDepth*.tar.gz",
            ],
        )
        has_ticks = _has_any_path(
            episode_root,
            [
                "*trade*tick*.parquet",
                "*trade*tick*.csv",
                "*trade*tick*.zip",
                "*trade*tick*.csv.gz",
                "*trades*.parquet",
                "*trades*.csv",
                "*trades*.zip",
                "*trades*.csv.gz",
                "**/trades/*.zip",
                "**/trades/*.csv.gz",
                "**/spot_trades/*.csv.gz",
                "*aggTrade*.parquet",
                "*aggTrade*.csv",
                "*aggTrade*.zip",
                "*aggTrade*.csv.gz",
                "**/aggTrades/*.zip",
                "**/aggTrades/*.csv.gz",
            ],
        )
        rows.append(
            {
                "episode": episode,
                "l2_orderbook_available": bool(has_book),
                "tick_trades_available": bool(has_ticks),
                "l2_ready": bool(has_book and has_ticks),
                "l2_root": str(episode_root),
            }
        )
    if not rows:
        return _empty_table(L2_COVERAGE_COLUMNS)
    return pd.DataFrame(rows, columns=L2_COVERAGE_COLUMNS)


def _normalize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", symbol.upper())


def _parse_symbol(symbol: str) -> tuple[str, str, str] | None:
    normalized = _normalize_symbol(symbol)
    for quote in ("USDC", "USDT"):
        idx = normalized.find(quote)
        if idx > 0:
            root = normalized[:idx]
            suffix = normalized[idx + len(quote) :]
            return root, quote, suffix
    return None


def _market_type_from_suffix(suffix: str | None) -> str:
    s = str(suffix or "").upper()
    if s == "SPOT":
        return "spot"
    if s in {"PERP", "SWAP", "FUTURES", "FUTURE", "PERPETUAL"} or s.isdigit():
        return "derivatives"
    return "unknown"


def _preference_from_delta(delta: float, *, tolerance: float) -> str:
    if not np.isfinite(delta):
        return "indeterminate"
    if delta < -tolerance:
        return "USDC"
    if delta > tolerance:
        return "USDT"
    return "indeterminate"


def _normalize_execution_frame(frame: pd.DataFrame, *, episode: str, source: str) -> pd.DataFrame:
    required = {"timestamp_utc", "price", "volume", "symbol", "venue"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame()

    out = frame.copy()
    out["timestamp_utc"] = parse_timestamp_utc(out["timestamp_utc"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").abs()
    out["symbol"] = out["symbol"].astype(str)
    out["venue"] = out["venue"].astype(str)
    out = out.dropna(subset=["timestamp_utc", "price", "symbol", "venue"]).copy()
    out = out[out["price"] > 0].copy()
    out = out.dropna(subset=["volume"]).copy()
    out["episode"] = episode
    out["execution_source"] = source

    parsed = out["symbol"].map(_parse_symbol)
    out["root"] = parsed.map(lambda x: x[0] if x else None)
    out["quote"] = parsed.map(lambda x: x[1] if x else None)
    out["suffix"] = parsed.map(lambda x: x[2] if x else None)
    out["market_type"] = out["suffix"].map(_market_type_from_suffix)
    out = out[out["quote"].isin(["USDC", "USDT"])].copy()
    out = out[out["market_type"].isin(["spot", "derivatives"])].copy()
    return out


def _infer_symbol_from_filename(name: str) -> str | None:
    patterns = [
        re.compile(r"^(?P<sym>[A-Z0-9]+)-(?:trades|aggTrades)-\d{4}-\d{2}-\d{2}\.zip$"),
        re.compile(r"^(?P<sym>[A-Z0-9]+)-\d{4}-\d{2}\.csv\.gz$"),
        re.compile(r"^(?P<sym>[A-Z0-9]+)\d{4}-\d{2}-\d{2}\.csv\.gz$"),
    ]
    for pattern in patterns:
        match = pattern.match(name)
        if match is not None:
            return str(match.group("sym"))
    return None


def _aggregate_tick_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    local = rows.copy()
    ts_raw = local["timestamp_utc"]
    ts_num = pd.to_numeric(ts_raw, errors="coerce")
    ts = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns, UTC]")
    abs_num = ts_num.abs()
    ns_mask = ts_num.notna() & abs_num.ge(1e17)
    us_mask = ts_num.notna() & abs_num.ge(1e14) & abs_num.lt(1e17)
    ms_mask = ts_num.notna() & abs_num.ge(1e11) & abs_num.lt(1e14)
    sec_mask = ts_num.notna() & abs_num.ge(1e9) & abs_num.lt(1e11)
    if bool(ns_mask.any()):
        ts.loc[ns_mask] = pd.to_datetime(ts_num.loc[ns_mask], unit="ns", utc=True, errors="coerce")
    if bool(us_mask.any()):
        ts.loc[us_mask] = pd.to_datetime(ts_num.loc[us_mask], unit="us", utc=True, errors="coerce")
    if bool(ms_mask.any()):
        ts.loc[ms_mask] = pd.to_datetime(ts_num.loc[ms_mask], unit="ms", utc=True, errors="coerce")
    if bool(sec_mask.any()):
        ts.loc[sec_mask] = pd.to_datetime(ts_num.loc[sec_mask], unit="s", utc=True, errors="coerce")
    other_mask = ts.isna()
    if bool(other_mask.any()):
        ts.loc[other_mask] = parse_timestamp_utc(ts_raw.loc[other_mask])
    local["timestamp_utc"] = ts
    local["timestamp_utc"] = pd.DatetimeIndex(local["timestamp_utc"]).floor("min")
    local["price"] = pd.to_numeric(local["price"], errors="coerce")
    local["volume"] = pd.to_numeric(local["volume"], errors="coerce").abs()
    local = local.dropna(subset=["timestamp_utc", "price", "volume", "symbol", "venue"]).copy()
    local = local[(local["price"] > 0) & (local["volume"] > 0)].copy()
    if local.empty:
        return local
    local["weighted_price"] = local["price"] * local["volume"]
    grouped = local.groupby(["timestamp_utc", "symbol", "venue"], as_index=False).agg(
        weighted_price=("weighted_price", "sum"),
        volume=("volume", "sum"),
    )
    grouped = grouped[grouped["volume"] > 0].copy()
    grouped["price"] = grouped["weighted_price"] / grouped["volume"]
    return grouped[["timestamp_utc", "price", "volume", "symbol", "venue"]]


def _load_binance_tick_file(path: Path) -> pd.DataFrame:
    name = path.name
    symbol = _infer_symbol_from_filename(name)
    if symbol is None:
        return pd.DataFrame()

    if "-aggTrades-" in name:
        usecols = [1, 2, 5]
        colnames = ["price", "volume", "timestamp_utc"]
    elif "-trades-" in name:
        usecols = [1, 2, 4]
        colnames = ["price", "volume", "timestamp_utc"]
    else:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="zip",
        header=None,
        usecols=usecols,
        names=colnames,
        dtype="string",
        chunksize=500_000,
    ):
        part = chunk.copy()
        part["symbol"] = f"{symbol}-PERP"
        part["venue"] = "binance"
        chunks.append(part)

    if not chunks:
        return pd.DataFrame()
    return _aggregate_tick_rows(pd.concat(chunks, ignore_index=True))


def _load_bybit_tick_file(path: Path) -> pd.DataFrame:
    symbol_guess = _infer_symbol_from_filename(path.name)
    if symbol_guess is None:
        return pd.DataFrame()

    header = pd.read_csv(path, compression="gzip", nrows=0)
    cols = {str(col).strip().lower(): str(col) for col in header.columns}
    ts_col = cols.get("timestamp") or cols.get("time")
    price_col = cols.get("price")
    vol_col = cols.get("size") or cols.get("volume") or cols.get("qty")
    sym_col = cols.get("symbol")
    if ts_col is None or price_col is None or vol_col is None:
        return pd.DataFrame()

    usecols = [ts_col, price_col, vol_col] + ([sym_col] if sym_col else [])
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=500_000):
        part = pd.DataFrame()
        part["timestamp_utc"] = chunk[ts_col]
        part["price"] = chunk[price_col]
        part["volume"] = chunk[vol_col]
        if sym_col and sym_col in chunk.columns:
            part["symbol"] = chunk[sym_col].astype(str).str.upper()
        else:
            part["symbol"] = symbol_guess
        # Bybit public spot files map to spot symbols; derivatives map to perpetual symbols.
        if "spot_trades" in path.as_posix():
            part["symbol"] = part["symbol"].astype(str) + "-SPOT"
        else:
            part["symbol"] = part["symbol"].astype(str) + "-PERP"
        part["venue"] = "bybit"
        chunks.append(part)

    if not chunks:
        return pd.DataFrame()
    return _aggregate_tick_rows(pd.concat(chunks, ignore_index=True))


def _load_okx_tick_file(path: Path) -> pd.DataFrame:
    if "trades" not in path.name.lower():
        return pd.DataFrame()

    header = pd.read_csv(path, compression="zip", nrows=0, encoding="utf-8", encoding_errors="ignore")
    cols = {str(col).split("/")[0].strip().lower(): str(col) for col in header.columns}
    ts_col = cols.get("created_time") or cols.get("createdtime") or cols.get("timestamp") or cols.get("ts")
    price_col = cols.get("price")
    vol_col = cols.get("size") or cols.get("qty") or cols.get("volume")
    sym_col = cols.get("instrument_name") or cols.get("instid") or cols.get("symbol")
    if ts_col is None or price_col is None or vol_col is None or sym_col is None:
        return pd.DataFrame()

    usecols = [ts_col, price_col, vol_col, sym_col]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="zip",
        usecols=usecols,
        chunksize=500_000,
        dtype="string",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        part = pd.DataFrame()
        part["timestamp_utc"] = chunk[ts_col]
        part["price"] = chunk[price_col]
        part["volume"] = chunk[vol_col]
        part["symbol"] = chunk[sym_col].astype(str).str.upper()
        part["venue"] = "okx"
        chunks.append(part)

    if not chunks:
        return pd.DataFrame()
    return _aggregate_tick_rows(pd.concat(chunks, ignore_index=True))


def _load_episode_tick_resampled(episode: str, l2_root: Path) -> pd.DataFrame | None:
    episode_root = l2_root / episode
    if not episode_root.exists():
        return None

    tables: list[pd.DataFrame] = []
    for path in sorted(episode_root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        try:
            if name.endswith(".zip") and "okx" in path.as_posix() and "trades" in name.lower():
                parsed = _load_okx_tick_file(path)
            elif name.endswith(".zip") and ("-trades-" in name or "-aggTrades-" in name):
                parsed = _load_binance_tick_file(path)
            elif name.endswith(".csv.gz") and "bybit" in path.as_posix():
                parsed = _load_bybit_tick_file(path)
            else:
                continue
        except Exception:
            continue
        if not parsed.empty:
            tables.append(parsed)

    if not tables:
        return None
    merged = pd.concat(tables, ignore_index=True)
    merged = merged.sort_values(["timestamp_utc", "symbol", "venue"]).drop_duplicates(
        subset=["timestamp_utc", "symbol", "venue"],
        keep="last",
    )
    return _normalize_execution_frame(merged, episode=episode, source="ticks")


def _parse_boolean(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _load_binance_trade_ticks(path: Path) -> pd.DataFrame:
    name = path.name
    match = re.match(r"^(?P<sym>[A-Z0-9]+)-trades-\d{4}-\d{2}-\d{2}\.zip$", name)
    if match is None:
        return pd.DataFrame()
    symbol = str(match.group("sym"))

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="zip",
        header=None,
        usecols=[1, 2, 3, 4, 5],
        names=["price", "volume", "trade_notional", "timestamp_utc", "is_buyer_maker"],
        dtype="string",
        chunksize=500_000,
    ):
        part = chunk.copy()
        part["timestamp_utc"] = pd.to_datetime(
            pd.to_numeric(part["timestamp_utc"], errors="coerce"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
        part["price"] = pd.to_numeric(part["price"], errors="coerce")
        part["volume"] = pd.to_numeric(part["volume"], errors="coerce").abs()
        part["trade_notional"] = pd.to_numeric(part["trade_notional"], errors="coerce").abs()
        maker_flag = _parse_boolean(part["is_buyer_maker"])
        part["aggressor_side"] = np.where(maker_flag, "sell", "buy")
        part["symbol"] = f"{symbol}-PERP"
        part["venue"] = "binance"
        part = part.dropna(subset=["timestamp_utc", "price", "volume"]).copy()
        part = part[(part["price"] > 0) & (part["volume"] > 0)].copy()
        if part.empty:
            continue
        fallback_notional = (part["price"] * part["volume"]).abs()
        part["trade_notional"] = part["trade_notional"].where(part["trade_notional"].gt(0), fallback_notional)
        part = part.dropna(subset=["trade_notional"]).copy()
        part = part[part["trade_notional"] > 0].copy()
        if part.empty:
            continue
        chunks.append(
            part[
                [
                    "timestamp_utc",
                    "price",
                    "volume",
                    "trade_notional",
                    "aggressor_side",
                    "symbol",
                    "venue",
                ]
            ]
        )

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _normalize_trade_ticks_frame(frame: pd.DataFrame, *, episode: str, source: str) -> pd.DataFrame:
    required = ["timestamp_utc", "price", "volume", "symbol", "venue"]
    normalized = _normalize_execution_frame(frame[required], episode=episode, source=source)
    if normalized.empty:
        return normalized
    extras = frame.loc[normalized.index, ["trade_notional", "aggressor_side"]].copy()
    normalized["trade_notional"] = pd.to_numeric(extras["trade_notional"], errors="coerce").abs()
    fallback_notional = (normalized["price"].astype(float) * normalized["volume"].astype(float)).abs()
    normalized["trade_notional"] = normalized["trade_notional"].where(normalized["trade_notional"].gt(0), fallback_notional)
    normalized["aggressor_side"] = extras["aggressor_side"].astype(str).str.lower()
    normalized = normalized[normalized["aggressor_side"].isin(["buy", "sell"])].copy()
    normalized = normalized.dropna(subset=["trade_notional"]).copy()
    normalized = normalized[normalized["trade_notional"] > 0].copy()
    return normalized


def _load_episode_trade_ticks(episode: str, l2_root: Path) -> pd.DataFrame | None:
    episode_root = l2_root / episode
    if not episode_root.exists():
        return None

    tables: list[pd.DataFrame] = []
    for path in sorted(episode_root.rglob("*-trades-*.zip")):
        if not path.is_file():
            continue
        # Snapshot slippage is currently computed where aggressor side is available (Binance trade files).
        if "binance" not in path.as_posix():
            continue
        try:
            parsed = _load_binance_trade_ticks(path)
        except Exception:
            continue
        if not parsed.empty:
            tables.append(parsed)

    if not tables:
        return None
    merged = pd.concat(tables, ignore_index=True)
    merged = merged.sort_values(["timestamp_utc", "symbol", "venue"]).reset_index(drop=True)
    normalized = _normalize_trade_ticks_frame(merged, episode=episode, source="trade_ticks")
    if normalized.empty:
        return None
    return normalized


def _load_binance_bookdepth_snapshot_file(path: Path) -> pd.DataFrame:
    name = path.name
    match = re.match(r"^(?P<sym>[A-Z0-9]+)-bookDepth-\d{4}-\d{2}-\d{2}\.zip$", name)
    if match is None:
        return pd.DataFrame()
    symbol = str(match.group("sym"))

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="zip",
        header=0,
        usecols=[0, 1, 3],
        names=["timestamp_utc", "percentage", "notional"],
        dtype="string",
        chunksize=500_000,
    ):
        part = chunk.copy()
        part["timestamp_utc"] = pd.to_datetime(
            part["timestamp_utc"].astype("string"),
            format="%Y-%m-%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        part["percentage"] = pd.to_numeric(part["percentage"], errors="coerce")
        part["notional"] = pd.to_numeric(part["notional"], errors="coerce").abs()
        part = part.dropna(subset=["timestamp_utc", "percentage", "notional"]).copy()
        part = part[(part["notional"] > 0) & (part["percentage"].abs().isin(list(BOOK_LEVEL_PCTS)))].copy()
        if part.empty:
            continue
        part["symbol"] = f"{symbol}-PERP"
        part["venue"] = "binance"
        chunks.append(part[["timestamp_utc", "symbol", "venue", "percentage", "notional"]])

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["timestamp_utc", "symbol", "venue", "percentage"]).drop_duplicates(
        subset=["timestamp_utc", "symbol", "venue", "percentage"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _build_bookdepth_snapshot_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    local = raw.copy()
    local["side"] = np.where(local["percentage"] > 0, "ask", "bid")
    local["pct_level"] = local["percentage"].abs().astype(int)
    local = local[local["pct_level"].isin(BOOK_LEVEL_PCTS)].copy()
    if local.empty:
        return pd.DataFrame()
    pivot = local.pivot_table(
        index=["timestamp_utc", "symbol", "venue"],
        columns=["side", "pct_level"],
        values="notional",
        aggfunc="last",
    )
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [f"{str(side)}_notional_{int(level)}pct" for side, level in pivot.columns]
    pivot = pivot.sort_index().reset_index()
    for side in ("ask", "bid"):
        side_cols = [f"{side}_notional_{int(level)}pct" for level in BOOK_LEVEL_PCTS]
        for col in side_cols:
            if col not in pivot.columns:
                pivot[col] = np.nan
        side_data = pivot[side_cols].apply(pd.to_numeric, errors="coerce")
        side_data = side_data.where(side_data > 0)
        side_data = side_data.cummax(axis=1)
        pivot[side_cols] = side_data
    return pivot.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def _load_episode_bookdepth_snapshots(episode: str, l2_root: Path) -> pd.DataFrame | None:
    episode_root = l2_root / episode
    if not episode_root.exists():
        return None
    tables: list[pd.DataFrame] = []
    for path in sorted(episode_root.rglob("*-bookDepth-*.zip")):
        if not path.is_file():
            continue
        try:
            parsed = _load_binance_bookdepth_snapshot_file(path)
        except Exception:
            continue
        if not parsed.empty:
            tables.append(parsed)
    if not tables:
        return None
    merged = pd.concat(tables, ignore_index=True)
    table = _build_bookdepth_snapshot_table(merged)
    if table.empty:
        return None
    return table


def _compute_book_walk_bps(
    trade_notional: np.ndarray,
    notional_1: np.ndarray,
    notional_2: np.ndarray,
    notional_3: np.ndarray,
    notional_4: np.ndarray,
    notional_5: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n1 = np.asarray(notional_1, dtype=float)
    n2 = np.asarray(notional_2, dtype=float)
    n3 = np.asarray(notional_3, dtype=float)
    n4 = np.asarray(notional_4, dtype=float)
    n5 = np.asarray(notional_5, dtype=float)
    q = np.asarray(trade_notional, dtype=float)

    walk = np.full(q.shape, np.nan, dtype=float)
    valid = np.isfinite(q) & (q > 0) & np.isfinite(n1) & (n1 > 0)
    if not bool(valid.any()):
        return walk, np.zeros(q.shape, dtype=bool)

    def _segment(mask: np.ndarray, left: np.ndarray, right: np.ndarray, base_bps: float) -> None:
        denom = right - left
        seg = mask & np.isfinite(denom) & (denom > 0)
        if bool(seg.any()):
            walk[seg] = base_bps + 100.0 * (q[seg] - left[seg]) / denom[seg]

    seg1 = valid & (q <= n1)
    if bool(seg1.any()):
        walk[seg1] = 100.0 * q[seg1] / n1[seg1]
    _segment(valid & (q > n1) & (q <= n2), n1, n2, 100.0)
    _segment(valid & (q > n2) & (q <= n3), n2, n3, 200.0)
    _segment(valid & (q > n3) & (q <= n4), n3, n4, 300.0)
    _segment(valid & (q > n4) & (q <= n5), n4, n5, 400.0)
    over = valid & (q > n5)
    _segment(over, n5, n5 + (n5 - n4), 500.0)
    if bool(over.any()):
        fallback = over & np.isnan(walk) & np.isfinite(n5) & (n5 > 0)
        walk[fallback] = 500.0 * q[fallback] / n5[fallback]
    insufficient = valid & (q > n5)
    return walk, insufficient


def build_snapshot_trade_frame(
    trades: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    snapshot_match_tolerance_sec: int = 300,
) -> pd.DataFrame:
    if trades.empty or snapshots.empty:
        return pd.DataFrame()

    tables: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(seconds=max(1, int(snapshot_match_tolerance_sec)))
    for symbol, trade_group in trades.groupby("symbol", sort=False):
        snap_group = snapshots[snapshots["symbol"] == symbol].copy()
        if snap_group.empty:
            unmatched = trade_group.copy()
            unmatched["snapshot_timestamp_utc"] = pd.NaT
            tables.append(unmatched)
            continue
        g_trade = trade_group.copy()
        g_snap = snap_group.copy()
        g_trade["timestamp_utc"] = (
            pd.to_datetime(g_trade["timestamp_utc"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        )
        g_snap["timestamp_utc"] = (
            pd.to_datetime(g_snap["timestamp_utc"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        )
        g_trade = g_trade.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        g_snap = g_snap.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        if g_trade.empty or g_snap.empty:
            continue
        g_snap = g_snap.rename(columns={"timestamp_utc": "snapshot_timestamp_utc"})
        merged = pd.merge_asof(
            g_trade,
            g_snap,
            left_on="timestamp_utc",
            right_on="snapshot_timestamp_utc",
            by=["symbol", "venue"],
            direction="backward",
            tolerance=tolerance,
        )
        tables.append(merged)

    if not tables:
        return pd.DataFrame()
    aligned = pd.concat(tables, ignore_index=True)
    aligned["snapshot_matched"] = aligned["snapshot_timestamp_utc"].notna()

    aligned["trade_second"] = pd.DatetimeIndex(aligned["timestamp_utc"]).floor("s")
    side_flow = (
        aligned.groupby(["episode", "venue", "symbol", "aggressor_side", "trade_second"], as_index=False)["trade_notional"]
        .sum()
        .rename(columns={"trade_notional": "same_second_side_notional"})
    )
    aligned = aligned.merge(
        side_flow,
        on=["episode", "venue", "symbol", "aggressor_side", "trade_second"],
        how="left",
    )
    aligned["same_second_side_notional"] = pd.to_numeric(aligned["same_second_side_notional"], errors="coerce").abs()

    for level in BOOK_LEVEL_PCTS:
        ask_col = f"ask_notional_{int(level)}pct"
        bid_col = f"bid_notional_{int(level)}pct"
        side_col = f"book_notional_{int(level)}pct"
        aligned[side_col] = np.where(
            aligned["aggressor_side"].eq("buy"),
            pd.to_numeric(aligned.get(ask_col), errors="coerce"),
            pd.to_numeric(aligned.get(bid_col), errors="coerce"),
        )

    walk_bps, insufficient = _compute_book_walk_bps(
        trade_notional=pd.to_numeric(aligned["trade_notional"], errors="coerce").to_numpy(dtype=float),
        notional_1=pd.to_numeric(aligned["book_notional_1pct"], errors="coerce").to_numpy(dtype=float),
        notional_2=pd.to_numeric(aligned["book_notional_2pct"], errors="coerce").to_numpy(dtype=float),
        notional_3=pd.to_numeric(aligned["book_notional_3pct"], errors="coerce").to_numpy(dtype=float),
        notional_4=pd.to_numeric(aligned["book_notional_4pct"], errors="coerce").to_numpy(dtype=float),
        notional_5=pd.to_numeric(aligned["book_notional_5pct"], errors="coerce").to_numpy(dtype=float),
    )
    aligned["book_walk_bps"] = walk_bps
    aligned["insufficient_depth"] = insufficient.astype(bool)
    aligned["dnl"] = aligned["trade_notional"] / aligned["book_notional_1pct"]
    aligned["linear_baseline_bps"] = 100.0 * aligned["dnl"]
    aligned["book_walk_excess_bps"] = aligned["book_walk_bps"] - aligned["linear_baseline_bps"]
    aligned["queue_load"] = aligned["same_second_side_notional"] / aligned["book_notional_1pct"]
    aligned["rel_size"] = aligned["dnl"]
    for col in ["book_walk_bps", "dnl", "linear_baseline_bps", "book_walk_excess_bps", "queue_load", "rel_size"]:
        aligned.loc[~np.isfinite(pd.to_numeric(aligned[col], errors="coerce")), col] = np.nan
    aligned = aligned.sort_values(["episode", "venue", "symbol", "timestamp_utc"]).reset_index(drop=True)
    return aligned


def build_snapshot_slippage(aligned: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty:
        return _empty_table(SLIPPAGE_COLUMNS)
    rows: list[dict[str, Any]] = []
    keys = ["episode", "venue", "market_type", "root", "quote", "symbol"]
    for key, group in aligned.groupby(keys, sort=False):
        total_obs = int(group.shape[0])
        valid = group[
            group["snapshot_matched"].astype(bool)
            & pd.to_numeric(group["book_walk_bps"], errors="coerce").notna()
            & pd.to_numeric(group["rel_size"], errors="coerce").notna()
        ].copy()
        if valid.empty:
            continue
        q90 = float(valid["rel_size"].quantile(0.90))
        large = valid["rel_size"] >= q90
        large_count = int(large.sum())
        rows.append(
            {
                "episode": key[0],
                "venue": key[1],
                "market_type": key[2],
                "root": key[3],
                "quote": key[4],
                "symbol": key[5],
                "n_obs": int(valid.shape[0]),
                "median_volume": float(pd.to_numeric(valid["volume"], errors="coerce").median()),
                "median_rel_size": float(pd.to_numeric(valid["rel_size"], errors="coerce").median()),
                "q90_rel_size": q90,
                "baseline_vol_median_bps": float(pd.to_numeric(valid["linear_baseline_bps"], errors="coerce").median()),
                "impact_all_mean_bps": float(pd.to_numeric(valid["book_walk_bps"], errors="coerce").mean()),
                "impact_all_median_bps": float(pd.to_numeric(valid["book_walk_bps"], errors="coerce").median()),
                "impact_large_mean_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_bps"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "impact_large_median_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_bps"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "impact_all_mean_excess_bps": float(pd.to_numeric(valid["book_walk_excess_bps"], errors="coerce").mean()),
                "impact_all_median_excess_bps": float(
                    pd.to_numeric(valid["book_walk_excess_bps"], errors="coerce").median()
                ),
                "impact_large_mean_excess_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_excess_bps"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "impact_large_median_excess_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_excess_bps"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "impact_all_mean_norm": float(pd.to_numeric(valid["dnl"], errors="coerce").mean()),
                "impact_all_median_norm": float(pd.to_numeric(valid["dnl"], errors="coerce").median()),
                "impact_large_mean_norm": (
                    float(pd.to_numeric(valid.loc[large, "dnl"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "impact_large_median_norm": (
                    float(pd.to_numeric(valid.loc[large, "dnl"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "large_count": large_count,
                "slippage_method": "orderbook_snapshot_bookwalk",
                "snapshot_match_ratio": float(valid.shape[0] / total_obs) if total_obs > 0 else np.nan,
                "insufficient_depth_ratio": float(valid["insufficient_depth"].astype(bool).mean()),
                "book_walk_all_mean_bps": float(pd.to_numeric(valid["book_walk_bps"], errors="coerce").mean()),
                "book_walk_all_median_bps": float(pd.to_numeric(valid["book_walk_bps"], errors="coerce").median()),
                "book_walk_large_mean_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_bps"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "book_walk_large_median_bps": (
                    float(pd.to_numeric(valid.loc[large, "book_walk_bps"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "dnl_all_mean": float(pd.to_numeric(valid["dnl"], errors="coerce").mean()),
                "dnl_all_median": float(pd.to_numeric(valid["dnl"], errors="coerce").median()),
                "dnl_large_mean": (
                    float(pd.to_numeric(valid.loc[large, "dnl"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "dnl_large_median": (
                    float(pd.to_numeric(valid.loc[large, "dnl"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "queue_load_all_mean": float(pd.to_numeric(valid["queue_load"], errors="coerce").mean()),
                "queue_load_all_median": float(pd.to_numeric(valid["queue_load"], errors="coerce").median()),
                "queue_load_large_mean": (
                    float(pd.to_numeric(valid.loc[large, "queue_load"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "queue_load_large_median": (
                    float(pd.to_numeric(valid.loc[large, "queue_load"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
            }
        )
    if not rows:
        return _empty_table(SLIPPAGE_COLUMNS)
    out = pd.DataFrame(rows)
    for col in SLIPPAGE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SLIPPAGE_COLUMNS].sort_values(["episode", "venue", "market_type", "root", "quote"]).reset_index(drop=True)


def build_slippage_vs_size(aligned: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    if aligned.empty:
        return _empty_table(SLIPPAGE_CURVE_COLUMNS)
    rows: list[dict[str, Any]] = []
    keys = ["episode", "venue", "market_type", "root", "quote", "symbol"]
    for key, group in aligned.groupby(keys, sort=False):
        valid = group[
            group["snapshot_matched"].astype(bool)
            & pd.to_numeric(group["dnl"], errors="coerce").notna()
            & pd.to_numeric(group["book_walk_bps"], errors="coerce").notna()
        ].copy()
        if valid.shape[0] < 20:
            continue
        bins = min(max(2, int(n_bins)), int(valid["dnl"].nunique()))
        if bins < 2:
            continue
        try:
            valid["size_bin"] = pd.qcut(valid["dnl"], q=bins, duplicates="drop")
        except Exception:
            continue
        grouped = (
            valid.groupby("size_bin", observed=False, as_index=False)
            .agg(
                n_obs=("dnl", "count"),
                dnl_median=("dnl", "median"),
                book_walk_mean_bps=("book_walk_bps", "mean"),
                book_walk_median_bps=("book_walk_bps", "median"),
                queue_load_mean=("queue_load", "mean"),
            )
            .sort_values("size_bin")
            .reset_index(drop=True)
        )
        for _, row in grouped.iterrows():
            interval = row["size_bin"]
            rows.append(
                {
                    "episode": key[0],
                    "venue": key[1],
                    "market_type": key[2],
                    "root": key[3],
                    "quote": key[4],
                    "symbol": key[5],
                    "size_bin": str(interval),
                    "dnl_bin_low": float(interval.left) if hasattr(interval, "left") else np.nan,
                    "dnl_bin_high": float(interval.right) if hasattr(interval, "right") else np.nan,
                    "n_obs": int(row["n_obs"]),
                    "dnl_median": float(row["dnl_median"]),
                    "book_walk_mean_bps": float(row["book_walk_mean_bps"]),
                    "book_walk_median_bps": float(row["book_walk_median_bps"]),
                    "queue_load_mean": float(row["queue_load_mean"]),
                }
            )
    if not rows:
        return _empty_table(SLIPPAGE_CURVE_COLUMNS)
    return pd.DataFrame(rows, columns=SLIPPAGE_CURVE_COLUMNS).sort_values(
        ["episode", "venue", "market_type", "root", "quote", "dnl_bin_low", "dnl_bin_high"]
    ).reset_index(drop=True)


def _load_binance_bookdepth_file(path: Path) -> pd.DataFrame:
    name = path.name
    match = re.match(r"^(?P<sym>[A-Z0-9]+)-bookDepth-\d{4}-\d{2}-\d{2}\.zip$", name)
    if match is None:
        return pd.DataFrame()
    symbol = str(match.group("sym"))

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="zip",
        header=None,
        usecols=[0, 1, 3],
        names=["timestamp_utc", "percentage", "notional"],
        dtype="string",
        chunksize=500_000,
    ):
        part = chunk.copy()
        part["timestamp_utc"] = pd.to_datetime(
            part["timestamp_utc"].astype("string"),
            format="%Y-%m-%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        part["timestamp_utc"] = pd.DatetimeIndex(part["timestamp_utc"]).floor("min")
        part["percentage"] = pd.to_numeric(part["percentage"], errors="coerce")
        part["notional"] = pd.to_numeric(part["notional"], errors="coerce").abs()
        part = part.dropna(subset=["timestamp_utc", "percentage", "notional"]).copy()
        part = part[(part["notional"] > 0) & (part["percentage"].abs().le(1.0))].copy()
        if part.empty:
            continue
        part["symbol"] = f"{symbol}-PERP"
        part["venue"] = "binance"
        chunks.append(part)

    if not chunks:
        return pd.DataFrame()
    full = pd.concat(chunks, ignore_index=True)
    grouped = full.groupby(["timestamp_utc", "symbol", "venue"], as_index=False).agg(
        depth_notional_1pct=("notional", "sum")
    )
    return grouped


def _load_episode_bookdepth_minute(episode: str, l2_root: Path) -> pd.DataFrame | None:
    episode_root = l2_root / episode
    if not episode_root.exists():
        return None
    tables: list[pd.DataFrame] = []
    for path in sorted(episode_root.rglob("*-bookDepth-*.zip")):
        if not path.is_file():
            continue
        try:
            table = _load_binance_bookdepth_file(path)
        except Exception:
            continue
        if not table.empty:
            tables.append(table)
    if not tables:
        return None
    merged = pd.concat(tables, ignore_index=True)
    merged = merged.sort_values(["timestamp_utc", "symbol", "venue"]).drop_duplicates(
        subset=["timestamp_utc", "symbol", "venue"],
        keep="last",
    )
    return merged


def _load_episode_resampled(episode: str, processed_root: Path) -> pd.DataFrame | None:
    path = processed_root / episode / "prices_resampled.csv"
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    out = _normalize_execution_frame(frame, episode=episode, source="bars")
    if out.empty:
        return None
    return out


def build_enriched_trade_frame(
    raw: pd.DataFrame,
    *,
    size_window: int = 60,
    min_size_periods: int = 20,
    vol_window: int | None = None,
    min_vol_periods: int | None = None,
    norm_floor_bps: float = 1.0,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    vol_window_eff = max(2, int(size_window if vol_window is None else vol_window))
    min_vol_periods_eff = max(2, int(min_size_periods if min_vol_periods is None else min_vol_periods))
    group_cols = ["episode", "venue", "market_type", "symbol"]
    for _, group in raw.groupby(group_cols, sort=False):
        g = group.sort_values("timestamp_utc").copy()
        g["log_price"] = np.log(g["price"].astype(float).clip(lower=1e-12))
        g["ret_bps"] = g["log_price"].diff() * 1e4
        g["abs_ret_bps"] = g["ret_bps"].abs()
        g["fwd_abs_ret_bps"] = g["abs_ret_bps"].shift(-1)
        # Robust local volatility baseline to reduce volatility/impact confounding.
        rolling_absret_med = (
            g["abs_ret_bps"]
            .astype(float)
            .rolling(window=vol_window_eff, min_periods=min_vol_periods_eff)
            .median()
            .shift(1)
        )
        g["rolling_median_abs_ret_bps"] = rolling_absret_med
        g["fwd_abs_ret_excess_bps"] = g["fwd_abs_ret_bps"] - rolling_absret_med
        denom_floor = max(1e-6, float(norm_floor_bps))
        denom = rolling_absret_med.clip(lower=denom_floor)
        g["fwd_abs_ret_norm"] = g["fwd_abs_ret_bps"] / denom

        rolling_med_vol = (
            g["volume"]
            .astype(float)
            .rolling(window=max(2, int(size_window)), min_periods=max(2, int(min_size_periods)))
            .median()
            .shift(1)
        )
        g["rolling_median_volume"] = rolling_med_vol
        g["rel_size"] = g["volume"] / rolling_med_vol
        if "depth_notional_1pct" in g.columns:
            depth = pd.to_numeric(g["depth_notional_1pct"], errors="coerce")
            trade_notional = (g["price"].astype(float) * g["volume"].astype(float)).abs()
            g["rel_size_depth"] = trade_notional / depth
            g.loc[depth.le(0), "rel_size_depth"] = np.nan
            g["rel_size"] = g["rel_size_depth"].where(g["rel_size_depth"].notna(), g["rel_size"])
        g.loc[~np.isfinite(g["rel_size"]), "rel_size"] = np.nan
        rows.append(g)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["episode", "venue", "symbol", "timestamp_utc"]).reset_index(drop=True)
    return out


def build_slippage_proxy(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["episode", "venue", "market_type", "root", "quote", "symbol"]
    for key, group in enriched.groupby(keys, sort=False):
        g = group.copy()
        valid = g["fwd_abs_ret_bps"].notna() & g["rel_size"].notna() & g["volume"].notna()
        g = g.loc[valid]
        if g.empty:
            continue

        q90 = float(g["rel_size"].quantile(0.90))
        large = g["rel_size"] >= q90
        large_count = int(large.sum())

        rows.append(
            {
                "episode": key[0],
                "venue": key[1],
                "market_type": key[2],
                "root": key[3],
                "quote": key[4],
                "symbol": key[5],
                "n_obs": int(g.shape[0]),
                "median_volume": float(g["volume"].median()),
                "median_rel_size": float(g["rel_size"].median()),
                "q90_rel_size": q90,
                "baseline_vol_median_bps": float(pd.to_numeric(g["rolling_median_abs_ret_bps"], errors="coerce").median()),
                "impact_all_mean_bps": float(g["fwd_abs_ret_bps"].mean()),
                "impact_all_median_bps": float(g["fwd_abs_ret_bps"].median()),
                "impact_large_mean_bps": float(g.loc[large, "fwd_abs_ret_bps"].mean()) if large_count > 0 else np.nan,
                "impact_large_median_bps": float(g.loc[large, "fwd_abs_ret_bps"].median()) if large_count > 0 else np.nan,
                "impact_all_mean_excess_bps": float(pd.to_numeric(g["fwd_abs_ret_excess_bps"], errors="coerce").mean()),
                "impact_all_median_excess_bps": float(pd.to_numeric(g["fwd_abs_ret_excess_bps"], errors="coerce").median()),
                "impact_large_mean_excess_bps": (
                    float(pd.to_numeric(g.loc[large, "fwd_abs_ret_excess_bps"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "impact_large_median_excess_bps": (
                    float(pd.to_numeric(g.loc[large, "fwd_abs_ret_excess_bps"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "impact_all_mean_norm": float(pd.to_numeric(g["fwd_abs_ret_norm"], errors="coerce").mean()),
                "impact_all_median_norm": float(pd.to_numeric(g["fwd_abs_ret_norm"], errors="coerce").median()),
                "impact_large_mean_norm": (
                    float(pd.to_numeric(g.loc[large, "fwd_abs_ret_norm"], errors="coerce").mean())
                    if large_count > 0
                    else np.nan
                ),
                "impact_large_median_norm": (
                    float(pd.to_numeric(g.loc[large, "fwd_abs_ret_norm"], errors="coerce").median())
                    if large_count > 0
                    else np.nan
                ),
                "large_count": large_count,
                "slippage_method": "bar_proxy_next_bar_abs_return",
            }
        )

    if not rows:
        return _empty_table(SLIPPAGE_COLUMNS)
    out = pd.DataFrame(rows)
    for col in SLIPPAGE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SLIPPAGE_COLUMNS].sort_values(["episode", "venue", "market_type", "root", "quote"]).reset_index(
        drop=True
    )


def build_cross_quote_comparison(slippage: pd.DataFrame, *, norm_delta_tolerance: float = 0.05) -> pd.DataFrame:
    if slippage.empty:
        return _empty_table(COMPARISON_COLUMNS)

    def _as_float(row: pd.Series, column: str) -> float:
        if column not in row.index:
            return float("nan")
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else float("nan")

    rows: list[dict[str, Any]] = []
    for (episode, venue, market_type, root), group in slippage.groupby(
        ["episode", "venue", "market_type", "root"], sort=False
    ):
        by_quote = {quote: q for quote, q in group.groupby("quote")}
        if "USDC" not in by_quote or "USDT" not in by_quote:
            continue

        usdc = by_quote["USDC"].iloc[0]
        usdt = by_quote["USDT"].iloc[0]
        usdc_large_raw = _as_float(usdc, "impact_large_mean_bps")
        usdt_large_raw = _as_float(usdt, "impact_large_mean_bps")
        usdc_large_excess = _as_float(usdc, "impact_large_mean_excess_bps")
        usdt_large_excess = _as_float(usdt, "impact_large_mean_excess_bps")
        usdc_large_norm = _as_float(usdc, "impact_large_mean_norm")
        usdt_large_norm = _as_float(usdt, "impact_large_mean_norm")
        delta_raw = float(usdc_large_raw - usdt_large_raw)
        delta_excess = float(usdc_large_excess - usdt_large_excess)
        delta_norm = float(usdc_large_norm - usdt_large_norm)
        preferred_norm = _preference_from_delta(delta_norm, tolerance=norm_delta_tolerance)
        usdc_dnl = _as_float(usdc, "dnl_large_mean")
        usdt_dnl = _as_float(usdt, "dnl_large_mean")
        usdc_queue = _as_float(usdc, "queue_load_large_mean")
        usdt_queue = _as_float(usdt, "queue_load_large_mean")
        usdc_snapshot_ratio = _as_float(usdc, "snapshot_match_ratio")
        usdt_snapshot_ratio = _as_float(usdt, "snapshot_match_ratio")

        rows.append(
            {
                "episode": episode,
                "venue": venue,
                "market_type": market_type,
                "root": root,
                "impact_large_mean_bps_usdc": usdc_large_raw,
                "impact_large_mean_bps_usdt": usdt_large_raw,
                "impact_large_delta_usdc_minus_usdt_bps": delta_raw,
                "impact_large_mean_excess_bps_usdc": usdc_large_excess,
                "impact_large_mean_excess_bps_usdt": usdt_large_excess,
                "impact_large_delta_excess_usdc_minus_usdt_bps": delta_excess,
                "impact_large_mean_norm_usdc": usdc_large_norm,
                "impact_large_mean_norm_usdt": usdt_large_norm,
                "impact_large_delta_norm_usdc_minus_usdt": delta_norm,
                "impact_all_mean_bps_usdc": _as_float(usdc, "impact_all_mean_bps"),
                "impact_all_mean_bps_usdt": _as_float(usdt, "impact_all_mean_bps"),
                "preferred_quote_on_large_norm": preferred_norm,
                "dnl_large_mean_usdc": usdc_dnl,
                "dnl_large_mean_usdt": usdt_dnl,
                "dnl_large_delta_usdc_minus_usdt": float(usdc_dnl - usdt_dnl),
                "queue_load_large_mean_usdc": usdc_queue,
                "queue_load_large_mean_usdt": usdt_queue,
                "queue_load_large_delta_usdc_minus_usdt": float(usdc_queue - usdt_queue),
                "snapshot_match_ratio_usdc": usdc_snapshot_ratio,
                "snapshot_match_ratio_usdt": usdt_snapshot_ratio,
                "slippage_method_usdc": str(usdc.get("slippage_method", "")),
                "slippage_method_usdt": str(usdt.get("slippage_method", "")),
            }
        )

    if not rows:
        return _empty_table(COMPARISON_COLUMNS)
    out = pd.DataFrame(rows)
    for col in COMPARISON_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[COMPARISON_COLUMNS].sort_values(["episode", "venue", "market_type", "root"]).reset_index(drop=True)


def build_resilience_table(
    enriched: pd.DataFrame,
    *,
    horizon_bars: int = 60,
    shock_quantile: float = 0.99,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["episode", "venue", "market_type", "root", "quote", "symbol"]
    for key, group in enriched.groupby(keys, sort=False):
        g = group.sort_values("timestamp_utc").reset_index(drop=True)
        ret = g["abs_ret_bps"].astype(float)
        if ret.notna().sum() < 50:
            continue

        threshold = float(ret.quantile(shock_quantile))
        baseline = float(ret.median())
        shock_idx = ret[ret >= threshold].index.tolist()
        if not shock_idx:
            continue

        recoveries: list[float] = []
        unrecovered = 0
        for i in shock_idx:
            start = i + 1
            end = min(i + 1 + max(1, int(horizon_bars)), len(g))
            if start >= end:
                continue
            future = ret.iloc[start:end]
            recovered = future[future <= baseline]
            if recovered.empty:
                unrecovered += 1
                continue
            recovery_bars = float(int(recovered.index[0] - i))
            recoveries.append(recovery_bars)

        rows.append(
            {
                "episode": key[0],
                "venue": key[1],
                "market_type": key[2],
                "root": key[3],
                "quote": key[4],
                "symbol": key[5],
                "shock_threshold_bps": threshold,
                "baseline_abs_ret_bps": baseline,
                "n_shocks": int(len(shock_idx)),
                "n_recovered": int(len(recoveries)),
                "unrecovered_ratio": float(unrecovered / len(shock_idx)),
                "recovery_median_bars": float(np.median(recoveries)) if recoveries else np.nan,
                "recovery_p90_bars": float(np.quantile(recoveries, 0.9)) if recoveries else np.nan,
            }
        )

    if not rows:
        return _empty_table(RESILIENCE_COLUMNS)
    return pd.DataFrame(rows).sort_values(["episode", "venue", "market_type", "root", "quote"]).reset_index(
        drop=True
    )


def build_venue_summary(comparison: pd.DataFrame, resilience: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return _empty_table(VENUE_COLUMNS)

    cmp_agg = (
        comparison.groupby(["venue", "market_type"], as_index=False)
        .agg(
            n_root_episode_pairs=("root", "count"),
            mean_delta_large_raw_bps=("impact_large_delta_usdc_minus_usdt_bps", "mean"),
            median_delta_large_raw_bps=("impact_large_delta_usdc_minus_usdt_bps", "median"),
            mean_delta_large_excess_bps=("impact_large_delta_excess_usdc_minus_usdt_bps", "mean"),
            median_delta_large_excess_bps=("impact_large_delta_excess_usdc_minus_usdt_bps", "median"),
            mean_delta_large_norm=("impact_large_delta_norm_usdc_minus_usdt", "mean"),
            median_delta_large_norm=("impact_large_delta_norm_usdc_minus_usdt", "median"),
            n_indeterminate_norm=("preferred_quote_on_large_norm", lambda s: int((s == "indeterminate").sum())),
        )
        .sort_values(["venue", "market_type"])
        .reset_index(drop=True)
    )

    if resilience.empty:
        return cmp_agg

    res_agg = (
        resilience.groupby(["venue", "market_type", "quote"], as_index=False)
        .agg(
            median_recovery_bars=("recovery_median_bars", "median"),
            mean_unrecovered_ratio=("unrecovered_ratio", "mean"),
        )
    )
    res_pivot = res_agg.pivot(index=["venue", "market_type"], columns="quote")
    res_pivot.columns = [f"{metric}_{quote.lower()}" for metric, quote in res_pivot.columns]
    res_pivot = res_pivot.reset_index()

    merged = cmp_agg.merge(res_pivot, on=["venue", "market_type"], how="left")
    return merged.sort_values(["venue", "market_type"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build execution/slippage/resilience diagnostics from episode resampled data."
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=DEFAULT_EPISODES,
        help="Episode ids to include.",
    )
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Processed episodes root.")
    parser.add_argument(
        "--l2-root",
        default="data/processed/orderbook",
        help="Root folder containing per-episode L2 orderbook and tick-trade files.",
    )
    parser.add_argument("--output-dir", default="reports/final", help="Output folder for execution diagnostics.")
    parser.add_argument(
        "--allow-bar-proxy-without-l2",
        action="store_true",
        help=(
            "Compatibility switch. Lightweight bar-proxy diagnostics are now enabled by default "
            "unless --strict-l2-required is set."
        ),
    )
    parser.add_argument(
        "--strict-l2-required",
        action="store_true",
        help="Fail closed when L2/tick coverage is incomplete.",
    )
    parser.add_argument("--size-window", type=int, default=60, help="Rolling window for relative size proxy.")
    parser.add_argument("--min-size-periods", type=int, default=20, help="Min periods for rolling size baseline.")
    parser.add_argument(
        "--norm-floor-bps",
        type=float,
        default=1.0,
        help="Lower bound in bps for volatility normalization denominator.",
    )
    parser.add_argument(
        "--norm-delta-tolerance",
        type=float,
        default=0.05,
        help="Indifference band for normalized USDC-USDT large-size delta classification.",
    )
    parser.add_argument(
        "--snapshot-match-tolerance-sec",
        type=int,
        default=300,
        help="Max backward gap (seconds) when matching trades to orderbook snapshots.",
    )
    parser.add_argument(
        "--size-bins",
        type=int,
        default=10,
        help="Quantile bins for slippage-vs-size summary table.",
    )
    parser.add_argument("--resilience-horizon", type=int, default=60, help="Forward bars to look for recovery.")
    parser.add_argument("--shock-quantile", type=float, default=0.99, help="Shock threshold quantile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    l2_root = Path(args.l2_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slippage_path = out_dir / "execution_slippage_proxy.csv"
    slippage_curve_path = out_dir / "execution_slippage_vs_size.csv"
    comparison_path = out_dir / "execution_cross_quote_comparison.csv"
    resilience_path = out_dir / "execution_resilience.csv"
    venue_path = out_dir / "execution_venue_comparison.csv"
    coverage_path = out_dir / "execution_l2_coverage.csv"
    report_path = out_dir / "execution_quality_report.md"

    l2_coverage = build_l2_coverage(list(args.episodes), l2_root)
    l2_coverage.to_csv(coverage_path, index=False)
    missing_l2 = l2_coverage[~l2_coverage["l2_ready"].astype(bool)].copy()
    ready_l2 = l2_coverage[l2_coverage["l2_ready"].astype(bool)].copy()

    selected_episodes = list(args.episodes)
    partial_l2_mode = False
    allow_bar_proxy_without_l2 = bool(args.allow_bar_proxy_without_l2) or (not bool(args.strict_l2_required))

    if (not allow_bar_proxy_without_l2) and (not missing_l2.empty):
        if not ready_l2.empty:
            selected_episodes = ready_l2["episode"].astype(str).tolist()
            partial_l2_mode = True
        else:
            _empty_table(SLIPPAGE_COLUMNS).to_csv(slippage_path, index=False)
            _empty_table(SLIPPAGE_CURVE_COLUMNS).to_csv(slippage_curve_path, index=False)
            _empty_table(COMPARISON_COLUMNS).to_csv(comparison_path, index=False)
            _empty_table(RESILIENCE_COLUMNS).to_csv(resilience_path, index=False)
            _empty_table(VENUE_COLUMNS).to_csv(venue_path, index=False)

            with report_path.open("w", encoding="utf-8") as handle:
                handle.write("# Execution Diagnostics Blocked (L2 Missing)\n\n")
                handle.write(
                    "Execution-quality conclusions are disabled because required L2 orderbook + tick-trade "
                    "coverage is incomplete for selected episodes.\n\n"
                )
                handle.write("## L2 Coverage\n\n")
                handle.write("```text\n")
                handle.write(l2_coverage.to_string(index=False))
                handle.write("\n```\n")
                handle.write(
                    "\nNo bar-proxy fallback was produced because `--strict-l2-required` was set.\n"
                )
                handle.write(
                    "This fail-closed behavior prevents unsupported venue/quote liquidity rankings.\n"
                )
                handle.write("\n## Artifacts\n\n")
                handle.write(f"- `{coverage_path}`\n")
                handle.write(f"- `{slippage_path}`\n")
                handle.write(f"- `{slippage_curve_path}`\n")
                handle.write(f"- `{comparison_path}`\n")
                handle.write(f"- `{resilience_path}`\n")
                handle.write(f"- `{venue_path}`\n")

            print("Execution diagnostics blocked: missing L2/tick inputs for all selected episodes.")
            print(f"- l2_coverage: {coverage_path}")
            print(f"- report: {report_path}")
            return

    frames: list[pd.DataFrame] = []
    snapshot_trade_tables: list[pd.DataFrame] = []
    missing: list[str] = []
    for episode in selected_episodes:
        bars = _load_episode_resampled(episode, processed_root)
        ticks = _load_episode_tick_resampled(episode, l2_root)
        depth = _load_episode_bookdepth_minute(episode, l2_root)
        tick_trades = _load_episode_trade_ticks(episode, l2_root)
        snapshots = _load_episode_bookdepth_snapshots(episode, l2_root)

        if tick_trades is not None and snapshots is not None and (not tick_trades.empty) and (not snapshots.empty):
            aligned = build_snapshot_trade_frame(
                tick_trades,
                snapshots,
                snapshot_match_tolerance_sec=args.snapshot_match_tolerance_sec,
            )
            if not aligned.empty:
                snapshot_trade_tables.append(aligned)

        has_bars = bars is not None and (not bars.empty)
        has_ticks = ticks is not None and (not ticks.empty)
        if (not has_bars) and (not has_ticks):
            missing.append(episode)
            continue

        if has_bars and has_ticks:
            merged = pd.concat([bars, ticks], ignore_index=True)
            merged = merged.sort_values(["timestamp_utc", "symbol", "venue", "execution_source"]).drop_duplicates(
                subset=["timestamp_utc", "symbol", "venue", "episode"],
                keep="last",
            )
            frame = merged
        elif has_ticks:
            frame = ticks
        else:
            frame = bars

        if depth is not None and not depth.empty:
            frame = frame.merge(depth, on=["timestamp_utc", "symbol", "venue"], how="left")
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No valid episode resampled files were found for execution diagnostics. selected={selected_episodes}"
        )

    raw = pd.concat(frames, ignore_index=True)
    enriched = build_enriched_trade_frame(
        raw,
        size_window=args.size_window,
        min_size_periods=args.min_size_periods,
        norm_floor_bps=args.norm_floor_bps,
    )
    proxy_slippage = build_slippage_proxy(enriched)
    snapshot_trade_frame = (
        pd.concat(snapshot_trade_tables, ignore_index=True) if snapshot_trade_tables else pd.DataFrame()
    )
    snapshot_slippage = build_snapshot_slippage(snapshot_trade_frame)
    slippage_vs_size = build_slippage_vs_size(snapshot_trade_frame, n_bins=args.size_bins)

    if snapshot_slippage.empty:
        slippage = proxy_slippage
    elif proxy_slippage.empty:
        slippage = snapshot_slippage
    else:
        key_cols = ["episode", "venue", "market_type", "root", "quote", "symbol"]
        snap_keys = snapshot_slippage[key_cols].drop_duplicates()
        proxy_only = proxy_slippage.merge(snap_keys, on=key_cols, how="left", indicator=True)
        proxy_only = proxy_only[proxy_only["_merge"] == "left_only"].drop(columns=["_merge"])
        slippage = pd.concat([snapshot_slippage, proxy_only], ignore_index=True)
    slippage = slippage.sort_values(["episode", "venue", "market_type", "root", "quote"]).reset_index(drop=True)

    comparison = build_cross_quote_comparison(
        slippage,
        norm_delta_tolerance=args.norm_delta_tolerance,
    )
    resilience = build_resilience_table(
        enriched,
        horizon_bars=args.resilience_horizon,
        shock_quantile=args.shock_quantile,
    )
    venue_summary = build_venue_summary(comparison, resilience)

    slippage.to_csv(slippage_path, index=False)
    slippage_vs_size.to_csv(slippage_curve_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    resilience.to_csv(resilience_path, index=False)
    venue_summary.to_csv(venue_path, index=False)

    snapshot_rows = int((slippage["slippage_method"].astype(str) == "orderbook_snapshot_bookwalk").sum())
    bar_rows = int((slippage["slippage_method"].astype(str) == "bar_proxy_next_bar_abs_return").sum())
    used_snapshot_mode = snapshot_rows > 0

    with report_path.open("w", encoding="utf-8") as handle:
        if used_snapshot_mode:
            handle.write("# Execution Diagnostics (Order-Book Snapshot)\n\n")
            handle.write("This report computes slippage-vs-size from L2 snapshot book-walk metrics.\n\n")
        else:
            handle.write("# Execution Proxy Diagnostics (Bar-Level Fallback)\n\n")
            handle.write(
                "No usable trade-to-snapshot alignment was available, so diagnostics use bar-level proxy metrics.\n\n"
            )
        if partial_l2_mode:
            handle.write(
                "Partial-L2 mode: only episodes with both orderbook+tick coverage were included "
                "because `--strict-l2-required` was set.\n\n"
            )
        handle.write("## Scope\n\n")
        for ep in selected_episodes:
            handle.write(f"- `{ep}`\n")
        skipped_l2 = sorted(set(args.episodes) - set(selected_episodes))
        if skipped_l2:
            handle.write("\nSkipped due to missing L2 coverage:\n")
            for ep in skipped_l2:
                handle.write(f"- `{ep}`\n")
        if missing:
            handle.write("\nMissing episodes (skipped):\n")
            for ep in missing:
                handle.write(f"- `{ep}`\n")

        handle.write("\n## L2 Coverage\n\n")
        handle.write("```text\n")
        handle.write(l2_coverage.to_string(index=False))
        handle.write("\n```\n")
        if used_snapshot_mode:
            handle.write(
                "\nSnapshot coverage note: rows with `slippage_method=orderbook_snapshot_bookwalk` use true "
                "trade-to-orderbook matching; remaining rows are explicit bar-proxy fallback.\n"
            )
        else:
            handle.write("\nL2 coverage is provided for transparency. Diagnostics below are bar-level fallback only.\n")

        handle.write("\n## Method Notes\n\n")
        if used_snapshot_mode:
            handle.write(
                "- Trade source: Binance tick trades (`*-trades-*.zip`) with aggressor side from `is_buyer_maker`.\n"
            )
            handle.write(
                "- Snapshot source: Binance `bookDepth` snapshots (`*-bookDepth-*.zip`) on `±1..±5%` depth levels.\n"
            )
            handle.write(
                "- Matching: each trade is aligned to the latest snapshot at or before trade time "
                f"within `{int(args.snapshot_match_tolerance_sec)}` seconds.\n"
            )
            handle.write(
                "- Book-walk slippage (`book_walk_bps`): piecewise interpolation over cumulative notional "
                "levels (1% to 5%) on the consumed side (asks for buy taker, bids for sell taker).\n"
            )
            handle.write(
                "- DNL (`dnl`): `trade_notional / side_notional_at_1pct`.\n"
            )
            handle.write(
                "- Queue proxy (`queue_load`): same-second side taker notional divided by side 1% notional.\n"
            )
            handle.write("- Large-size bucket: top 10% `dnl` per symbol.\n")
            if bar_rows > 0:
                handle.write(
                    f"- Mixed mode: `{bar_rows}` grouped rows fall back to bar-proxy because snapshot matching was unavailable.\n"
                )
        else:
            handle.write(
                "- Data source: `prices_resampled.csv` bars with per-minute tick overlays where available.\n"
            )
            handle.write(
                "- Slippage proxy: next-bar absolute return (bps) conditioned on relative size (volume/depth proxy).\n"
            )
            handle.write(
                "- Volatility control: excess bps and normalized metrics use local median absolute return "
                f"(floor `{args.norm_floor_bps:.3f}` bps).\n"
            )
            handle.write("- Large-size bucket: top 10% relative-size bars per symbol.\n")
        handle.write("- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.\n")
        handle.write("- Comparability guardrail: cross-venue tables are segmented by `market_type` (`spot` vs `derivatives`).\n")

        if not slippage_vs_size.empty:
            handle.write("\n## Slippage vs Size\n\n")
            handle.write("```text\n")
            handle.write(slippage_vs_size.to_string(index=False))
            handle.write("\n```\n")

        if not comparison.empty:
            handle.write("\n## Cross-Quote Comparison (USDC vs USDT)\n\n")
            handle.write("```text\n")
            handle.write(comparison.to_string(index=False))
            handle.write("\n```\n")
            for market_type, group in comparison.groupby("market_type", sort=True):
                usdc_better = int((group["preferred_quote_on_large_norm"] == "USDC").sum())
                usdt_better = int((group["preferred_quote_on_large_norm"] == "USDT").sum())
                indeterminate = int((group["preferred_quote_on_large_norm"] == "indeterminate").sum())
                total = int(group.shape[0])
                handle.write(
                    f"\n- `{market_type}` (normalized delta, tolerance={args.norm_delta_tolerance:.3f}): "
                    f"USDC lower-impact `{usdc_better}/{total}`, "
                    f"USDT lower-impact `{usdt_better}/{total}`, "
                    f"indeterminate `{indeterminate}/{total}`.\n"
                )
            handle.write("\nInterpretation guardrail: these are descriptive diagnostics, not a full fee/funding-adjusted TCA ranking.\n")

        if not resilience.empty:
            handle.write("\n## Resilience Summary\n\n")
            handle.write("```text\n")
            handle.write(resilience.to_string(index=False))
            handle.write("\n```\n")

        if not venue_summary.empty:
            handle.write("\n## Venue Summary (Within Market Type)\n\n")
            handle.write("```text\n")
            handle.write(venue_summary.to_string(index=False))
            handle.write("\n```\n")

        handle.write("\n## Artifacts\n\n")
        handle.write(f"- `{coverage_path}`\n")
        handle.write(f"- `{slippage_path}`\n")
        handle.write(f"- `{slippage_curve_path}`\n")
        handle.write(f"- `{comparison_path}`\n")
        handle.write(f"- `{resilience_path}`\n")
        handle.write(f"- `{venue_path}`\n")

    print("Execution quality diagnostics completed.")
    print(f"- l2_coverage: {coverage_path}")
    print(f"- slippage_proxy: {slippage_path}")
    print(f"- slippage_vs_size: {slippage_curve_path}")
    print(f"- cross_quote: {comparison_path}")
    print(f"- resilience: {resilience_path}")
    print(f"- venue_summary: {venue_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
