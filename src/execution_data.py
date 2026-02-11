from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests

from src.execution_quality import DEFAULT_EPISODES


PREFERRED_ROOT_ORDER = ("BTC", "ETH", "SOL", "BNB")
SYMBOL_RE = re.compile(r"^(?P<root>[A-Z0-9]+)(?P<quote>USDT|USDC)(?:-[A-Z0-9]+)?$")


@dataclass(frozen=True)
class FetchTask:
    episode: str
    venue: str
    kind: str
    symbol: str | None
    day: date
    url: str
    destination: Path


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _read_episode_resampled(episode: str, processed_root: Path) -> pd.DataFrame:
    path = processed_root / episode / "prices_resampled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Episode resampled file not found: {path}")
    frame = pd.read_csv(path)
    if "timestamp_utc" not in frame.columns:
        raise ValueError(f"Missing timestamp_utc in {path}")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).copy()
    if frame.empty:
        raise ValueError(f"No valid timestamps in {path}")
    return frame


def infer_episode_bounds(
    episode: str,
    processed_root: Path,
    *,
    override_start: date | None,
    override_end: date | None,
    max_days_per_episode: int,
) -> tuple[date, date]:
    if override_start is not None and override_end is not None:
        if override_end < override_start:
            raise ValueError("end date must be >= start date")
        return override_start, override_end

    frame = _read_episode_resampled(episode, processed_root)
    start = frame["timestamp_utc"].min().date()
    end = frame["timestamp_utc"].max().date()
    if max_days_per_episode > 0:
        allowed_end = start + timedelta(days=max_days_per_episode - 1)
        end = min(end, allowed_end)
    return start, end


def _episode_days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def infer_episode_venue(episode: str, processed_root: Path) -> str:
    frame = _read_episode_resampled(episode, processed_root)
    if "venue" not in frame.columns:
        return "unknown"
    venue = frame["venue"].astype(str).str.lower()
    if venue.empty:
        return "unknown"
    return str(venue.mode().iloc[0]) if not venue.mode().empty else "unknown"


def _sort_roots(roots: set[str]) -> list[str]:
    preferred = [root for root in PREFERRED_ROOT_ORDER if root in roots]
    others = sorted(root for root in roots if root not in set(PREFERRED_ROOT_ORDER))
    return preferred + others


def infer_episode_symbols(
    episode: str,
    processed_root: Path,
    *,
    max_symbols: int,
) -> list[str]:
    frame = _read_episode_resampled(episode, processed_root)
    if "symbol" not in frame.columns:
        return []
    raw_symbols = sorted(set(frame["symbol"].astype(str).tolist()))

    grouped: dict[str, set[str]] = {}
    for sym in raw_symbols:
        m = SYMBOL_RE.match(sym.upper())
        if m is None:
            continue
        grouped.setdefault(m.group("root"), set()).add(m.group("quote"))

    selected: list[str] = []
    for root in _sort_roots(set(grouped.keys())):
        quotes = grouped[root]
        if "USDC" in quotes:
            selected.append(f"{root}USDC")
        if "USDT" in quotes:
            selected.append(f"{root}USDT")
        if max_symbols > 0 and len(selected) >= max_symbols:
            break

    if max_symbols > 0:
        selected = selected[:max_symbols]
    return selected


def build_binance_trades_url(symbol: str, day: date) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{d}.zip"


def build_binance_aggtrades_url(symbol: str, day: date) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip"


def build_binance_bookdepth_url(symbol: str, day: date) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{d}.zip"


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


def build_bybit_spot_trades_url(symbol: str, month: date) -> str:
    return f"https://public.bybit.com/spot/{symbol}/{symbol}-{month.strftime('%Y-%m')}.csv.gz"


def build_bybit_derivatives_trades_url(symbol: str, day: date) -> str:
    return f"https://public.bybit.com/trading/{symbol}/{symbol}{day.strftime('%Y-%m-%d')}.csv.gz"


def _month_start(day: date) -> date:
    return date(day.year, day.month, 1)


def build_fetch_tasks_for_episode(
    *,
    episode: str,
    venue: str,
    symbols: list[str],
    start: date,
    end: date,
    output_root: Path,
    include_agg_trades: bool,
) -> list[FetchTask]:
    tasks: list[FetchTask] = []
    episode_root = output_root / episode
    days = _episode_days(start, end)
    venue_l = venue.lower()

    if venue_l == "binance":
        for sym in symbols:
            for day in days:
                tasks.append(
                    FetchTask(
                        episode=episode,
                        venue="binance",
                        kind="trades",
                        symbol=sym,
                        day=day,
                        url=build_binance_trades_url(sym, day),
                        destination=episode_root / "binance" / "trades" / f"{sym}-trades-{day.isoformat()}.zip",
                    )
                )
                if include_agg_trades:
                    tasks.append(
                        FetchTask(
                            episode=episode,
                            venue="binance",
                            kind="aggTrades",
                            symbol=sym,
                            day=day,
                            url=build_binance_aggtrades_url(sym, day),
                            destination=episode_root / "binance" / "aggTrades" / f"{sym}-aggTrades-{day.isoformat()}.zip",
                        )
                    )
                tasks.append(
                    FetchTask(
                        episode=episode,
                        venue="binance",
                        kind="bookDepth",
                        symbol=sym,
                        day=day,
                        url=build_binance_bookdepth_url(sym, day),
                        destination=episode_root / "binance" / "bookDepth" / f"{sym}-bookDepth-{day.isoformat()}.zip",
                    )
                )
        return tasks

    if venue_l == "okx":
        for day in days:
            tasks.append(
                FetchTask(
                    episode=episode,
                    venue="okx",
                    kind="trades",
                    symbol=None,
                    day=day,
                    url=build_okx_trades_url(day),
                    destination=episode_root / "okx" / "trades" / f"allfuture-trades-{day.isoformat()}.zip",
                )
            )
            tasks.append(
                FetchTask(
                    episode=episode,
                    venue="okx",
                    kind="funding",
                    symbol=None,
                    day=day,
                    url=build_okx_funding_url(day),
                    destination=episode_root / "okx" / "funding" / f"allswaprate-swaprate-{day.isoformat()}.zip",
                )
            )
        return tasks

    if venue_l == "bybit":
        spot_mode = any(sym.endswith("USDC") for sym in symbols)
        if spot_mode:
            months = sorted({_month_start(day) for day in days})
            for sym in symbols:
                for month in months:
                    tasks.append(
                        FetchTask(
                            episode=episode,
                            venue="bybit",
                            kind="trades",
                            symbol=sym,
                            day=month,
                            url=build_bybit_spot_trades_url(sym, month),
                            destination=episode_root / "bybit" / "spot_trades" / f"{sym}-{month.strftime('%Y-%m')}.csv.gz",
                        )
                    )
        else:
            for sym in symbols:
                for day in days:
                    tasks.append(
                        FetchTask(
                            episode=episode,
                            venue="bybit",
                            kind="trades",
                            symbol=sym,
                            day=day,
                            url=build_bybit_derivatives_trades_url(sym, day),
                            destination=episode_root / "bybit" / "trades" / f"{sym}{day.strftime('%Y-%m-%d')}.csv.gz",
                        )
                    )
        return tasks

    return tasks


def fetch_one(task: FetchTask, *, timeout: int, skip_existing: bool) -> dict[str, Any]:
    task.destination.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and task.destination.exists() and task.destination.stat().st_size > 0:
        return {
            "episode": task.episode,
            "venue": task.venue,
            "kind": task.kind,
            "symbol": task.symbol or "",
            "day": task.day.isoformat(),
            "url": task.url,
            "local_path": str(task.destination),
            "status": "skipped_existing",
            "http_status": 200,
            "bytes": int(task.destination.stat().st_size),
        }

    try:
        with requests.get(task.url, timeout=timeout, stream=True) as response:
            status = int(response.status_code)
            if status != 200:
                return {
                    "episode": task.episode,
                    "venue": task.venue,
                    "kind": task.kind,
                    "symbol": task.symbol or "",
                    "day": task.day.isoformat(),
                    "url": task.url,
                    "local_path": str(task.destination),
                    "status": "http_error",
                    "http_status": status,
                    "bytes": 0,
                }
            size = 0
            with task.destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    size += len(chunk)
            return {
                "episode": task.episode,
                "venue": task.venue,
                "kind": task.kind,
                "symbol": task.symbol or "",
                "day": task.day.isoformat(),
                "url": task.url,
                "local_path": str(task.destination),
                "status": "downloaded",
                "http_status": status,
                "bytes": size,
            }
    except Exception as exc:
        return {
            "episode": task.episode,
            "venue": task.venue,
            "kind": task.kind,
            "symbol": task.symbol or "",
            "day": task.day.isoformat(),
            "url": task.url,
            "local_path": str(task.destination),
            "status": f"error:{type(exc).__name__}",
            "http_status": -1,
            "bytes": 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download publicly available execution datasets (orderbook/trades where available) "
            "under data/processed/orderbook/<episode>/..."
        )
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=DEFAULT_EPISODES,
        help="Episode ids to bootstrap from data/processed/episodes/<episode>/prices_resampled.csv",
    )
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Root containing episode tables.")
    parser.add_argument("--output-root", default="data/processed/orderbook", help="Destination root for execution data.")
    parser.add_argument("--start", default=None, help="Optional override start date YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional override end date YYYY-MM-DD.")
    parser.add_argument(
        "--max-days-per-episode",
        type=int,
        default=3,
        help="Safety cap on number of days downloaded per episode (0 = no cap).",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=2,
        help="Max symbols per episode to fetch (for Binance/Bybit).",
    )
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already present.")
    parser.add_argument(
        "--include-agg-trades",
        action="store_true",
        help="Also fetch Binance aggTrades files in addition to trades.",
    )
    parser.add_argument(
        "--manifest",
        default="reports/final/execution_data_manifest.csv",
        help="Output CSV manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    override_start = parse_iso_date(args.start) if args.start else None
    override_end = parse_iso_date(args.end) if args.end else None

    all_rows: list[dict[str, Any]] = []
    total_tasks = 0
    for episode in args.episodes:
        try:
            venue = infer_episode_venue(episode, processed_root)
            start, end = infer_episode_bounds(
                episode,
                processed_root,
                override_start=override_start,
                override_end=override_end,
                max_days_per_episode=int(args.max_days_per_episode),
            )
            symbols = infer_episode_symbols(
                episode,
                processed_root,
                max_symbols=int(args.max_symbols),
            )
            tasks = build_fetch_tasks_for_episode(
                episode=episode,
                venue=venue,
                symbols=symbols,
                start=start,
                end=end,
                output_root=output_root,
                include_agg_trades=bool(args.include_agg_trades),
            )
        except Exception as exc:
            all_rows.append(
                {
                    "episode": episode,
                    "venue": "",
                    "kind": "",
                    "symbol": "",
                    "day": "",
                    "url": "",
                    "local_path": "",
                    "status": f"episode_error:{type(exc).__name__}",
                    "http_status": -1,
                    "bytes": 0,
                }
            )
            print(f"- {episode}: episode_error ({exc})")
            continue

        total_tasks += len(tasks)
        print(
            f"- {episode}: venue={venue}, start={start}, end={end}, "
            f"symbols={symbols}, tasks={len(tasks)}"
        )
        for task in tasks:
            row = fetch_one(
                task,
                timeout=int(args.timeout_sec),
                skip_existing=bool(args.skip_existing),
            )
            all_rows.append(row)

    manifest = pd.DataFrame(all_rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "episode",
                "venue",
                "kind",
                "symbol",
                "day",
                "url",
                "local_path",
                "status",
                "http_status",
                "bytes",
            ]
        )
    manifest.to_csv(manifest_path, index=False)

    downloaded = int((manifest["status"] == "downloaded").sum()) if "status" in manifest else 0
    skipped = int((manifest["status"] == "skipped_existing").sum()) if "status" in manifest else 0
    errors = int(
        (~manifest["status"].astype(str).isin({"downloaded", "skipped_existing", "http_error"})).sum()
    ) if "status" in manifest else 0
    http_errors = int((manifest["status"] == "http_error").sum()) if "status" in manifest else 0
    print("Execution data bootstrap completed.")
    print(f"- tasks: {total_tasks}")
    print(f"- downloaded: {downloaded}")
    print(f"- skipped_existing: {skipped}")
    print(f"- http_errors: {http_errors}")
    print(f"- other_errors: {errors}")
    print(f"- manifest: {manifest_path}")


if __name__ == "__main__":
    main()
