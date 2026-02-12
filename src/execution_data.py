from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from src.execution_quality import DEFAULT_EPISODES


PREFERRED_ROOT_ORDER = ("BTC", "ETH", "SOL", "BNB")
OKX_PRIAPI_DOWNLOAD_LINK = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"


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


def _extract_root_quote_from_symbol(raw_symbol: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[^A-Z0-9]", "", str(raw_symbol).upper())
    for quote in ("USDC", "USDT"):
        idx = normalized.find(quote)
        if idx > 0:
            root = normalized[:idx]
            if root:
                return root, quote
    return None


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
        parsed = _extract_root_quote_from_symbol(sym)
        if parsed is None:
            continue
        root, quote = parsed
        grouped.setdefault(root, set()).add(quote)

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


def _symbols_to_okx_instruments(symbols: list[str]) -> tuple[list[str], list[str]]:
    inst_ids: set[str] = set()
    inst_families: set[str] = set()
    for symbol in symbols:
        parsed = _extract_root_quote_from_symbol(symbol)
        if parsed is None:
            continue
        root, quote = parsed
        inst = f"{root}-{quote}"
        inst_ids.add(inst)
        inst_families.add(inst)
    ordered = sorted(inst_ids)
    return ordered, sorted(inst_families)


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


def _day_end_epoch_ms(day: date) -> str:
    dt = datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)
    return str(int(dt.timestamp() * 1000))


def _as_okx_inst_query(inst_type: str, instruments: list[str]) -> dict[str, list[str]]:
    inst_type_u = str(inst_type).upper()
    if inst_type_u == "SPOT":
        return {"instIdList": instruments}
    return {"instFamilyList": instruments}


def query_okx_priapi_download_links(
    *,
    module: str,
    inst_type: str,
    instruments: list[str],
    day: date,
    timeout_sec: int,
) -> list[dict[str, str]]:
    if not instruments:
        return []
    day_end_ms = _day_end_epoch_ms(day)
    payload = {
        "module": str(module),
        "instType": str(inst_type).upper(),
        "instQueryParam": _as_okx_inst_query(inst_type, instruments),
        "dateQuery": {
            "dateAggrType": "daily",
            "begin": day_end_ms,
            "end": day_end_ms,
        },
    }
    try:
        response = requests.post(
            OKX_PRIAPI_DOWNLOAD_LINK,
            json=payload,
            timeout=max(10, int(timeout_sec)),
        )
    except Exception:
        return []

    if int(response.status_code) != 200:
        return []

    try:
        body = response.json()
    except Exception:
        return []

    if str(body.get("code", "")) != "0":
        return []

    details = body.get("data", {}).get("details", []) or []
    rows: list[dict[str, str]] = []
    for detail in details:
        group_details = detail.get("groupDetails", []) or []
        for group in group_details:
            url = str(group.get("url", "")).strip()
            filename = str(group.get("filename", "")).strip()
            if not url:
                continue
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name
            if not filename:
                continue
            rows.append(
                {
                    "url": url,
                    "filename": filename,
                    "inst_type": str(detail.get("instType", "")).upper(),
                    "inst_family": str(detail.get("instFamily", "")).upper(),
                    "inst_id": str(detail.get("instId", "")).upper(),
                }
            )
    return rows


def build_okx_priapi_orderbook_tasks(
    *,
    episode: str,
    symbols: list[str],
    days: list[date],
    output_root: Path,
    timeout_sec: int,
    modules: list[str],
    inst_types: list[str],
) -> list[FetchTask]:
    episode_root = output_root / episode
    inst_ids, inst_families = _symbols_to_okx_instruments(symbols)
    tasks: list[FetchTask] = []
    seen: set[tuple[str, str]] = set()

    for module in modules:
        module_s = str(module)
        for inst_type in inst_types:
            inst_type_u = str(inst_type).upper()
            instruments = inst_ids if inst_type_u == "SPOT" else inst_families
            for day in days:
                links = query_okx_priapi_download_links(
                    module=module_s,
                    inst_type=inst_type_u,
                    instruments=instruments,
                    day=day,
                    timeout_sec=timeout_sec,
                )
                for link in links:
                    filename = link["filename"]
                    key = (module_s, filename)
                    if key in seen:
                        continue
                    seen.add(key)
                    folder = "orderbook_l2_400lv" if module_s == "4" else f"orderbook_l2_module_{module_s}"
                    tasks.append(
                        FetchTask(
                            episode=episode,
                            venue="okx",
                            kind=folder,
                            symbol=link.get("inst_family") or link.get("inst_id") or None,
                            day=day,
                            url=link["url"],
                            destination=episode_root / "okx" / folder / filename,
                        )
                    )
    return tasks


def build_external_manifest_tasks(
    *,
    manifest_path: Path,
    output_root: Path,
    episodes_filter: set[str],
) -> dict[str, list[FetchTask]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"External URL manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required = {"episode", "url"}
    if not required.issubset(set(frame.columns)):
        raise ValueError(
            f"External URL manifest must include columns: {sorted(required)}. "
            f"Found: {sorted(frame.columns.tolist())}"
        )

    out: dict[str, list[FetchTask]] = {}
    for _, row in frame.iterrows():
        episode = str(row.get("episode", "")).strip()
        url = str(row.get("url", "")).strip()
        if not episode or not url:
            continue
        if episodes_filter and episode not in episodes_filter:
            continue

        venue = str(row.get("venue", "external")).strip() or "external"
        kind = str(row.get("kind", "external")).strip() or "external"
        symbol = str(row.get("symbol", "")).strip() or None

        day_raw = str(row.get("day", "")).strip()
        day = parse_iso_date(day_raw) if day_raw else date.today()

        local_rel = str(row.get("local_path", "")).strip()
        if local_rel:
            destination = (output_root / episode / local_rel).resolve()
        else:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                filename = Path(urlparse(url).path).name
            if not filename:
                digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
                filename = f"external_{digest}.bin"
            destination = output_root / episode / venue / kind / filename

        out.setdefault(episode, []).append(
            FetchTask(
                episode=episode,
                venue=venue,
                kind=kind,
                symbol=symbol,
                day=day,
                url=url,
                destination=destination,
            )
        )
    return out


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


def dedupe_tasks(tasks: list[FetchTask]) -> list[FetchTask]:
    out: list[FetchTask] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.url, str(task.destination))
        if key in seen:
            continue
        seen.add(key)
        out.append(task)
    return out


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
    parser.add_argument(
        "--external-url-manifest",
        default=None,
        help=(
            "Optional CSV with direct file URLs to ingest (columns: episode,url; optional: "
            "venue,kind,symbol,day,filename,local_path). Useful for manual Bybit/OKX portal exports."
        ),
    )
    parser.add_argument(
        "--disable-okx-priapi-orderbook",
        action="store_true",
        help="Disable OKX priapi orderbook discovery (module 4).",
    )
    parser.add_argument(
        "--okx-orderbook-modules",
        nargs="+",
        default=["4"],
        help="OKX priapi orderbook module list (default: 4 for 400-level L2).",
    )
    parser.add_argument(
        "--okx-priapi-inst-types",
        nargs="+",
        default=["SWAP", "SPOT"],
        help="OKX inst types to query for orderbook modules (default: SWAP SPOT).",
    )
    parser.add_argument(
        "--okx-priapi-timeout-sec",
        type=int,
        default=30,
        help="Timeout in seconds for OKX priapi metadata discovery calls.",
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
    episodes_filter = set(str(ep) for ep in args.episodes)

    external_tasks_by_episode: dict[str, list[FetchTask]] = {}
    if args.external_url_manifest:
        manifest_in = Path(args.external_url_manifest)
        external_tasks_by_episode = build_external_manifest_tasks(
            manifest_path=manifest_in,
            output_root=output_root,
            episodes_filter=episodes_filter,
        )

    all_rows: list[dict[str, Any]] = []
    total_tasks = 0
    for episode in args.episodes:
        tasks: list[FetchTask] = []
        symbols: list[str] = []
        venue = "unknown"
        start: date | None = None
        end: date | None = None
        episode_external = list(external_tasks_by_episode.get(str(episode), []))
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
            if venue.lower() == "okx" and (not bool(args.disable_okx_priapi_orderbook)):
                days = _episode_days(start, end)
                okx_priapi_tasks = build_okx_priapi_orderbook_tasks(
                    episode=episode,
                    symbols=symbols,
                    days=days,
                    output_root=output_root,
                    timeout_sec=int(args.okx_priapi_timeout_sec),
                    modules=[str(m) for m in args.okx_orderbook_modules],
                    inst_types=[str(s).upper() for s in args.okx_priapi_inst_types],
                )
                tasks.extend(okx_priapi_tasks)
        except Exception as exc:
            if not episode_external:
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
            print(f"- {episode}: warning episode metadata unavailable ({exc}); using external URL manifest only.")

        if episode_external:
            tasks.extend(episode_external)
        tasks = dedupe_tasks(tasks)
        if not tasks:
            print(f"- {episode}: no fetch tasks generated (venue={venue}, symbols={symbols})")
            all_rows.append(
                {
                    "episode": episode,
                    "venue": venue,
                    "kind": "",
                    "symbol": "",
                    "day": "",
                    "url": "",
                    "local_path": "",
                    "status": "no_tasks",
                    "http_status": 204,
                    "bytes": 0,
                }
            )
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
        (~manifest["status"].astype(str).isin({"downloaded", "skipped_existing", "http_error", "no_tasks"})).sum()
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
