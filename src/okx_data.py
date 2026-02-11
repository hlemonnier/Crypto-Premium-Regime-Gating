from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta
import re
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd
import requests

from src.data_ingest import IngestConfig, build_processed_prices, save_processed
from src.pipeline import export_outputs, load_config, run_pipeline
from src.premium import infer_cross_asset_pairs

INSTRUMENT_RE = re.compile(r"^(?P<root>[A-Z0-9]+)-(?P<quote>USDT|USDC)-(?P<suffix>[A-Z0-9]+)$")


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    out: list[date] = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def build_okx_trades_url(day: date) -> str:
    yyyymm = day.strftime("%Y%m")
    day_str = day.strftime("%Y-%m-%d")
    return (
        "https://www.okx.com/cdn/okex/traderecords/trades/monthly/"
        f"{yyyymm}/allfuture-trades-{day_str}.zip"
    )


def download_to_path(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=60, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def parse_instrument(inst: str) -> tuple[str, str, str] | None:
    m = INSTRUMENT_RE.match(inst)
    if not m:
        return None
    return m.group("root"), m.group("quote"), m.group("suffix")


def normalize_instrument_symbol(inst: str) -> str | None:
    parsed = parse_instrument(inst)
    if parsed is None:
        return None
    root, quote, suffix = parsed
    return f"{root}{quote}-{suffix}"


def _resolve_okx_columns(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in columns:
        english = str(col).split("/", 1)[0].strip().lower()
        mapping[english] = str(col)
    required = {"instrument_name", "size", "price", "created_time"}
    missing = required.difference(mapping)
    if missing:
        raise ValueError(f"Could not resolve OKX trade columns. Missing={sorted(missing)}")
    return {
        "instrument": mapping["instrument_name"],
        "size": mapping["size"],
        "price": mapping["price"],
        "created_time": mapping["created_time"],
    }


def _open_okx_csv(zip_path: Path):
    archive = ZipFile(zip_path)
    names = [name for name in archive.namelist() if not name.endswith("/")]
    if not names:
        archive.close()
        raise ValueError(f"Empty zip archive: {zip_path}")
    return archive, names[0]


def auto_select_symbols(
    zip_paths: list[Path],
    *,
    roots: list[str],
) -> list[str]:
    volumes: dict[tuple[str, str, str], float] = defaultdict(float)
    roots_set = set(roots)

    for zip_path in zip_paths:
        archive, csv_name = _open_okx_csv(zip_path)
        try:
            with archive.open(csv_name) as handle:
                header = pd.read_csv(handle, encoding="gbk", nrows=0)
            cols = _resolve_okx_columns(header.columns)

            with archive.open(csv_name) as handle:
                for chunk in pd.read_csv(
                    handle,
                    encoding="gbk",
                    usecols=[cols["instrument"], cols["size"]],
                    chunksize=200_000,
                ):
                    inst = chunk[cols["instrument"]].astype(str)
                    size = pd.to_numeric(chunk[cols["size"]], errors="coerce").fillna(0.0)
                    tmp = pd.DataFrame({"inst": inst, "size": size})
                    tmp = tmp[tmp["inst"].str.contains("USDC|USDT", regex=True, na=False)]
                    if tmp.empty:
                        continue
                    grouped = tmp.groupby("inst", as_index=False)["size"].sum()
                    for row in grouped.itertuples(index=False):
                        parsed = parse_instrument(row.inst)
                        if parsed is None:
                            continue
                        root, quote, suffix = parsed
                        if root not in roots_set or quote not in {"USDC", "USDT"}:
                            continue
                        volumes[(root, suffix, quote)] += float(row.size)
        finally:
            archive.close()

    selected: list[str] = []
    for root in roots:
        suffix_scores: dict[str, float] = {}
        for (r, suffix, quote), vol in volumes.items():
            if r != root:
                continue
            if quote == "USDC":
                usdt_vol = volumes.get((root, suffix, "USDT"), 0.0)
                if usdt_vol > 0 and vol > 0:
                    suffix_scores[suffix] = vol + usdt_vol
        if not suffix_scores:
            continue
        best_suffix = max(suffix_scores.items(), key=lambda kv: kv[1])[0]
        selected.append(f"{root}-USDC-{best_suffix}")
        selected.append(f"{root}-USDT-{best_suffix}")
    return selected


def load_selected_trades(zip_paths: list[Path], symbols: list[str]) -> pd.DataFrame:
    symbol_set = set(symbols)
    tables: list[pd.DataFrame] = []

    for zip_path in zip_paths:
        archive, csv_name = _open_okx_csv(zip_path)
        try:
            with archive.open(csv_name) as handle:
                header = pd.read_csv(handle, encoding="gbk", nrows=0)
            cols = _resolve_okx_columns(header.columns)

            with archive.open(csv_name) as handle:
                for chunk in pd.read_csv(
                    handle,
                    encoding="gbk",
                    usecols=[cols["instrument"], cols["size"], cols["price"], cols["created_time"]],
                    chunksize=200_000,
                ):
                    inst = chunk[cols["instrument"]].astype(str)
                    mask = inst.isin(symbol_set)
                    if not mask.any():
                        continue

                    local = pd.DataFrame()
                    local["instrument"] = inst[mask]
                    local["size"] = pd.to_numeric(chunk.loc[mask, cols["size"]], errors="coerce")
                    local["price"] = pd.to_numeric(chunk.loc[mask, cols["price"]], errors="coerce")
                    local["created_time"] = pd.to_numeric(
                        chunk.loc[mask, cols["created_time"]], errors="coerce"
                    )
                    local = local.dropna(subset=["instrument", "size", "price", "created_time"])
                    if local.empty:
                        continue

                    local["timestamp_utc"] = pd.to_datetime(
                        local["created_time"], unit="ms", utc=True, errors="coerce"
                    ).dt.floor("min")
                    local = local.dropna(subset=["timestamp_utc"])
                    local["weighted_price"] = local["price"] * local["size"]
                    grouped = local.groupby(["timestamp_utc", "instrument"], as_index=False).agg(
                        weighted_price=("weighted_price", "sum"),
                        volume=("size", "sum"),
                    )
                    grouped = grouped[grouped["volume"] > 0]
                    if grouped.empty:
                        continue
                    grouped["price"] = grouped["weighted_price"] / grouped["volume"]
                    grouped["symbol"] = grouped["instrument"].map(normalize_instrument_symbol)
                    grouped = grouped.dropna(subset=["symbol"])
                    grouped["venue"] = "okx"
                    tables.append(grouped[["timestamp_utc", "symbol", "price", "volume", "venue"]])
        finally:
            archive.close()

    if not tables:
        raise RuntimeError("No OKX trades loaded for selected symbols.")
    out = pd.concat(tables, ignore_index=True)
    out = out.sort_values(["timestamp_utc", "symbol"]).reset_index(drop=True)
    return out


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
        description="Download OKX futures trade archives, build matrix, and optionally run the pipeline."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Exact OKX instrument names, e.g. BTC-USDC-230630 BTC-USDT-230630",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["BTC", "ETH"],
        help="Roots for automatic symbol selection when --symbols is omitted.",
    )
    parser.add_argument("--raw-dir", default="data/raw/okx/trades", help="Raw trade zip destination.")
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Processed episode root.")
    parser.add_argument("--episode-name", default=None, help="Optional custom episode name.")
    parser.add_argument("--resample-rule", default="1min", help="Target resample rule.")
    parser.add_argument("--ffill-limit", type=int, default=2, help="Forward-fill limit.")
    parser.add_argument("--glitch-sigma-threshold", type=float, default=12.0, help="Glitch filter threshold.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip downloading files already present.")
    parser.add_argument("--run-pipeline", action="store_true", help="Run full premium pipeline after ingest.")
    parser.add_argument("--config", default="configs/config.yaml", help="Pipeline config path.")
    parser.add_argument("--reports-root", default="reports/episodes", help="Per-episode reports root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end)
    episode = args.episode_name or f"okx_{start.isoformat()}_{end.isoformat()}"
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_root) / episode
    reports_root = Path(args.reports_root) / episode

    days = daterange(start, end)
    zip_paths: list[Path] = []
    print(f"Collecting OKX trade archives for {episode}...")
    for day in days:
        url = build_okx_trades_url(day)
        path = raw_dir / Path(url).name
        if not (args.skip_existing and path.exists()):
            download_to_path(url, path)
        zip_paths.append(path)

    symbols = args.symbols
    if symbols is None:
        symbols = auto_select_symbols(zip_paths, roots=args.roots)
        if not symbols:
            raise SystemExit(
                "Could not auto-select OKX USDC/USDT futures symbols from trade archives."
            )
        print(f"Auto-selected symbols: {symbols}")

    raw_table = load_selected_trades(zip_paths, symbols)

    ingest_cfg = IngestConfig(
        resample_rule=args.resample_rule,
        ffill_limit=args.ffill_limit,
        glitch_sigma_threshold=args.glitch_sigma_threshold,
    )
    resampled, matrix = build_processed_prices(raw_table, ingest_cfg)
    save_processed(resampled, matrix, processed_dir)
    matrix_path = find_matrix_path(processed_dir)

    print("Ingestion completed.")
    print(f"- symbols: {symbols}")
    print(f"- raw rows: {raw_table.shape[0]}")
    print(f"- resampled rows: {resampled.shape[0]}")
    print(f"- matrix shape: {matrix.shape}")
    print(f"- matrix path: {matrix_path}")

    if not args.run_pipeline:
        return

    target_usdc_candidates = [c for c in matrix.columns if c.startswith("BTCUSDC-")]
    target_usdt_candidates = [c for c in matrix.columns if c.startswith("BTCUSDT-")]
    if not target_usdc_candidates or not target_usdt_candidates:
        available = ", ".join(sorted(matrix.columns))
        raise SystemExit(
            "Missing BTCUSDC/BTCUSDT target symbols in OKX matrix. "
            f"Available=[{available}]"
        )
    target_usdc = sorted(target_usdc_candidates)[0]
    target_usdt = sorted(target_usdt_candidates)[0]

    config = load_config(args.config)
    run_cfg = deepcopy(config)
    run_cfg.setdefault("data", {})
    run_cfg["data"]["resample_rule"] = args.resample_rule
    run_cfg["data"]["price_matrix_path"] = str(matrix_path)
    run_cfg.setdefault("outputs", {})
    run_cfg["outputs"]["tables_dir"] = str(reports_root / "tables")
    run_cfg["outputs"]["figures_dir"] = str(reports_root / "figures")
    run_cfg.setdefault("premium", {})
    run_cfg["premium"]["target_usdc_symbol"] = target_usdc
    run_cfg["premium"]["target_usdt_symbol"] = target_usdt
    run_cfg["premium"]["proxy_pairs"] = _build_proxy_pairs(list(matrix.columns), (target_usdc, target_usdt))

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
