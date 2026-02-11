from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EPISODES = [
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


def _empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _has_any_path(root: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(root.glob(pattern)):
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
                "*orderbook*l2*.parquet",
                "*orderbook*l2*.csv",
                "*book*l2*.parquet",
                "*book*l2*.csv",
                "orderbook*.parquet",
                "orderbook*.csv",
                "depth*.parquet",
                "depth*.csv",
            ],
        )
        has_ticks = _has_any_path(
            episode_root,
            [
                "*trade*tick*.parquet",
                "*trade*tick*.csv",
                "*trades*.parquet",
                "*trades*.csv",
                "*aggTrade*.parquet",
                "*aggTrade*.csv",
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
    if s == "PERP" or s.isdigit():
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


def _load_episode_resampled(episode: str, processed_root: Path) -> pd.DataFrame | None:
    path = processed_root / episode / "prices_resampled.csv"
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    required = {"timestamp_utc", "price", "volume", "symbol", "venue"}
    if not required.issubset(set(frame.columns)):
        return None

    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["venue"] = frame["venue"].astype(str)
    frame = frame.dropna(subset=["timestamp_utc", "price", "symbol", "venue"]).copy()
    frame = frame[frame["price"] > 0].copy()
    frame["episode"] = episode

    parsed = frame["symbol"].map(_parse_symbol)
    frame["root"] = parsed.map(lambda x: x[0] if x else None)
    frame["quote"] = parsed.map(lambda x: x[1] if x else None)
    frame["suffix"] = parsed.map(lambda x: x[2] if x else None)
    frame["market_type"] = frame["suffix"].map(_market_type_from_suffix)
    frame = frame[frame["quote"].isin(["USDC", "USDT"])].copy()
    frame = frame[frame["market_type"].isin(["spot", "derivatives"])].copy()
    return frame


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
            }
        )

    if not rows:
        return _empty_table(SLIPPAGE_COLUMNS)
    out = pd.DataFrame(rows)
    return out.sort_values(["episode", "venue", "market_type", "root", "quote"]).reset_index(drop=True)


def build_cross_quote_comparison(slippage: pd.DataFrame, *, norm_delta_tolerance: float = 0.05) -> pd.DataFrame:
    if slippage.empty:
        return _empty_table(COMPARISON_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (episode, venue, market_type, root), group in slippage.groupby(
        ["episode", "venue", "market_type", "root"], sort=False
    ):
        by_quote = {quote: q for quote, q in group.groupby("quote")}
        if "USDC" not in by_quote or "USDT" not in by_quote:
            continue

        usdc = by_quote["USDC"].iloc[0]
        usdt = by_quote["USDT"].iloc[0]
        delta_raw = float(usdc["impact_large_mean_bps"] - usdt["impact_large_mean_bps"])
        delta_excess = float(usdc["impact_large_mean_excess_bps"] - usdt["impact_large_mean_excess_bps"])
        delta_norm = float(usdc["impact_large_mean_norm"] - usdt["impact_large_mean_norm"])
        preferred_norm = _preference_from_delta(delta_norm, tolerance=norm_delta_tolerance)

        rows.append(
            {
                "episode": episode,
                "venue": venue,
                "market_type": market_type,
                "root": root,
                "impact_large_mean_bps_usdc": float(usdc["impact_large_mean_bps"]),
                "impact_large_mean_bps_usdt": float(usdt["impact_large_mean_bps"]),
                "impact_large_delta_usdc_minus_usdt_bps": delta_raw,
                "impact_large_mean_excess_bps_usdc": float(usdc["impact_large_mean_excess_bps"]),
                "impact_large_mean_excess_bps_usdt": float(usdt["impact_large_mean_excess_bps"]),
                "impact_large_delta_excess_usdc_minus_usdt_bps": delta_excess,
                "impact_large_mean_norm_usdc": float(usdc["impact_large_mean_norm"]),
                "impact_large_mean_norm_usdt": float(usdt["impact_large_mean_norm"]),
                "impact_large_delta_norm_usdc_minus_usdt": delta_norm,
                "impact_all_mean_bps_usdc": float(usdc["impact_all_mean_bps"]),
                "impact_all_mean_bps_usdt": float(usdt["impact_all_mean_bps"]),
                "preferred_quote_on_large_norm": preferred_norm,
            }
        )

    if not rows:
        return _empty_table(COMPARISON_COLUMNS)
    return pd.DataFrame(rows).sort_values(["episode", "venue", "market_type", "root"]).reset_index(drop=True)


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
            "Allow bar-level proxy diagnostics even when L2/tick data are missing. "
            "Default is fail-closed (no execution ranking) without L2 readiness."
        ),
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
    comparison_path = out_dir / "execution_cross_quote_comparison.csv"
    resilience_path = out_dir / "execution_resilience.csv"
    venue_path = out_dir / "execution_venue_comparison.csv"
    coverage_path = out_dir / "execution_l2_coverage.csv"
    report_path = out_dir / "execution_quality_report.md"

    l2_coverage = build_l2_coverage(list(args.episodes), l2_root)
    l2_coverage.to_csv(coverage_path, index=False)
    missing_l2 = l2_coverage[~l2_coverage["l2_ready"].astype(bool)].copy()

    if (not args.allow_bar_proxy_without_l2) and (not missing_l2.empty):
        _empty_table(SLIPPAGE_COLUMNS).to_csv(slippage_path, index=False)
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
                "\nNo bar-proxy fallback was produced because `--allow-bar-proxy-without-l2` was not set.\n"
            )
            handle.write(
                "This fail-closed behavior prevents unsupported venue/quote liquidity rankings.\n"
            )
            handle.write("\n## Artifacts\n\n")
            handle.write(f"- `{coverage_path}`\n")
            handle.write(f"- `{slippage_path}`\n")
            handle.write(f"- `{comparison_path}`\n")
            handle.write(f"- `{resilience_path}`\n")
            handle.write(f"- `{venue_path}`\n")

        print("Execution diagnostics blocked: missing L2/tick inputs for at least one episode.")
        print(f"- l2_coverage: {coverage_path}")
        print(f"- report: {report_path}")
        return

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for episode in args.episodes:
        frame = _load_episode_resampled(episode, processed_root)
        if frame is None or frame.empty:
            missing.append(episode)
            continue
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No valid episode resampled files were found for execution diagnostics.")

    raw = pd.concat(frames, ignore_index=True)
    enriched = build_enriched_trade_frame(
        raw,
        size_window=args.size_window,
        min_size_periods=args.min_size_periods,
        norm_floor_bps=args.norm_floor_bps,
    )
    slippage = build_slippage_proxy(enriched)
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
    comparison.to_csv(comparison_path, index=False)
    resilience.to_csv(resilience_path, index=False)
    venue_summary.to_csv(venue_path, index=False)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Execution Proxy Diagnostics (Bar-Level)\n\n")
        handle.write("This report extends stablecoin analysis with bar-level execution proxies.\n\n")
        handle.write("## Scope\n\n")
        for ep in args.episodes:
            handle.write(f"- `{ep}`\n")
        if missing:
            handle.write("\nMissing episodes (skipped):\n")
            for ep in missing:
                handle.write(f"- `{ep}`\n")

        handle.write("\n## L2 Coverage\n\n")
        handle.write("```text\n")
        handle.write(l2_coverage.to_string(index=False))
        handle.write("\n```\n")
        handle.write(
            "\nNote: L2 coverage is provided for transparency. Current diagnostics below remain bar-level proxies.\n"
        )

        handle.write("\n## Method Notes\n\n")
        handle.write("- Data source: `prices_resampled.csv` (price + volume bars).\n")
        handle.write("- Slippage proxy: next-bar absolute return (bps) conditioned on relative size (`volume / rolling median volume`).\n")
        handle.write(
            "- Volatility control: report large-size deltas in raw bps, excess bps (`next_bar_abs_ret - local_median_abs_ret`), "
            "and normalized units (`next_bar_abs_ret / local_median_abs_ret`).\n"
        )
        handle.write(f"- Normalization floor: local volatility denominator floored at `{args.norm_floor_bps:.3f}` bps.\n")
        handle.write("- Large-size bucket: top 10% relative-size bars per symbol.\n")
        handle.write("- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.\n")
        handle.write(
            "- Limitation: this is **not** order-book snapshot slippage/depth "
            "(book-walk, DNL, queueing); it is a trade-bar proxy given available data.\n"
        )
        handle.write("- Comparability guardrail: cross-venue tables are segmented by `market_type` (`spot` vs `derivatives`); avoid mixing them for venue ranking.\n")

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
                    f"USDC lower-proxy-impact `{usdc_better}/{total}`, "
                    f"USDT lower-proxy-impact `{usdt_better}/{total}`, "
                    f"indeterminate `{indeterminate}/{total}`.\n"
                )
            handle.write(
                "\nInterpretation guardrail: these are descriptive bar-level proxy gaps only; "
                "they are insufficient to rank venues for execution quality.\n"
            )

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
        handle.write(f"- `{comparison_path}`\n")
        handle.write(f"- `{resilience_path}`\n")
        handle.write(f"- `{venue_path}`\n")

    print("Execution quality diagnostics completed.")
    print(f"- l2_coverage: {coverage_path}")
    print(f"- slippage_proxy: {slippage_path}")
    print(f"- cross_quote: {comparison_path}")
    print(f"- resilience: {resilience_path}")
    print(f"- venue_summary: {venue_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
