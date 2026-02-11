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
    frame = frame[frame["quote"].isin(["USDC", "USDT"])].copy()
    return frame


def build_enriched_trade_frame(
    raw: pd.DataFrame,
    *,
    size_window: int = 60,
    min_size_periods: int = 20,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    group_cols = ["episode", "venue", "symbol"]
    for _, group in raw.groupby(group_cols, sort=False):
        g = group.sort_values("timestamp_utc").copy()
        g["log_price"] = np.log(g["price"].astype(float).clip(lower=1e-12))
        g["ret_bps"] = g["log_price"].diff() * 1e4
        g["abs_ret_bps"] = g["ret_bps"].abs()
        g["fwd_abs_ret_bps"] = g["abs_ret_bps"].shift(-1)

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
    keys = ["episode", "venue", "root", "quote", "symbol"]
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
                "root": key[2],
                "quote": key[3],
                "symbol": key[4],
                "n_obs": int(g.shape[0]),
                "median_volume": float(g["volume"].median()),
                "median_rel_size": float(g["rel_size"].median()),
                "q90_rel_size": q90,
                "impact_all_mean_bps": float(g["fwd_abs_ret_bps"].mean()),
                "impact_all_median_bps": float(g["fwd_abs_ret_bps"].median()),
                "impact_large_mean_bps": float(g.loc[large, "fwd_abs_ret_bps"].mean()) if large_count > 0 else np.nan,
                "impact_large_median_bps": float(g.loc[large, "fwd_abs_ret_bps"].median()) if large_count > 0 else np.nan,
                "large_count": large_count,
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["episode", "venue", "root", "quote"]).reset_index(drop=True)


def build_cross_quote_comparison(slippage: pd.DataFrame) -> pd.DataFrame:
    if slippage.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (episode, venue, root), group in slippage.groupby(["episode", "venue", "root"], sort=False):
        by_quote = {quote: q for quote, q in group.groupby("quote")}
        if "USDC" not in by_quote or "USDT" not in by_quote:
            continue

        usdc = by_quote["USDC"].iloc[0]
        usdt = by_quote["USDT"].iloc[0]
        delta = float(usdc["impact_large_mean_bps"] - usdt["impact_large_mean_bps"])
        if np.isnan(delta):
            preferred = "undetermined"
        elif delta < 0:
            preferred = "USDC"
        elif delta > 0:
            preferred = "USDT"
        else:
            preferred = "tie"

        rows.append(
            {
                "episode": episode,
                "venue": venue,
                "root": root,
                "impact_large_mean_bps_usdc": float(usdc["impact_large_mean_bps"]),
                "impact_large_mean_bps_usdt": float(usdt["impact_large_mean_bps"]),
                "impact_large_delta_usdc_minus_usdt_bps": delta,
                "impact_all_mean_bps_usdc": float(usdc["impact_all_mean_bps"]),
                "impact_all_mean_bps_usdt": float(usdt["impact_all_mean_bps"]),
                "preferred_quote_on_large": preferred,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["episode", "venue", "root"]).reset_index(drop=True)


def build_resilience_table(
    enriched: pd.DataFrame,
    *,
    horizon_bars: int = 60,
    shock_quantile: float = 0.99,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["episode", "venue", "root", "quote", "symbol"]
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
                "root": key[2],
                "quote": key[3],
                "symbol": key[4],
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
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["episode", "venue", "root", "quote"]).reset_index(drop=True)


def build_venue_summary(comparison: pd.DataFrame, resilience: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()

    cmp_agg = (
        comparison.groupby("venue", as_index=False)
        .agg(
            n_root_episode_pairs=("root", "count"),
            usdc_preferred_count=("preferred_quote_on_large", lambda s: int((s == "USDC").sum())),
            usdt_preferred_count=("preferred_quote_on_large", lambda s: int((s == "USDT").sum())),
            mean_delta_usdc_minus_usdt_bps=("impact_large_delta_usdc_minus_usdt_bps", "mean"),
        )
        .sort_values("venue")
        .reset_index(drop=True)
    )

    if resilience.empty:
        return cmp_agg

    res_agg = (
        resilience.groupby(["venue", "quote"], as_index=False)
        .agg(
            median_recovery_bars=("recovery_median_bars", "median"),
            mean_unrecovered_ratio=("unrecovered_ratio", "mean"),
        )
    )
    res_pivot = res_agg.pivot(index="venue", columns="quote")
    res_pivot.columns = [f"{metric}_{quote.lower()}" for metric, quote in res_pivot.columns]
    res_pivot = res_pivot.reset_index()

    merged = cmp_agg.merge(res_pivot, on="venue", how="left")
    return merged.sort_values("venue").reset_index(drop=True)


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
    parser.add_argument("--output-dir", default="reports/final", help="Output folder for execution diagnostics.")
    parser.add_argument("--size-window", type=int, default=60, help="Rolling window for relative size proxy.")
    parser.add_argument("--min-size-periods", type=int, default=20, help="Min periods for rolling size baseline.")
    parser.add_argument("--resilience-horizon", type=int, default=60, help="Forward bars to look for recovery.")
    parser.add_argument("--shock-quantile", type=float, default=0.99, help="Shock threshold quantile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    )
    slippage = build_slippage_proxy(enriched)
    comparison = build_cross_quote_comparison(slippage)
    resilience = build_resilience_table(
        enriched,
        horizon_bars=args.resilience_horizon,
        shock_quantile=args.shock_quantile,
    )
    venue_summary = build_venue_summary(comparison, resilience)

    slippage_path = out_dir / "execution_slippage_proxy.csv"
    comparison_path = out_dir / "execution_cross_quote_comparison.csv"
    resilience_path = out_dir / "execution_resilience.csv"
    venue_path = out_dir / "execution_venue_comparison.csv"
    report_path = out_dir / "execution_quality_report.md"

    slippage.to_csv(slippage_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    resilience.to_csv(resilience_path, index=False)
    venue_summary.to_csv(venue_path, index=False)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Execution Quality Diagnostics\n\n")
        handle.write("This report extends stablecoin analysis with execution-quality proxies.\n\n")
        handle.write("## Scope\n\n")
        for ep in args.episodes:
            handle.write(f"- `{ep}`\n")
        if missing:
            handle.write("\nMissing episodes (skipped):\n")
            for ep in missing:
                handle.write(f"- `{ep}`\n")

        handle.write("\n## Method Notes\n\n")
        handle.write("- Data source: `prices_resampled.csv` (price + volume bars).\n")
        handle.write("- Slippage proxy: next-bar absolute return (bps) conditioned on relative size (`volume / rolling median volume`).\n")
        handle.write("- Large-size bucket: top 10% relative-size bars per symbol.\n")
        handle.write("- Resilience proxy: after shock bars (`abs_ret_bps` >= quantile), bars to return to median absolute-return baseline.\n")
        handle.write("- Limitation: this is **not** order-book snapshot slippage/depth; it is a trade-bar proxy given available data.\n")

        if not comparison.empty:
            handle.write("\n## Cross-Quote Comparison (USDC vs USDT)\n\n")
            handle.write("```text\n")
            handle.write(comparison.to_string(index=False))
            handle.write("\n```\n")
            usdc_better = int((comparison["preferred_quote_on_large"] == "USDC").sum())
            usdt_better = int((comparison["preferred_quote_on_large"] == "USDT").sum())
            total = int(comparison.shape[0])
            handle.write(f"\n- USDC preferred on large-size proxy impact: `{usdc_better}/{total}`\n")
            handle.write(f"- USDT preferred on large-size proxy impact: `{usdt_better}/{total}`\n")

        if not resilience.empty:
            handle.write("\n## Resilience Summary\n\n")
            handle.write("```text\n")
            handle.write(resilience.to_string(index=False))
            handle.write("\n```\n")

        if not venue_summary.empty:
            handle.write("\n## Venue Summary\n\n")
            handle.write("```text\n")
            handle.write(venue_summary.to_string(index=False))
            handle.write("\n```\n")

        handle.write("\n## Artifacts\n\n")
        handle.write(f"- `{slippage_path}`\n")
        handle.write(f"- `{comparison_path}`\n")
        handle.write(f"- `{resilience_path}`\n")
        handle.write(f"- `{venue_path}`\n")

    print("Execution quality diagnostics completed.")
    print(f"- slippage_proxy: {slippage_path}")
    print(f"- cross_quote: {comparison_path}")
    print(f"- resilience: {resilience_path}")
    print(f"- venue_summary: {venue_path}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
