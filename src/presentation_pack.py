from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EPISODES = [
    "bybit_usdc_depeg_2023",
    "okx_usdc_depeg_2023",
    "march_vol_2024_binance",
    "yen_unwind_2024_binance",
    "yen_followthrough_2024_binance",
]


def _read_metrics_for_episode(episode: str, reports_root: Path) -> pd.DataFrame | None:
    metrics_path = reports_root / "episodes" / episode / "tables" / "metrics.csv"
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path, index_col=0)
    frame["episode"] = episode
    frame["variant"] = frame.index
    frame = frame.reset_index(drop=True)
    return frame


def _read_onchain_snapshot(episode: str, reports_root: Path) -> dict[str, Any] | None:
    signal_path = reports_root / "episodes" / episode / "tables" / "signal_frame.csv"
    if not signal_path.exists():
        return None
    frame = pd.read_csv(signal_path)
    needed = {
        "onchain_proxy",
        "onchain_divergence",
        "onchain_depeg_flag",
        "depeg_flag",
        "onchain_usdc_minus_1",
        "onchain_usdt_minus_1",
    }
    if not needed.issubset(set(frame.columns)):
        return None
    source_ts = (
        pd.to_datetime(frame["onchain_source_timestamp_utc"], utc=True, errors="coerce")
        if "onchain_source_timestamp_utc" in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    )
    source_age = (
        pd.to_numeric(frame["onchain_source_age_hours"], errors="coerce")
        if "onchain_source_age_hours" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    stress_col = (
        frame["stress_source"].astype(str)
        if "stress_source" in frame.columns
        else pd.Series("", index=frame.index, dtype="object")
    )
    decision_col = (
        frame["decision"].astype(str)
        if "decision" in frame.columns
        else pd.Series("", index=frame.index, dtype="object")
    )
    confidence = (
        pd.to_numeric(frame["confidence_score"], errors="coerce")
        if "confidence_score" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    size = (
        pd.to_numeric(frame["position_size"], errors="coerce")
        if "position_size" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    trade_mask = decision_col.eq("Trade")
    return {
        "episode": episode,
        "onchain_data_ratio": float(pd.to_numeric(frame["onchain_proxy"], errors="coerce").notna().mean()),
        "onchain_usdc_minus_1_abs_mean": float(
            pd.to_numeric(frame["onchain_usdc_minus_1"], errors="coerce").abs().mean(skipna=True)
        ),
        "onchain_usdt_minus_1_abs_mean": float(
            pd.to_numeric(frame["onchain_usdt_minus_1"], errors="coerce").abs().mean(skipna=True)
        ),
        "onchain_divergence_abs_mean": float(
            pd.to_numeric(frame["onchain_divergence"], errors="coerce").abs().mean(skipna=True)
        ),
        "onchain_depeg_count": int(pd.to_numeric(frame["onchain_depeg_flag"], errors="coerce").fillna(0).astype(bool).sum()),
        "combined_depeg_count": int(pd.to_numeric(frame["depeg_flag"], errors="coerce").fillna(0).astype(bool).sum()),
        "onchain_source_timestamp_ratio": float(source_ts.notna().mean()),
        "onchain_source_age_hours_median": float(source_age.median(skipna=True)),
        "stress_usdc_depeg_count": int((stress_col == "usdc_depeg_stress").sum()),
        "stress_usdt_concern_count": int((stress_col == "usdt_backing_concern").sum()),
        "stress_technical_flow_count": int((stress_col == "technical_flow_imbalance").sum()),
        "avg_trade_confidence": float(confidence.loc[trade_mask].mean(skipna=True)) if bool(trade_mask.any()) else 0.0,
        "avg_trade_position_size": float(size.loc[trade_mask].mean(skipna=True)) if bool(trade_mask.any()) else 0.0,
    }


def _read_proxy_coverage(episode: str, reports_root: Path) -> dict[str, Any] | None:
    proxy_path = reports_root / "episodes" / episode / "tables" / "stablecoin_proxy_components.csv"
    if not proxy_path.exists():
        return None
    frame = pd.read_csv(proxy_path, nrows=1)
    cols = [
        c
        for c in frame.columns
        if c not in {"timestamp_utc", "Unnamed: 0"} and not str(c).startswith("Unnamed:")
    ]
    return {"episode": episode, "proxy_component_count": int(len(cols))}


def _read_pnl_localization(
    episode: str,
    variant: str,
    reports_root: Path,
    *,
    window_bars: int = 10,
) -> dict[str, Any] | None:
    path = reports_root / "episodes" / episode / "tables" / f"trade_log_{variant}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "net_pnl" not in frame.columns:
        return None

    pnl = pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0.0)
    net = float(pnl.sum())
    abs_sum = float(pnl.abs().sum())
    positive_sum = float(pnl[pnl > 0].sum())
    abs_sorted = pnl.abs().sort_values(ascending=False)
    top1_share = float(abs_sorted.head(1).sum() / abs_sum) if abs_sum > 0 else 0.0
    top3_share = float(abs_sorted.head(3).sum() / abs_sum) if abs_sum > 0 else 0.0
    top5_share = float(abs_sorted.head(5).sum() / abs_sum) if abs_sum > 0 else 0.0

    best_window_sum = np.nan
    best_window_share_of_net = np.nan
    best_window_share_of_positive = np.nan
    best_window_start = None
    best_window_end = None
    w = max(2, int(window_bars))
    if len(pnl) >= w:
        rolling = pnl.rolling(w).sum()
        best_idx = int(rolling.idxmax())
        best_window_sum = float(rolling.iloc[best_idx])
        if net > 0:
            best_window_share_of_net = float(best_window_sum / net)
        if positive_sum > 0:
            best_window_share_of_positive = float(best_window_sum / positive_sum)
        if "timestamp_utc" in frame.columns:
            best_window_start = frame.iloc[best_idx - w + 1]["timestamp_utc"]
            best_window_end = frame.iloc[best_idx]["timestamp_utc"]

    localized_positive = bool(
        net > 1e-3
        and np.isfinite(best_window_share_of_net)
        and best_window_share_of_net >= 0.5
    )
    return {
        "episode": episode,
        "variant": variant,
        "net_pnl": net,
        "abs_pnl_sum": abs_sum,
        "positive_pnl_sum": positive_sum,
        "top1_abs_pnl_share": top1_share,
        "top3_abs_pnl_share": top3_share,
        "top5_abs_pnl_share": top5_share,
        "best_window_bars": w,
        "best_window_sum_pnl": best_window_sum,
        "best_window_share_of_net_pnl": best_window_share_of_net,
        "best_window_share_of_positive_pnl": best_window_share_of_positive,
        "best_window_start": best_window_start,
        "best_window_end": best_window_end,
        "localized_positive_pnl_flag": localized_positive,
    }


def _plot_metric_comparison(wide: pd.DataFrame, metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = list(wide.index)
    naive = wide.get(f"{metric}_naive", pd.Series(np.nan, index=wide.index))
    gated = wide.get(f"{metric}_gated", pd.Series(np.nan, index=wide.index))

    x = np.arange(len(episodes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.bar(x - width / 2, naive.values, width, label="naive", color="#7f7f7f", alpha=0.8)
    ax.bar(x + width / 2, gated.values, width, label="gated", color="#003f5c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(episodes, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Naive vs Gated - {metric}")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final polished presentation pack.")
    parser.add_argument("--reports-root", default="reports", help="Root folder containing episode outputs.")
    parser.add_argument("--output-dir", default="reports/final", help="Output folder for final pack.")
    parser.add_argument("--episodes", nargs="+", default=DEFAULT_EPISODES, help="Episode ids to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_frames: list[pd.DataFrame] = []
    onchain_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    for episode in args.episodes:
        frame = _read_metrics_for_episode(episode, reports_root)
        if frame is not None:
            metric_frames.append(frame)
        onchain = _read_onchain_snapshot(episode, reports_root)
        if onchain is not None:
            onchain_rows.append(onchain)
        proxy = _read_proxy_coverage(episode, reports_root)
        if proxy is not None:
            proxy_rows.append(proxy)
        for variant in ("naive", "gated"):
            loc = _read_pnl_localization(episode, variant, reports_root)
            if loc is not None:
                localization_rows.append(loc)

    if not metric_frames:
        raise FileNotFoundError("No episode metrics found for selected episodes.")

    long_metrics = pd.concat(metric_frames, ignore_index=True)
    base_cols = [
        "episode",
        "variant",
        "sharpe",
        "pnl_net",
        "max_drawdown",
        "turnover",
        "flip_rate",
        "active_ratio",
        "hit_rate",
        "n_bars",
        "n_active_bars",
        "horizon_days",
        "avg_active_position_size",
        "edge_net_trade_median_bps",
        "edge_net_trade_p10_bps",
        "optimal_size_trade_mean",
        "break_even_premium_median_bps",
        "cost_bps_applied_mean",
    ]
    optional_cols = [
        "sharpe_full_annualized",
        "sharpe_active",
        "sharpe_active_annualized",
        "annualization_factor",
    ]
    selected_cols = [col for col in base_cols + optional_cols if col in long_metrics.columns]
    long_metrics = long_metrics[selected_cols]
    long_path = output_dir / "final_episode_metrics_long.csv"
    long_metrics.to_csv(long_path, index=False)

    pivot = long_metrics.pivot(index="episode", columns="variant")
    pivot.columns = [f"{metric}_{variant}" for metric, variant in pivot.columns]
    pivot = pivot.reset_index()
    wide_path = output_dir / "final_episode_metrics_wide.csv"
    pivot.to_csv(wide_path, index=False)

    wide_for_plot = pivot.set_index("episode")
    _plot_metric_comparison(wide_for_plot, "sharpe", figures_dir / "sharpe_naive_vs_gated.png")
    _plot_metric_comparison(wide_for_plot, "pnl_net", figures_dir / "pnl_naive_vs_gated.png")
    _plot_metric_comparison(wide_for_plot, "flip_rate", figures_dir / "fliprate_naive_vs_gated.png")

    onchain_df = pd.DataFrame(onchain_rows).sort_values("episode") if onchain_rows else pd.DataFrame()
    proxy_df = pd.DataFrame(proxy_rows).sort_values("episode") if proxy_rows else pd.DataFrame()
    onchain_path = output_dir / "final_onchain_snapshot.csv"
    if not onchain_df.empty:
        onchain_df.to_csv(onchain_path, index=False)
    proxy_path = output_dir / "final_proxy_coverage.csv"
    if not proxy_df.empty:
        proxy_df.to_csv(proxy_path, index=False)
    localization_df = pd.DataFrame(localization_rows).sort_values(["episode", "variant"]) if localization_rows else pd.DataFrame()
    localization_path = output_dir / "final_pnl_localization.csv"
    if not localization_df.empty:
        localization_df.to_csv(localization_path, index=False)

    calibration_details_path = output_dir / "calibration_details.csv"
    calibration_agg_path = output_dir / "calibration_aggregate.csv"
    calibration_exists = calibration_details_path.exists() and calibration_agg_path.exists()
    execution_report_path = output_dir / "execution_quality_report.md"
    execution_slippage_path = output_dir / "execution_slippage_proxy.csv"
    edge_net_curve_path = output_dir / "edge_net_size_curve.csv"
    break_even_curve_path = output_dir / "break_even_premium_curve.csv"
    edge_net_summary_path = output_dir / "edge_net_summary.csv"
    edge_net_figure_path = output_dir / "figures" / "figure_4_edge_net.png"
    execution_comparison_path = output_dir / "execution_cross_quote_comparison.csv"
    execution_resilience_path = output_dir / "execution_resilience.csv"
    execution_venue_path = output_dir / "execution_venue_comparison.csv"
    execution_l2_coverage_path = output_dir / "execution_l2_coverage.csv"
    execution_venue_df = (
        pd.read_csv(execution_venue_path).sort_values(["venue", "market_type"])
        if execution_venue_path.exists()
        else pd.DataFrame()
    )
    execution_l2_coverage_df = (
        pd.read_csv(execution_l2_coverage_path).sort_values("episode")
        if execution_l2_coverage_path.exists()
        else pd.DataFrame()
    )
    localized_naive_episodes: set[str] = set()
    if not localization_df.empty:
        localized_naive_episodes = set(
            localization_df.loc[
                (localization_df["variant"] == "naive")
                & localization_df["localized_positive_pnl_flag"].astype(bool),
                "episode",
            ].astype(str).tolist()
        )
    robust_filter_df = pd.DataFrame(
        {
            "episode": list(wide_for_plot.index.astype(str)),
            "exclude_from_robust_aggregate": [ep in localized_naive_episodes for ep in wide_for_plot.index.astype(str)],
            "exclude_reason": [
                "localized_naive_positive_pnl" if ep in localized_naive_episodes else ""
                for ep in wide_for_plot.index.astype(str)
            ],
        }
    )
    robust_filter_path = output_dir / "final_robust_filter.csv"
    robust_filter_df.to_csv(robust_filter_path, index=False)

    md_path = output_dir / "executive_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Executive Summary\n\n")
        handle.write("This report consolidates final episode performance and on-chain diagnostics.\n\n")

        handle.write("## Included Episodes\n\n")
        for ep in args.episodes:
            handle.write(f"- `{ep}`\n")

        handle.write("\n## Performance Snapshot (Gated)\n\n")
        gated = long_metrics[long_metrics["variant"] == "gated"].copy()
        handle.write("```text\n")
        handle.write(gated.to_string(index=False))
        handle.write("\n```\n")
        handle.write(
            "\nMetric convention: `sharpe` is full-series and non-annualized. "
            "Annualized Sharpe columns are exported for reference only.\n"
        )

        if "sharpe_gated" in wide_for_plot.columns and "sharpe_naive" in wide_for_plot.columns:
            sharpe_delta = wide_for_plot["sharpe_gated"] - wide_for_plot["sharpe_naive"]
            sharpe_valid = sharpe_delta.dropna()
            if not sharpe_valid.empty:
                sharpe_up = int((sharpe_valid > 0).sum())
                sharpe_down = int((sharpe_valid < 0).sum())
                sharpe_total = int(sharpe_valid.shape[0])
                handle.write(
                    f"\n- Raw mean Sharpe delta (gated - naive, full-series non-annualized): "
                    f"`{float(sharpe_valid.mean()):.4f}`\n"
                )
                handle.write(
                    f"- Raw median Sharpe delta (gated - naive, full-series non-annualized): "
                    f"`{float(sharpe_valid.median()):.4f}`\n"
                )
                handle.write(f"- Raw episodes with Sharpe improvement: `{sharpe_up}/{sharpe_total}`\n")
                handle.write(f"- Raw episodes with Sharpe degradation: `{sharpe_down}/{sharpe_total}`\n")

                sharpe_eval = sharpe_valid
                if localized_naive_episodes:
                    sharpe_eval = sharpe_valid.loc[~sharpe_valid.index.isin(localized_naive_episodes)]
                    excluded = sorted(set(sharpe_valid.index) - set(sharpe_eval.index))
                    if excluded:
                        handle.write(
                            f"- Robust Sharpe aggregate excludes localized naive episodes: `{excluded}`\n"
                        )
                        handle.write(
                            f"- Robust mean Sharpe delta (gated - naive): "
                            f"`{float(sharpe_eval.mean()):.4f}`\n"
                        )
                        handle.write(
                            f"- Robust median Sharpe delta (gated - naive): "
                            f"`{float(sharpe_eval.median()):.4f}`\n"
                        )

                if float(sharpe_eval.mean()) > 0:
                    handle.write("- Conclusion (Sharpe): gated improvement is positive on robust aggregate.\n")
                else:
                    handle.write(
                        "- Conclusion (Sharpe): gated improvement is **not** demonstrated on robust aggregate.\n"
                    )

        if "pnl_net_gated" in wide_for_plot.columns and "pnl_net_naive" in wide_for_plot.columns:
            pnl_delta = wide_for_plot["pnl_net_gated"] - wide_for_plot["pnl_net_naive"]
            pnl_valid = pnl_delta.dropna()
            if not pnl_valid.empty:
                pnl_up = int((pnl_valid > 0).sum())
                pnl_down = int((pnl_valid < 0).sum())
                pnl_total = int(pnl_valid.shape[0])
                handle.write(
                    f"- Raw mean PnL delta (gated - naive): `{float(pnl_valid.mean()):.6f}`\n"
                )
                handle.write(
                    f"- Raw median PnL delta (gated - naive): `{float(pnl_valid.median()):.6f}`\n"
                )
                handle.write(f"- Raw episodes with PnL improvement: `{pnl_up}/{pnl_total}`\n")
                handle.write(f"- Raw episodes with PnL degradation: `{pnl_down}/{pnl_total}`\n")

                pnl_eval = pnl_valid
                if localized_naive_episodes:
                    pnl_eval = pnl_valid.loc[~pnl_valid.index.isin(localized_naive_episodes)]
                    excluded = sorted(set(pnl_valid.index) - set(pnl_eval.index))
                    if excluded:
                        handle.write(
                            f"- Robust PnL aggregate excludes localized naive episodes: `{excluded}`\n"
                        )
                        handle.write(
                            f"- Robust mean PnL delta (gated - naive): "
                            f"`{float(pnl_eval.mean()):.6f}`\n"
                        )
                        handle.write(
                            f"- Robust median PnL delta (gated - naive): "
                            f"`{float(pnl_eval.median()):.6f}`\n"
                        )

                if float(pnl_eval.mean()) > 0:
                    handle.write("- Conclusion (PnL): gated improvement is positive on robust aggregate.\n")
                else:
                    handle.write(
                        "- Conclusion (PnL): gated improvement is **not** demonstrated on robust aggregate.\n"
                    )

        if not onchain_df.empty:
            handle.write("\n## On-Chain Validation Snapshot\n\n")
            handle.write("```text\n")
            handle.write(onchain_df.to_string(index=False))
            handle.write("\n```\n")

        if not proxy_df.empty:
            handle.write("\n## Proxy Coverage Notes\n\n")
            handle.write("```text\n")
            handle.write(proxy_df.to_string(index=False))
            handle.write("\n```\n")
            handle.write(
                "\nInterpretation: debiased premium is strongest when proxy_component_count > 0. "
                "When coverage is missing, treat the episode primarily as depeg safety/on-chain validation.\n"
            )

        if not localization_df.empty:
            handle.write("\n## PnL Localization Diagnostics\n\n")
            show_cols = [
                "episode",
                "variant",
                "net_pnl",
                "top1_abs_pnl_share",
                "top3_abs_pnl_share",
                "top5_abs_pnl_share",
                "best_window_bars",
                "best_window_share_of_net_pnl",
                "best_window_share_of_positive_pnl",
                "localized_positive_pnl_flag",
                "best_window_start",
                "best_window_end",
            ]
            show_cols = [c for c in show_cols if c in localization_df.columns]
            handle.write("```text\n")
            handle.write(localization_df[show_cols].to_string(index=False))
            handle.write("\n```\n")
            naive_loc = localization_df[localization_df["variant"] == "naive"].copy()
            naive_pos = naive_loc[naive_loc["net_pnl"] > 0]
            localized_pos = naive_pos["localized_positive_pnl_flag"].astype(bool).sum()
            total_pos = int(naive_pos.shape[0])
            if total_pos > 0:
                handle.write(
                    f"\n- Naive positive-PnL episodes with >50% of net PnL explained by one "
                    f"`{int(localization_df['best_window_bars'].iloc[0])}`-bar window: "
                    f"`{int(localized_pos)}/{total_pos}`.\n"
                )
            handle.write(
                "Interpretation: when `localized_positive_pnl_flag` is true, performance is structurally fragile "
                "and should not be treated as robust signal quality.\n"
            )
            handle.write(f"Robust aggregate exclusion map is exported to: `{robust_filter_path}`.\n")

        if not execution_l2_coverage_df.empty:
            handle.write("\n## Execution Data Readiness (L2)\n\n")
            handle.write("```text\n")
            handle.write(execution_l2_coverage_df.to_string(index=False))
            handle.write("\n```\n")
            ready = execution_l2_coverage_df["l2_ready"].astype(bool)
            if not bool(ready.all()):
                handle.write(
                    "L2 coverage is incomplete for at least one episode. "
                    "Execution tables below are lightweight proxies and should be treated as descriptive.\n"
                )

        if not execution_venue_df.empty:
            handle.write("\n## Execution Proxy Snapshot (Bar-Level)\n\n")
            handle.write("```text\n")
            handle.write(execution_venue_df.to_string(index=False))
            handle.write("\n```\n")
            handle.write(
                "\nInterpretation: compare raw, excess, and normalized deltas jointly. "
                "A negative delta means lower proxy impact in USDC quotes versus USDT for the same root/venue.\n"
            )
            handle.write(
                "Scope note: this section is a bar-level proxy and does not validate order-book "
                "microstructure items from the Mike brief.\n"
            )
            handle.write(
                "Comparability note: venue comparisons are only defensible within the same `market_type` "
                "(`spot` vs `derivatives`).\n"
            )
            handle.write(
                "Decision guardrail: do not conclude 'better liquidity' without L2 order-book replay "
                "(book-walk), and normalization of tick/lot/fees/funding/contract specs.\n"
            )
        elif not execution_l2_coverage_df.empty:
            handle.write(
                "\n## Execution Proxy Snapshot (Bar-Level)\n\n"
                "No proxy table is reported because fail-closed mode blocked execution diagnostics "
                "without full L2 readiness.\n"
            )

        handle.write("\n## Generated Artifacts\n\n")
        handle.write(f"- `{long_path}`\n")
        handle.write(f"- `{wide_path}`\n")
        handle.write(f"- `{figures_dir / 'sharpe_naive_vs_gated.png'}`\n")
        handle.write(f"- `{figures_dir / 'pnl_naive_vs_gated.png'}`\n")
        handle.write(f"- `{figures_dir / 'fliprate_naive_vs_gated.png'}`\n")
        if not onchain_df.empty:
            handle.write(f"- `{onchain_path}`\n")
        if not proxy_df.empty:
            handle.write(f"- `{proxy_path}`\n")
        if not localization_df.empty:
            handle.write(f"- `{localization_path}`\n")
        if execution_report_path.exists():
            handle.write(f"- `{execution_report_path}`\n")
        handle.write(f"- `{robust_filter_path}`\n")
        if execution_l2_coverage_path.exists():
            handle.write(f"- `{execution_l2_coverage_path}`\n")
        if execution_slippage_path.exists():
            handle.write(f"- `{execution_slippage_path}`\n")
        if edge_net_curve_path.exists():
            handle.write(f"- `{edge_net_curve_path}`\n")
        if break_even_curve_path.exists():
            handle.write(f"- `{break_even_curve_path}`\n")
        if edge_net_summary_path.exists():
            handle.write(f"- `{edge_net_summary_path}`\n")
        if edge_net_figure_path.exists():
            handle.write(f"- `{edge_net_figure_path}`\n")
        if execution_comparison_path.exists():
            handle.write(f"- `{execution_comparison_path}`\n")
        if execution_resilience_path.exists():
            handle.write(f"- `{execution_resilience_path}`\n")
        if execution_venue_path.exists():
            handle.write(f"- `{execution_venue_path}`\n")

        if calibration_exists:
            handle.write(f"- `{calibration_details_path}`\n")
            handle.write(f"- `{calibration_agg_path}`\n")
            handle.write(
                "\nCalibration outputs are available and can be referenced directly in the deck.\n"
            )

    print("Presentation pack completed.")
    print(f"- metrics_long: {long_path}")
    print(f"- metrics_wide: {wide_path}")
    if not onchain_df.empty:
        print(f"- onchain_snapshot: {onchain_path}")
    print(f"- summary: {md_path}")


if __name__ == "__main__":
    main()
