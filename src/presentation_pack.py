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

    calibration_details_path = output_dir / "calibration_details.csv"
    calibration_agg_path = output_dir / "calibration_aggregate.csv"
    calibration_exists = calibration_details_path.exists() and calibration_agg_path.exists()
    execution_report_path = output_dir / "execution_quality_report.md"
    execution_slippage_path = output_dir / "execution_slippage_proxy.csv"
    execution_comparison_path = output_dir / "execution_cross_quote_comparison.csv"
    execution_resilience_path = output_dir / "execution_resilience.csv"
    execution_venue_path = output_dir / "execution_venue_comparison.csv"
    execution_venue_df = (
        pd.read_csv(execution_venue_path).sort_values("venue")
        if execution_venue_path.exists()
        else pd.DataFrame()
    )

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
                    f"\n- Mean Sharpe delta (gated - naive, full-series non-annualized): "
                    f"`{float(sharpe_valid.mean()):.4f}`\n"
                )
                handle.write(
                    f"- Median Sharpe delta (gated - naive, full-series non-annualized): "
                    f"`{float(sharpe_valid.median()):.4f}`\n"
                )
                handle.write(f"- Episodes with Sharpe improvement: `{sharpe_up}/{sharpe_total}`\n")
                handle.write(f"- Episodes with Sharpe degradation: `{sharpe_down}/{sharpe_total}`\n")

        if "pnl_net_gated" in wide_for_plot.columns and "pnl_net_naive" in wide_for_plot.columns:
            pnl_delta = wide_for_plot["pnl_net_gated"] - wide_for_plot["pnl_net_naive"]
            pnl_valid = pnl_delta.dropna()
            if not pnl_valid.empty:
                pnl_up = int((pnl_valid > 0).sum())
                pnl_down = int((pnl_valid < 0).sum())
                pnl_total = int(pnl_valid.shape[0])
                handle.write(
                    f"- Mean PnL delta (gated - naive): `{float(pnl_valid.mean()):.6f}`\n"
                )
                handle.write(
                    f"- Median PnL delta (gated - naive): `{float(pnl_valid.median()):.6f}`\n"
                )
                handle.write(f"- Episodes with PnL improvement: `{pnl_up}/{pnl_total}`\n")
                handle.write(f"- Episodes with PnL degradation: `{pnl_down}/{pnl_total}`\n")

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

        if not execution_venue_df.empty:
            handle.write("\n## Execution Quality Snapshot\n\n")
            handle.write("```text\n")
            handle.write(execution_venue_df.to_string(index=False))
            handle.write("\n```\n")
            handle.write(
                "\nInterpretation: lower `mean_delta_usdc_minus_usdt_bps` indicates lower large-size impact "
                "in USDC quotes versus USDT for the same root/venue.\n"
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
        if execution_report_path.exists():
            handle.write(f"- `{execution_report_path}`\n")
        if execution_slippage_path.exists():
            handle.write(f"- `{execution_slippage_path}`\n")
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
