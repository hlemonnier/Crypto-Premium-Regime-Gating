from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.ablation_core import build_dataclass, compute_core_frames, simple_decision
from src.backtest import BacktestConfig, run_backtest, run_naive_baseline
from src.hawkes import HawkesConfig, estimate_hawkes_rolling, evaluate_hawkes_quality
from src.pipeline import load_config, load_price_matrix
from src.strategy import StrategyConfig, build_decisions

VARIANT_ORDER = ["naive", "debias_only", "plus_robust", "plus_regime", "plus_hawkes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation stack: debias only / +robust / +regime / +hawkes."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config path.")
    parser.add_argument("--price-matrix", default=None, help="Optional matrix override.")
    parser.add_argument(
        "--output-dir",
        default="reports/tables",
        help="Directory for ablation metrics and trade logs.",
    )
    parser.add_argument(
        "--skip-hawkes",
        action="store_true",
        help="Skip the +hawkes variant.",
    )
    parser.add_argument(
        "--proxy-methods",
        nargs="+",
        choices=["median", "pw_rolling"],
        default=None,
        help=(
            "Optional premium proxy method override(s). "
            "Pass both to evaluate side-by-side: --proxy-methods median pw_rolling."
        ),
    )
    return parser.parse_args()


def _run_ablation_once(
    config: dict[str, Any],
    matrix: pd.DataFrame,
    *,
    skip_hawkes: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = compute_core_frames(config, matrix, premium_leg="debiased")
    premium_frame = frames["premium_frame"]
    onchain_frame = frames["onchain_frame"]
    robust_frame = frames["robust_frame"]
    state_frame = frames["state_frame"]
    regime_frame = frames["regime_frame"]
    m_t = frames["m_t"]
    freq = str(frames["freq"].iloc[0])

    strategy_cfg = build_dataclass(StrategyConfig, config.get("strategy"))
    backtest_cfg = build_dataclass(BacktestConfig, config.get("backtest"))

    metrics_rows: list[dict[str, float]] = []
    logs: dict[str, pd.DataFrame] = {}

    naive_decision = run_naive_baseline(
        premium_frame["p_naive"],
        threshold=backtest_cfg.naive_threshold,
    )["decision"]
    naive_log, naive_metrics = run_backtest(
        premium_frame["p_naive"],
        naive_decision,
        premium_frame["p_naive"],
        freq=freq,
        cost_bps=backtest_cfg.cost_bps,
        position_mode=backtest_cfg.position_mode,
        exit_on_widen=backtest_cfg.exit_on_widen,
        exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
        min_holding_bars=backtest_cfg.min_holding_bars,
        max_holding_bars=backtest_cfg.max_holding_bars,
    )
    metrics_rows.append({"variant": "naive", **naive_metrics})
    logs["naive"] = naive_log

    debias_decision = simple_decision(
        premium_frame["p"],
        threshold=backtest_cfg.naive_threshold,
        depeg_flag=premium_frame["depeg_flag"],
    )
    debias_log, debias_metrics = run_backtest(
        premium_frame["p"],
        debias_decision,
        premium_frame["p"],
        freq=freq,
        cost_bps=backtest_cfg.cost_bps,
        position_mode=backtest_cfg.position_mode,
        exit_on_widen=backtest_cfg.exit_on_widen,
        exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
        min_holding_bars=backtest_cfg.min_holding_bars,
        max_holding_bars=backtest_cfg.max_holding_bars,
    )
    metrics_rows.append({"variant": "debias_only", **debias_metrics})
    logs["debias_only"] = debias_log

    robust_threshold = (strategy_cfg.entry_k * robust_frame["sigma_hat"]).rename("robust_threshold")
    robust_decision = simple_decision(
        robust_frame["p_smooth"],
        threshold=robust_threshold,
        depeg_flag=premium_frame["depeg_flag"],
    )
    robust_log, robust_metrics = run_backtest(
        robust_frame["p_smooth"],
        robust_decision,
        robust_frame["p_smooth"],
        freq=freq,
        cost_bps=backtest_cfg.cost_bps,
        position_mode=backtest_cfg.position_mode,
        exit_on_widen=backtest_cfg.exit_on_widen,
        exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
        min_holding_bars=backtest_cfg.min_holding_bars,
        max_holding_bars=backtest_cfg.max_holding_bars,
    )
    metrics_rows.append({"variant": "plus_robust", **robust_metrics})
    logs["plus_robust"] = robust_log

    regime_decision_frame = build_decisions(
        m_t=m_t,
        T_t=state_frame["T_t"],
        chi_t=state_frame["chi_t"],
        sigma_hat=robust_frame["sigma_hat"],
        regime=regime_frame["regime"],
        depeg_flag=premium_frame["depeg_flag"],
        event=robust_frame["event"],
        stablecoin_proxy=premium_frame["stablecoin_proxy"],
        onchain_proxy=onchain_frame.get("onchain_proxy"),
        onchain_usdc_minus_1=onchain_frame.get("onchain_usdc_minus_1"),
        onchain_usdt_minus_1=onchain_frame.get("onchain_usdt_minus_1"),
        n_t=None,
        cfg=strategy_cfg,
    )
    regime_log, regime_metrics = run_backtest(
        premium_frame["p"],
        regime_decision_frame["decision"],
        m_t,
        freq=freq,
        cost_bps=backtest_cfg.cost_bps,
        position_size=regime_decision_frame.get("position_size"),
        position_mode=backtest_cfg.position_mode,
        exit_on_widen=backtest_cfg.exit_on_widen,
        exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
        min_holding_bars=backtest_cfg.min_holding_bars,
        max_holding_bars=backtest_cfg.max_holding_bars,
    )
    metrics_rows.append({"variant": "plus_regime", **regime_metrics})
    logs["plus_regime"] = regime_log

    if not skip_hawkes:
        hawkes_cfg = build_dataclass(HawkesConfig, config.get("hawkes"))
        hawkes_cfg = replace(hawkes_cfg, enabled=True)
        hawkes_frame = estimate_hawkes_rolling(robust_frame["event"], hawkes_cfg)
        hawkes_quality_pass, _, _ = evaluate_hawkes_quality(hawkes_frame, hawkes_cfg)
        hawkes_decision_frame = build_decisions(
            m_t=m_t,
            T_t=state_frame["T_t"],
            chi_t=state_frame["chi_t"],
            sigma_hat=robust_frame["sigma_hat"],
            regime=regime_frame["regime"],
            depeg_flag=premium_frame["depeg_flag"],
            event=robust_frame["event"],
            stablecoin_proxy=premium_frame["stablecoin_proxy"],
            onchain_proxy=onchain_frame.get("onchain_proxy"),
            onchain_usdc_minus_1=onchain_frame.get("onchain_usdc_minus_1"),
            onchain_usdt_minus_1=onchain_frame.get("onchain_usdt_minus_1"),
            n_t=hawkes_frame["n_t"] if (hawkes_quality_pass and "n_t" in hawkes_frame) else None,
            cfg=strategy_cfg,
        )
        hawkes_log, hawkes_metrics = run_backtest(
            premium_frame["p"],
            hawkes_decision_frame["decision"],
            m_t,
            freq=freq,
            cost_bps=backtest_cfg.cost_bps,
            position_size=hawkes_decision_frame.get("position_size"),
            position_mode=backtest_cfg.position_mode,
            exit_on_widen=backtest_cfg.exit_on_widen,
            exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
            min_holding_bars=backtest_cfg.min_holding_bars,
            max_holding_bars=backtest_cfg.max_holding_bars,
        )
        metrics_rows.append({"variant": "plus_hawkes", **hawkes_metrics})
        logs["plus_hawkes"] = hawkes_log

    metrics = pd.DataFrame(metrics_rows).set_index("variant")
    metrics = metrics.reindex([v for v in VARIANT_ORDER if v in metrics.index])
    return metrics, logs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    matrix_path = args.price_matrix or config.get("data", {}).get("price_matrix_path")
    if not matrix_path:
        raise ValueError("No price matrix path configured. Use --price-matrix or set data.price_matrix_path.")

    matrix = load_price_matrix(matrix_path)
    configured_method = str(config.get("premium", {}).get("proxy_method", "median")).strip().lower()
    proxy_methods = args.proxy_methods or [configured_method]
    proxy_methods = [str(method).strip().lower() for method in proxy_methods]
    if len(proxy_methods) == 0:
        proxy_methods = ["median"]
    invalid = sorted({method for method in proxy_methods if method not in {"median", "pw_rolling"}})
    if invalid:
        raise ValueError(f"Unsupported proxy methods: {invalid}. Use median and/or pw_rolling.")

    method_runs: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]] = []
    for method in proxy_methods:
        run_cfg = deepcopy(config)
        run_cfg.setdefault("premium", {})
        run_cfg["premium"]["proxy_method"] = method
        metrics, logs = _run_ablation_once(run_cfg, matrix, skip_hawkes=args.skip_hawkes)
        method_runs.append((method, metrics, logs))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "ablation_metrics.csv"
    if len(method_runs) == 1:
        method, metrics, logs = method_runs[0]
        metrics.to_csv(metrics_path, index=True)
        for variant, log in logs.items():
            log.to_csv(out_dir / f"ablation_trade_log_{variant}.csv", index=True)
        metrics_display = metrics
    else:
        combined_rows: list[pd.DataFrame] = []
        for method, metrics, _ in method_runs:
            block = metrics.reset_index().rename(columns={"index": "variant"})
            block.insert(0, "proxy_method", method)
            combined_rows.append(block)
        metrics_multi = pd.concat(combined_rows, axis=0, ignore_index=True)
        metrics_multi["variant"] = pd.Categorical(metrics_multi["variant"], categories=VARIANT_ORDER, ordered=True)
        metrics_multi = metrics_multi.sort_values(["proxy_method", "variant"])
        metrics_display = metrics_multi.set_index(["proxy_method", "variant"])
        metrics_display.to_csv(metrics_path, index=True)

        for method, _, logs in method_runs:
            for variant, log in logs.items():
                log.to_csv(out_dir / f"ablation_trade_log_{method}_{variant}.csv", index=True)

    summary = out_dir / "ablation_summary.md"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("# Ablation Summary\n\n")
        handle.write(f"- price_matrix: `{matrix_path}`\n")
        handle.write(f"- config: `{args.config}`\n\n")
        handle.write(f"- proxy_methods: `{', '.join(proxy_methods)}`\n\n")
        handle.write("## Metrics\n\n")
        handle.write("```text\n")
        handle.write(metrics_display.to_string())
        handle.write("\n```\n")
        handle.write("\n")

    print("Ablation completed.")
    print(f"- metrics: {metrics_path}")
    print(f"- summary: {summary}")
    print(metrics_display)


if __name__ == "__main__":
    main()
