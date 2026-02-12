from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import BacktestConfig, run_backtest, run_naive_baseline
from src.hawkes import HawkesConfig, estimate_hawkes_rolling, evaluate_hawkes_quality
from src.onchain import OnchainConfig, build_onchain_validation_frame, empty_onchain_frame
from src.pipeline import load_config, load_price_matrix
from src.premium import PremiumConfig, build_premium_frame
from src.regimes import RegimeConfig, build_regime_frame
from src.robust_filter import RobustFilterConfig, build_robust_frame
from src.statmech import StatMechConfig, build_statmech_frame
from src.strategy import StrategyConfig, build_decisions

VARIANT_ORDER = ["naive", "debias_only", "plus_robust", "plus_regime", "plus_hawkes"]


def _build_dataclass(cls: Any, data: dict[str, Any] | None) -> Any:
    data = data or {}
    valid_fields = {field.name for field in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**kwargs)


def _simple_decision(
    signal: pd.Series,
    threshold: pd.Series | float,
    depeg_flag: pd.Series,
) -> pd.Series:
    decision = pd.Series("Widen", index=signal.index, dtype="object")
    trade = signal.abs().gt(threshold)
    decision.loc[trade] = "Trade"
    decision.loc[depeg_flag.astype(bool)] = "Risk-off"
    return decision.rename("decision")


def _compute_core_frames(config: dict[str, Any], matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    freq = config.get("data", {}).get("resample_rule", "1min")

    premium_cfg = _build_dataclass(PremiumConfig, config.get("premium"))
    premium_raw = config.get("premium", {})
    proxy_pairs_raw = premium_raw.get("proxy_pairs", [])
    proxy_pairs = [tuple(pair) for pair in proxy_pairs_raw] if proxy_pairs_raw else None
    premium_frame, proxy_components = build_premium_frame(
        matrix,
        premium_cfg,
        proxy_pairs=proxy_pairs,
        freq=freq,
    )

    onchain_cfg = _build_dataclass(OnchainConfig, config.get("onchain"))
    if onchain_cfg.enabled:
        try:
            onchain_frame = build_onchain_validation_frame(
                index=premium_frame.index,
                stablecoin_proxy=premium_frame["stablecoin_proxy"],
                cfg=onchain_cfg,
            )
        except Exception:
            onchain_frame = empty_onchain_frame(premium_frame.index)
    else:
        onchain_frame = empty_onchain_frame(premium_frame.index)
    market_depeg_flag = premium_frame["depeg_flag"].fillna(False).astype(bool).rename("market_depeg_flag")
    premium_frame["market_depeg_flag"] = market_depeg_flag
    onchain_effective = onchain_frame.get(
        "onchain_depeg_flag_effective",
        pd.Series(False, index=premium_frame.index, name="onchain_depeg_flag_effective"),
    )
    onchain_effective = onchain_effective.fillna(False).astype(bool)
    premium_frame["depeg_flag"] = (market_depeg_flag | onchain_effective).rename("depeg_flag")

    robust_cfg = _build_dataclass(RobustFilterConfig, config.get("robust_filter"))
    robust_frame = build_robust_frame(premium_frame["p"], cfg=robust_cfg, freq=freq)
    m_t = robust_frame["p_smooth"].rename("m_t")

    stat_cfg = _build_dataclass(StatMechConfig, config.get("statmech"))
    state_frame = build_statmech_frame(robust_frame["z_t"], m_t, cfg=stat_cfg, freq=freq)

    regime_cfg = _build_dataclass(RegimeConfig, config.get("regimes"))
    regime_frame = build_regime_frame(state_frame, regime_cfg)

    return {
        "premium_frame": premium_frame,
        "proxy_components": proxy_components,
        "onchain_frame": onchain_frame,
        "robust_frame": robust_frame,
        "state_frame": state_frame,
        "regime_frame": regime_frame,
        "m_t": m_t,
        "freq": pd.Series([freq]),
    }


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
    frames = _compute_core_frames(config, matrix)
    premium_frame = frames["premium_frame"]
    onchain_frame = frames["onchain_frame"]
    robust_frame = frames["robust_frame"]
    state_frame = frames["state_frame"]
    regime_frame = frames["regime_frame"]
    m_t = frames["m_t"]
    freq = str(frames["freq"].iloc[0])

    strategy_cfg = _build_dataclass(StrategyConfig, config.get("strategy"))
    backtest_cfg = _build_dataclass(BacktestConfig, config.get("backtest"))

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

    debias_decision = _simple_decision(
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
    robust_decision = _simple_decision(
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
        hawkes_cfg = _build_dataclass(HawkesConfig, config.get("hawkes"))
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
