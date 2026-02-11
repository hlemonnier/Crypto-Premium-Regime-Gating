from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import yaml

from src.backtest import BacktestConfig, compare_strategies, export_metrics
from src.data_ingest import parse_timestamp_utc, sanitize_single_bar_spikes
from src.hawkes import HawkesConfig, estimate_hawkes_rolling
from src.onchain import OnchainConfig, build_onchain_validation_frame, empty_onchain_frame
from src.plots import (
    PlotConfig,
    plot_figure_1_timeline,
    plot_figure_2_panel,
    plot_figure_3_phase_space,
)
from src.premium import PremiumConfig, build_premium_frame
from src.regimes import RegimeConfig, build_regime_frame
from src.robust_filter import RobustFilterConfig, build_robust_frame
from src.statmech import StatMechConfig, build_statmech_frame
from src.strategy import StrategyConfig, build_decisions


def _build_dataclass(cls: Any, data: dict[str, Any] | None) -> Any:
    data = data or {}
    valid_fields = {field.name for field in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**kwargs)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping at top level.")
    return data


def load_price_matrix(
    path: str | Path,
    *,
    sanitize_pair_spikes: bool = True,
    single_bar_spike_jump_log: float = 0.015,
    single_bar_spike_reversion_log: float = 0.003,
    single_bar_spike_counterpart_max_log: float = 0.002,
    single_bar_spike_min_cross_pairs: int = 1,
) -> pd.DataFrame:
    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Price matrix not found: {matrix_path}")
    suffixes = {s.lower() for s in matrix_path.suffixes}
    if ".parquet" in suffixes:
        frame = pd.read_parquet(matrix_path)
    elif ".csv" in suffixes or ".gz" in suffixes:
        frame = pd.read_csv(matrix_path)
    else:
        raise ValueError(
            f"Unsupported matrix format for {matrix_path.name}. Use parquet or csv."
        )

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp_utc" in frame.columns:
            frame = frame.set_index("timestamp_utc")
        frame.index = pd.DatetimeIndex(parse_timestamp_utc(pd.Series(frame.index)))
    else:
        frame.index = pd.DatetimeIndex(parse_timestamp_utc(pd.Series(frame.index)))

    nat_rows = int(frame.index.isna().sum())
    if nat_rows > 0:
        warnings.warn(
            f"Dropping {nat_rows} rows with invalid timestamps from {matrix_path}.",
            stacklevel=2,
        )
        frame = frame.loc[frame.index.notna()]

    duplicated = int(frame.index.duplicated(keep="last").sum())
    if duplicated > 0:
        warnings.warn(
            f"Dropping {duplicated} duplicate timestamps from {matrix_path} (keep='last').",
            stacklevel=2,
        )
        frame = frame.loc[~frame.index.duplicated(keep="last")]

    frame = frame.sort_index()

    if sanitize_pair_spikes:
        frame, diagnostics = sanitize_single_bar_spikes(
            frame,
            jump_threshold_log=single_bar_spike_jump_log,
            reversion_tolerance_log=single_bar_spike_reversion_log,
            counterpart_max_move_log=single_bar_spike_counterpart_max_log,
            min_cross_confirm_pairs=single_bar_spike_min_cross_pairs,
        )
        if not diagnostics.empty:
            warnings.warn(
                f"Corrected {diagnostics.shape[0]} single-bar stablecoin spike points from {matrix_path}.",
                stacklevel=2,
            )
    return frame


def run_pipeline(config: dict[str, Any], price_matrix: pd.DataFrame) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    freq = data_cfg.get("resample_rule", "1min")

    premium_cfg = _build_dataclass(PremiumConfig, config.get("premium"))
    premium_raw = config.get("premium", {})
    proxy_pairs_raw = premium_raw.get("proxy_pairs", [])
    proxy_pairs = [tuple(pair) for pair in proxy_pairs_raw] if proxy_pairs_raw else None

    premium_frame, proxy_components = build_premium_frame(
        price_matrix,
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
        except Exception as exc:
            warnings.warn(f"On-chain validation failed, continuing without it: {exc}")
            onchain_frame = empty_onchain_frame(premium_frame.index)
    else:
        onchain_frame = empty_onchain_frame(premium_frame.index)

    # Safety override integrates both market-implied and on-chain depeg detection.
    premium_frame["depeg_flag"] = (
        premium_frame["depeg_flag"].astype(bool) | onchain_frame["onchain_depeg_flag"].astype(bool)
    )

    robust_cfg = _build_dataclass(RobustFilterConfig, config.get("robust_filter"))
    robust_frame = build_robust_frame(premium_frame["p"], cfg=robust_cfg, freq=freq)

    m_t = robust_frame["p_smooth"].rename("m_t")
    stat_cfg = _build_dataclass(StatMechConfig, config.get("statmech"))
    state_frame = build_statmech_frame(
        robust_frame["z_t"],
        m_t,
        cfg=stat_cfg,
        freq=freq,
    )

    regime_cfg = _build_dataclass(RegimeConfig, config.get("regimes"))
    regime_frame = build_regime_frame(state_frame, regime_cfg)

    hawkes_cfg = _build_dataclass(HawkesConfig, config.get("hawkes"))
    hawkes_frame = (
        estimate_hawkes_rolling(robust_frame["event"], hawkes_cfg)
        if hawkes_cfg.enabled
        else pd.DataFrame(index=premium_frame.index)
    )
    n_t = hawkes_frame["n_t"] if "n_t" in hawkes_frame.columns else None

    strategy_cfg = _build_dataclass(StrategyConfig, config.get("strategy"))
    decision_frame = build_decisions(
        m_t=m_t,
        T_t=state_frame["T_t"],
        chi_t=state_frame["chi_t"],
        sigma_hat=robust_frame["sigma_hat"],
        regime=regime_frame["regime"],
        depeg_flag=premium_frame["depeg_flag"],
        n_t=n_t,
        cfg=strategy_cfg,
    )

    signal_frame = pd.concat(
        [premium_frame, onchain_frame, robust_frame, m_t, state_frame, regime_frame, hawkes_frame, decision_frame],
        axis=1,
    )
    signal_frame = signal_frame.loc[~signal_frame.index.duplicated(keep="last")]
    signal_frame = signal_frame.sort_index()

    backtest_cfg = _build_dataclass(BacktestConfig, config.get("backtest"))
    metrics, gated_log, naive_log = compare_strategies(
        p_naive=signal_frame["p_naive"],
        p_debiased=signal_frame["p"],
        decision_gated=signal_frame["decision"],
        m_t=signal_frame["m_t"],
        freq=freq,
        cfg=backtest_cfg,
    )

    return {
        "signal_frame": signal_frame,
        "proxy_components": proxy_components,
        "metrics": metrics,
        "gated_log": gated_log,
        "naive_log": naive_log,
    }


def export_outputs(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    outputs_cfg = config.get("outputs", {})
    tables_dir = Path(outputs_cfg.get("tables_dir", "reports/tables"))
    figures_dir = Path(outputs_cfg.get("figures_dir", "reports/figures"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    signal_frame = results["signal_frame"]
    metrics = results["metrics"]

    metrics_path = export_metrics(metrics, tables_dir / "metrics.csv")
    gated_path = tables_dir / "trade_log_gated.csv"
    naive_path = tables_dir / "trade_log_naive.csv"
    signal_path = tables_dir / "signal_frame.parquet"
    proxy_path = tables_dir / "stablecoin_proxy_components.parquet"

    results["gated_log"].to_csv(gated_path, index=True)
    results["naive_log"].to_csv(naive_path, index=True)
    signal_path = _save_frame(signal_frame, signal_path)
    proxy_path = _save_frame(results["proxy_components"], proxy_path)

    plot_cfg = _build_dataclass(PlotConfig, config.get("plots"))
    fig1 = plot_figure_1_timeline(
        signal_frame,
        figures_dir / "figure_1_timeline.png",
        plot_cfg,
    )
    fig2 = plot_figure_2_panel(
        signal_frame,
        figures_dir / "figure_2_panel.png",
        plot_cfg,
    )
    fig3 = plot_figure_3_phase_space(
        signal_frame,
        figures_dir / "figure_3_phase_space.png",
        plot_cfg,
        entry_k=float(config.get("strategy", {}).get("entry_k", 2.0)),
    )

    return {
        "metrics": metrics_path,
        "trade_log_gated": gated_path,
        "trade_log_naive": naive_path,
        "signal_frame": signal_path,
        "proxy_components": proxy_path,
        "figure_1": fig1,
        "figure_2": fig2,
        "figure_3": fig3,
    }


def _save_frame(frame: pd.DataFrame, preferred_path: Path) -> Path:
    try:
        frame.to_parquet(preferred_path)
        return preferred_path
    except Exception:
        fallback = preferred_path.with_suffix(".csv")
        frame.to_csv(fallback, index=True)
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run premium regime gating pipeline.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--price-matrix",
        default=None,
        help="Optional override for data.price_matrix_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    matrix_path = args.price_matrix or config.get("data", {}).get("price_matrix_path")
    if not matrix_path:
        raise ValueError(
            "No price matrix path provided. Set data.price_matrix_path in config or pass --price-matrix."
        )

    price_matrix = load_price_matrix(
        matrix_path,
        sanitize_pair_spikes=bool(data_cfg.get("sanitize_single_bar_spikes", True)),
        single_bar_spike_jump_log=float(data_cfg.get("single_bar_spike_jump_log", 0.015)),
        single_bar_spike_reversion_log=float(data_cfg.get("single_bar_spike_reversion_log", 0.003)),
        single_bar_spike_counterpart_max_log=float(data_cfg.get("single_bar_spike_counterpart_max_log", 0.002)),
        single_bar_spike_min_cross_pairs=int(data_cfg.get("single_bar_spike_min_cross_pairs", 1)),
    )
    results = run_pipeline(config, price_matrix)
    exported = export_outputs(results, config)

    print("Pipeline completed.")
    for name, path in exported.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
