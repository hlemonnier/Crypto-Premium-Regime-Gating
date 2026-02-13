from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import yaml

from src.backtest import BacktestConfig, compare_strategies, export_metrics
from src.data_ingest import parse_timestamp_utc, sanitize_single_bar_spikes
from src.hawkes import HawkesConfig, estimate_hawkes_rolling, evaluate_hawkes_quality
from src.onchain import OnchainConfig, build_onchain_validation_frame, empty_onchain_frame
from src.plots import (
    PlotConfig,
    plot_figure_1_timeline,
    plot_figure_2_panel,
    plot_figure_3_phase_space,
    plot_figure_4_edge_net,
)
from src.premium import PremiumConfig, build_premium_frame
from src.regimes import RegimeConfig, build_regime_frame
from src.robust_filter import RobustFilterConfig, build_robust_frame
from src.statmech import StatMechConfig, build_statmech_frame
from src.strategy import ExecutionUnifierConfig, StrategyConfig, build_decisions


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


def _load_optional_slippage_curve(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    curve_path = Path(path)
    if not curve_path.exists():
        warnings.warn(
            f"Execution unifier slippage curve not found at {curve_path}; using fallback linear cost model.",
            stacklevel=2,
        )
        return None
    try:
        return pd.read_csv(curve_path)
    except Exception as exc:
        warnings.warn(
            f"Could not read execution unifier slippage curve at {curve_path}; using fallback linear model: {exc}",
            stacklevel=2,
        )
        return None


def _select_mispricing_signal(
    *,
    premium_frame: pd.DataFrame,
    robust_frame: pd.DataFrame,
    source: str,
) -> pd.Series:
    key = str(source).strip().lower()
    if key in {"p_smooth", "smooth", "smoothed"}:
        return pd.to_numeric(robust_frame["p_smooth"], errors="coerce").rename("m_t")
    if key in {"p", "raw", "p_raw", "debiased"}:
        return pd.to_numeric(premium_frame["p"], errors="coerce").rename("m_t")
    if key in {"residual", "p_minus_p_smooth", "deviation"}:
        residual = pd.to_numeric(premium_frame["p"], errors="coerce") - pd.to_numeric(
            robust_frame["p_smooth"], errors="coerce"
        )
        return residual.rename("m_t")
    raise ValueError(
        "Unsupported strategy.m_t_source="
        f"{source!r}. Expected one of: p_smooth, p, residual."
    )


def _extract_size_curve_columns(frame: pd.DataFrame, prefix: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    marker = f"{prefix}_s"
    for col in frame.columns:
        if not str(col).startswith(marker):
            continue
        token = str(col)[len(marker) :]
        if len(token) == 0 or (not token.isdigit()):
            continue
        size = float(int(token)) / 100.0
        out.append((size, str(col)))
    out.sort(key=lambda x: x[0])
    return out


def _nan_quantile_safe(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, float(q)))


def _build_unifier_artifacts(signal_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if signal_frame.empty:
        empty = pd.DataFrame()
        return {"edge_net_size_curve": empty, "break_even_premium_curve": empty, "edge_net_summary": empty}

    expected_cols = _extract_size_curve_columns(signal_frame, "expected_net_pnl_bps")
    break_even_cols = _extract_size_curve_columns(signal_frame, "break_even_premium_bps")
    if len(expected_cols) == 0 and len(break_even_cols) == 0:
        empty = pd.DataFrame()
        return {"edge_net_size_curve": empty, "break_even_premium_curve": empty, "edge_net_summary": empty}

    regime = signal_frame.get("regime", pd.Series("unknown", index=signal_frame.index)).astype(str)
    idx_name = signal_frame.index.name or "timestamp_utc"

    curve_rows: list[pd.DataFrame] = []
    for size, col in expected_cols:
        values = pd.to_numeric(signal_frame[col], errors="coerce")
        local = pd.DataFrame(
            {
                idx_name: signal_frame.index,
                "regime": regime,
                "size": float(size),
                "expected_net_pnl_bps": values,
            }
        )
        curve_rows.append(local)
    edge_curve = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    break_rows: list[pd.DataFrame] = []
    for size, col in break_even_cols:
        values = pd.to_numeric(signal_frame[col], errors="coerce")
        local = pd.DataFrame(
            {
                idx_name: signal_frame.index,
                "regime": regime,
                "size": float(size),
                "break_even_premium_bps": values,
            }
        )
        break_rows.append(local)
    break_curve = pd.concat(break_rows, ignore_index=True) if break_rows else pd.DataFrame()

    summary_parts: list[pd.DataFrame] = []
    if not edge_curve.empty:
        edge_summary = (
            edge_curve.groupby(["regime", "size"], as_index=False)
            .agg(
                expected_net_pnl_median_bps=("expected_net_pnl_bps", "median"),
                expected_net_pnl_p10_bps=("expected_net_pnl_bps", lambda s: _nan_quantile_safe(s, 0.10)),
                n_obs=("expected_net_pnl_bps", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            )
        )
        summary_parts.append(edge_summary)
    if (not edge_curve.empty) and (not break_curve.empty):
        break_summary = (
            break_curve.groupby(["regime", "size"], as_index=False)
            .agg(
                break_even_premium_median_bps=("break_even_premium_bps", "median"),
                break_even_premium_p90_bps=("break_even_premium_bps", lambda s: _nan_quantile_safe(s, 0.90)),
            )
        )
        merged = summary_parts[0].merge(break_summary, on=["regime", "size"], how="outer")
        summary_parts = [merged]
    elif not break_curve.empty:
        summary_parts.append(
            break_curve.groupby(["regime", "size"], as_index=False).agg(
                break_even_premium_median_bps=("break_even_premium_bps", "median"),
                break_even_premium_p90_bps=("break_even_premium_bps", lambda s: _nan_quantile_safe(s, 0.90)),
            )
        )

    edge_summary = summary_parts[0] if summary_parts else pd.DataFrame()
    if not edge_summary.empty:
        edge_summary = edge_summary.sort_values(["regime", "size"]).reset_index(drop=True)
    if not edge_curve.empty:
        edge_curve = edge_curve.sort_values([idx_name, "regime", "size"]).reset_index(drop=True)
    if not break_curve.empty:
        break_curve = break_curve.sort_values([idx_name, "regime", "size"]).reset_index(drop=True)

    return {
        "edge_net_size_curve": edge_curve,
        "break_even_premium_curve": break_curve,
        "edge_net_summary": edge_summary,
    }


def _segment_lengths(labels: pd.Series, *, fallback_label: str = "unknown") -> pd.DataFrame:
    series = labels.copy()
    if series.empty:
        return pd.DataFrame(columns=["label", "length_bars"])
    series = series.where(series.notna(), fallback_label).astype(str)
    segment_id = series.ne(series.shift(1)).cumsum()
    frame = (
        pd.DataFrame({"label": series, "segment_id": segment_id})
        .groupby("segment_id", as_index=False)
        .agg(label=("label", "first"), length_bars=("label", "size"))
    )
    return frame[["label", "length_bars"]]


def _build_coherence_diagnostics(signal_frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    n_bars = int(signal_frame.shape[0])
    event = signal_frame.get("event", pd.Series(False, index=signal_frame.index)).fillna(False).astype(bool)
    regime = signal_frame.get("regime", pd.Series("unknown", index=signal_frame.index)).astype(str)
    decision = signal_frame.get("decision", pd.Series("unknown", index=signal_frame.index)).astype(str)

    regime_segments = _segment_lengths(regime, fallback_label="unknown")
    decision_segments = _segment_lengths(decision, fallback_label="unknown")
    event_segments = _segment_lengths(event.astype(str), fallback_label="False")
    event_true_segments = event_segments.loc[event_segments["label"].eq("True"), "length_bars"]

    stress_segments = regime_segments.loc[regime_segments["label"].eq("stress"), "length_bars"]
    transient_segments = regime_segments.loc[regime_segments["label"].eq("transient"), "length_bars"]

    if decision.empty:
        decision_flip_count = 0
    else:
        decision_flip_count = int(decision.ne(decision.shift(1)).iloc[1:].sum())

    decision_flip_rate = float("nan")
    if ("gated" in metrics.index) and ("flip_rate" in metrics.columns):
        decision_flip_rate = float(pd.to_numeric(metrics.loc["gated", "flip_rate"], errors="coerce"))
    elif n_bars > 1:
        decision_flip_rate = float(decision_flip_count / float(n_bars - 1))

    row = {
        "n_bars": n_bars,
        "event_bars": int(event.sum()),
        "event_rate": float(event.mean()) if n_bars > 0 else float("nan"),
        "event_segment_count": int(event_true_segments.shape[0]),
        "event_segment_mean_bars": float(event_true_segments.mean()) if not event_true_segments.empty else float("nan"),
        "event_segment_median_bars": float(event_true_segments.median()) if not event_true_segments.empty else float("nan"),
        "regime_segment_count": int(regime_segments.shape[0]),
        "regime_segment_mean_bars": float(regime_segments["length_bars"].mean()) if not regime_segments.empty else float("nan"),
        "regime_segment_median_bars": float(regime_segments["length_bars"].median()) if not regime_segments.empty else float("nan"),
        "stress_segment_count": int(stress_segments.shape[0]),
        "stress_segment_mean_bars": float(stress_segments.mean()) if not stress_segments.empty else float("nan"),
        "stress_segment_median_bars": float(stress_segments.median()) if not stress_segments.empty else float("nan"),
        "transient_segment_count": int(transient_segments.shape[0]),
        "transient_segment_mean_bars": float(transient_segments.mean()) if not transient_segments.empty else float("nan"),
        "transient_segment_median_bars": float(transient_segments.median()) if not transient_segments.empty else float("nan"),
        "decision_segment_count": int(decision_segments.shape[0]),
        "decision_segment_mean_bars": float(decision_segments["length_bars"].mean()) if not decision_segments.empty else float("nan"),
        "decision_segment_median_bars": float(decision_segments["length_bars"].median()) if not decision_segments.empty else float("nan"),
        "decision_flip_count": decision_flip_count,
        "decision_flip_rate": decision_flip_rate,
    }
    return pd.DataFrame([row])


def _build_scope_audit(
    *,
    signal_frame: pd.DataFrame,
    signal_path: Path,
    metrics_path: Path,
    safety_diag_path: Path,
    figure_1_path: Path,
    figure_2_path: Path,
    figure_3_path: Path,
    depeg_bars: int,
    riskoff_bars: int,
    depeg_without_riskoff_bars: int,
) -> pd.DataFrame:
    scope_ref = "original_email_request_baseline"
    required_decisions = {"Trade", "Widen", "Risk-off"}
    decision_series = signal_frame.get("decision", pd.Series("", index=signal_frame.index)).astype(str)
    observed_decisions = {value for value in decision_series.unique() if value}

    rows = [
        {
            "scope_reference": scope_ref,
            "requirement_id": "premium_debias_columns",
            "requirement": "Debiased premium outputs exist (p_naive, stablecoin_proxy, p, depeg_flag).",
            "status": "pass"
            if {"p_naive", "stablecoin_proxy", "p", "depeg_flag"}.issubset(set(signal_frame.columns))
            else "fail",
            "evidence": "columns="
            + str(sorted([col for col in ["p_naive", "stablecoin_proxy", "p", "depeg_flag"] if col in signal_frame.columns])),
            "artifact_path": str(signal_path),
        },
        {
            "scope_reference": scope_ref,
            "requirement_id": "robust_filter_columns",
            "requirement": "Robust filter outputs exist (p_smooth, sigma_hat, z_t, event).",
            "status": "pass"
            if {"p_smooth", "sigma_hat", "z_t", "event"}.issubset(set(signal_frame.columns))
            else "fail",
            "evidence": "columns="
            + str(sorted([col for col in ["p_smooth", "sigma_hat", "z_t", "event"] if col in signal_frame.columns])),
            "artifact_path": str(signal_path),
        },
        {
            "scope_reference": scope_ref,
            "requirement_id": "state_variables_columns",
            "requirement": "State variables exist (H_t, T_t, chi_t) with regime labels.",
            "status": "pass"
            if {"H_t", "T_t", "chi_t", "regime"}.issubset(set(signal_frame.columns))
            else "fail",
            "evidence": "columns="
            + str(sorted([col for col in ["H_t", "T_t", "chi_t", "regime"] if col in signal_frame.columns])),
            "artifact_path": str(signal_path),
        },
        {
            "scope_reference": scope_ref,
            "requirement_id": "decision_outputs",
            "requirement": "Decision output is present and only uses Trade/Widen/Risk-off states.",
            "status": "pass"
            if observed_decisions.issubset(required_decisions) and len(observed_decisions) > 0
            else "fail",
            "evidence": f"observed_decisions={sorted(observed_decisions)}",
            "artifact_path": str(signal_path),
        },
        {
            "scope_reference": scope_ref,
            "requirement_id": "stablecoin_depeg_safety",
            "requirement": "Depeg windows are demonstrated and always Risk-off on flagged bars.",
            "status": "pass"
            if (depeg_bars > 0 and riskoff_bars > 0 and depeg_without_riskoff_bars == 0)
            else "fail",
            "evidence": (
                f"depeg_bars={depeg_bars};"
                f"riskoff_bars={riskoff_bars};"
                f"depeg_without_riskoff_bars={depeg_without_riskoff_bars}"
            ),
            "artifact_path": str(safety_diag_path),
        },
        {
            "scope_reference": scope_ref,
            "requirement_id": "required_exports",
            "requirement": "Core deliverables exported (metrics.csv + figure_1/2/3).",
            "status": "pass"
            if (
                metrics_path.exists()
                and figure_1_path.exists()
                and figure_2_path.exists()
                and figure_3_path.exists()
                and safety_diag_path.exists()
            )
            else "fail",
            "evidence": (
                f"metrics_exists={metrics_path.exists()};"
                f"figure_1_exists={figure_1_path.exists()};"
                f"figure_2_exists={figure_2_path.exists()};"
                f"figure_3_exists={figure_3_path.exists()};"
                f"safety_diag_exists={safety_diag_path.exists()}"
            ),
            "artifact_path": ";".join(
                [
                    str(metrics_path),
                    str(figure_1_path),
                    str(figure_2_path),
                    str(figure_3_path),
                    str(safety_diag_path),
                ]
            ),
        },
    ]
    out = pd.DataFrame(rows)
    out["pass"] = out["status"].eq("pass")
    return out


def _parse_positive_timedelta(value: str, *, field_name: str) -> pd.Timedelta:
    try:
        td = pd.to_timedelta(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid timedelta string, got {value!r}.") from exc
    if td <= pd.Timedelta(0):
        raise ValueError(f"{field_name} must be positive, got {value!r}.")
    return td


def validate_price_matrix_frequency(
    index: pd.Index,
    expected_freq: str,
    *,
    context: str = "price_matrix",
) -> pd.Timedelta:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{context} must use a DatetimeIndex for frequency validation.")
    if index.size < 2:
        raise ValueError(
            f"{context} needs at least 2 rows to validate spacing against data.resample_rule={expected_freq!r}."
        )

    expected_td = _parse_positive_timedelta(str(expected_freq), field_name="data.resample_rule")
    deltas = index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError(f"{context} has no positive timestamp deltas; cannot validate matrix spacing.")

    mismatched = positive[positive != expected_td]
    if mismatched.empty:
        return expected_td

    observed = positive.value_counts().sort_values(ascending=False)
    preview = ", ".join(
        f"{str(delta)} x{int(count)}"
        for delta, count in observed.iloc[:5].items()
    )
    raise ValueError(
        "Configured data.resample_rule does not match price matrix spacing. "
        f"expected={str(expected_td)} context={context!r} observed_top_deltas=[{preview}]"
    )


def _observed_index_delta(index: pd.Index) -> pd.Timedelta | None:
    if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
        return None
    deltas = index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return None
    observed = positive.value_counts().sort_values(ascending=False)
    return observed.index[0]


def _is_missing_parquet_engine(exc: Exception) -> bool:
    if isinstance(exc, ImportError):
        return True
    message = str(exc).lower()
    return (
        "unable to find a usable engine" in message
        or "missing optional dependency 'pyarrow'" in message
        or "missing optional dependency 'fastparquet'" in message
    )


def _next_unique_name(base: str, taken: set[object]) -> str:
    candidate = base
    suffix = 1
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _assert_no_duplicate_columns(frame: pd.DataFrame, *, context: str) -> None:
    duplicated_cols = frame.columns[frame.columns.duplicated(keep=False)]
    if len(duplicated_cols) == 0:
        return
    duplicates = sorted({str(col) for col in duplicated_cols.tolist()})
    raise ValueError(
        f"Duplicate column names found in {context}: {duplicates}. "
        "Refusing to apply drop-first semantics."
    )


def _prepare_frame_for_parquet(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    out = frame.copy()
    _assert_no_duplicate_columns(out, context=f"{context} (parquet export)")

    if isinstance(out.index, pd.MultiIndex):
        names: list[object | None] = list(out.index.names)
    else:
        names = [out.index.name]

    taken: set[object] = set(out.columns.tolist())
    seen_index_names: set[object] = set()
    renamed = False
    resolved_names: list[object | None] = []
    for i, name in enumerate(names):
        if name is None:
            resolved_names.append(None)
            continue
        if (name in taken) or (name in seen_index_names):
            base = f"{name}_index"
            resolved = _next_unique_name(base, taken)
            warnings.warn(
                f"Renaming index level {i} from {name!r} to {resolved!r} before parquet export for {context}.",
                stacklevel=2,
            )
            resolved_names.append(resolved)
            seen_index_names.add(resolved)
            renamed = True
            continue
        resolved_names.append(name)
        seen_index_names.add(name)
        taken.add(name)

    if renamed:
        if isinstance(out.index, pd.MultiIndex):
            out.index = out.index.rename(resolved_names)
        else:
            out.index = out.index.rename(resolved_names[0])
    return out


def load_price_matrix(
    path: str | Path,
    *,
    sanitize_pair_spikes: bool = True,
    single_bar_spike_jump_log: float = 0.015,
    single_bar_spike_reversion_log: float = 0.003,
    single_bar_spike_counterpart_max_log: float = 0.002,
    single_bar_spike_min_cross_pairs: int = 1,
    expected_freq: str | None = None,
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

    if expected_freq is not None:
        validate_price_matrix_frequency(
            frame.index,
            str(expected_freq),
            context=str(matrix_path),
        )

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
    freq = str(data_cfg.get("resample_rule", "1min"))
    backtest_cfg = _build_dataclass(BacktestConfig, config.get("backtest"))
    execution_unifier_cfg = _build_dataclass(ExecutionUnifierConfig, config.get("execution_unifier"))
    execution_unifier_raw = config.get("execution_unifier", {})
    slippage_curve_path = execution_unifier_raw.get(
        "slippage_curve_path",
        execution_unifier_cfg.slippage_curve_path,
    )
    slippage_curve = _load_optional_slippage_curve(slippage_curve_path) if execution_unifier_cfg.enabled else None

    validate_price_matrix_frequency(
        price_matrix.index,
        freq,
        context="run_pipeline.price_matrix",
    )

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
    onchain_fail_closed_on_error = bool(getattr(onchain_cfg, "fail_closed_on_error", True))
    if onchain_cfg.enabled:
        try:
            onchain_frame = build_onchain_validation_frame(
                index=premium_frame.index,
                stablecoin_proxy=premium_frame["stablecoin_proxy"],
                cfg=onchain_cfg,
            )
        except Exception as exc:
            if onchain_fail_closed_on_error:
                warnings.warn(
                    "On-chain validation failed, fail-closed guardrail engaged "
                    f"(forcing Risk-off until recovery): {exc}"
                )
            else:
                warnings.warn(
                    "On-chain validation failed, continuing fail-open without on-chain guardrail: "
                    f"{exc}"
                )
            onchain_frame = empty_onchain_frame(
                premium_frame.index,
                fail_closed=onchain_fail_closed_on_error,
            )
    else:
        onchain_frame = empty_onchain_frame(premium_frame.index)

    market_depeg_flag = premium_frame["depeg_flag"].fillna(False).astype(bool).rename("market_depeg_flag")
    premium_frame["market_depeg_flag"] = market_depeg_flag
    onchain_effective = onchain_frame.get(
        "onchain_depeg_flag_effective",
        pd.Series(False, index=premium_frame.index, name="onchain_depeg_flag_effective"),
    )
    onchain_effective = onchain_effective.fillna(False).astype(bool).rename("onchain_depeg_flag_effective")
    premium_frame["depeg_flag"] = (market_depeg_flag | onchain_effective).rename("depeg_flag")

    robust_cfg = _build_dataclass(RobustFilterConfig, config.get("robust_filter"))
    robust_frame = build_robust_frame(premium_frame["p"], cfg=robust_cfg, freq=freq)

    strategy_cfg = _build_dataclass(StrategyConfig, config.get("strategy"))
    m_t = _select_mispricing_signal(
        premium_frame=premium_frame,
        robust_frame=robust_frame,
        source=str(strategy_cfg.m_t_source),
    )
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
    if hawkes_cfg.enabled:
        hawkes_frame = estimate_hawkes_rolling(robust_frame["event"], hawkes_cfg)
        hawkes_quality_pass, hawkes_quality_reason, hawkes_quality_metrics = evaluate_hawkes_quality(
            hawkes_frame,
            hawkes_cfg,
        )
    else:
        hawkes_frame = pd.DataFrame(index=premium_frame.index)
        hawkes_quality_pass = False
        hawkes_quality_reason = "disabled"
        hawkes_quality_metrics = {
            "hawkes_refit_count": 0.0,
            "hawkes_fit_ok_ratio": 0.0,
            "hawkes_n_unique": 0.0,
            "hawkes_n_std": 0.0,
        }

    n_t = hawkes_frame["n_t"] if (hawkes_quality_pass and "n_t" in hawkes_frame.columns) else None
    hawkes_diag_frame = pd.DataFrame(index=premium_frame.index)
    hawkes_diag_frame["hawkes_quality_pass"] = bool(hawkes_quality_pass)
    hawkes_diag_frame["hawkes_quality_reason"] = str(hawkes_quality_reason)
    for key, value in hawkes_quality_metrics.items():
        if key in hawkes_frame.columns:
            continue
        hawkes_diag_frame[key] = float(value)

    decision_frame = build_decisions(
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
        n_t=n_t,
        premium=premium_frame["p"],
        slippage_curve=slippage_curve,
        execution_unifier_cfg=execution_unifier_cfg,
        base_cost_bps=backtest_cfg.cost_bps,
        cfg=strategy_cfg,
    )

    signal_frame = pd.concat(
        [
            premium_frame,
            onchain_frame,
            robust_frame,
            m_t,
            state_frame,
            regime_frame,
            hawkes_frame,
            hawkes_diag_frame,
            decision_frame,
        ],
        axis=1,
    )
    _assert_no_duplicate_columns(signal_frame, context="run_pipeline.signal_frame")
    signal_frame = signal_frame.loc[~signal_frame.index.duplicated(keep="last")]
    signal_frame = signal_frame.sort_index()

    metrics, gated_log, naive_log = compare_strategies(
        p_naive=signal_frame["p_naive"],
        p_debiased=signal_frame["p"],
        decision_gated=signal_frame["decision"],
        m_t=signal_frame["m_t"],
        size_gated=signal_frame.get("position_size"),
        dynamic_cost_gated=signal_frame.get("edge_cost_bps_opt"),
        expected_edge_gated=signal_frame.get("edge_gross_bps"),
        expected_cost_gated=signal_frame.get("edge_cost_bps_opt"),
        expected_net_gated=signal_frame.get("edge_net_bps_opt"),
        freq=freq,
        cfg=backtest_cfg,
    )

    if "gated" in metrics.index:
        trade_mask = signal_frame.get("decision", pd.Series("", index=signal_frame.index)).astype(str).eq("Trade")
        edge_net_trade = pd.to_numeric(signal_frame.get("edge_net_bps_opt"), errors="coerce").loc[trade_mask]
        optimal_trade = pd.to_numeric(signal_frame.get("optimal_size"), errors="coerce").loc[trade_mask]
        break_even_trade = pd.to_numeric(signal_frame.get("break_even_premium_bps_opt"), errors="coerce").loc[trade_mask]
        metrics.loc["gated", "edge_net_trade_median_bps"] = (
            float(edge_net_trade.median()) if not edge_net_trade.dropna().empty else float("nan")
        )
        metrics.loc["gated", "edge_net_trade_p10_bps"] = (
            float(edge_net_trade.quantile(0.10)) if not edge_net_trade.dropna().empty else float("nan")
        )
        metrics.loc["gated", "optimal_size_trade_mean"] = (
            float(optimal_trade.mean()) if not optimal_trade.dropna().empty else 0.0
        )
        metrics.loc["gated", "break_even_premium_median_bps"] = (
            float(break_even_trade.median()) if not break_even_trade.dropna().empty else float("nan")
        )
        if "cost_bps_applied" in gated_log.columns:
            cost_series = pd.to_numeric(gated_log["cost_bps_applied"], errors="coerce")
            metrics.loc["gated", "cost_bps_applied_mean"] = (
                float(cost_series.mean()) if not cost_series.dropna().empty else float("nan")
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
    data_cfg = config.get("data", {})
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
    safety_diag_path = tables_dir / "safety_diagnostics.csv"
    coherence_diag_path = tables_dir / "coherence_diagnostics.csv"
    scope_audit_path = tables_dir / "scope_audit.csv"
    edge_net_curve_path = tables_dir / "edge_net_size_curve.csv"
    break_even_curve_path = tables_dir / "break_even_premium_curve.csv"
    edge_net_summary_path = tables_dir / "edge_net_summary.csv"

    results["gated_log"].to_csv(gated_path, index=True)
    results["naive_log"].to_csv(naive_path, index=True)
    signal_path = _save_frame(signal_frame, signal_path)
    proxy_path = _save_frame(results["proxy_components"], proxy_path)
    configured_freq = str(data_cfg.get("resample_rule", "1min"))
    expected_td = _parse_positive_timedelta(configured_freq, field_name="data.resample_rule")
    observed_td = _observed_index_delta(signal_frame.index)
    observed_delta = str(observed_td) if observed_td is not None else "n/a"
    frequency_consistent = bool(observed_td == expected_td) if observed_td is not None else False
    depeg_flag = signal_frame.get("depeg_flag", pd.Series(False, index=signal_frame.index)).astype(bool)
    riskoff_flag = signal_frame.get("riskoff_flag", pd.Series(False, index=signal_frame.index)).astype(bool)
    trade_flag = signal_frame.get("decision", pd.Series("", index=signal_frame.index)).astype(str).eq("Trade")
    trade_signal = signal_frame.get("trade_signal", pd.Series(False, index=signal_frame.index)).astype(bool)
    entry_threshold = pd.to_numeric(signal_frame.get("entry_threshold", pd.Series(np.nan, index=signal_frame.index)), errors="coerce")
    temperature = pd.to_numeric(signal_frame.get("T_t", pd.Series(np.nan, index=signal_frame.index)), errors="coerce")
    onchain_stale = signal_frame.get("onchain_data_stale", pd.Series(False, index=signal_frame.index)).astype(bool)
    gated_active_ratio = float(pd.to_numeric(metrics.loc["gated", "active_ratio"], errors="coerce")) if "gated" in metrics.index else float("nan")
    naive_active_ratio = float(pd.to_numeric(metrics.loc["naive", "active_ratio"], errors="coerce")) if "naive" in metrics.index else float("nan")
    gated_comparable_vs_naive = (
        int(pd.to_numeric(metrics.loc["gated", "comparable_vs_naive"], errors="coerce"))
        if ("gated" in metrics.index and "comparable_vs_naive" in metrics.columns)
        else 0
    )
    gated_degenerate_no_trade = (
        int(pd.to_numeric(metrics.loc["gated", "degenerate_no_trade"], errors="coerce"))
        if ("gated" in metrics.index and "degenerate_no_trade" in metrics.columns)
        else 0
    )
    safety_diag = pd.DataFrame(
        [
            {
                "depeg_bars": int(depeg_flag.sum()),
                "riskoff_bars": int(riskoff_flag.sum()),
                "depeg_without_riskoff_bars": int((depeg_flag & ~riskoff_flag).sum()),
                "trade_but_trade_signal_false": int((trade_flag & ~trade_signal).sum()),
                "trade_with_entry_threshold_nan": int((trade_flag & entry_threshold.isna()).sum()),
                "trade_with_T_t_nan": int((trade_flag & temperature.isna()).sum()),
                "gated_active_ratio": gated_active_ratio,
                "naive_active_ratio": naive_active_ratio,
                "gated_comparable_vs_naive": gated_comparable_vs_naive,
                "gated_degenerate_no_trade": gated_degenerate_no_trade,
                "hawkes_quality_pass": bool(signal_frame.get("hawkes_quality_pass", pd.Series(False)).iloc[0])
                if not signal_frame.empty
                else False,
                "hawkes_quality_reason": str(signal_frame.get("hawkes_quality_reason", pd.Series(["n/a"])).iloc[0])
                if not signal_frame.empty
                else "n/a",
                "onchain_stale_ratio": float(onchain_stale.mean()) if len(onchain_stale) > 0 else 0.0,
                "configured_resample_rule": configured_freq,
                "expected_index_delta": str(expected_td),
                "observed_index_delta": observed_delta,
                "index_delta_matches_config": frequency_consistent,
            }
        ]
    )
    safety_diag.to_csv(safety_diag_path, index=False)
    coherence_diag = _build_coherence_diagnostics(signal_frame, metrics)
    coherence_diag.to_csv(coherence_diag_path, index=False)

    unifier_artifacts = _build_unifier_artifacts(signal_frame)
    unifier_artifacts["edge_net_size_curve"].to_csv(edge_net_curve_path, index=False)
    unifier_artifacts["break_even_premium_curve"].to_csv(break_even_curve_path, index=False)
    unifier_artifacts["edge_net_summary"].to_csv(edge_net_summary_path, index=False)

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
    fig4 = plot_figure_4_edge_net(
        signal_frame,
        figures_dir / "figure_4_edge_net.png",
        plot_cfg,
    )
    scope_audit = _build_scope_audit(
        signal_frame=signal_frame,
        signal_path=signal_path,
        metrics_path=metrics_path,
        safety_diag_path=safety_diag_path,
        figure_1_path=fig1,
        figure_2_path=fig2,
        figure_3_path=fig3,
        depeg_bars=int(depeg_flag.sum()),
        riskoff_bars=int(riskoff_flag.sum()),
        depeg_without_riskoff_bars=int((depeg_flag & ~riskoff_flag).sum()),
    )
    scope_audit.to_csv(scope_audit_path, index=False)

    return {
        "metrics": metrics_path,
        "trade_log_gated": gated_path,
        "trade_log_naive": naive_path,
        "signal_frame": signal_path,
        "proxy_components": proxy_path,
        "safety_diagnostics": safety_diag_path,
        "coherence_diagnostics": coherence_diag_path,
        "scope_audit": scope_audit_path,
        "edge_net_size_curve": edge_net_curve_path,
        "break_even_premium_curve": break_even_curve_path,
        "edge_net_summary": edge_net_summary_path,
        "figure_1": fig1,
        "figure_2": fig2,
        "figure_3": fig3,
        "figure_4": fig4,
    }


def _save_frame(frame: pd.DataFrame, preferred_path: Path) -> Path:
    prepared = _prepare_frame_for_parquet(frame, context=str(preferred_path))
    try:
        prepared.to_parquet(preferred_path)
        return preferred_path
    except Exception as exc:
        if not _is_missing_parquet_engine(exc):
            raise
        fallback = preferred_path.with_suffix(".csv")
        prepared.to_csv(fallback, index=True)
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
        expected_freq=str(data_cfg.get("resample_rule", "1min")),
    )
    results = run_pipeline(config, price_matrix)
    exported = export_outputs(results, config)

    print("Pipeline completed.")
    for name, path in exported.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
