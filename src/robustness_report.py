from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import glob
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from src.ablation_core import (
    FactorialVariant,
    build_dataclass,
    build_factorial_variants,
    compute_core_frames,
    simple_decision,
)
from src.backtest import BacktestConfig, run_backtest
from src.hawkes import HawkesConfig, estimate_hawkes_rolling, evaluate_hawkes_quality
from src.pipeline import load_config, load_price_matrix, run_pipeline
from src.strategy import ExecutionUnifierConfig, StrategyConfig, build_decisions
from src.tune_gating import (
    CandidateGrid,
    build_param_grid,
    classify_compatibility_reason,
    compatibility_skip_table,
    evaluate_dataset,
    parse_float_list,
)

SCENARIO_ORDER = ["base", "fees_x2", "spread_x2", "latency_1bar", "liquidity_half", "combined_worst"]
SINGLE_STRESS_SCENARIOS = ["fees_x2", "spread_x2", "latency_1bar", "liquidity_half"]
REFERENCE_VARIANT_ID = FactorialVariant(
    premium="debiased",
    gating=True,
    statmech=True,
    hawkes=True,
).variant_id
SMOKE_TOKENS = ("smoke",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run walk-forward OOS robustness with full 2x2x2x2 ablations and stress scenarios."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config path.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["data/processed/episodes/*/prices_matrix.*"],
        help="Episode matrix globs.",
    )
    parser.add_argument("--output-dir", default="reports/robustness", help="Output folder.")
    parser.add_argument(
        "--min-train-episodes",
        type=int,
        default=2,
        help="Minimum train episode count for walk-forward splits.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke episodes (excluded by default).",
    )
    parser.add_argument("--entry-k", default="0.5,0.75,1.0,1.25", help="Candidate list.")
    parser.add_argument("--t-widen", default="0.95,0.97,0.99", help="Candidate list.")
    parser.add_argument("--chi-widen", default="0.95,0.97,0.99", help="Candidate list.")
    parser.add_argument("--stress", default="0.9,0.95,0.99", help="Candidate list.")
    parser.add_argument("--recovery", default="0.6,0.8", help="Candidate list.")
    parser.add_argument("--min-active-ratio", type=float, default=0.002, help="Penalty threshold.")
    parser.add_argument("--max-combos", type=int, default=0, help="Randomly sample N combos (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for combo sampling.")
    return parser.parse_args()


def resolve_episode_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        raw = str(pattern)
        if Path(raw).is_absolute():
            matches = [Path(p) for p in glob.glob(raw)]
        else:
            matches = list(Path().glob(raw))
        files.extend(sorted(matches))
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(f"No episode files matched patterns: {patterns}")
    return deduped


def _is_smoke_episode(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return any(token in lowered for token in SMOKE_TOKENS)


def apply_default_episode_filters(
    files: list[Path],
    *,
    include_smoke: bool,
) -> tuple[list[Path], list[Path]]:
    if include_smoke:
        return list(files), []
    kept = [path for path in files if not _is_smoke_episode(path)]
    dropped = [path for path in files if _is_smoke_episode(path)]
    return kept, dropped


def _matrix_start_end(matrix: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if matrix.empty:
        ts = pd.Timestamp("1970-01-01", tz="UTC")
        return ts, ts
    idx = matrix.index
    return idx.min(), idx.max()


def _sort_matrix_items(matrices: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    return sorted(matrices.items(), key=lambda item: (_matrix_start_end(item[1])[0], item[0]))


def build_walkforward_splits(
    items: list[tuple[str, pd.DataFrame]],
    *,
    min_train_episodes: int = 1,
) -> list[dict[str, Any]]:
    min_train = max(1, int(min_train_episodes))
    if len(items) <= min_train:
        return []

    out: list[dict[str, Any]] = []
    split_id = 1
    for test_pos in range(min_train, len(items)):
        train_items = items[:test_pos]
        test_item = items[test_pos]
        train_ids = [path for path, _ in train_items]
        test_id, test_matrix = test_item
        train_matrix_map = {path: matrix for path, matrix in train_items}

        train_starts = [_matrix_start_end(matrix)[0] for _, matrix in train_items]
        train_ends = [_matrix_start_end(matrix)[1] for _, matrix in train_items]
        test_start, test_end = _matrix_start_end(test_matrix)

        out.append(
            {
                "split_id": split_id,
                "train_ids": train_ids,
                "test_id": test_id,
                "train_matrices": train_matrix_map,
                "test_matrix": test_matrix,
                "train_start": min(train_starts),
                "train_end": max(train_ends),
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        split_id += 1
    return out


def _load_matrix_with_config(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    return load_price_matrix(
        path,
        sanitize_pair_spikes=bool(data_cfg.get("sanitize_single_bar_spikes", True)),
        single_bar_spike_jump_log=float(data_cfg.get("single_bar_spike_jump_log", 0.015)),
        single_bar_spike_reversion_log=float(data_cfg.get("single_bar_spike_reversion_log", 0.003)),
        single_bar_spike_counterpart_max_log=float(data_cfg.get("single_bar_spike_counterpart_max_log", 0.002)),
        single_bar_spike_min_cross_pairs=int(data_cfg.get("single_bar_spike_min_cross_pairs", 1)),
        expected_freq=str(data_cfg.get("resample_rule", "1min")),
    )


def _load_matrices(files: list[Path], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {str(path): _load_matrix_with_config(path, config) for path in files}


def filter_compatible_matrices(
    base_config: dict[str, Any],
    matrices: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    compatible: dict[str, pd.DataFrame] = {}
    skipped: dict[str, dict[str, str]] = {}
    for path, matrix in matrices.items():
        try:
            results = run_pipeline(base_config, matrix)
        except Exception as exc:
            skipped[path] = classify_compatibility_reason(exc)
            continue
        metrics = results.get("metrics")
        if isinstance(metrics, pd.DataFrame) and ("gated" in metrics.index):
            gated = metrics.loc["gated"]
            degenerate_raw = float(pd.to_numeric(gated.get("degenerate_no_trade"), errors="coerce"))
            comparable_raw = float(pd.to_numeric(gated.get("comparable_vs_naive"), errors="coerce"))
            degenerate_no_trade = int(degenerate_raw) if np.isfinite(degenerate_raw) else 0
            comparable_vs_naive = int(comparable_raw) if np.isfinite(comparable_raw) else 0
            if degenerate_no_trade > 0 or comparable_vs_naive <= 0:
                skipped[path] = {
                    "reason_code": "degenerate_strategy",
                    "reason": (
                        "Gated strategy produced degenerate/non-comparable outputs for this episode "
                        f"(degenerate_no_trade={degenerate_no_trade}, comparable_vs_naive={comparable_vs_naive})."
                    ),
                }
                continue
        compatible[path] = matrix
    return compatible, skipped


def _write_graceful_compatibility_outputs(
    *,
    output_dir: Path,
    skipped: dict[str, dict[str, str]],
    reason_code: str,
    reason: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    walkforward_df = pd.DataFrame()
    ablation_df = pd.DataFrame()
    stress_df = pd.DataFrame()
    verdict_df = pd.DataFrame(columns=["split_id", "test_episode", "variant_id", "verdict_pass"])
    calibration_df = pd.DataFrame()

    walkforward_path = output_dir / "walkforward_split_metrics.csv"
    ablation_path = output_dir / "ablation_factorial_oos.csv"
    stress_path = output_dir / "stress_matrix_oos.csv"
    verdict_path = output_dir / "robustness_verdict.csv"
    calibration_path = output_dir / "walkforward_calibration_details.csv"

    walkforward_df.to_csv(walkforward_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)
    stress_df.to_csv(stress_path, index=False)
    verdict_df.to_csv(verdict_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)

    skip_table = compatibility_skip_table(skipped)
    run_row = pd.DataFrame(
        [{"matrix_path": "__RUN__", "reason_code": str(reason_code), "reason": str(reason)}]
    )
    skip_table = pd.concat([skip_table, run_row], axis=0, ignore_index=True)
    skipped_path = output_dir / "compatibility_skipped.csv"
    skip_table.to_csv(skipped_path, index=False)

    summary_path = _write_summary_markdown(
        output_dir=output_dir,
        walkforward_df=walkforward_df,
        ablation_df=ablation_df,
        stress_df=stress_df,
        verdict_df=verdict_df,
    )

    print("Robustness sweep skipped gracefully.")
    print(f"- reason_code: {reason_code}")
    print(f"- reason: {reason}")
    print(f"- walkforward_split_metrics: {walkforward_path}")
    print(f"- ablation_factorial_oos: {ablation_path}")
    print(f"- stress_matrix_oos: {stress_path}")
    print(f"- robustness_verdict: {verdict_path}")
    print(f"- calibration_details: {calibration_path}")
    print(f"- compatibility_skipped: {skipped_path}")
    print(f"- summary: {summary_path}")


def _apply_combo(base_config: dict[str, Any], combo: dict[str, float]) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg.setdefault("strategy", {})
    cfg.setdefault("regimes", {})
    cfg["strategy"]["entry_k"] = float(combo["entry_k"])
    cfg["strategy"]["t_widen_quantile"] = float(combo["t_widen_quantile"])
    cfg["strategy"]["chi_widen_quantile"] = float(combo["chi_widen_quantile"])
    cfg["regimes"]["stress_quantile"] = float(combo["stress_quantile"])
    cfg["regimes"]["recovery_quantile"] = float(combo["recovery_quantile"])
    return cfg


def _calibrate_split(
    *,
    base_config: dict[str, Any],
    train_matrices: dict[str, pd.DataFrame],
    combos: list[dict[str, float]],
    min_active_ratio: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    def _num_col(table: pd.DataFrame, name: str, *, default: float = 0.0) -> pd.Series:
        if name not in table.columns:
            return pd.Series(float(default), index=table.index, dtype="float64")
        return pd.to_numeric(table[name], errors="coerce").fillna(float(default))

    def _robust_calibration_score(table: pd.DataFrame) -> pd.Series:
        base_score = _num_col(table, "train_score")
        mean_sharpe = _num_col(table, "train_mean_sharpe_full_raw")
        mean_pnl = _num_col(table, "train_mean_pnl_net")
        mean_turnover = _num_col(table, "train_mean_turnover")
        min_sharpe = _num_col(table, "train_min_sharpe_full_raw")
        mean_active = _num_col(table, "train_mean_active_ratio")

        downside_min_sharpe = (-min_sharpe).clip(lower=0.0)
        over_active = (mean_active - 0.30).clip(lower=0.0)
        negative_pnl = (-mean_pnl).clip(lower=0.0)

        score = (
            base_score
            + (30.0 * mean_pnl)
            + (0.50 * mean_sharpe)
            - (0.004 * mean_turnover)
            - (2.0 * downside_min_sharpe)
            - (0.20 * over_active)
            - (80.0 * negative_pnl)
        )
        return score.rename("train_robust_score")

    rows: list[dict[str, Any]] = []
    for combo in combos:
        stats = evaluate_dataset(
            base_config=base_config,
            matrices=train_matrices,
            combo=combo,
            min_active_ratio=min_active_ratio,
            prefix="train",
        )
        rows.append({**combo, **stats})

    table = pd.DataFrame(rows)
    table["train_robust_score"] = _robust_calibration_score(table)
    table = table.sort_values(
        ["train_robust_score", "train_mean_pnl_net", "train_mean_sharpe_full_raw", "train_score"],
        ascending=False,
    ).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    tuned_cfg = _apply_combo(base_config, best)
    return best, table, tuned_cfg


def _resolve_spread_fee_bps(
    *,
    base_cost_bps: float,
    spread_roundtrip_bps: float | None,
    fees_roundtrip_bps: float | None,
) -> tuple[float, float]:
    base = max(0.0, float(base_cost_bps))
    spread = 0.4 * base if spread_roundtrip_bps is None else max(0.0, float(spread_roundtrip_bps))
    fees = 0.6 * base if fees_roundtrip_bps is None else max(0.0, float(fees_roundtrip_bps))
    return float(spread), float(fees)


def scenario_costs(
    dynamic_cost_bps: pd.Series | None,
    *,
    index: pd.Index,
    base_cost_bps: float,
    spread_roundtrip_bps: float | None,
    fees_roundtrip_bps: float | None,
    scenario: str,
) -> pd.Series:
    sc = str(scenario).strip().lower()
    if sc not in SCENARIO_ORDER:
        raise ValueError(f"Unsupported scenario={scenario!r}")

    spread_bps, fees_bps = _resolve_spread_fee_bps(
        base_cost_bps=base_cost_bps,
        spread_roundtrip_bps=spread_roundtrip_bps,
        fees_roundtrip_bps=fees_roundtrip_bps,
    )
    fixed_cost = spread_bps + fees_bps

    if dynamic_cost_bps is None:
        base_dynamic = pd.Series(float(base_cost_bps), index=index, dtype="float64")
    else:
        base_dynamic = pd.to_numeric(dynamic_cost_bps.reindex(index), errors="coerce")
        base_dynamic = base_dynamic.fillna(float(base_cost_bps)).clip(lower=0.0).astype("float64")

    slippage_component = (base_dynamic - float(fixed_cost)).clip(lower=0.0)
    if sc in {"base", "latency_1bar"}:
        return base_dynamic.rename("dynamic_cost_bps")
    if sc == "fees_x2":
        out = (spread_bps + (2.0 * fees_bps)) + slippage_component
        return out.rename("dynamic_cost_bps")
    if sc == "spread_x2":
        out = ((2.0 * spread_bps) + fees_bps) + slippage_component
        return out.rename("dynamic_cost_bps")
    if sc == "liquidity_half":
        out = float(fixed_cost) + (2.0 * slippage_component)
        return out.rename("dynamic_cost_bps")
    if sc == "combined_worst":
        out = ((2.0 * spread_bps) + (2.0 * fees_bps)) + (2.0 * slippage_component)
        return out.rename("dynamic_cost_bps")
    raise AssertionError("unreachable")


def apply_latency_one_bar(
    decision: pd.Series,
    position_size: pd.Series | None,
    dynamic_cost_bps: pd.Series | None,
    m_t: pd.Series,
) -> tuple[pd.Series, pd.Series | None, pd.Series | None, pd.Series]:
    delayed_decision = decision.shift(1)
    if len(delayed_decision) > 0:
        delayed_decision.iloc[0] = "Risk-off"
    delayed_decision = delayed_decision.fillna("Risk-off").astype("object").rename("decision")

    delayed_size: pd.Series | None = None
    if position_size is not None:
        delayed_size = pd.to_numeric(position_size, errors="coerce").shift(1)
        delayed_size = delayed_size.fillna(0.0).clip(lower=0.0, upper=1.0).rename("position_size")

    delayed_cost: pd.Series | None = None
    if dynamic_cost_bps is not None:
        delayed_cost = pd.to_numeric(dynamic_cost_bps, errors="coerce").shift(1)
        if delayed_cost.dropna().empty:
            fill_value = 0.0
        else:
            fill_value = float(delayed_cost.dropna().iloc[0])
        delayed_cost = delayed_cost.fillna(fill_value).clip(lower=0.0).rename("dynamic_cost_bps")

    delayed_m = pd.to_numeric(m_t, errors="coerce").shift(1).rename("m_t")
    if len(delayed_m) > 0:
        delayed_m.iloc[0] = np.nan
    return delayed_decision, delayed_size, delayed_cost, delayed_m


def _build_aux_regime_series(
    *,
    index: pd.Index,
    regime_frame: pd.DataFrame,
    enabled: bool,
) -> pd.Series:
    if enabled:
        return regime_frame["regime"]
    return pd.Series("transient", index=index, name="regime", dtype="object")


def _build_aux_state_series(
    *,
    index: pd.Index,
    state_frame: pd.DataFrame,
    enabled: bool,
) -> tuple[pd.Series, pd.Series]:
    if enabled:
        return state_frame["T_t"], state_frame["chi_t"]
    return (
        pd.Series(1.0, index=index, name="T_t", dtype="float64"),
        pd.Series(0.0, index=index, name="chi_t", dtype="float64"),
    )


def build_variant_payload(
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    variant: FactorialVariant,
) -> dict[str, Any]:
    premium_frame = frames["premium_frame"]
    signal_premium = frames["signal_premium"]
    onchain_frame = frames["onchain_frame"]
    robust_frame = frames["robust_frame"]
    state_frame = frames["state_frame"]
    regime_frame = frames["regime_frame"]
    m_t = frames["m_t"]
    freq = str(frames["freq"].iloc[0])

    strategy_cfg = build_dataclass(StrategyConfig, config.get("strategy"))
    backtest_cfg = build_dataclass(BacktestConfig, config.get("backtest"))
    exec_cfg = build_dataclass(ExecutionUnifierConfig, config.get("execution_unifier"))
    hawkes_cfg = build_dataclass(HawkesConfig, config.get("hawkes"))
    hawkes_cfg = replace(hawkes_cfg, enabled=True)

    hawkes_quality_pass = False
    hawkes_quality_reason = "disabled_by_variant"
    hawkes_quality_metrics = {
        "hawkes_refit_count": 0.0,
        "hawkes_fit_ok_ratio": 0.0,
        "hawkes_n_unique": 0.0,
        "hawkes_n_std": 0.0,
    }
    hawkes_frame = pd.DataFrame(index=m_t.index)
    if variant.hawkes:
        hawkes_frame = estimate_hawkes_rolling(robust_frame["event"], hawkes_cfg)
        hawkes_quality_pass, hawkes_quality_reason, hawkes_quality_metrics = evaluate_hawkes_quality(
            hawkes_frame,
            hawkes_cfg,
        )

    statmech_effective_on = bool(variant.statmech)
    gating_effective_on = bool(variant.gating and variant.statmech)
    if not variant.gating:
        gating_effective_reason = "disabled_by_variant"
    elif not variant.statmech:
        gating_effective_reason = "requires_statmech"
    else:
        gating_effective_reason = "ok"

    if not variant.hawkes:
        hawkes_effective_on = False
        hawkes_effective_reason = "disabled_by_variant"
    elif not gating_effective_on:
        hawkes_effective_on = False
        hawkes_effective_reason = "gating_effective_off"
    elif not hawkes_quality_pass:
        hawkes_effective_on = False
        hawkes_effective_reason = f"quality_fail:{hawkes_quality_reason}"
    else:
        hawkes_effective_on = True
        hawkes_effective_reason = "ok"

    aux_T_t, aux_chi_t = _build_aux_state_series(
        index=m_t.index,
        state_frame=state_frame,
        enabled=statmech_effective_on,
    )
    aux_regime = _build_aux_regime_series(
        index=m_t.index,
        regime_frame=regime_frame,
        enabled=statmech_effective_on,
    )
    aux_frame = build_decisions(
        m_t=m_t,
        T_t=aux_T_t,
        chi_t=aux_chi_t,
        sigma_hat=robust_frame["sigma_hat"],
        regime=aux_regime,
        depeg_flag=premium_frame["depeg_flag"],
        event=robust_frame["event"],
        stablecoin_proxy=premium_frame["stablecoin_proxy"],
        onchain_proxy=onchain_frame.get("onchain_proxy"),
        onchain_usdc_minus_1=onchain_frame.get("onchain_usdc_minus_1"),
        onchain_usdt_minus_1=onchain_frame.get("onchain_usdt_minus_1"),
        n_t=None,
        premium=signal_premium,
        slippage_curve=None,
        execution_unifier_cfg=exec_cfg,
        base_cost_bps=backtest_cfg.cost_bps,
        cfg=strategy_cfg,
    )

    if gating_effective_on:
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
            n_t=(hawkes_frame["n_t"] if (hawkes_effective_on and "n_t" in hawkes_frame.columns) else None),
            premium=signal_premium,
            slippage_curve=None,
            execution_unifier_cfg=exec_cfg,
            base_cost_bps=backtest_cfg.cost_bps,
            cfg=strategy_cfg,
        )
        decision = decision_frame["decision"]
        position_size = decision_frame.get("position_size")
        dynamic_cost_bps = decision_frame.get("edge_cost_bps_opt")
    else:
        if statmech_effective_on:
            threshold = (strategy_cfg.entry_k * state_frame["T_t"] * robust_frame["sigma_hat"]).rename("entry_threshold")
        else:
            threshold = (strategy_cfg.entry_k * robust_frame["sigma_hat"]).rename("entry_threshold")
        decision = simple_decision(
            m_t,
            threshold=threshold,
            depeg_flag=premium_frame["depeg_flag"],
        )
        dynamic_cost_bps = aux_frame.get("edge_cost_bps_opt")
        if exec_cfg.enabled:
            optimal_size = pd.to_numeric(aux_frame.get("optimal_size"), errors="coerce").clip(lower=0.0, upper=1.0)
            optimal_size = optimal_size.fillna(0.0)
            position_size = pd.Series(0.0, index=m_t.index, dtype="float64")
            trade_mask = decision.eq("Trade")
            riskoff_mask = decision.eq("Risk-off")
            position_size.loc[trade_mask] = optimal_size.loc[trade_mask]
            position_size.loc[(~trade_mask) & (~riskoff_mask)] = float(np.clip(exec_cfg.widen_floor_size, 0.0, 1.0))
            position_size = position_size.rename("position_size")
        else:
            position_size = None

    return {
        "variant_id": variant.variant_id,
        "decision": decision.rename("decision"),
        "position_size": position_size,
        "dynamic_cost_bps": dynamic_cost_bps,
        "m_t": m_t.rename("m_t"),
        "premium": signal_premium.rename("premium"),
        "freq": freq,
        "backtest_cfg": backtest_cfg,
        "execution_unifier_cfg": exec_cfg,
        "statmech_effective_on": bool(statmech_effective_on),
        "gating_effective_on": bool(gating_effective_on),
        "gating_effective_reason": str(gating_effective_reason),
        "hawkes_requested": bool(variant.hawkes),
        "hawkes_quality_pass": bool(hawkes_quality_pass),
        "hawkes_quality_reason": str(hawkes_quality_reason),
        "hawkes_effective_on": bool(hawkes_effective_on),
        "hawkes_effective_reason": str(hawkes_effective_reason),
        "hawkes_refit_count": float(hawkes_quality_metrics.get("hawkes_refit_count", 0.0)),
        "hawkes_fit_ok_ratio": float(hawkes_quality_metrics.get("hawkes_fit_ok_ratio", 0.0)),
        "hawkes_n_unique": float(hawkes_quality_metrics.get("hawkes_n_unique", 0.0)),
        "hawkes_n_std": float(hawkes_quality_metrics.get("hawkes_n_std", 0.0)),
    }


def _run_variant_scenario(payload: dict[str, Any], scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_name = str(scenario).strip().lower()
    if scenario_name not in SCENARIO_ORDER:
        raise ValueError(f"Unsupported scenario={scenario!r}")

    decision = payload["decision"]
    position_size = payload["position_size"]
    m_t = payload["m_t"]
    premium = payload["premium"]
    backtest_cfg: BacktestConfig = payload["backtest_cfg"]
    exec_cfg: ExecutionUnifierConfig = payload["execution_unifier_cfg"]
    dynamic_cost_raw = payload["dynamic_cost_bps"]

    dynamic_cost = scenario_costs(
        dynamic_cost_raw,
        index=premium.index,
        base_cost_bps=float(backtest_cfg.cost_bps),
        spread_roundtrip_bps=exec_cfg.spread_roundtrip_bps,
        fees_roundtrip_bps=exec_cfg.fees_roundtrip_bps,
        scenario=scenario_name,
    )

    decision_exec = decision
    size_exec = position_size
    m_exec = m_t
    cost_exec: pd.Series | None = dynamic_cost
    if scenario_name in {"latency_1bar", "combined_worst"}:
        decision_exec, size_exec, cost_exec, m_exec = apply_latency_one_bar(
            decision_exec,
            size_exec,
            cost_exec,
            m_exec,
        )

    trade_log, metrics = run_backtest(
        premium,
        decision_exec,
        m_exec,
        freq=str(payload["freq"]),
        cost_bps=float(backtest_cfg.cost_bps),
        position_size=size_exec,
        dynamic_cost_bps=cost_exec,
        position_mode=backtest_cfg.position_mode,
        exit_on_widen=backtest_cfg.exit_on_widen,
        exit_on_mean_reversion=backtest_cfg.exit_on_mean_reversion,
        min_holding_bars=backtest_cfg.min_holding_bars,
        max_holding_bars=backtest_cfg.max_holding_bars,
        widen_floor_size=backtest_cfg.widen_floor_size,
    )

    out = {
        "scenario": scenario_name,
        **metrics,
        "statmech_effective_on": bool(payload["statmech_effective_on"]),
        "gating_effective_on": bool(payload["gating_effective_on"]),
        "gating_effective_reason": str(payload["gating_effective_reason"]),
        "hawkes_requested": bool(payload["hawkes_requested"]),
        "hawkes_quality_pass": bool(payload["hawkes_quality_pass"]),
        "hawkes_quality_reason": str(payload["hawkes_quality_reason"]),
        "hawkes_effective_on": bool(payload["hawkes_effective_on"]),
        "hawkes_effective_reason": str(payload["hawkes_effective_reason"]),
        "hawkes_refit_count": float(payload["hawkes_refit_count"]),
        "hawkes_fit_ok_ratio": float(payload["hawkes_fit_ok_ratio"]),
        "hawkes_n_unique": float(payload["hawkes_n_unique"]),
        "hawkes_n_std": float(payload["hawkes_n_std"]),
    }
    return trade_log, out


def run_factorial_stress_for_matrix(
    *,
    config: dict[str, Any],
    matrix: pd.DataFrame,
    split_id: int,
    test_episode: str,
) -> pd.DataFrame:
    frames_by_premium = {
        "naive": compute_core_frames(config, matrix, premium_leg="naive"),
        "debiased": compute_core_frames(config, matrix, premium_leg="debiased"),
    }

    rows: list[dict[str, Any]] = []
    for variant in build_factorial_variants():
        frames = frames_by_premium[str(variant.premium).strip().lower()]
        payload = build_variant_payload(config, frames, variant)

        for scenario in SCENARIO_ORDER:
            _, metrics_row = _run_variant_scenario(payload, scenario)
            rows.append(
                {
                    "split_id": int(split_id),
                    "test_episode": str(test_episode),
                    "variant_id": str(variant.variant_id),
                    "premium_mode": str(variant.premium),
                    "gating_requested": bool(variant.gating),
                    "statmech_requested": bool(variant.statmech),
                    "hawkes_requested_variant": bool(variant.hawkes),
                    **metrics_row,
                }
            )
    return pd.DataFrame(rows)


def _scenario_pass(row: pd.Series) -> bool:
    sharpe = float(pd.to_numeric(row.get("sharpe"), errors="coerce"))
    pnl_net = float(pd.to_numeric(row.get("pnl_net"), errors="coerce"))
    return bool(np.isfinite(sharpe) and np.isfinite(pnl_net) and sharpe > 0.0 and pnl_net > 0.0)


def compute_strict_verdict_table(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = stress_df.groupby(["split_id", "test_episode", "variant_id"], sort=True)
    for (split_id, test_episode, variant_id), block in grouped:
        scenario_rows = {str(s): g.iloc[0] for s, g in block.groupby("scenario", sort=False, observed=False)}
        base_row = scenario_rows.get("base")
        base_pass = _scenario_pass(base_row) if base_row is not None else False

        single_flags: dict[str, bool] = {}
        for sc in SINGLE_STRESS_SCENARIOS:
            row_sc = scenario_rows.get(sc)
            single_flags[f"pass_{sc}"] = _scenario_pass(row_sc) if row_sc is not None else False
        single_pass_count = int(sum(bool(v) for v in single_flags.values()))
        singles_majority_pass = bool(single_pass_count >= 3)

        combined_row = scenario_rows.get("combined_worst")
        combined_pass = _scenario_pass(combined_row) if combined_row is not None else False
        verdict_pass = bool(base_pass and singles_majority_pass)

        base_sharpe = float(pd.to_numeric(base_row.get("sharpe"), errors="coerce")) if base_row is not None else np.nan
        base_pnl = float(pd.to_numeric(base_row.get("pnl_net"), errors="coerce")) if base_row is not None else np.nan
        combined_sharpe = (
            float(pd.to_numeric(combined_row.get("sharpe"), errors="coerce")) if combined_row is not None else np.nan
        )
        combined_pnl = (
            float(pd.to_numeric(combined_row.get("pnl_net"), errors="coerce")) if combined_row is not None else np.nan
        )

        rows.append(
            {
                "split_id": int(split_id),
                "test_episode": str(test_episode),
                "variant_id": str(variant_id),
                "base_sharpe": base_sharpe,
                "base_pnl_net": base_pnl,
                "base_pass": bool(base_pass),
                **single_flags,
                "single_pass_count": int(single_pass_count),
                "singles_majority_pass": bool(singles_majority_pass),
                "combined_worst_sharpe": combined_sharpe,
                "combined_worst_pnl_net": combined_pnl,
                "combined_worst_pass": bool(combined_pass),
                "verdict_pass": bool(verdict_pass),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["split_id", "variant_id"]).reset_index(drop=True)


def _summarize_walkforward_row(
    *,
    split: dict[str, Any],
    best: dict[str, Any],
    split_stress: pd.DataFrame,
) -> dict[str, Any]:
    reference = split_stress.loc[
        split_stress["variant_id"].eq(REFERENCE_VARIANT_ID) & split_stress["scenario"].eq("base")
    ]
    if reference.empty:
        reference_sharpe = np.nan
        reference_pnl = np.nan
    else:
        reference_sharpe = float(reference.iloc[0]["sharpe"])
        reference_pnl = float(reference.iloc[0]["pnl_net"])

    return {
        "split_id": int(split["split_id"]),
        "train_episode_count": int(len(split["train_ids"])),
        "train_episode_ids": "|".join(split["train_ids"]),
        "test_episode_id": str(split["test_id"]),
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "test_start": split["test_start"],
        "test_end": split["test_end"],
        "selected_entry_k": float(best["entry_k"]),
        "selected_t_widen_quantile": float(best["t_widen_quantile"]),
        "selected_chi_widen_quantile": float(best["chi_widen_quantile"]),
        "selected_stress_quantile": float(best["stress_quantile"]),
        "selected_recovery_quantile": float(best["recovery_quantile"]),
        "selected_train_robust_score": float(best["train_robust_score"]),
        "train_score": float(best["train_score"]),
        "train_mean_sharpe_full_raw": float(best["train_mean_sharpe_full_raw"]),
        "train_mean_pnl_net": float(best["train_mean_pnl_net"]),
        "reference_variant_id": REFERENCE_VARIANT_ID,
        "reference_base_sharpe": reference_sharpe,
        "reference_base_pnl_net": reference_pnl,
    }


def _write_summary_markdown(
    *,
    output_dir: Path,
    walkforward_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    verdict_df: pd.DataFrame,
) -> Path:
    summary_path = output_dir / "robustness_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Robustness Summary\n\n")
        handle.write("Strict verdict rule:\n")
        handle.write("- `PASS` iff base scenario has `Sharpe > 0` and `PnL net > 0`\n")
        handle.write("- and at least 3/4 single stress scenarios pass (`fees_x2`, `spread_x2`, `latency_1bar`, `liquidity_half`).\n")
        handle.write("\n")

        handle.write("## Run Statistics\n\n")
        handle.write(f"- splits: `{int(walkforward_df.shape[0])}`\n")
        handle.write(f"- base ablation rows: `{int(ablation_df.shape[0])}`\n")
        handle.write(f"- stress rows: `{int(stress_df.shape[0])}`\n")
        handle.write(f"- verdict rows: `{int(verdict_df.shape[0])}`\n")
        handle.write("\n")

        if not verdict_df.empty:
            overall_pass_rate = float(verdict_df["verdict_pass"].mean())
            ref = verdict_df.loc[verdict_df["variant_id"].eq(REFERENCE_VARIANT_ID)]
            ref_pass_rate = float(ref["verdict_pass"].mean()) if not ref.empty else float("nan")
            handle.write("## Verdict Rates\n\n")
            handle.write(f"- overall verdict pass rate: `{overall_pass_rate:.3f}`\n")
            handle.write(f"- reference variant (`{REFERENCE_VARIANT_ID}`) pass rate: `{ref_pass_rate:.3f}`\n")
            handle.write("\n")

        handle.write("## Walk-Forward Splits\n\n")
        if walkforward_df.empty:
            handle.write("- none\n\n")
        else:
            handle.write("```text\n")
            handle.write(walkforward_df.to_string(index=False))
            handle.write("\n```\n\n")

        handle.write("## Reference Variant Verdict\n\n")
        ref_rows = verdict_df.loc[verdict_df["variant_id"].eq(REFERENCE_VARIANT_ID)]
        if ref_rows.empty:
            handle.write("- no rows\n")
        else:
            handle.write("```text\n")
            handle.write(ref_rows.to_string(index=False))
            handle.write("\n```\n")
    return summary_path


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = Path(args.output_dir)

    files = resolve_episode_files(args.episodes)
    files, dropped = apply_default_episode_filters(
        files,
        include_smoke=bool(args.include_smoke),
    )
    if dropped:
        print(f"Excluded smoke episodes by default: {len(dropped)}")
        for path in dropped:
            print(f"- {path}")
    if not files:
        raise RuntimeError("No episode files available after applying default filters.")

    all_matrices = _load_matrices(files, base_config)
    compatible, skipped = filter_compatible_matrices(base_config, all_matrices)
    if skipped:
        print("Skipped incompatible episodes:")
        for path, detail in skipped.items():
            print(f"- {path} [{detail.get('reason_code', 'pipeline_error')}]: {detail.get('reason', '')}")
    if len(compatible) < max(2, int(args.min_train_episodes) + 1):
        reason = (
            "Not enough compatible episodes for walk-forward. "
            f"compatible={len(compatible)} min_required={max(2, int(args.min_train_episodes) + 1)}"
        )
        _write_graceful_compatibility_outputs(
            output_dir=output_dir,
            skipped=skipped,
            reason_code="insufficient_compatible_episodes",
            reason=reason,
        )
        return

    sorted_items = _sort_matrix_items(compatible)
    splits = build_walkforward_splits(
        sorted_items,
        min_train_episodes=int(args.min_train_episodes),
    )
    if not splits:
        reason = "No walk-forward split generated after compatibility filtering."
        _write_graceful_compatibility_outputs(
            output_dir=output_dir,
            skipped=skipped,
            reason_code="no_walkforward_splits",
            reason=reason,
        )
        return

    grid = CandidateGrid(
        entry_k=parse_float_list(args.entry_k),
        t_widen_quantile=parse_float_list(args.t_widen),
        chi_widen_quantile=parse_float_list(args.chi_widen),
        stress_quantile=parse_float_list(args.stress),
        recovery_quantile=parse_float_list(args.recovery),
    )
    combos = build_param_grid(grid)
    if args.max_combos and args.max_combos > 0 and len(combos) > int(args.max_combos):
        rnd = random.Random(int(args.seed))
        combos = rnd.sample(combos, int(args.max_combos))

    walkforward_rows: list[dict[str, Any]] = []
    stress_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []

    print(f"Walk-forward splits: {len(splits)}")
    print(f"Calibration combinations per split: {len(combos)}")

    for split in splits:
        split_id = int(split["split_id"])
        train_matrices = split["train_matrices"]
        test_matrix = split["test_matrix"]
        test_id = str(split["test_id"])
        print(f"[split {split_id}] calibrating on {len(train_matrices)} episodes; testing on {test_id}")

        best, calibration_table, tuned_config = _calibrate_split(
            base_config=base_config,
            train_matrices=train_matrices,
            combos=combos,
            min_active_ratio=float(args.min_active_ratio),
        )
        calibration_block = calibration_table.copy()
        calibration_block.insert(0, "split_id", split_id)
        calibration_rows.append(calibration_block)

        split_stress = run_factorial_stress_for_matrix(
            config=tuned_config,
            matrix=test_matrix,
            split_id=split_id,
            test_episode=test_id,
        )
        stress_rows.append(split_stress)
        walkforward_rows.append(
            _summarize_walkforward_row(
                split=split,
                best=best,
                split_stress=split_stress,
            )
        )

    walkforward_df = pd.DataFrame(walkforward_rows).sort_values("split_id").reset_index(drop=True)
    stress_df = pd.concat(stress_rows, axis=0, ignore_index=True).copy()
    stress_df["scenario"] = pd.Categorical(stress_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    stress_df = stress_df.sort_values(["split_id", "variant_id", "scenario"]).reset_index(drop=True)

    ablation_df = stress_df.loc[stress_df["scenario"].eq("base")].copy()
    ablation_df["scenario"] = ablation_df["scenario"].astype(str)
    verdict_df = compute_strict_verdict_table(stress_df)
    stress_df["scenario"] = stress_df["scenario"].astype(str)

    output_dir.mkdir(parents=True, exist_ok=True)

    walkforward_path = output_dir / "walkforward_split_metrics.csv"
    ablation_path = output_dir / "ablation_factorial_oos.csv"
    stress_path = output_dir / "stress_matrix_oos.csv"
    verdict_path = output_dir / "robustness_verdict.csv"
    calibration_path = output_dir / "walkforward_calibration_details.csv"

    walkforward_df.to_csv(walkforward_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)
    stress_df.to_csv(stress_path, index=False)
    verdict_df.to_csv(verdict_path, index=False)
    if calibration_rows:
        pd.concat(calibration_rows, axis=0, ignore_index=True).to_csv(calibration_path, index=False)

    summary_path = _write_summary_markdown(
        output_dir=output_dir,
        walkforward_df=walkforward_df,
        ablation_df=ablation_df,
        stress_df=stress_df,
        verdict_df=verdict_df,
    )

    print("Robustness report completed.")
    print(f"- walkforward_split_metrics: {walkforward_path}")
    print(f"- ablation_factorial_oos: {ablation_path}")
    print(f"- stress_matrix_oos: {stress_path}")
    print(f"- robustness_verdict: {verdict_path}")
    print(f"- calibration_details: {calibration_path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
