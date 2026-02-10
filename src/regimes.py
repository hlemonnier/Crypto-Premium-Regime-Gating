from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.thresholds import quantile_threshold


@dataclass(frozen=True)
class RegimeConfig:
    weight_temperature: float = 0.45
    weight_susceptibility: float = 0.45
    weight_entropy: float = 0.10
    smooth_span: int = 60
    cp_window: int = 30
    stress_quantile: float = 0.85
    recovery_quantile: float = 0.65
    cp_quantile: float = 0.90
    threshold_mode: str = "expanding"
    threshold_min_periods: int = 120
    threshold_window: int | None = None


def _robust_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.nanstd(values))
    scale = max(scale, 1e-12)
    return (values - median) / scale


def build_regime_score(
    H_t: pd.Series,
    T_t: pd.Series,
    chi_t: pd.Series,
    *,
    w_T: float,
    w_chi: float,
    w_H: float,
) -> pd.Series:
    z_H = _robust_zscore(H_t)
    z_T = _robust_zscore(T_t)
    z_chi = _robust_zscore(chi_t)
    score = (w_T * z_T) + (w_chi * z_chi) + (w_H * z_H)
    return score.rename("regime_score")


def detect_regime(
    regime_score: pd.Series,
    *,
    smooth_span: int = 60,
    cp_window: int = 30,
    stress_quantile: float = 0.85,
    recovery_quantile: float = 0.65,
    cp_quantile: float = 0.90,
    threshold_mode: str = "expanding",
    threshold_min_periods: int = 120,
    threshold_window: int | None = None,
) -> pd.DataFrame:
    score = regime_score.astype(float).copy()
    smooth = score.ewm(span=max(2, smooth_span), adjust=False, min_periods=5).mean()
    level_shift = smooth.diff().abs().rolling(max(2, cp_window), min_periods=5).mean()

    high = quantile_threshold(
        score,
        stress_quantile,
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="stress_threshold",
    )
    low = quantile_threshold(
        score,
        recovery_quantile,
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="recovery_threshold",
    )
    cp_high = quantile_threshold(
        level_shift,
        cp_quantile,
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="cp_stress_threshold",
    )
    cp_low = quantile_threshold(
        level_shift,
        max(0.5, recovery_quantile - 0.1),
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="cp_recovery_threshold",
    )

    states: list[str] = []
    state = "transient"
    for s, cp, high_t, low_t, cp_high_t, cp_low_t in zip(
        score.to_numpy(),
        level_shift.to_numpy(),
        high.to_numpy(),
        low.to_numpy(),
        cp_high.to_numpy(),
        cp_low.to_numpy(),
    ):
        if not np.isfinite(s):
            states.append(state)
            continue
        if state == "transient":
            stress_from_score = np.isfinite(high_t) and s >= high_t
            stress_from_shift = np.isfinite(cp) and np.isfinite(cp_high_t) and cp >= cp_high_t
            if stress_from_score or stress_from_shift:
                state = "stress"
        else:
            recover_score = np.isfinite(low_t) and s <= low_t
            recover_shift = (not np.isfinite(cp)) or (np.isfinite(cp_low_t) and cp <= cp_low_t)
            if recover_score and recover_shift:
                state = "transient"
        states.append(state)

    regime = pd.Series(states, index=score.index, name="regime")
    boundary = regime.ne(regime.shift(1)).fillna(False)
    segment_id = boundary.cumsum().astype(int).rename("segment_id")
    change_point = (boundary | (level_shift.ge(cp_high) & cp_high.notna())).fillna(False).rename("change_point")

    return pd.concat(
        [
            score.rename("regime_score"),
            smooth.rename("regime_score_smooth"),
            level_shift.rename("regime_level_shift"),
            high,
            low,
            cp_high,
            cp_low,
            change_point,
            regime,
            segment_id,
        ],
        axis=1,
    )


def build_regime_frame(states: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    score = build_regime_score(
        states["H_t"],
        states["T_t"],
        states["chi_t"],
        w_T=cfg.weight_temperature,
        w_chi=cfg.weight_susceptibility,
        w_H=cfg.weight_entropy,
    )
    return detect_regime(
        score,
        smooth_span=cfg.smooth_span,
        cp_window=cfg.cp_window,
        stress_quantile=cfg.stress_quantile,
        recovery_quantile=cfg.recovery_quantile,
        cp_quantile=cfg.cp_quantile,
        threshold_mode=cfg.threshold_mode,
        threshold_min_periods=cfg.threshold_min_periods,
        threshold_window=cfg.threshold_window,
    )
