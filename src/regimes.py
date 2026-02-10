from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.thresholds import quantile_threshold

VALID_ZSCORE_MODES = {"expanding", "rolling"}


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
    zscore_mode: str = "expanding"
    zscore_min_periods: int = 120
    zscore_window: int | None = None
    zscore_scale_floor: float = 1e-12


def _window_mad(window: np.ndarray) -> float:
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return np.nan
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def _causal_robust_zscore(
    series: pd.Series,
    *,
    mode: str = "expanding",
    min_periods: int = 120,
    window: int | None = None,
    scale_floor: float = 1e-12,
) -> pd.Series:
    if mode not in VALID_ZSCORE_MODES:
        raise ValueError(f"Unsupported zscore mode: {mode}. Expected one of {sorted(VALID_ZSCORE_MODES)}")

    values = series.astype(float)
    min_periods = max(2, int(min_periods))
    floor = max(float(scale_floor), 1e-12)

    if mode == "expanding":
        median = values.expanding(min_periods=min_periods).median()
        mad = values.expanding(min_periods=min_periods).apply(_window_mad, raw=True)
        fallback_std = values.expanding(min_periods=min_periods).std(ddof=0)
    else:
        if window is None:
            raise ValueError("window must be provided when zscore_mode='rolling'")
        window = max(2, int(window))
        roll_min = min(min_periods, window)
        median = values.rolling(window=window, min_periods=roll_min).median()
        mad = values.rolling(window=window, min_periods=roll_min).apply(_window_mad, raw=True)
        fallback_std = values.rolling(window=window, min_periods=roll_min).std(ddof=0)

    # Strictly causal: statistics at t are computed from history up to t-1.
    median = median.shift(1)
    mad = mad.shift(1)
    fallback_std = fallback_std.shift(1)

    scale = 1.4826 * mad
    scale = scale.where(scale > floor, fallback_std)
    scale = scale.clip(lower=floor)
    z = (values - median) / scale
    return z.rename(f"{series.name or 'series'}_z")


def build_regime_score(
    H_t: pd.Series,
    T_t: pd.Series,
    chi_t: pd.Series,
    *,
    w_T: float,
    w_chi: float,
    w_H: float,
    zscore_mode: str = "expanding",
    zscore_min_periods: int = 120,
    zscore_window: int | None = None,
    zscore_scale_floor: float = 1e-12,
) -> pd.Series:
    z_H = _causal_robust_zscore(
        H_t,
        mode=zscore_mode,
        min_periods=zscore_min_periods,
        window=zscore_window,
        scale_floor=zscore_scale_floor,
    )
    z_T = _causal_robust_zscore(
        T_t,
        mode=zscore_mode,
        min_periods=zscore_min_periods,
        window=zscore_window,
        scale_floor=zscore_scale_floor,
    )
    z_chi = _causal_robust_zscore(
        chi_t,
        mode=zscore_mode,
        min_periods=zscore_min_periods,
        window=zscore_window,
        scale_floor=zscore_scale_floor,
    )
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
        zscore_mode=cfg.zscore_mode,
        zscore_min_periods=cfg.zscore_min_periods,
        zscore_window=cfg.zscore_window,
        zscore_scale_floor=cfg.zscore_scale_floor,
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
