from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.robust_filter import window_to_observations


@dataclass(frozen=True)
class StatMechConfig:
    window: str = "1h"
    entropy_bins: int = 17
    entropy_clip: float = 6.0
    min_period_fraction: float = 0.25


def _entropy_from_window(values: np.ndarray, bins: int, clip: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.shape[0] < 3:
        return np.nan
    clipped = np.clip(finite, -clip, clip)
    hist, _ = np.histogram(clipped, bins=bins, range=(-clip, clip))
    total = hist.sum()
    if total <= 0:
        return np.nan
    probs = hist.astype(float) / total
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / np.log(bins)


def _temperature_from_window(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.shape[0] < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def _variance_from_window(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.shape[0] < 2:
        return np.nan
    return float(np.var(finite))


def compute_state_variables(
    z_t: pd.Series,
    m_t: pd.Series,
    *,
    window_obs: int,
    entropy_bins: int = 17,
    entropy_clip: float = 6.0,
    min_period_fraction: float = 0.25,
) -> pd.DataFrame:
    if window_obs < 2:
        raise ValueError("window_obs must be >= 2")
    if entropy_bins < 5:
        raise ValueError("entropy_bins must be >= 5")

    min_periods = max(2, int(np.ceil(window_obs * min_period_fraction)))
    entropy = z_t.rolling(window_obs, min_periods=min_periods).apply(
        _entropy_from_window,
        raw=True,
        args=(entropy_bins, entropy_clip),
    )
    temperature = z_t.rolling(window_obs, min_periods=min_periods).apply(
        _temperature_from_window,
        raw=True,
    )
    susceptibility = m_t.rolling(window_obs, min_periods=min_periods).apply(
        _variance_from_window,
        raw=True,
    )

    return pd.DataFrame(
        {
            "H_t": entropy.astype(float),
            "T_t": temperature.astype(float),
            "chi_t": susceptibility.astype(float),
        },
        index=z_t.index,
    )


def build_statmech_frame(
    z_t: pd.Series,
    m_t: pd.Series,
    *,
    cfg: StatMechConfig,
    freq: str,
) -> pd.DataFrame:
    window_obs = window_to_observations(cfg.window, freq)
    return compute_state_variables(
        z_t,
        m_t,
        window_obs=window_obs,
        entropy_bins=cfg.entropy_bins,
        entropy_clip=cfg.entropy_clip,
        min_period_fraction=cfg.min_period_fraction,
    )
