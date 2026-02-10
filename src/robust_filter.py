from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustFilterConfig:
    window: str = "1h"
    z_threshold: float = 3.0
    sigma_floor: float = 1e-6
    min_period_fraction: float = 0.25


def window_to_observations(window: str, freq: str) -> int:
    win_td = pd.to_timedelta(window)
    freq_td = pd.to_timedelta(freq)
    obs = int(np.ceil(win_td / freq_td))
    return max(obs, 2)


def robust_filter(
    premium: pd.Series,
    *,
    window_obs: int,
    z_threshold: float = 3.0,
    sigma_floor: float = 1e-6,
    min_period_fraction: float = 0.25,
) -> pd.DataFrame:
    if window_obs < 2:
        raise ValueError("window_obs must be >= 2")
    if not 0 < min_period_fraction <= 1:
        raise ValueError("min_period_fraction must be in (0, 1]")

    p = premium.astype(float).copy()
    min_periods = max(2, int(np.ceil(window_obs * min_period_fraction)))

    p_smooth = p.rolling(window_obs, min_periods=min_periods).median().rename("p_smooth")
    abs_dev = (p - p_smooth).abs()
    sigma_hat = (1.4826 * abs_dev.rolling(window_obs, min_periods=min_periods).median()).rename(
        "sigma_hat"
    )

    finite_sigma = sigma_hat[np.isfinite(sigma_hat)]
    dynamic_floor = (
        float(np.nanpercentile(finite_sigma.to_numpy(), 5) * 0.25)
        if finite_sigma.shape[0] >= 20
        else sigma_floor
    )
    floor = max(sigma_floor, dynamic_floor, 1e-12)
    sigma_hat = sigma_hat.clip(lower=floor)

    z_t = ((p - p_smooth) / sigma_hat).rename("z_t")
    events = z_t.abs().gt(z_threshold).fillna(False).rename("event")

    return pd.concat([p_smooth, sigma_hat, z_t, events], axis=1)


def build_robust_frame(
    premium: pd.Series,
    *,
    cfg: RobustFilterConfig,
    freq: str,
) -> pd.DataFrame:
    window_obs = window_to_observations(cfg.window, freq)
    return robust_filter(
        premium,
        window_obs=window_obs,
        z_threshold=cfg.z_threshold,
        sigma_floor=cfg.sigma_floor,
        min_period_fraction=cfg.min_period_fraction,
    )
