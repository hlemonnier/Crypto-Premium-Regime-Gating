from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class HawkesConfig:
    enabled: bool = True
    warmup: str = "48h"
    rolling_window: str = "12h"
    refit_interval: str = "30min"
    stability_epsilon: float = 0.01
    min_events_for_fit: int = 20
    auto_warmup_fraction: float = 0.35
    auto_min_warmup: str = "4h"
    auto_adjust_min_events: bool = True
    auto_min_events_floor: int = 5


@dataclass(frozen=True)
class HawkesParams:
    mu: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta


def _negative_log_likelihood(
    unconstrained: np.ndarray,
    event_times: np.ndarray,
    horizon: float,
    epsilon: float,
) -> float:
    if event_times.size == 0 or horizon <= 0:
        return 1e12

    mu = float(np.exp(unconstrained[0]))
    alpha = float(np.exp(unconstrained[1]))
    beta = float(alpha + epsilon + np.exp(unconstrained[2]))

    if not np.isfinite(mu + alpha + beta):
        return 1e12

    loglik = 0.0
    decay_sum = 0.0
    prev = 0.0
    for t in event_times:
        dt = t - prev
        decay_sum *= np.exp(-beta * dt)
        intensity = mu + alpha * decay_sum
        if intensity <= 0 or not np.isfinite(intensity):
            return 1e12
        loglik += np.log(intensity)
        decay_sum += 1.0
        prev = t

    integral = mu * horizon + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (horizon - event_times)))
    nll = -(loglik - integral)
    if not np.isfinite(nll):
        return 1e12
    return float(nll)


def fit_exponential_hawkes(
    event_times: np.ndarray,
    horizon: float,
    *,
    stability_epsilon: float = 0.01,
    maxiter: int = 300,
) -> HawkesParams | None:
    if event_times.size < 2 or horizon <= 0:
        return None

    inter = np.diff(event_times)
    finite_inter = inter[np.isfinite(inter) & (inter > 0)]
    median_inter = float(np.median(finite_inter)) if finite_inter.size else max(horizon / event_times.size, 1.0)
    beta0 = 1.0 / max(median_inter, 1.0)
    alpha0 = 0.4 * beta0
    rate = event_times.size / horizon
    mu0 = max(rate * 0.6, 1e-8)

    x0 = np.array([np.log(mu0), np.log(alpha0), np.log(max(beta0 - alpha0, 1e-6))], dtype=float)
    result = minimize(
        _negative_log_likelihood,
        x0,
        args=(event_times, horizon, stability_epsilon),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    if not result.success:
        return None

    mu = float(np.exp(result.x[0]))
    alpha = float(np.exp(result.x[1]))
    beta = float(alpha + stability_epsilon + np.exp(result.x[2]))
    if not np.isfinite(mu + alpha + beta) or beta <= alpha + stability_epsilon:
        return None
    return HawkesParams(mu=mu, alpha=alpha, beta=beta)


def _window_events(
    events: pd.Series,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> np.ndarray:
    window = events.loc[(events.index > window_start) & (events.index <= window_end)]
    idx = window[window].index
    if idx.empty:
        return np.array([], dtype=float)
    return (idx - window_start).total_seconds().to_numpy(dtype=float)


def estimate_hawkes_rolling(events: pd.Series, cfg: HawkesConfig) -> pd.DataFrame:
    if not isinstance(events.index, pd.DatetimeIndex):
        raise TypeError("events must use a DatetimeIndex")

    event_series = events.fillna(False).astype(bool).sort_index()
    index = event_series.index
    if index.empty:
        return pd.DataFrame(index=index)

    warmup = pd.to_timedelta(cfg.warmup)
    rolling_window = pd.to_timedelta(cfg.rolling_window)
    refit = pd.to_timedelta(cfg.refit_interval)
    total_span = index[-1] - index[0]

    if total_span > pd.Timedelta(0):
        fraction = float(np.clip(cfg.auto_warmup_fraction, 0.0, 1.0))
        warmup_cap = total_span * fraction
        warmup_floor = min(pd.to_timedelta(cfg.auto_min_warmup), total_span)
        if warmup_cap > pd.Timedelta(0):
            warmup = min(warmup, warmup_cap)
        warmup = max(warmup, warmup_floor)
        warmup = min(warmup, total_span)
        rolling_window = min(rolling_window, total_span)

    if rolling_window <= pd.Timedelta(0):
        rolling_window = pd.to_timedelta("1min")
    if refit <= pd.Timedelta(0):
        refit = pd.to_timedelta("1min")

    lambda_t = pd.Series(np.nan, index=index, name="lambda_t")
    n_t = pd.Series(np.nan, index=index, name="n_t")
    mu_t = pd.Series(np.nan, index=index, name="mu_t")
    alpha_t = pd.Series(np.nan, index=index, name="alpha_t")
    beta_t = pd.Series(np.nan, index=index, name="beta_t")
    fit_ok = pd.Series(False, index=index, name="hawkes_fit_ok")

    start_time = index[0]
    effective_min_events = max(2, int(cfg.min_events_for_fit))
    if cfg.auto_adjust_min_events:
        eligible_idx = event_series.index[(event_series.index - start_time) >= warmup]
        if len(eligible_idx) > 0:
            rolling_events = event_series.rolling(rolling_window).sum().loc[eligible_idx]
            max_events = int(np.nanmax(rolling_events.to_numpy(dtype=float))) if not rolling_events.empty else 0
            floor = max(2, int(cfg.auto_min_events_floor))
            if max_events > 0:
                effective_min_events = min(
                    max_events,
                    max(floor, min(effective_min_events, max_events)),
                )
            else:
                effective_min_events = floor

    params: HawkesParams | None = None
    last_refit: pd.Timestamp | None = None

    event_index = event_series[event_series].index
    for ts in index:
        if ts - start_time < warmup:
            continue

        needs_refit = last_refit is None or (ts - last_refit) >= refit
        if needs_refit:
            window_start = ts - rolling_window
            event_times = _window_events(event_series, window_start, ts)
            horizon = float((ts - window_start).total_seconds())
            if event_times.size >= effective_min_events:
                fitted = fit_exponential_hawkes(
                    event_times,
                    horizon,
                    stability_epsilon=cfg.stability_epsilon,
                )
                if fitted is not None:
                    params = fitted
                    last_refit = ts

        if params is None:
            continue

        window_start = ts - rolling_window
        past_events = event_index[(event_index > window_start) & (event_index <= ts)]
        if past_events.empty:
            excitation = 0.0
        else:
            dt = (ts - past_events).total_seconds().to_numpy(dtype=float)
            excitation = float(np.exp(-params.beta * dt).sum())

        intensity = params.mu + params.alpha * excitation
        lambda_t.loc[ts] = intensity
        n_t.loc[ts] = params.branching_ratio
        mu_t.loc[ts] = params.mu
        alpha_t.loc[ts] = params.alpha
        beta_t.loc[ts] = params.beta
        fit_ok.loc[ts] = True

    return pd.concat([lambda_t, n_t, mu_t, alpha_t, beta_t, fit_ok], axis=1)
