from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import warnings

import pandas as pd

from src.thresholds import quantile_threshold


@dataclass(frozen=True)
class StrategyConfig:
    entry_k: float = 2.0
    t_widen_quantile: float = 0.80
    chi_widen_quantile: float = 0.80
    threshold_mode: str = "expanding"
    threshold_min_periods: int = 120
    threshold_window: int | None = None
    hawkes_threshold_mode: str = "fixed"
    hawkes_threshold_min_periods: int = 120
    hawkes_threshold_window: int | None = None
    hawkes_widen_quantile: float = 0.80
    hawkes_risk_off_quantile: float = 0.95
    hawkes_widen_threshold: float = 0.70
    hawkes_risk_off_threshold: float = 0.85


def _fallback_sigma_scale(m_t: pd.Series, *, min_periods: int = 20, floor: float = 1e-8) -> pd.Series:
    abs_m = pd.to_numeric(m_t, errors="coerce").abs()
    expanding_med = abs_m.expanding(min_periods=max(2, int(min_periods))).median().shift(1)
    fallback = expanding_med.clip(lower=float(floor))
    fallback = fallback.fillna(abs_m.expanding(min_periods=1).median().shift(1))
    fallback = fallback.clip(lower=float(floor))
    return fallback.rename("fallback_sigma_scale")


def compute_widen_flag(
    T_t: pd.Series,
    chi_t: pd.Series,
    *,
    t_quantile: float = 0.80,
    chi_quantile: float = 0.80,
    threshold_mode: str = "expanding",
    threshold_min_periods: int = 120,
    threshold_window: int | None = None,
) -> pd.DataFrame:
    t_thr = quantile_threshold(
        T_t,
        t_quantile,
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="t_widen_threshold",
    )
    chi_thr = quantile_threshold(
        chi_t,
        chi_quantile,
        mode=threshold_mode,
        min_periods=threshold_min_periods,
        window=threshold_window,
        shift=1,
        name="chi_widen_threshold",
    )
    widen_flag = (T_t.ge(t_thr) | chi_t.ge(chi_thr)).fillna(False).rename("widen_flag")
    return pd.concat([t_thr, chi_thr, widen_flag], axis=1)


def build_decisions(
    *,
    m_t: pd.Series,
    T_t: pd.Series,
    chi_t: pd.Series,
    sigma_hat: pd.Series | None = None,
    regime: pd.Series,
    depeg_flag: pd.Series,
    n_t: pd.Series | None = None,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    widen_frame = compute_widen_flag(
        T_t,
        chi_t,
        t_quantile=cfg.t_widen_quantile,
        chi_quantile=cfg.chi_widen_quantile,
        threshold_mode=cfg.threshold_mode,
        threshold_min_periods=cfg.threshold_min_periods,
        threshold_window=cfg.threshold_window,
    )
    widen_flag = widen_frame["widen_flag"].astype(bool)
    sigma_scale = pd.Series(np.nan, index=m_t.index, dtype="float64")
    if sigma_hat is not None:
        sigma_scale = pd.to_numeric(sigma_hat, errors="coerce").rename("sigma_scale")
    sigma_fallback = _fallback_sigma_scale(m_t)
    sigma_scale = sigma_scale.where(sigma_scale.gt(0.0), sigma_fallback)
    entry_threshold = (cfg.entry_k * T_t * sigma_scale).rename("entry_threshold")

    decision = pd.Series("Widen", index=m_t.index, name="decision", dtype="object")
    side = pd.Series("Flat", index=m_t.index, name="trade_side", dtype="object")
    trade_signal = (m_t.abs().gt(entry_threshold)) & entry_threshold.notna()

    riskoff = depeg_flag.astype(bool)
    hawkes_widen_threshold = pd.Series(np.nan, index=m_t.index, name="hawkes_widen_threshold", dtype="float64")
    hawkes_riskoff_threshold = pd.Series(
        np.nan, index=m_t.index, name="hawkes_riskoff_threshold", dtype="float64"
    )
    hawkes_widen_signal = pd.Series(False, index=m_t.index, name="hawkes_widen_signal")
    hawkes_riskoff_signal = pd.Series(False, index=m_t.index, name="hawkes_riskoff_signal")
    if n_t is not None:
        if cfg.hawkes_threshold_mode == "fixed":
            hawkes_widen_threshold = pd.Series(
                cfg.hawkes_widen_threshold,
                index=m_t.index,
                name="hawkes_widen_threshold",
                dtype="float64",
            )
            hawkes_riskoff_threshold = pd.Series(
                cfg.hawkes_risk_off_threshold,
                index=m_t.index,
                name="hawkes_riskoff_threshold",
                dtype="float64",
            )
        else:
            hawkes_widen_threshold = quantile_threshold(
                n_t.astype(float),
                cfg.hawkes_widen_quantile,
                mode=cfg.hawkes_threshold_mode,
                min_periods=cfg.hawkes_threshold_min_periods,
                window=cfg.hawkes_threshold_window,
                shift=1,
                name="hawkes_widen_threshold",
            )
            hawkes_riskoff_threshold = quantile_threshold(
                n_t.astype(float),
                cfg.hawkes_risk_off_quantile,
                mode=cfg.hawkes_threshold_mode,
                min_periods=cfg.hawkes_threshold_min_periods,
                window=cfg.hawkes_threshold_window,
                shift=1,
                name="hawkes_riskoff_threshold",
            )
            hawkes_riskoff_threshold = pd.concat(
                [hawkes_riskoff_threshold, hawkes_widen_threshold], axis=1
            ).max(axis=1).rename("hawkes_riskoff_threshold")

        hawkes_widen_signal = n_t.ge(hawkes_widen_threshold).fillna(False).rename("hawkes_widen_signal")
        hawkes_riskoff_signal = n_t.ge(hawkes_riskoff_threshold).fillna(False).rename("hawkes_riskoff_signal")
        riskoff = riskoff | hawkes_riskoff_signal
    riskoff = riskoff | regime.eq("stress")
    decision.loc[riskoff] = "Risk-off"

    widen = (~riskoff) & widen_flag
    decision.loc[widen] = "Widen"

    trade = (~riskoff) & (~widen) & trade_signal
    decision.loc[trade] = "Trade"

    side.loc[trade & m_t.gt(0)] = "ShortPremium"
    side.loc[trade & m_t.lt(0)] = "LongPremium"

    if n_t is not None:
        hawkes_widen = (~riskoff) & hawkes_widen_signal
        decision.loc[hawkes_widen] = "Widen"
        side.loc[hawkes_widen] = "Flat"

    trade_count = int(trade.sum())
    if trade_count == 0 and m_t.notna().any():
        warnings.warn(
            "No Trade decisions were generated for this run (all bars gated to Widen/Risk-off). "
            "Review entry_k and widen quantiles if this is unintended.",
            stacklevel=2,
        )

    return pd.concat(
        [
            entry_threshold,
            widen_frame,
            hawkes_widen_threshold,
            hawkes_riskoff_threshold,
            hawkes_widen_signal,
            hawkes_riskoff_signal,
            trade_signal.rename("trade_signal"),
            decision,
            side,
        ],
        axis=1,
    )
