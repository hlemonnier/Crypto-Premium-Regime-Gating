from __future__ import annotations

from dataclasses import dataclass

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
    hawkes_widen_threshold: float = 0.70
    hawkes_risk_off_threshold: float = 0.85


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
    if sigma_hat is not None:
        entry_threshold = (cfg.entry_k * T_t * sigma_hat).rename("entry_threshold")
    else:
        entry_threshold = (cfg.entry_k * T_t).rename("entry_threshold")

    decision = pd.Series("Widen", index=m_t.index, name="decision", dtype="object")
    side = pd.Series("Flat", index=m_t.index, name="trade_side", dtype="object")
    trade_signal = (m_t.abs().gt(entry_threshold)) & entry_threshold.notna()

    riskoff = depeg_flag.astype(bool)
    if n_t is not None:
        riskoff = riskoff | n_t.ge(cfg.hawkes_risk_off_threshold)
    riskoff = riskoff | regime.eq("stress")
    decision.loc[riskoff] = "Risk-off"

    widen = (~riskoff) & widen_flag
    decision.loc[widen] = "Widen"

    trade = (~riskoff) & (~widen) & trade_signal
    decision.loc[trade] = "Trade"

    side.loc[trade & m_t.gt(0)] = "ShortPremium"
    side.loc[trade & m_t.lt(0)] = "LongPremium"

    if n_t is not None:
        hawkes_widen = (~riskoff) & n_t.ge(cfg.hawkes_widen_threshold)
        decision.loc[hawkes_widen] = "Widen"
        side.loc[hawkes_widen] = "Flat"

    return pd.concat(
        [
            entry_threshold,
            widen_frame,
            trade_signal.rename("trade_signal"),
            decision,
            side,
        ],
        axis=1,
    )
