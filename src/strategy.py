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
    confidence_floor: float = 0.10
    confidence_ceiling: float = 1.00
    event_confidence_penalty: float = 0.70
    technical_stress_penalty: float = 0.70
    usdt_concern_penalty: float = 0.50
    hawkes_widen_penalty: float = 0.75
    proxy_stress_threshold: float = 0.002
    onchain_stress_threshold: float = 0.003


def _clip_penalty(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


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


def classify_stress_source(
    *,
    regime: pd.Series,
    depeg_flag: pd.Series,
    stablecoin_proxy: pd.Series | None = None,
    onchain_proxy: pd.Series | None = None,
    onchain_usdc_minus_1: pd.Series | None = None,
    onchain_usdt_minus_1: pd.Series | None = None,
    proxy_threshold: float = 0.002,
    onchain_threshold: float = 0.003,
) -> pd.Series:
    idx = regime.index
    source = pd.Series("none", index=idx, name="stress_source", dtype="object")
    stress_like = regime.eq("stress").fillna(False) | depeg_flag.astype(bool).fillna(False)

    proxy_thr = abs(float(proxy_threshold))
    onchain_thr = abs(float(onchain_threshold))

    if stablecoin_proxy is not None:
        proxy = pd.to_numeric(stablecoin_proxy, errors="coerce").reindex(idx)
        source.loc[stress_like & proxy.ge(proxy_thr)] = "usdc_depeg_stress"
        source.loc[stress_like & proxy.le(-proxy_thr)] = "usdt_backing_concern"

    if onchain_proxy is not None:
        proxy = pd.to_numeric(onchain_proxy, errors="coerce").reindex(idx)
        source.loc[stress_like & proxy.ge(proxy_thr)] = "usdc_depeg_stress"
        source.loc[stress_like & proxy.le(-proxy_thr)] = "usdt_backing_concern"

    if onchain_usdc_minus_1 is not None and onchain_usdt_minus_1 is not None:
        usdc_dev = pd.to_numeric(onchain_usdc_minus_1, errors="coerce").reindex(idx)
        usdt_dev = pd.to_numeric(onchain_usdt_minus_1, errors="coerce").reindex(idx)
        usdc_dom = usdc_dev.abs().ge(usdt_dev.abs())
        usdt_dom = usdt_dev.abs().gt(usdc_dev.abs())
        usdc_stress = stress_like & usdc_dev.le(-onchain_thr) & usdc_dom
        usdt_stress = stress_like & usdt_dev.le(-onchain_thr) & usdt_dom
        source.loc[usdc_stress] = "usdc_depeg_stress"
        source.loc[usdt_stress] = "usdt_backing_concern"

    source.loc[stress_like & source.eq("none")] = "technical_flow_imbalance"
    return source


def _build_confidence(
    *,
    m_t: pd.Series,
    entry_threshold: pd.Series,
    trade_signal: pd.Series,
    event: pd.Series | None,
    stress_source: pd.Series,
    hawkes_widen_signal: pd.Series,
    cfg: StrategyConfig,
) -> pd.Series:
    ratio = pd.Series(np.nan, index=m_t.index, dtype="float64")
    denom = pd.to_numeric(entry_threshold, errors="coerce")
    numer = pd.to_numeric(m_t, errors="coerce").abs()
    valid = denom.gt(0.0) & denom.notna() & numer.notna()
    ratio.loc[valid] = numer.loc[valid] / denom.loc[valid]

    margin = ratio.sub(1.0).clip(lower=0.0)
    base = pd.Series(1.0 - np.exp(-margin.to_numpy(dtype=float)), index=m_t.index, dtype="float64")
    base = base.clip(lower=0.0, upper=1.0).fillna(0.0)

    penalty = pd.Series(1.0, index=m_t.index, dtype="float64")
    if event is not None:
        event_flag = event.astype(bool).reindex(m_t.index).fillna(False)
        penalty.loc[event_flag] *= _clip_penalty(cfg.event_confidence_penalty)
    penalty.loc[stress_source.eq("technical_flow_imbalance")] *= _clip_penalty(cfg.technical_stress_penalty)
    penalty.loc[stress_source.eq("usdt_backing_concern")] *= _clip_penalty(cfg.usdt_concern_penalty)
    penalty.loc[hawkes_widen_signal.astype(bool).reindex(m_t.index).fillna(False)] *= _clip_penalty(
        cfg.hawkes_widen_penalty
    )
    penalty = penalty.clip(lower=0.0, upper=1.0)

    confidence = (base * penalty).clip(lower=0.0, upper=1.0)
    trade_mask = trade_signal.astype(bool).reindex(m_t.index).fillna(False)
    floor = float(np.clip(cfg.confidence_floor, 0.0, 1.0))
    ceil = float(np.clip(cfg.confidence_ceiling, floor, 1.0))
    confidence.loc[trade_mask] = confidence.loc[trade_mask].clip(lower=floor, upper=ceil)
    confidence.loc[~trade_mask] = 0.0
    return confidence.rename("confidence_score")


def build_decisions(
    *,
    m_t: pd.Series,
    T_t: pd.Series,
    chi_t: pd.Series,
    sigma_hat: pd.Series | None = None,
    regime: pd.Series,
    depeg_flag: pd.Series,
    event: pd.Series | None = None,
    stablecoin_proxy: pd.Series | None = None,
    onchain_proxy: pd.Series | None = None,
    onchain_usdc_minus_1: pd.Series | None = None,
    onchain_usdt_minus_1: pd.Series | None = None,
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

    stress_source = classify_stress_source(
        regime=regime,
        depeg_flag=depeg_flag,
        stablecoin_proxy=stablecoin_proxy,
        onchain_proxy=onchain_proxy,
        onchain_usdc_minus_1=onchain_usdc_minus_1,
        onchain_usdt_minus_1=onchain_usdt_minus_1,
        proxy_threshold=cfg.proxy_stress_threshold,
        onchain_threshold=cfg.onchain_stress_threshold,
    )
    usdc_stress = stress_source.eq("usdc_depeg_stress")
    usdt_concern = stress_source.eq("usdt_backing_concern")
    technical_flow = stress_source.eq("technical_flow_imbalance")

    riskoff = usdc_stress.copy()
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
    riskoff = riskoff.fillna(False).rename("riskoff_flag")
    decision.loc[riskoff] = "Risk-off"

    stress_widen = ((usdt_concern | technical_flow) & (~riskoff)).rename("stress_widen_flag")
    widen = (~riskoff) & (widen_flag | stress_widen | hawkes_widen_signal)
    decision.loc[widen] = "Widen"

    trade = (~riskoff) & (~widen) & trade_signal
    decision.loc[trade] = "Trade"

    side.loc[trade & m_t.gt(0)] = "ShortPremium"
    side.loc[trade & m_t.lt(0)] = "LongPremium"

    confidence = _build_confidence(
        m_t=m_t,
        entry_threshold=entry_threshold,
        trade_signal=trade,
        event=event,
        stress_source=stress_source,
        hawkes_widen_signal=hawkes_widen_signal,
        cfg=cfg,
    )
    position_size = confidence.rename("position_size")

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
            sigma_scale.rename("sigma_scale"),
            widen_frame,
            hawkes_widen_threshold,
            hawkes_riskoff_threshold,
            hawkes_widen_signal,
            hawkes_riskoff_signal,
            stress_source,
            riskoff,
            stress_widen,
            trade_signal.rename("trade_signal"),
            confidence,
            position_size,
            decision,
            side,
        ],
        axis=1,
    )
