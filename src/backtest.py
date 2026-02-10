from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 1.0
    naive_threshold: float = 0.001


def periods_per_year_from_freq(freq: str) -> float:
    seconds = pd.to_timedelta(freq).total_seconds()
    if seconds <= 0:
        raise ValueError("freq must map to a positive timedelta")
    return (365.25 * 24 * 3600) / seconds


def decision_to_position(decision: pd.Series, m_t: pd.Series) -> pd.Series:
    pos = pd.Series(0.0, index=decision.index, name="position")
    trade_idx = decision.eq("Trade")
    pos.loc[trade_idx] = np.sign(m_t.loc[trade_idx]).astype(float)
    return pos


def run_backtest(
    premium: pd.Series,
    decision: pd.Series,
    m_t: pd.Series,
    *,
    freq: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    p = premium.astype(float).copy()
    pos = decision_to_position(decision, m_t)

    dp = p.diff().fillna(0.0)
    gross_pnl = (pos.shift(1).fillna(0.0) * (-dp)).rename("gross_pnl")
    turnover = pos.diff().abs().fillna(pos.abs()).rename("turnover")
    costs = (turnover * (cost_bps * 1e-4)).rename("costs")
    net_pnl = (gross_pnl - costs).rename("net_pnl")
    cum_pnl = net_pnl.cumsum().rename("cum_pnl")
    drawdown = (cum_pnl - cum_pnl.cummax()).rename("drawdown")

    out = pd.concat([p.rename("premium"), pos, decision.rename("decision"), gross_pnl, costs, net_pnl, cum_pnl, drawdown], axis=1)
    out.index.name = "timestamp_utc"

    in_market = pos.shift(1).abs() > 0
    pnl_when_active = net_pnl.where(in_market)
    active_std = float(pnl_when_active.std(ddof=0))
    annualization = np.sqrt(periods_per_year_from_freq(freq))

    sharpe = float((pnl_when_active.mean() / active_std) * annualization) if active_std > 0 else 0.0
    hit_rate = float((pnl_when_active > 0).mean()) if in_market.any() else 0.0
    flip_rate = float(decision.ne(decision.shift(1)).mean())

    metrics = {
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.sum()),
        "pnl_net": float(net_pnl.sum()),
        "flip_rate": flip_rate,
        "hit_rate": hit_rate,
        "active_ratio": float(in_market.mean()),
    }
    return out, metrics


def run_naive_baseline(
    p_naive: pd.Series,
    *,
    threshold: float,
) -> pd.DataFrame:
    decision = pd.Series("Widen", index=p_naive.index, name="decision", dtype="object")
    decision.loc[p_naive.abs().gt(threshold)] = "Trade"
    return pd.DataFrame({"decision": decision})


def compare_strategies(
    *,
    p_naive: pd.Series,
    p_debiased: pd.Series,
    decision_gated: pd.Series,
    m_t: pd.Series,
    freq: str,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gated_log, gated_metrics = run_backtest(
        p_debiased,
        decision_gated,
        m_t,
        freq=freq,
        cost_bps=cfg.cost_bps,
    )
    naive_signal = run_naive_baseline(p_naive, threshold=cfg.naive_threshold)
    naive_log, naive_metrics = run_backtest(
        p_naive,
        naive_signal["decision"],
        p_naive,
        freq=freq,
        cost_bps=cfg.cost_bps,
    )

    metrics = pd.DataFrame(
        [naive_metrics, gated_metrics],
        index=["naive", "gated"],
    )
    return metrics, gated_log, naive_log


def export_metrics(metrics: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=True)
    return output
