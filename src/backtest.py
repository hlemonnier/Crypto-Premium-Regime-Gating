from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 1.0
    naive_threshold: float = 0.001
    position_mode: str = "stateful"
    exit_on_widen: bool = True
    exit_on_mean_reversion: bool = True
    min_holding_bars: int = 5
    max_holding_bars: int | None = None


def periods_per_year_from_freq(freq: str) -> float:
    seconds = pd.to_timedelta(freq).total_seconds()
    if seconds <= 0:
        raise ValueError("freq must map to a positive timedelta")
    return (365.25 * 24 * 3600) / seconds


def _sharpe_ratio(pnl: pd.Series) -> float:
    clean = pnl.loc[pnl.notna()].astype(float)
    if clean.empty:
        return 0.0
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return 0.0
    mean = float(clean.mean())
    if not np.isfinite(mean):
        return 0.0
    return mean / std


def _position_from_trade_signal(decision_i: object, m_i: float) -> float:
    if str(decision_i) != "Trade" or not np.isfinite(m_i):
        return 0.0
    sign = float(np.sign(m_i))
    return sign if sign != 0.0 else 0.0


def _decision_to_position_one_bar(decision: pd.Series, m_t: pd.Series) -> pd.DataFrame:
    pos = pd.Series(0.0, index=decision.index, name="position")
    trade_idx = decision.eq("Trade") & m_t.notna()
    pos.loc[trade_idx] = np.sign(m_t.loc[trade_idx]).astype(float)

    prev = pos.shift(1).fillna(0.0)
    event = pd.Series("Flat", index=decision.index, name="position_event", dtype="object")
    event.loc[(prev == 0.0) & (pos != 0.0)] = "Entry"
    event.loc[(prev != 0.0) & (pos == 0.0)] = "Exit"
    event.loc[(prev != 0.0) & (pos != 0.0) & (np.sign(prev) != np.sign(pos))] = "Flip"
    event.loc[(prev != 0.0) & (pos != 0.0) & (np.sign(prev) == np.sign(pos))] = "Hold"
    hold_bars = pd.Series(0, index=decision.index, name="holding_bars", dtype=int)
    hold_bars.loc[pos.ne(0)] = 1
    return pd.concat([pos, event, hold_bars], axis=1)


def _decision_to_position_stateful(
    decision: pd.Series,
    m_t: pd.Series,
    *,
    exit_on_widen: bool,
    exit_on_mean_reversion: bool,
    min_holding_bars: int,
    max_holding_bars: int | None,
) -> pd.DataFrame:
    idx = decision.index
    pos_values = np.zeros(len(idx), dtype=float)
    hold_values = np.zeros(len(idx), dtype=int)
    events = np.full(len(idx), "Flat", dtype=object)

    prev_pos = 0.0
    prev_holding = 0
    min_hold = max(1, int(min_holding_bars))
    max_hold = None if max_holding_bars is None else max(1, int(max_holding_bars))

    for i, (decision_i, m_i) in enumerate(zip(decision.to_numpy(), m_t.to_numpy())):
        current = prev_pos
        exited_this_bar = False

        if current != 0.0:
            should_exit = False
            decision_str = str(decision_i)
            if decision_str == "Risk-off":
                should_exit = True
            elif exit_on_widen and decision_str == "Widen" and prev_holding >= min_hold:
                should_exit = True
            elif exit_on_mean_reversion and np.isfinite(m_i) and (m_i * current <= 0.0):
                should_exit = True
            elif max_hold is not None and prev_holding >= max_hold:
                should_exit = True
            if should_exit:
                current = 0.0
                exited_this_bar = True

        # Enforce exit conditions for this bar; otherwise max-holding and safety exits
        # can be bypassed by immediate same-bar re-entry.
        if not exited_this_bar:
            target = _position_from_trade_signal(decision_i, float(m_i) if np.isfinite(m_i) else np.nan)
            if target != 0.0:
                if current == 0.0:
                    current = target
                elif np.sign(current) != np.sign(target):
                    current = target

        if current == 0.0:
            holding = 0
        elif prev_pos == 0.0 or np.sign(prev_pos) != np.sign(current):
            holding = 1
        else:
            holding = prev_holding + 1

        if prev_pos == 0.0 and current != 0.0:
            event = "Entry"
        elif prev_pos != 0.0 and current == 0.0:
            event = "Exit"
        elif prev_pos != 0.0 and current != 0.0 and np.sign(prev_pos) != np.sign(current):
            event = "Flip"
        elif current != 0.0:
            event = "Hold"
        else:
            event = "Flat"

        pos_values[i] = current
        hold_values[i] = holding
        events[i] = event
        prev_pos = current
        prev_holding = holding

    return pd.DataFrame(
        {
            "position": pos_values,
            "position_event": events,
            "holding_bars": hold_values,
        },
        index=idx,
    )


def decision_to_position(
    decision: pd.Series,
    m_t: pd.Series,
    *,
    mode: str = "stateful",
    exit_on_widen: bool = True,
    exit_on_mean_reversion: bool = True,
    min_holding_bars: int = 5,
    max_holding_bars: int | None = None,
) -> pd.DataFrame:
    mode = str(mode).strip().lower()
    if mode == "one_bar":
        return _decision_to_position_one_bar(decision, m_t)
    if mode == "stateful":
        return _decision_to_position_stateful(
            decision,
            m_t,
            exit_on_widen=exit_on_widen,
            exit_on_mean_reversion=exit_on_mean_reversion,
            min_holding_bars=min_holding_bars,
            max_holding_bars=max_holding_bars,
        )
    raise ValueError(f"Unsupported position mode: {mode}. Expected 'stateful' or 'one_bar'.")


def run_backtest(
    premium: pd.Series,
    decision: pd.Series,
    m_t: pd.Series,
    *,
    freq: str,
    cost_bps: float,
    position_size: pd.Series | None = None,
    position_mode: str = "stateful",
    exit_on_widen: bool = True,
    exit_on_mean_reversion: bool = True,
    min_holding_bars: int = 5,
    max_holding_bars: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    p = premium.astype(float).copy()
    position_frame = decision_to_position(
        decision,
        m_t,
        mode=position_mode,
        exit_on_widen=exit_on_widen,
        exit_on_mean_reversion=exit_on_mean_reversion,
        min_holding_bars=min_holding_bars,
        max_holding_bars=max_holding_bars,
    )
    pos_sign = pd.to_numeric(position_frame["position"], errors="coerce").fillna(0.0).rename("position_sign")
    if position_size is None:
        size = pd.Series(1.0, index=pos_sign.index, dtype="float64")
    else:
        size = pd.to_numeric(position_size.reindex(pos_sign.index), errors="coerce")
        size = size.fillna(0.0).clip(lower=0.0, upper=1.0)
    size = size.where(pos_sign.ne(0.0), 0.0).rename("position_size")
    pos = (pos_sign * size).rename("position")
    position_frame = position_frame.drop(columns=["position"]).copy()
    position_frame.insert(0, "position_sign", pos_sign)
    position_frame.insert(1, "position_size", size)
    position_frame.insert(2, "position", pos)

    dp = p.diff().fillna(0.0)
    gross_pnl = (pos.shift(1).fillna(0.0) * (-dp)).rename("gross_pnl")
    turnover = pos.diff().abs().fillna(pos.abs()).rename("turnover")
    costs = (turnover * (cost_bps * 1e-4)).rename("costs")
    net_pnl = (gross_pnl - costs).rename("net_pnl")
    cum_pnl = net_pnl.cumsum().rename("cum_pnl")
    drawdown = (cum_pnl - cum_pnl.cummax()).rename("drawdown")

    out = pd.concat(
        [
            p.rename("premium"),
            position_frame,
            decision.rename("decision"),
            gross_pnl,
            costs,
            net_pnl,
            cum_pnl,
            drawdown,
        ],
        axis=1,
    )
    out.index.name = "timestamp_utc"

    in_market = pos.shift(1).abs() > 0
    active_mask = in_market & net_pnl.notna()
    pnl_when_active = net_pnl.loc[active_mask]
    avg_active_size = float(pos.shift(1).abs().where(in_market).mean(skipna=True))
    if not np.isfinite(avg_active_size):
        avg_active_size = 0.0
    annualization = np.sqrt(periods_per_year_from_freq(freq))
    sharpe_full = _sharpe_ratio(net_pnl)
    sharpe_full_annualized = float(sharpe_full * annualization)
    sharpe_active = _sharpe_ratio(pnl_when_active)
    sharpe_active_annualized = float(sharpe_active * annualization)
    hit_rate = float((pnl_when_active > 0).mean()) if not pnl_when_active.empty else 0.0
    decision_prev = decision.shift(1)
    valid_decision_transition = decision.notna() & decision_prev.notna()
    decision_flip = decision.ne(decision_prev) & valid_decision_transition
    flip_rate = (
        float(decision_flip.sum()) / float(valid_decision_transition.sum())
        if int(valid_decision_transition.sum()) > 0
        else 0.0
    )

    pos_num = pd.to_numeric(pos, errors="coerce")
    pos_prev = pos_num.shift(1)
    valid_position_transition = pos_num.notna() & pos_prev.notna()
    position_flip = pos_num.ne(pos_prev) & valid_position_transition
    position_flip_rate = (
        float(position_flip.sum()) / float(valid_position_transition.sum())
        if int(valid_position_transition.sum()) > 0
        else 0.0
    )

    avg_holding = float(position_frame["holding_bars"].replace(0, np.nan).mean())
    if not np.isfinite(avg_holding):
        avg_holding = 0.0

    horizon_days = 0.0
    if len(p.index) > 1:
        horizon_seconds = (p.index[-1] - p.index[0]).total_seconds()
        horizon_days = float(max(horizon_seconds, 0.0) / (24 * 3600))

    metrics = {
        # Primary Sharpe is full-series and non-annualized for short-horizon comparability.
        "sharpe": sharpe_full,
        "sharpe_full_annualized": sharpe_full_annualized,
        "sharpe_active": sharpe_active,
        "sharpe_active_annualized": sharpe_active_annualized,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.sum()),
        "pnl_net": float(net_pnl.sum()),
        "flip_rate": flip_rate,
        "hit_rate": hit_rate,
        "active_ratio": float(in_market.mean()),
        "avg_active_position_size": avg_active_size,
        "position_flip_rate": position_flip_rate,
        "avg_holding_bars": avg_holding,
        "n_bars": int(len(net_pnl)),
        "n_active_bars": int(active_mask.sum()),
        "horizon_days": horizon_days,
        "annualization_factor": float(annualization),
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
    size_gated: pd.Series | None = None,
    freq: str,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gated_log, gated_metrics = run_backtest(
        p_debiased,
        decision_gated,
        m_t,
        freq=freq,
        cost_bps=cfg.cost_bps,
        position_size=size_gated,
        position_mode=cfg.position_mode,
        exit_on_widen=cfg.exit_on_widen,
        exit_on_mean_reversion=cfg.exit_on_mean_reversion,
        min_holding_bars=cfg.min_holding_bars,
        max_holding_bars=cfg.max_holding_bars,
    )
    naive_signal = run_naive_baseline(p_naive, threshold=cfg.naive_threshold)
    naive_log, naive_metrics = run_backtest(
        p_naive,
        naive_signal["decision"],
        p_naive,
        freq=freq,
        cost_bps=cfg.cost_bps,
        position_mode=cfg.position_mode,
        exit_on_widen=cfg.exit_on_widen,
        exit_on_mean_reversion=cfg.exit_on_mean_reversion,
        min_holding_bars=cfg.min_holding_bars,
        max_holding_bars=cfg.max_holding_bars,
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
