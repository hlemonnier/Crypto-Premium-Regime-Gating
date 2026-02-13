from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class ExecutionUnifierConfig:
    enabled: bool = False
    slippage_curve_path: str = "reports/final/execution_slippage_vs_size.csv"
    min_history: int = 120
    min_regime_history: int = 60
    edge_bins: int = 12
    horizon_min_bars: int = 5
    horizon_max_bars: int = 60
    horizon_default_bars: int = 20
    horizon_estimation_min_obs: int = 30
    size_grid_step: float = 0.05
    trade_min_size: float = 0.10
    widen_floor_size: float = 0.25
    spread_roundtrip_bps: float | None = None
    fees_roundtrip_bps: float | None = None
    fallback_linear_slippage_bps: float = 1.0
    transient_cost_multiplier: float = 1.0
    stress_cost_multiplier: float = 1.35
    slippage_root: str = "BTC"
    slippage_quote_preference: str = "USDT"


def _clip_penalty(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _fallback_sigma_scale(m_t: pd.Series, *, min_periods: int = 20, floor: float = 1e-8) -> pd.Series:
    abs_m = pd.to_numeric(m_t, errors="coerce").abs()
    expanding_med = abs_m.expanding(min_periods=max(2, int(min_periods))).median().shift(1)
    fallback = expanding_med.clip(lower=float(floor))
    fallback = fallback.fillna(abs_m.expanding(min_periods=1).median().shift(1))
    fallback = fallback.clip(lower=float(floor))
    return fallback.rename("fallback_sigma_scale")


def _size_grid(step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0:
        step = 0.05
    steps = int(np.ceil(1.0 / float(step)))
    grid = np.linspace(0.0, 1.0, steps + 1)
    return np.clip(grid, 0.0, 1.0)


def _size_token(size: float) -> str:
    return f"{int(round(float(size) * 100.0)):03d}"


def _size_column_name(prefix: str, size: float) -> str:
    return f"{prefix}_s{_size_token(size)}"


def _estimate_ar1_rho(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    centered = values - float(np.mean(values))
    lag = centered[:-1]
    den = float(np.dot(lag, lag))
    if den <= 1e-20:
        return 0.0
    rho = float(np.dot(centered[1:], lag) / den)
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip(rho, -0.999, 0.999))


def _estimate_adaptive_horizon_bars(m_t: pd.Series, cfg: ExecutionUnifierConfig) -> pd.Series:
    values = pd.to_numeric(m_t, errors="coerce").to_numpy(dtype=float)
    n = values.shape[0]
    out = np.full(n, int(max(1, int(cfg.horizon_default_bars))), dtype=int)

    h_min = int(max(1, int(cfg.horizon_min_bars)))
    h_max = int(max(h_min, int(cfg.horizon_max_bars)))
    h_default = int(np.clip(int(cfg.horizon_default_bars), h_min, h_max))
    min_obs = int(max(3, int(cfg.horizon_estimation_min_obs)))

    for t in range(n):
        hist = values[:t]
        finite = hist[np.isfinite(hist)]
        if finite.size < min_obs:
            out[t] = h_default
            continue

        rho = abs(_estimate_ar1_rho(finite))
        if (not np.isfinite(rho)) or rho <= 1e-8 or rho >= 0.999:
            out[t] = h_default
            continue

        denom = -math.log(rho)
        if not np.isfinite(denom) or denom <= 1e-12:
            out[t] = h_default
            continue

        half_life = float(math.log(2.0) / denom)
        if not np.isfinite(half_life):
            out[t] = h_default
            continue

        out[t] = int(np.clip(int(round(half_life)), h_min, h_max))

    return pd.Series(out, index=m_t.index, name="edge_horizon_bars")


def _fit_monotone_edge_curve(
    abs_m_bps_samples: np.ndarray,
    edge_samples_bps: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(abs_m_bps_samples) & np.isfinite(edge_samples_bps)
    x = abs_m_bps_samples[valid]
    y = edge_samples_bps[valid]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if x.size == 1:
        return np.array([float(x[0])], dtype=float), np.array([float(y[0])], dtype=float)

    bins = max(2, int(min(n_bins, x.size)))
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)

    if edges.size < 2:
        x_med = float(np.median(x))
        y_med = float(np.median(y))
        return np.array([x_med], dtype=float), np.array([y_med], dtype=float)

    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, edges.size - 2)

    x_bins: list[float] = []
    y_bins: list[float] = []
    for b in range(edges.size - 1):
        mask = idx == b
        if not bool(mask.any()):
            continue
        xb = float(np.median(x[mask]))
        yb = float(np.median(y[mask]))
        x_bins.append(xb)
        y_bins.append(yb)

    if len(x_bins) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_arr = np.asarray(x_bins, dtype=float)
    y_arr = np.asarray(y_bins, dtype=float)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    y_monotone = np.maximum.accumulate(y_arr)
    return x_arr, y_monotone


def _interp_monotone_curve(x: np.ndarray, y: np.ndarray, value: float) -> float:
    if x.size == 0 or y.size == 0 or not np.isfinite(value):
        return float("nan")
    if x.size == 1:
        return float(y[0])
    return float(np.interp(float(value), x, y, left=y[0], right=y[-1]))


def _invert_monotone_curve(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if x.size == 0 or y.size == 0 or not np.isfinite(target):
        return float("nan")
    if x.size == 1:
        return float(x[0]) if float(y[0]) >= float(target) else float("nan")

    y_unique: list[float] = []
    x_unique: list[float] = []
    for xi, yi in zip(x, y):
        xi_f = float(xi)
        yi_f = float(yi)
        if len(y_unique) == 0:
            y_unique.append(yi_f)
            x_unique.append(xi_f)
            continue
        if yi_f > y_unique[-1]:
            y_unique.append(yi_f)
            x_unique.append(xi_f)
        else:
            x_unique[-1] = max(x_unique[-1], xi_f)

    y_arr = np.asarray(y_unique, dtype=float)
    x_arr = np.asarray(x_unique, dtype=float)

    if target <= y_arr[0]:
        return float(x_arr[0])
    if target > y_arr[-1]:
        return float("nan")
    return float(np.interp(float(target), y_arr, x_arr))


def _resolve_fixed_cost_bps(cfg: ExecutionUnifierConfig, *, base_cost_bps: float) -> float:
    baseline = max(0.0, float(base_cost_bps))
    spread = cfg.spread_roundtrip_bps
    fees = cfg.fees_roundtrip_bps
    spread_bps = 0.4 * baseline if spread is None else max(0.0, float(spread))
    fees_bps = 0.6 * baseline if fees is None else max(0.0, float(fees))
    return float(spread_bps + fees_bps)


def _choose_slippage_curve(
    slippage_curve: pd.DataFrame | None,
    *,
    root_hint: str,
    quote_preference: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if slippage_curve is None or slippage_curve.empty:
        return None, None, "fallback_linear"

    frame = slippage_curve.copy()
    if "book_walk_mean_bps" not in frame.columns:
        return None, None, "fallback_linear"

    frame["book_walk_mean_bps"] = pd.to_numeric(frame["book_walk_mean_bps"], errors="coerce")
    frame = frame.loc[frame["book_walk_mean_bps"].notna()].copy()
    if frame.empty:
        return None, None, "fallback_linear"

    if "root" in frame.columns:
        root = str(root_hint).upper()
        root_rows = frame.loc[frame["root"].astype(str).str.upper().eq(root)].copy()
        if not root_rows.empty:
            frame = root_rows

    source = "slippage_curve"
    if "quote" in frame.columns:
        preferred_quote = str(quote_preference).upper()
        quote_rows = frame.loc[frame["quote"].astype(str).str.upper().eq(preferred_quote)].copy()
        if not quote_rows.empty:
            frame = quote_rows
            source = f"slippage_curve_{preferred_quote.lower()}"

    if "dnl_median" in frame.columns:
        frame["_size_raw"] = pd.to_numeric(frame["dnl_median"], errors="coerce").abs()
    else:
        frame["_size_raw"] = np.nan

    if frame["_size_raw"].notna().any():
        frame = frame.sort_values("_size_raw")
    else:
        frame = frame.sort_values("book_walk_mean_bps")

    n = frame.shape[0]
    if n == 1:
        y_single = max(0.0, float(frame["book_walk_mean_bps"].iloc[0]))
        return np.array([0.0, 1.0], dtype=float), np.array([0.0, y_single], dtype=float), source

    y_raw = frame["book_walk_mean_bps"].to_numpy(dtype=float)
    y_raw = np.where(np.isfinite(y_raw), np.maximum(y_raw, 0.0), np.nan)
    if not np.isfinite(y_raw).any():
        return None, None, "fallback_linear"

    y_raw = pd.Series(y_raw).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
    y_monotone = np.maximum.accumulate(y_raw)

    x_native = np.linspace(0.0, 1.0, n)
    x = np.linspace(0.0, 1.0, 21)
    y = np.interp(x, x_native, y_monotone)
    return x, y, source


def _slippage_bps_at_size(size: float, x: np.ndarray | None, y: np.ndarray | None, fallback_linear_bps: float) -> float:
    s = float(np.clip(size, 0.0, 1.0))
    if x is None or y is None:
        return float(max(0.0, float(fallback_linear_bps)) * s)
    return float(np.interp(s, x, y, left=y[0], right=y[-1]))


def _empty_unifier_frame(index: pd.Index, size_grid: np.ndarray, *, source: str = "disabled") -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["edge_gross_bps"] = np.nan
    out["edge_cost_bps_opt"] = np.nan
    out["edge_net_bps_opt"] = np.nan
    out["expected_net_pnl_opt_bps"] = 0.0
    out["optimal_size"] = 0.0
    out["break_even_premium_bps_opt"] = np.nan
    out["edge_horizon_bars"] = np.nan
    out["edge_model_source"] = source
    out["cost_model_source"] = source
    out["cost_regime_multiplier"] = np.nan
    out["fixed_cost_bps"] = np.nan
    for s in size_grid:
        out[_size_column_name("expected_net_pnl_bps", float(s))] = np.nan
        out[_size_column_name("break_even_premium_bps", float(s))] = np.nan
    return out


def _compute_execution_unifier(
    *,
    m_t: pd.Series,
    premium: pd.Series | None,
    regime: pd.Series,
    cfg: ExecutionUnifierConfig,
    base_cost_bps: float,
    slippage_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    size_grid = _size_grid(cfg.size_grid_step)
    if (not cfg.enabled) or premium is None:
        return _empty_unifier_frame(m_t.index, size_grid, source="disabled")

    p = pd.to_numeric(premium.reindex(m_t.index), errors="coerce")
    m = pd.to_numeric(m_t, errors="coerce")
    reg = regime.reindex(m_t.index).astype(str)

    horizon = _estimate_adaptive_horizon_bars(m, cfg)
    horizon_vals = pd.to_numeric(horizon, errors="coerce").fillna(float(cfg.horizon_default_bars)).astype(int).to_numpy()

    m_vals = m.to_numpy(dtype=float)
    p_vals = p.to_numpy(dtype=float)
    reg_vals = reg.to_numpy(dtype=object)
    abs_m_bps = np.abs(m_vals) * 1e4

    n = len(m_vals)
    realized_edge = np.full(n, np.nan, dtype=float)
    for i in range(n):
        h_i = int(max(1, horizon_vals[i]))
        j = i + h_i
        if j >= n:
            continue
        if not (np.isfinite(m_vals[i]) and np.isfinite(p_vals[i]) and np.isfinite(p_vals[j])):
            continue
        sign_i = float(np.sign(m_vals[i]))
        if sign_i == 0.0:
            continue
        realized_edge[i] = sign_i * (p_vals[i] - p_vals[j]) * 1e4

    maturity_time = np.full(n, -1, dtype=int)
    for i in range(n):
        h_i = int(max(1, horizon_vals[i]))
        maturity = i + h_i + 1
        if maturity < n and np.isfinite(realized_edge[i]) and np.isfinite(abs_m_bps[i]):
            maturity_time[i] = maturity

    matured_at: dict[int, list[int]] = {}
    for i, t_mature in enumerate(maturity_time):
        if t_mature < 0:
            continue
        matured_at.setdefault(int(t_mature), []).append(i)

    fixed_cost_bps = _resolve_fixed_cost_bps(cfg, base_cost_bps=base_cost_bps)
    slip_x, slip_y, cost_source = _choose_slippage_curve(
        slippage_curve,
        root_hint=cfg.slippage_root,
        quote_preference=cfg.slippage_quote_preference,
    )

    expected_opt = np.full(n, 0.0, dtype=float)
    edge_gross = np.full(n, np.nan, dtype=float)
    edge_cost_opt = np.full(n, np.nan, dtype=float)
    edge_net_opt = np.full(n, np.nan, dtype=float)
    size_opt = np.zeros(n, dtype=float)
    be_opt = np.full(n, np.nan, dtype=float)
    model_source = np.full(n, "insufficient_history", dtype=object)
    cost_source_arr = np.full(n, cost_source, dtype=object)
    regime_mult_arr = np.full(n, np.nan, dtype=float)

    per_size_expected = {float(s): np.full(n, np.nan, dtype=float) for s in size_grid}
    per_size_break_even = {float(s): np.full(n, np.nan, dtype=float) for s in size_grid}

    matured_global: list[int] = []
    matured_by_regime: dict[str, list[int]] = {}

    min_history = int(max(10, int(cfg.min_history)))
    min_regime_history = int(max(5, int(cfg.min_regime_history)))

    for t in range(n):
        new_samples = matured_at.get(t, [])
        if new_samples:
            matured_global.extend(new_samples)
            for i in new_samples:
                key = str(reg_vals[i])
                matured_by_regime.setdefault(key, []).append(i)

        if not np.isfinite(abs_m_bps[t]):
            continue

        if len(matured_global) == 0:
            edge_est = 0.0
            x_curve = np.array([], dtype=float)
            y_curve = np.array([], dtype=float)
            src = "insufficient_history"
        else:
            sample_idx_global = np.asarray(matured_global, dtype=int)
            sample_idx_regime = np.asarray(matured_by_regime.get(str(reg_vals[t]), []), dtype=int)

            if sample_idx_regime.size >= min_regime_history:
                selected = sample_idx_regime
                src = "regime"
            else:
                selected = sample_idx_global
                src = "global_fallback"

            if selected.size < min_history:
                src = "insufficient_history"

            x_curve, y_curve = _fit_monotone_edge_curve(
                abs_m_bps[selected],
                realized_edge[selected],
                n_bins=cfg.edge_bins,
            )
            if x_curve.size == 0:
                edge_est = 0.0
                src = "insufficient_history"
            else:
                edge_est = _interp_monotone_curve(x_curve, y_curve, float(abs_m_bps[t]))
                if not np.isfinite(edge_est):
                    edge_est = 0.0

        edge_gross[t] = float(edge_est)
        model_source[t] = src

        regime_is_stress = str(reg_vals[t]) == "stress"
        regime_mult = float(cfg.stress_cost_multiplier if regime_is_stress else cfg.transient_cost_multiplier)
        regime_mult_arr[t] = regime_mult

        cost_grid: list[float] = []
        net_grid: list[float] = []
        break_even_grid: list[float] = []
        for s in size_grid:
            s_f = float(s)
            slippage_bps = _slippage_bps_at_size(s_f, slip_x, slip_y, cfg.fallback_linear_slippage_bps)
            cost_bps = float(fixed_cost_bps + regime_mult * slippage_bps)
            cost_grid.append(cost_bps)
            net_val = float(s_f * (edge_est - cost_bps))
            net_grid.append(net_val)

            be_val = _invert_monotone_curve(x_curve, y_curve, cost_bps)
            break_even_grid.append(be_val)

            per_size_expected[s_f][t] = net_val
            per_size_break_even[s_f][t] = be_val

        if len(net_grid) == 0:
            continue

        best_idx = int(np.nanargmax(np.asarray(net_grid, dtype=float)))
        best_size = float(size_grid[best_idx])
        best_cost = float(cost_grid[best_idx])
        best_expected = float(net_grid[best_idx])
        best_net = float(edge_est - best_cost)

        size_opt[t] = best_size
        edge_cost_opt[t] = best_cost
        expected_opt[t] = best_expected
        edge_net_opt[t] = best_net
        be_opt[t] = float(break_even_grid[best_idx])

    out = pd.DataFrame(index=m_t.index)
    out["edge_gross_bps"] = edge_gross
    out["edge_cost_bps_opt"] = edge_cost_opt
    out["edge_net_bps_opt"] = edge_net_opt
    out["expected_net_pnl_opt_bps"] = expected_opt
    out["optimal_size"] = size_opt
    out["break_even_premium_bps_opt"] = be_opt
    out["edge_horizon_bars"] = horizon.astype(float)
    out["edge_model_source"] = pd.Series(model_source, index=m_t.index, dtype="object")
    out["cost_model_source"] = pd.Series(cost_source_arr, index=m_t.index, dtype="object")
    out["cost_regime_multiplier"] = regime_mult_arr
    out["fixed_cost_bps"] = float(fixed_cost_bps)

    for s in size_grid:
        s_f = float(s)
        out[_size_column_name("expected_net_pnl_bps", s_f)] = per_size_expected[s_f]
        out[_size_column_name("break_even_premium_bps", s_f)] = per_size_break_even[s_f]

    return out


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
    stress_like = regime.eq("stress").fillna(False) | depeg_flag.fillna(False).astype(bool)

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
    premium: pd.Series | None = None,
    slippage_curve: pd.DataFrame | None = None,
    execution_unifier_cfg: ExecutionUnifierConfig | None = None,
    base_cost_bps: float = 1.0,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    depeg = depeg_flag.fillna(False).astype(bool).rename("depeg_flag")

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
        depeg_flag=depeg,
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

    riskoff = depeg.copy()
    riskoff_reason = pd.Series("none", index=m_t.index, name="riskoff_reason", dtype="object")
    riskoff_reason.loc[depeg] = "depeg_flag"
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
        riskoff_reason.loc[hawkes_riskoff_signal & ~depeg] = "hawkes_riskoff"
    riskoff = riskoff.fillna(False).rename("riskoff_flag")
    decision.loc[riskoff] = "Risk-off"

    stress_widen = ((usdc_stress | usdt_concern | technical_flow) & (~riskoff)).rename("stress_widen_flag")
    widen = (~riskoff) & (widen_flag | stress_widen | hawkes_widen_signal)
    decision.loc[widen] = "Widen"

    unifier_cfg = execution_unifier_cfg or ExecutionUnifierConfig()
    unifier_frame = _compute_execution_unifier(
        m_t=m_t,
        premium=premium,
        regime=regime,
        cfg=unifier_cfg,
        base_cost_bps=base_cost_bps,
        slippage_curve=slippage_curve,
    )

    base_trade_gate = (~riskoff) & (~widen) & trade_signal
    if bool(unifier_cfg.enabled) and premium is not None:
        expected_opt = pd.to_numeric(unifier_frame["expected_net_pnl_opt_bps"], errors="coerce").fillna(0.0)
        optimal_size = pd.to_numeric(unifier_frame["optimal_size"], errors="coerce").clip(lower=0.0, upper=1.0)
        trade_min = float(np.clip(unifier_cfg.trade_min_size, 0.0, 1.0))
        net_edge_trade = expected_opt.gt(0.0) & optimal_size.ge(trade_min) & m_t.notna()
        trade = base_trade_gate & net_edge_trade
    else:
        trade = base_trade_gate

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

    if bool(unifier_cfg.enabled) and premium is not None:
        position_size = pd.Series(0.0, index=m_t.index, dtype="float64")
        optimal_size = pd.to_numeric(unifier_frame["optimal_size"], errors="coerce").clip(lower=0.0, upper=1.0)
        widen_floor = float(np.clip(unifier_cfg.widen_floor_size, 0.0, 1.0))
        position_size.loc[trade] = optimal_size.loc[trade]
        position_size.loc[(~riskoff) & (~trade)] = widen_floor
        position_size = position_size.rename("position_size")
    else:
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
            riskoff_reason,
            riskoff,
            stress_widen,
            trade_signal.rename("trade_signal"),
            confidence,
            unifier_frame,
            position_size,
            decision,
            side,
        ],
        axis=1,
    )
