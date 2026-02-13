from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

VALID_PROXY_METHODS = {"median", "pw_rolling"}


@dataclass(frozen=True)
class PremiumConfig:
    target_usdc_symbol: str = "BTCUSDC-PERP"
    target_usdt_symbol: str = "BTCUSDT-PERP"
    depeg_delta_log: float = 0.002
    depeg_min_consecutive: int = 5
    min_price: float = 1e-12
    auto_resolve_target_pair: bool = True
    allow_synthetic_usdc_from_usdt: bool = False
    allow_target_pair_as_proxy: bool = False
    fail_on_missing_proxy: bool = True
    proxy_method: str = "median"
    pw_window: str = "12h"
    pw_min_period_fraction: float = 0.5
    pw_rho_clip: float = 0.98
    pw_fallback_to_median: bool = True


def compute_naive_premium(
    price_matrix: pd.DataFrame,
    usdc_symbol: str,
    usdt_symbol: str,
    *,
    min_price: float = 1e-12,
) -> pd.Series:
    if usdc_symbol not in price_matrix.columns or usdt_symbol not in price_matrix.columns:
        raise KeyError(
            f"Missing target symbols in price matrix. Need {usdc_symbol} and {usdt_symbol}."
        )
    p_usdc = price_matrix[usdc_symbol].astype(float).clip(lower=min_price)
    p_usdt = price_matrix[usdt_symbol].astype(float).clip(lower=min_price)
    return np.log(p_usdc) - np.log(p_usdt)


def _normalize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", symbol.upper())


def _split_symbol(symbol: str) -> tuple[str, str, str] | None:
    normalized = _normalize_symbol(symbol)
    for stable in ("USDC", "USDT"):
        idx = normalized.find(stable)
        if idx > 0:
            root = normalized[:idx]
            suffix = normalized[idx + len(stable) :]
            return root, stable, suffix
    return None


def _replace_stable_token(symbol: str, old_token: str, new_token: str) -> str | None:
    upper = str(symbol).upper()
    idx = upper.find(str(old_token).upper())
    if idx < 0:
        return None
    old = str(old_token)
    return f"{symbol[:idx]}{new_token}{symbol[idx + len(old):]}"


def synthesize_usdc_legs_from_usdt(
    price_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    out = price_matrix.copy()
    available = set(out.columns)
    created_pairs: list[tuple[str, str]] = []

    for usdt_symbol in list(out.columns):
        parts = _split_symbol(str(usdt_symbol))
        if parts is None:
            continue
        _, stable, _ = parts
        if stable != "USDT":
            continue
        usdc_symbol = _replace_stable_token(str(usdt_symbol), "USDT", "USDC")
        if not usdc_symbol:
            continue
        if usdc_symbol in available:
            continue
        out[usdc_symbol] = out[usdt_symbol].astype(float)
        available.add(usdc_symbol)
        created_pairs.append((usdc_symbol, str(usdt_symbol)))

    return out, created_pairs


def infer_cross_asset_pairs(
    symbols: Iterable[str],
    *,
    exclude: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    buckets: dict[tuple[str, str], dict[str, str]] = {}
    for symbol in symbols:
        parts = _split_symbol(symbol)
        if parts is None:
            continue
        root, stable, suffix = parts
        key = (root, suffix)
        buckets.setdefault(key, {})[stable] = symbol

    pairs: list[tuple[str, str]] = []
    for key in sorted(buckets):
        stable_map = buckets[key]
        if {"USDC", "USDT"}.issubset(stable_map):
            pair = (stable_map["USDC"], stable_map["USDT"])
            if exclude and set(pair) == set(exclude):
                continue
            pairs.append(pair)
    return pairs


def _pair_symbol_set(pair: tuple[str, str]) -> frozenset[str]:
    return frozenset((pair[0], pair[1]))


def resolve_target_pair(
    price_matrix: pd.DataFrame,
    *,
    usdc_symbol: str,
    usdt_symbol: str,
    auto_resolve: bool = True,
) -> tuple[str, str]:
    configured_pair = (usdc_symbol, usdt_symbol)
    columns = set(price_matrix.columns)
    if usdc_symbol in columns and usdt_symbol in columns:
        return configured_pair

    missing = [symbol for symbol in configured_pair if symbol not in columns]
    if not auto_resolve:
        raise KeyError(
            f"Missing target symbols in price matrix. Need {usdc_symbol} and {usdt_symbol}. "
            f"Missing={missing}"
        )

    inferred_pairs = infer_cross_asset_pairs(price_matrix.columns)
    if not inferred_pairs:
        available = ", ".join(sorted(price_matrix.columns))
        raise ValueError(
            "No compatible USDC/USDT target pair found in price matrix. "
            "Expected a matched pair like BTCUSDC/BTCUSDT for the same market suffix. "
            f"Configured target={configured_pair}, available columns=[{available}]"
        )

    target_parts = _split_symbol(usdc_symbol)
    target_root = target_parts[0] if target_parts else "BTC"
    target_suffix = target_parts[2] if target_parts else ""

    scored_pairs: list[tuple[float, float, tuple[str, str]]] = []
    for pair in inferred_pairs:
        pair_parts = _split_symbol(pair[0])
        root = pair_parts[0] if pair_parts else ""
        suffix = pair_parts[2] if pair_parts else ""
        score = 0.0
        if root == target_root:
            score += 10.0
        if target_suffix and suffix == target_suffix:
            score += 3.0
        if root == "BTC":
            score += 1.0
        overlap = float(price_matrix[list(pair)].notna().all(axis=1).mean())
        scored_pairs.append((score, overlap, pair))

    _, _, best_pair = max(scored_pairs, key=lambda row: (row[0], row[1], row[2][0], row[2][1]))
    return best_pair


def _resolve_freq_timedelta(index: pd.Index, *, freq: str | None) -> pd.Timedelta:
    if freq is not None:
        freq_td = pd.to_timedelta(freq)
        if freq_td <= pd.Timedelta(0):
            raise ValueError(f"freq must be positive, got {freq!r}")
        return freq_td

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError(
            "pw_rolling proxy requires either `freq` or a DatetimeIndex with at least 2 rows."
        )
    deltas = index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError(
            "Could not infer positive sampling frequency from index for pw_rolling proxy."
        )
    return positive.median()


def _window_to_observations(window: str, freq_td: pd.Timedelta) -> int:
    window_td = pd.to_timedelta(window)
    if window_td <= pd.Timedelta(0):
        raise ValueError(f"pw_window must be positive, got {window!r}")
    obs = int(np.ceil(window_td / freq_td))
    return max(obs, 2)


def _estimate_ar1_rho(values: np.ndarray, *, rho_clip: float) -> float:
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
    clip = float(np.clip(abs(rho_clip), 1e-6, 0.999))
    return float(np.clip(rho, -clip, clip))


def _prais_winsten_intercept(values: np.ndarray, *, rho: float) -> float:
    if values.size == 0:
        return np.nan
    if values.size == 1:
        return float(values[0])

    r = float(np.clip(rho, -0.999, 0.999))
    gamma_sq = max(1.0 - (r * r), 0.0)
    gamma = float(np.sqrt(gamma_sq))

    x_star = np.ones_like(values, dtype=float)
    x_star[0] = gamma
    y_star = values.astype(float).copy()
    y_star[0] = gamma * y_star[0]
    y_star[1:] = values[1:] - (r * values[:-1])
    x_star[1:] = 1.0 - r

    den = float(np.dot(x_star, x_star))
    if den <= 1e-20:
        return float(np.nanmedian(values))
    return float(np.dot(x_star, y_star) / den)


def _rolling_pw_location(
    series: pd.Series,
    *,
    window_obs: int,
    min_periods: int,
    rho_clip: float,
    fallback_to_current: bool,
) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)

    for i in range(values.shape[0]):
        start = max(0, i - window_obs + 1)
        window = values[start : i + 1]
        finite = window[np.isfinite(window)]
        if finite.size < min_periods:
            if fallback_to_current and np.isfinite(values[i]):
                out[i] = values[i]
            continue

        rho = _estimate_ar1_rho(finite, rho_clip=rho_clip)
        mu_hat = _prais_winsten_intercept(finite, rho=rho)
        if np.isfinite(mu_hat):
            out[i] = mu_hat
        elif fallback_to_current and np.isfinite(values[i]):
            out[i] = values[i]

    return pd.Series(out, index=series.index, name="stablecoin_proxy")


def _diff_std(series: pd.Series) -> float:
    diffs = series.astype(float).diff().to_numpy(dtype=float)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=0))


def _ewma_smooth(series: pd.Series, *, span_obs: int) -> pd.Series:
    span = max(2, int(span_obs))
    smoothed = series.ewm(span=span, adjust=False, min_periods=1).mean()
    smoothed = smoothed.where(series.notna())
    return smoothed.rename(series.name)


def _enforce_proxy_smoothness(
    proxy: pd.Series,
    *,
    target_p_naive: pd.Series,
    freq_td: pd.Timedelta,
    max_ratio: float = 1.0,
) -> pd.Series:
    if not np.isfinite(max_ratio) or max_ratio <= 0.0:
        raise ValueError("max_ratio must be a positive finite float.")

    target_std = _diff_std(target_p_naive)
    proxy_std = _diff_std(proxy)
    if (not np.isfinite(target_std)) or target_std <= 0.0 or (not np.isfinite(proxy_std)):
        return proxy
    if proxy_std <= (max_ratio * target_std):
        return proxy

    base_span = max(2, int(np.ceil(pd.Timedelta("5min") / freq_td)))
    span_grid = sorted({max(2, base_span * mult) for mult in (1, 2, 4, 8, 16, 32)})

    best = proxy
    for span in span_grid:
        candidate = _ewma_smooth(proxy, span_obs=span)
        candidate_std = _diff_std(candidate)
        if not np.isfinite(candidate_std):
            continue
        best = candidate
        if candidate_std <= (max_ratio * target_std):
            return candidate
    return best


def compute_stablecoin_proxy(
    price_matrix: pd.DataFrame,
    *,
    proxy_pairs: Sequence[tuple[str, str]] | None = None,
    target_pair: tuple[str, str] | None = None,
    min_price: float = 1e-12,
    allow_target_pair_fallback: bool = False,
    fail_on_missing_proxy: bool = True,
    proxy_method: str = "median",
    freq: str | None = None,
    pw_window: str = "12h",
    pw_min_period_fraction: float = 0.5,
    pw_rho_clip: float = 0.98,
    pw_fallback_to_median: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    method = str(proxy_method).strip().lower()
    if method not in VALID_PROXY_METHODS:
        raise ValueError(
            f"Unsupported proxy_method={proxy_method!r}. Expected one of {sorted(VALID_PROXY_METHODS)}."
        )
    pairs = list(proxy_pairs) if proxy_pairs is not None else infer_cross_asset_pairs(
        price_matrix.columns, exclude=target_pair
    )

    available_columns = set(price_matrix.columns)
    effective_pairs = [
        (usdc_symbol, usdt_symbol)
        for usdc_symbol, usdt_symbol in pairs
        if usdc_symbol in available_columns and usdt_symbol in available_columns
    ]
    if target_pair is not None and not allow_target_pair_fallback:
        effective_pairs = [
            pair for pair in effective_pairs if _pair_symbol_set(pair) != _pair_symbol_set(target_pair)
        ]

    # If configured pairs are unavailable for this window, fallback to inferred pairs.
    if not effective_pairs:
        inferred = infer_cross_asset_pairs(price_matrix.columns, exclude=target_pair)
        if inferred:
            effective_pairs = inferred
        elif allow_target_pair_fallback and target_pair is not None:
            warnings.warn(
                "Stablecoin proxy fallback is using the target pair itself. "
                "Debiased premium can collapse to near-zero values.",
                stacklevel=2,
            )
            effective_pairs = [target_pair]

    components: dict[str, pd.Series] = {}
    for usdc_symbol, usdt_symbol in effective_pairs:
        component = compute_naive_premium(
            price_matrix, usdc_symbol, usdt_symbol, min_price=min_price
        )
        components[f"{usdc_symbol}__{usdt_symbol}"] = component

    if not components:
        if fail_on_missing_proxy:
            raise ValueError(
                "Could not compute stablecoin proxy without reusing the target pair. "
                "Provide valid cross-asset proxy_pairs, include additional USDC/USDT legs, "
                "or set premium.allow_target_pair_as_proxy=true / premium.fail_on_missing_proxy=false."
            )
        stablecoin_proxy = pd.Series(0.0, index=price_matrix.index, name="stablecoin_proxy")
        return stablecoin_proxy, pd.DataFrame(index=price_matrix.index)

    proxy_components = pd.DataFrame(components, index=price_matrix.index)
    median_proxy = proxy_components.median(axis=1, skipna=True).rename("stablecoin_proxy")

    if method == "median":
        stablecoin_proxy = median_proxy
    else:
        if not 0 < float(pw_min_period_fraction) <= 1:
            raise ValueError("pw_min_period_fraction must be in (0, 1].")
        freq_td = _resolve_freq_timedelta(price_matrix.index, freq=freq)
        window_obs = _window_to_observations(pw_window, freq_td)
        min_periods = max(2, int(np.ceil(window_obs * float(pw_min_period_fraction))))
        stablecoin_proxy = _rolling_pw_location(
            median_proxy,
            window_obs=window_obs,
            min_periods=min_periods,
            rho_clip=float(pw_rho_clip),
            fallback_to_current=bool(pw_fallback_to_median),
        )

    can_compare_to_target = (
        target_pair is not None
        and target_pair[0] in available_columns
        and target_pair[1] in available_columns
    )
    if can_compare_to_target:
        target_p_naive = compute_naive_premium(
            price_matrix,
            target_pair[0],
            target_pair[1],
            min_price=min_price,
        )
        try:
            freq_td = _resolve_freq_timedelta(price_matrix.index, freq=freq)
        except ValueError:
            freq_td = None
        if freq_td is not None:
            stablecoin_proxy = _enforce_proxy_smoothness(
                stablecoin_proxy.rename("stablecoin_proxy"),
                target_p_naive=target_p_naive,
                freq_td=freq_td,
                max_ratio=1.0,
            )

    return stablecoin_proxy, proxy_components


def compute_depeg_flag(
    stablecoin_proxy: pd.Series,
    *,
    delta_log: float = 0.002,
    min_consecutive: int = 5,
    freq: str | None = None,
) -> pd.Series:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")
    min_bars = int(min_consecutive)
    if freq is not None:
        freq_td = pd.to_timedelta(freq)
        if freq_td <= pd.Timedelta(0):
            raise ValueError(f"freq must map to a positive timedelta, got {freq!r}")
        freq_minutes = float(freq_td.total_seconds() / 60.0)
        if freq_minutes <= 0.0:
            raise ValueError(f"freq must map to a positive timedelta, got {freq!r}")
        # `min_consecutive` is specified in minutes to keep behavior stable
        # when changing bar frequency (1s, 1min, ...).
        min_bars = max(1, int(np.ceil(float(min_consecutive) / freq_minutes)))
    excursion = stablecoin_proxy.abs().ge(delta_log)
    active = (
        excursion.rolling(min_bars, min_periods=min_bars)
        .sum()
        .ge(min_bars)
    )
    return active.fillna(False).rename("depeg_flag")


def build_premium_frame(
    price_matrix: pd.DataFrame,
    cfg: PremiumConfig,
    *,
    proxy_pairs: Sequence[tuple[str, str]] | None = None,
    freq: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working_matrix = price_matrix
    target_pair: tuple[str, str]

    try:
        target_pair = resolve_target_pair(
            working_matrix,
            usdc_symbol=cfg.target_usdc_symbol,
            usdt_symbol=cfg.target_usdt_symbol,
            auto_resolve=cfg.auto_resolve_target_pair,
        )
    except ValueError as exc:
        no_pair_error = "No compatible USDC/USDT target pair found in price matrix."
        if (not cfg.allow_synthetic_usdc_from_usdt) or (no_pair_error not in str(exc)):
            raise

        synthesized, created_pairs = synthesize_usdc_legs_from_usdt(price_matrix)
        if not created_pairs:
            raise
        working_matrix = synthesized
        target_pair = resolve_target_pair(
            working_matrix,
            usdc_symbol=cfg.target_usdc_symbol,
            usdt_symbol=cfg.target_usdt_symbol,
            auto_resolve=cfg.auto_resolve_target_pair,
        )
        warnings.warn(
            "No native USDC leg detected; synthesized USDC symbols from USDT columns "
            f"for compatibility ({len(created_pairs)} synthetic pairs).",
            stacklevel=2,
        )

    if target_pair != (cfg.target_usdc_symbol, cfg.target_usdt_symbol):
        warnings.warn(
            "Configured target pair is unavailable for this episode. "
            f"Auto-selected {target_pair[0]} vs {target_pair[1]} instead.",
            stacklevel=2,
        )
    p_naive = compute_naive_premium(
        working_matrix,
        target_pair[0],
        target_pair[1],
        min_price=cfg.min_price,
    ).rename("p_naive")
    stablecoin_proxy, proxy_components = compute_stablecoin_proxy(
        working_matrix,
        proxy_pairs=proxy_pairs,
        target_pair=target_pair,
        min_price=cfg.min_price,
        allow_target_pair_fallback=cfg.allow_target_pair_as_proxy,
        fail_on_missing_proxy=cfg.fail_on_missing_proxy,
        proxy_method=cfg.proxy_method,
        freq=freq,
        pw_window=cfg.pw_window,
        pw_min_period_fraction=cfg.pw_min_period_fraction,
        pw_rho_clip=cfg.pw_rho_clip,
        pw_fallback_to_median=cfg.pw_fallback_to_median,
    )
    if proxy_components.empty:
        warnings.warn(
            "Stablecoin proxy components are empty. Debiasing is disabled for this run.",
            stacklevel=2,
        )
    debiased = (p_naive - stablecoin_proxy).rename("p")
    depeg_flag = compute_depeg_flag(
        stablecoin_proxy,
        delta_log=cfg.depeg_delta_log,
        min_consecutive=cfg.depeg_min_consecutive,
        freq=freq,
    )
    out = pd.concat([p_naive, stablecoin_proxy, debiased, depeg_flag], axis=1)
    return out, proxy_components
