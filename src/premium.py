from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PremiumConfig:
    target_usdc_symbol: str = "BTCUSDC-PERP"
    target_usdt_symbol: str = "BTCUSDT-PERP"
    depeg_delta_log: float = 0.002
    depeg_min_consecutive: int = 5
    min_price: float = 1e-12


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


def compute_stablecoin_proxy(
    price_matrix: pd.DataFrame,
    *,
    proxy_pairs: Sequence[tuple[str, str]] | None = None,
    target_pair: tuple[str, str] | None = None,
    min_price: float = 1e-12,
) -> tuple[pd.Series, pd.DataFrame]:
    pairs = list(proxy_pairs) if proxy_pairs is not None else infer_cross_asset_pairs(
        price_matrix.columns, exclude=target_pair
    )
    if not pairs and target_pair is not None:
        pairs = [target_pair]

    available_columns = set(price_matrix.columns)
    effective_pairs = [
        (usdc_symbol, usdt_symbol)
        for usdc_symbol, usdt_symbol in pairs
        if usdc_symbol in available_columns and usdt_symbol in available_columns
    ]

    # If configured pairs are unavailable for this window, fallback to inferred pairs.
    if not effective_pairs:
        inferred = infer_cross_asset_pairs(price_matrix.columns, exclude=target_pair)
        if inferred:
            effective_pairs = inferred
        elif target_pair is not None:
            effective_pairs = [target_pair]

    components: dict[str, pd.Series] = {}
    for usdc_symbol, usdt_symbol in effective_pairs:
        component = compute_naive_premium(
            price_matrix, usdc_symbol, usdt_symbol, min_price=min_price
        )
        components[f"{usdc_symbol}__{usdt_symbol}"] = component

    if not components:
        raise ValueError(
            "Could not compute stablecoin proxy. Provide valid proxy_pairs or "
            "include at least one USDC/USDT cross-asset pair in data."
        )

    proxy_components = pd.DataFrame(components, index=price_matrix.index)
    stablecoin_proxy = proxy_components.median(axis=1, skipna=True).rename("stablecoin_proxy")
    return stablecoin_proxy, proxy_components


def compute_depeg_flag(
    stablecoin_proxy: pd.Series,
    *,
    delta_log: float = 0.002,
    min_consecutive: int = 5,
) -> pd.Series:
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")
    excursion = stablecoin_proxy.abs().ge(delta_log)
    active = (
        excursion.rolling(min_consecutive, min_periods=min_consecutive)
        .sum()
        .ge(min_consecutive)
    )
    return active.fillna(False).rename("depeg_flag")


def build_premium_frame(
    price_matrix: pd.DataFrame,
    cfg: PremiumConfig,
    *,
    proxy_pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_pair = (cfg.target_usdc_symbol, cfg.target_usdt_symbol)
    p_naive = compute_naive_premium(
        price_matrix,
        cfg.target_usdc_symbol,
        cfg.target_usdt_symbol,
        min_price=cfg.min_price,
    ).rename("p_naive")
    stablecoin_proxy, proxy_components = compute_stablecoin_proxy(
        price_matrix,
        proxy_pairs=proxy_pairs,
        target_pair=target_pair,
        min_price=cfg.min_price,
    )
    debiased = (p_naive - stablecoin_proxy).rename("p")
    depeg_flag = compute_depeg_flag(
        stablecoin_proxy,
        delta_log=cfg.depeg_delta_log,
        min_consecutive=cfg.depeg_min_consecutive,
    )
    out = pd.concat([p_naive, stablecoin_proxy, debiased, depeg_flag], axis=1)
    return out, proxy_components
