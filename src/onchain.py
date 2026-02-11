from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class OnchainConfig:
    enabled: bool = True
    provider: str = "defillama"
    stablecoin_prices_url: str = "https://stablecoins.llama.fi/stablecoinprices"
    usdc_key: str = "usd-coin"
    usdt_key: str = "tether"
    cache_path: str = "data/processed/onchain/defillama_stablecoinprices.csv"
    cache_max_age_hours: int = 24
    request_timeout_sec: int = 40
    depeg_delta_log: float = 0.001
    depeg_min_consecutive: int = 5
    divergence_alert_log: float = 0.001


ONCHAIN_COLUMNS = [
    "onchain_usdc_price",
    "onchain_usdt_price",
    "onchain_usdc_minus_1",
    "onchain_usdt_minus_1",
    "onchain_log_usdc_dev",
    "onchain_log_usdt_dev",
    "onchain_proxy",
    "onchain_divergence",
    "onchain_depeg_flag",
    "onchain_divergence_flag",
    "onchain_data_available",
]


def empty_onchain_frame(index: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["onchain_usdc_price"] = np.nan
    frame["onchain_usdt_price"] = np.nan
    frame["onchain_usdc_minus_1"] = np.nan
    frame["onchain_usdt_minus_1"] = np.nan
    frame["onchain_log_usdc_dev"] = np.nan
    frame["onchain_log_usdt_dev"] = np.nan
    frame["onchain_proxy"] = np.nan
    frame["onchain_divergence"] = np.nan
    frame["onchain_depeg_flag"] = False
    frame["onchain_divergence_flag"] = False
    frame["onchain_data_available"] = False
    return frame


def _fetch_defillama_stablecoin_prices(cfg: OnchainConfig) -> pd.DataFrame:
    response = requests.get(cfg.stablecoin_prices_url, timeout=cfg.request_timeout_sec)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected DefiLlama payload for stablecoin prices.")

    rows: list[dict[str, float]] = []
    for point in payload:
        if not isinstance(point, dict):
            continue
        prices = point.get("prices", {})
        if not isinstance(prices, dict):
            continue
        ts = point.get("date")
        if ts is None:
            continue

        usdc = prices.get(cfg.usdc_key)
        usdt = prices.get(cfg.usdt_key)
        rows.append(
            {
                "timestamp_utc": pd.to_datetime(ts, unit="s", utc=True, errors="coerce"),
                "onchain_usdc_price": pd.to_numeric(usdc, errors="coerce"),
                "onchain_usdt_price": pd.to_numeric(usdt, errors="coerce"),
            }
        )

    frame = pd.DataFrame(rows).dropna(subset=["timestamp_utc"])
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return frame.reset_index(drop=True)


def _cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - modified <= timedelta(hours=max_age_hours)


def _load_onchain_daily(cfg: OnchainConfig) -> pd.DataFrame:
    cache_path = Path(cfg.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if _cache_is_fresh(cache_path, cfg.cache_max_age_hours):
        try:
            cached = pd.read_csv(cache_path)
            cached["timestamp_utc"] = pd.to_datetime(cached["timestamp_utc"], utc=True, errors="coerce")
            cached["onchain_usdc_price"] = pd.to_numeric(cached["onchain_usdc_price"], errors="coerce")
            cached["onchain_usdt_price"] = pd.to_numeric(cached["onchain_usdt_price"], errors="coerce")
            cached = cached.dropna(subset=["timestamp_utc"])
            if not cached.empty:
                return cached.sort_values("timestamp_utc").reset_index(drop=True)
        except Exception as exc:
            warnings.warn(f"Could not read on-chain cache {cache_path}: {exc}")

    fresh = _fetch_defillama_stablecoin_prices(cfg)
    if fresh.empty:
        return fresh
    fresh.to_csv(cache_path, index=False)
    return fresh


def build_onchain_validation_frame(
    *,
    index: pd.DatetimeIndex,
    stablecoin_proxy: pd.Series,
    cfg: OnchainConfig,
) -> pd.DataFrame:
    if not cfg.enabled or index.empty:
        return empty_onchain_frame(index)

    if cfg.provider != "defillama":
        raise ValueError(f"Unsupported on-chain provider: {cfg.provider}")

    daily = _load_onchain_daily(cfg)
    if daily.empty:
        return empty_onchain_frame(index)

    onchain = daily.set_index("timestamp_utc").sort_index()
    daily_usdc = onchain["onchain_usdc_price"].astype(float)
    daily_usdt = onchain["onchain_usdt_price"].astype(float)
    daily_proxy = pd.Series(np.nan, index=onchain.index, dtype=float, name="onchain_proxy")
    daily_valid = daily_usdc.gt(0) & daily_usdt.gt(0)
    daily_proxy.loc[daily_valid] = np.log(daily_usdt.loc[daily_valid]) - np.log(daily_usdc.loc[daily_valid])

    # Depeg persistence must be computed at the native on-chain sampling cadence
    # (daily for DefiLlama), not at the pipeline bar frequency after forward-fill.
    daily_excursion = daily_proxy.abs().ge(cfg.depeg_delta_log)
    daily_depeg_flag = (
        daily_excursion.rolling(cfg.depeg_min_consecutive, min_periods=cfg.depeg_min_consecutive)
        .sum()
        .ge(cfg.depeg_min_consecutive)
        .fillna(False)
    )

    aligned = onchain.reindex(index, method="ffill")

    usdc = aligned["onchain_usdc_price"].astype(float)
    usdt = aligned["onchain_usdt_price"].astype(float)
    usdc_minus_1 = (usdc - 1.0).rename("onchain_usdc_minus_1")
    usdt_minus_1 = (usdt - 1.0).rename("onchain_usdt_minus_1")
    log_usdc = pd.Series(np.nan, index=index, dtype=float, name="onchain_log_usdc_dev")
    log_usdt = pd.Series(np.nan, index=index, dtype=float, name="onchain_log_usdt_dev")
    pos_usdc = usdc.gt(0)
    pos_usdt = usdt.gt(0)
    log_usdc.loc[pos_usdc] = np.log(usdc.loc[pos_usdc])
    log_usdt.loc[pos_usdt] = np.log(usdt.loc[pos_usdt])
    proxy = daily_proxy.reindex(index, method="ffill").rename("onchain_proxy")

    divergence = (stablecoin_proxy.astype(float) - proxy).rename("onchain_divergence")
    onchain_depeg_flag = daily_depeg_flag.reindex(index, method="ffill").fillna(False).rename("onchain_depeg_flag")

    divergence_flag = divergence.abs().ge(cfg.divergence_alert_log).fillna(False).rename("onchain_divergence_flag")

    out = pd.DataFrame(
        {
            "onchain_usdc_price": usdc,
            "onchain_usdt_price": usdt,
            "onchain_usdc_minus_1": usdc_minus_1,
            "onchain_usdt_minus_1": usdt_minus_1,
            "onchain_log_usdc_dev": log_usdc,
            "onchain_log_usdt_dev": log_usdt,
            "onchain_proxy": proxy,
            "onchain_divergence": divergence,
            "onchain_depeg_flag": onchain_depeg_flag,
            "onchain_divergence_flag": divergence_flag,
            "onchain_data_available": proxy.notna(),
        },
        index=index,
    )
    return out
