from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
from typing import Any
import warnings

import pandas as pd

from src.onchain import OnchainConfig, build_onchain_validation_frame, empty_onchain_frame
from src.premium import PremiumConfig, build_premium_frame
from src.regimes import RegimeConfig, build_regime_frame
from src.robust_filter import RobustFilterConfig, build_robust_frame
from src.statmech import StatMechConfig, build_statmech_frame


@dataclass(frozen=True)
class FactorialVariant:
    premium: str
    gating: bool
    statmech: bool
    hawkes: bool

    @property
    def variant_id(self) -> str:
        premium = str(self.premium).strip().lower()
        gating = "on" if bool(self.gating) else "off"
        statmech = "on" if bool(self.statmech) else "off"
        hawkes = "on" if bool(self.hawkes) else "off"
        return f"premium_{premium}__gating_{gating}__statmech_{statmech}__hawkes_{hawkes}"


def build_factorial_variants() -> list[FactorialVariant]:
    out: list[FactorialVariant] = []
    for premium, gating, statmech, hawkes in product(
        ["naive", "debiased"],
        [False, True],
        [False, True],
        [False, True],
    ):
        out.append(
            FactorialVariant(
                premium=str(premium),
                gating=bool(gating),
                statmech=bool(statmech),
                hawkes=bool(hawkes),
            )
        )
    return out


def build_dataclass(cls: Any, data: dict[str, Any] | None) -> Any:
    data = data or {}
    valid_fields = {field.name for field in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**kwargs)


def simple_decision(
    signal: pd.Series,
    threshold: pd.Series | float,
    depeg_flag: pd.Series,
) -> pd.Series:
    decision = pd.Series("Widen", index=signal.index, dtype="object")
    trade = signal.abs().gt(threshold)
    decision.loc[trade] = "Trade"
    decision.loc[depeg_flag.astype(bool)] = "Risk-off"
    return decision.rename("decision")


def compute_core_frames(
    config: dict[str, Any],
    matrix: pd.DataFrame,
    *,
    premium_leg: str = "debiased",
) -> dict[str, pd.DataFrame]:
    premium_leg_norm = str(premium_leg).strip().lower()
    if premium_leg_norm not in {"naive", "debiased"}:
        raise ValueError(f"Unsupported premium_leg={premium_leg!r}; expected 'naive' or 'debiased'.")

    freq = str(config.get("data", {}).get("resample_rule", "1min"))

    premium_cfg = build_dataclass(PremiumConfig, config.get("premium"))
    premium_raw = config.get("premium", {})
    proxy_pairs_raw = premium_raw.get("proxy_pairs", [])
    proxy_pairs = [tuple(pair) for pair in proxy_pairs_raw] if proxy_pairs_raw else None
    premium_frame, proxy_components = build_premium_frame(
        matrix,
        premium_cfg,
        proxy_pairs=proxy_pairs,
        freq=freq,
    )

    onchain_cfg = build_dataclass(OnchainConfig, config.get("onchain"))
    onchain_fail_closed_on_error = bool(getattr(onchain_cfg, "fail_closed_on_error", True))
    if onchain_cfg.enabled:
        try:
            onchain_frame = build_onchain_validation_frame(
                index=premium_frame.index,
                stablecoin_proxy=premium_frame["stablecoin_proxy"],
                cfg=onchain_cfg,
            )
        except Exception as exc:
            if onchain_fail_closed_on_error:
                warnings.warn(
                    "On-chain validation failed in ablation run; fail-closed guardrail engaged "
                    f"(forcing Risk-off until recovery): {exc}"
                )
            else:
                warnings.warn(
                    "On-chain validation failed in ablation run; continuing fail-open without "
                    f"on-chain guardrail: {exc}"
                )
            onchain_frame = empty_onchain_frame(
                premium_frame.index,
                fail_closed=onchain_fail_closed_on_error,
            )
    else:
        onchain_frame = empty_onchain_frame(premium_frame.index)

    market_depeg_flag = premium_frame["depeg_flag"].fillna(False).astype(bool).rename("market_depeg_flag")
    premium_frame["market_depeg_flag"] = market_depeg_flag
    onchain_effective = onchain_frame.get(
        "onchain_depeg_flag_effective",
        pd.Series(False, index=premium_frame.index, name="onchain_depeg_flag_effective"),
    )
    onchain_effective = onchain_effective.fillna(False).astype(bool)
    premium_frame["depeg_flag"] = (market_depeg_flag | onchain_effective).rename("depeg_flag")

    signal_premium = premium_frame["p_naive"] if premium_leg_norm == "naive" else premium_frame["p"]
    signal_premium = signal_premium.rename("signal_premium")
    robust_cfg = build_dataclass(RobustFilterConfig, config.get("robust_filter"))
    robust_frame = build_robust_frame(signal_premium, cfg=robust_cfg, freq=freq)
    m_t = robust_frame["p_smooth"].rename("m_t")

    stat_cfg = build_dataclass(StatMechConfig, config.get("statmech"))
    state_frame = build_statmech_frame(robust_frame["z_t"], m_t, cfg=stat_cfg, freq=freq)

    regime_cfg = build_dataclass(RegimeConfig, config.get("regimes"))
    regime_frame = build_regime_frame(state_frame, regime_cfg)

    return {
        "premium_frame": premium_frame,
        "signal_premium": signal_premium,
        "proxy_components": proxy_components,
        "onchain_frame": onchain_frame,
        "robust_frame": robust_frame,
        "state_frame": state_frame,
        "regime_frame": regime_frame,
        "m_t": m_t,
        "freq": pd.Series([freq]),
    }
