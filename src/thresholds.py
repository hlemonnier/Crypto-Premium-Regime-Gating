from __future__ import annotations

import pandas as pd


VALID_THRESHOLD_MODES = {"fixed", "expanding", "rolling"}


def quantile_threshold(
    series: pd.Series,
    quantile: float,
    *,
    mode: str = "expanding",
    min_periods: int = 120,
    window: int | None = None,
    shift: int = 1,
    name: str | None = None,
) -> pd.Series:
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got: {quantile}")
    if mode not in VALID_THRESHOLD_MODES:
        raise ValueError(
            f"Unsupported threshold mode: {mode}. Expected one of {sorted(VALID_THRESHOLD_MODES)}"
        )

    values = series.astype(float)
    out_name = name or f"{series.name or 'series'}_q{quantile:.3f}"

    if mode == "fixed":
        value = float(values.quantile(quantile))
        return pd.Series(value, index=values.index, name=out_name, dtype=float)

    min_periods = max(2, int(min_periods))
    if mode == "expanding":
        threshold = values.expanding(min_periods=min_periods).quantile(quantile)
    else:
        if window is None:
            raise ValueError("window must be provided when mode='rolling'")
        window = max(2, int(window))
        threshold = values.rolling(window=window, min_periods=min(min_periods, window)).quantile(quantile)

    if shift > 0:
        threshold = threshold.shift(int(shift))
    return threshold.rename(out_name)
