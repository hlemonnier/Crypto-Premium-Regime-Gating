from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline import load_config, run_pipeline


class PipelineMispricingSourceTests(unittest.TestCase):
    @staticmethod
    def _build_matrix(n: int = 360) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        x = np.linspace(0.0, 1.0, n)
        btc_usdt = 43000.0 + (600.0 * np.sin(8.0 * np.pi * x)) + (250.0 * x)
        eth_usdt = 2400.0 + (80.0 * np.cos(7.0 * np.pi * x)) + (20.0 * x)
        usdc_offset = 1.0 + (0.0005 * np.sin(5.0 * np.pi * x))
        matrix = pd.DataFrame(
            {
                "BTCUSDT-PERP": btc_usdt,
                "BTCUSDC-PERP": btc_usdt * usdc_offset,
                "ETHUSDT-PERP": eth_usdt,
                "ETHUSDC-PERP": eth_usdt * usdc_offset,
            },
            index=idx,
        )
        matrix.index.name = "timestamp_utc"
        return matrix

    @staticmethod
    def _max_abs_diff(left: pd.Series, right: pd.Series) -> float:
        a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if not bool(mask.any()):
            return 0.0
        return float(np.max(np.abs(a[mask] - b[mask])))

    def test_m_t_source_selects_expected_series(self) -> None:
        matrix = self._build_matrix()
        config = load_config("configs/config.yaml")
        config.setdefault("onchain", {})["enabled"] = False
        config.setdefault("execution_unifier", {})["enabled"] = False
        config.setdefault("hawkes", {})["enabled"] = False

        # p_smooth (legacy default behavior)
        config.setdefault("strategy", {})["m_t_source"] = "p_smooth"
        smooth = run_pipeline(config, matrix)["signal_frame"]
        self.assertLess(
            self._max_abs_diff(smooth["m_t"], smooth["p_smooth"]),
            1e-12,
        )

        # raw debiased premium
        config["strategy"]["m_t_source"] = "p"
        raw = run_pipeline(config, matrix)["signal_frame"]
        self.assertLess(
            self._max_abs_diff(raw["m_t"], raw["p"]),
            1e-12,
        )

        # residual signal p - p_smooth
        config["strategy"]["m_t_source"] = "residual"
        residual = run_pipeline(config, matrix)["signal_frame"]
        expected_residual = residual["p"] - residual["p_smooth"]
        self.assertLess(
            self._max_abs_diff(residual["m_t"], expected_residual),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
