from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.onchain import empty_onchain_frame
from src.pipeline import load_price_matrix
from src.premium import PremiumConfig, build_premium_frame
from src.regimes import build_regime_score
from src.thresholds import quantile_threshold


class DataContractTests(unittest.TestCase):
    def test_auto_resolve_target_pair_when_config_symbols_missing(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDC-SPOT": [100.0, 101.0, 99.0, 100.5],
                "BTCUSDT-SPOT": [100.2, 101.1, 99.3, 100.9],
                "ETHUSDC-SPOT": [10.0, 10.2, 10.1, 10.15],
                "ETHUSDT-SPOT": [10.02, 10.22, 10.15, 10.20],
            },
            index=idx,
        )
        cfg = PremiumConfig()  # default target is PERP and should auto-resolve to SPOT here
        frame, proxy_components = build_premium_frame(
            matrix,
            cfg,
            proxy_pairs=[("ETHUSDC-SPOT", "ETHUSDT-SPOT")],
        )

        expected = (np.log(matrix["BTCUSDC-SPOT"]) - np.log(matrix["BTCUSDT-SPOT"])).rename("p_naive")
        pd.testing.assert_series_equal(frame["p_naive"], expected)
        self.assertEqual(proxy_components.shape[1], 1)

    def test_fail_closed_when_no_compatible_usdc_usdt_target_pair(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDT-PERP": [100.0, 100.1, 100.2],
                "ETHUSDT-PERP": [10.0, 10.1, 10.2],
            },
            index=idx,
        )
        with self.assertRaisesRegex(ValueError, "No compatible USDC/USDT target pair"):
            build_premium_frame(matrix, PremiumConfig())


class NoLookaheadTests(unittest.TestCase):
    def test_fixed_quantile_threshold_is_causal(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        series = pd.Series([0.0, 0.0, 0.0, 1000.0], index=idx, name="pressure")

        # Calibration uses only the first 3 points, so the future outlier must not change the threshold.
        threshold = quantile_threshold(
            series,
            quantile=0.9,
            mode="fixed",
            min_periods=3,
            shift=0,
            name="thr",
        )

        self.assertTrue(np.isnan(threshold.iloc[0]))
        self.assertTrue(np.isnan(threshold.iloc[1]))
        self.assertEqual(float(threshold.iloc[2]), 0.0)
        self.assertEqual(float(threshold.iloc[3]), 0.0)

    def test_regime_robust_zscore_is_causal(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        signal = pd.Series([0.0, 0.0, 0.0, 0.0, 1000.0], index=idx)
        score = build_regime_score(
            H_t=signal,
            T_t=signal,
            chi_t=signal,
            w_T=1.0,
            w_chi=0.0,
            w_H=0.0,
            zscore_mode="expanding",
            zscore_min_periods=3,
            zscore_window=None,
            zscore_scale_floor=1e-12,
        )
        self.assertTrue(np.isnan(score.iloc[0]))
        self.assertTrue(np.isnan(score.iloc[1]))
        self.assertTrue(np.isnan(score.iloc[2]))
        self.assertEqual(float(score.iloc[3]), 0.0)
        self.assertGreater(float(score.iloc[4]), 1e6)


class HitRateTests(unittest.TestCase):
    def test_hit_rate_counts_only_active_position_bars(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0], index=idx, name="premium")
        decision = pd.Series(["Widen", "Trade", "Widen", "Widen", "Widen"], index=idx, name="decision")
        m_t = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=idx, name="m_t")

        log, metrics = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="one_bar",
        )

        # One active bar only (the bar right after entry), and that PnL is positive.
        self.assertAlmostEqual(float(metrics["active_ratio"]), 0.2, places=9)
        self.assertAlmostEqual(float(metrics["hit_rate"]), 1.0, places=9)

        # Non-active bars should not pollute hit-rate.
        wrong_legacy = (log["net_pnl"].where(log["position"].shift(1).abs() > 0) > 0).mean()
        self.assertLess(float(wrong_legacy), float(metrics["hit_rate"]))


class PriceMatrixLoaderTests(unittest.TestCase):
    def test_loader_drops_nat_and_duplicate_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "matrix.csv"
            raw = pd.DataFrame(
                {
                    "timestamp_utc": [
                        "2024-01-01T00:00:00Z",
                        "not-a-date",
                        "2024-01-01T00:00:00Z",
                        "2024-01-01T00:01:00Z",
                    ],
                    "BTCUSDT-PERP": [100.0, 101.0, 102.0, 103.0],
                }
            )
            raw.to_csv(path, index=False)

            loaded = load_price_matrix(path)

            self.assertEqual(loaded.shape[0], 2)
            self.assertFalse(loaded.index.isna().any())
            self.assertEqual(
                float(loaded.loc[pd.Timestamp("2024-01-01T00:00:00Z"), "BTCUSDT-PERP"]),
                102.0,
            )


class OnchainColumnsTests(unittest.TestCase):
    def test_empty_onchain_frame_includes_usd_deviation_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        frame = empty_onchain_frame(idx)
        self.assertIn("onchain_usdc_minus_1", frame.columns)
        self.assertIn("onchain_usdt_minus_1", frame.columns)
        self.assertIn("onchain_log_usdc_dev", frame.columns)
        self.assertIn("onchain_log_usdt_dev", frame.columns)


if __name__ == "__main__":
    unittest.main()
