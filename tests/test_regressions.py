from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from src.backtest import periods_per_year_from_freq, run_backtest
from src.bybit_data import bybit_interval_to_ms
from src.data_ingest import _remove_glitches, read_market_file, sanitize_single_bar_spikes
from src.execution_data import _extract_root_quote_from_symbol
from src.execution_quality import (
    _load_okx_tick_file,
    build_cross_quote_comparison,
    build_l2_coverage,
    build_slippage_vs_size,
    build_snapshot_slippage,
    build_snapshot_trade_frame,
    build_venue_summary,
)
from src.hawkes import HawkesConfig, evaluate_hawkes_quality
from src.onchain import OnchainConfig, build_onchain_validation_frame, empty_onchain_frame
from src.pipeline import _prepare_frame_for_parquet, export_outputs, load_config, load_price_matrix, run_pipeline
from src.premium import PremiumConfig, build_premium_frame, compute_depeg_flag
from src.regimes import build_regime_score
from src.robust_filter import robust_filter
from src.strategy import StrategyConfig, build_decisions
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
    def test_pw_rolling_proxy_is_causal(self) -> None:
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=220, freq="1min", tz="UTC")
        n = len(idx)

        stable = np.cumsum(rng.normal(0.0, 2e-5, n))
        btc_base = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0, 7e-4, n)))
        eth_base = 2_000.0 * np.exp(np.cumsum(rng.normal(0.0, 9e-4, n)))
        sol_base = 80.0 * np.exp(np.cumsum(rng.normal(0.0, 12e-4, n)))

        btc_target_spread = stable + rng.normal(0.0, 8e-5, n)
        eth_spread = stable + rng.normal(0.0, 6e-5, n)
        sol_spread = stable + rng.normal(0.0, 7e-5, n)

        matrix = pd.DataFrame(
            {
                "BTCUSDT-PERP": btc_base,
                "BTCUSDC-PERP": btc_base * np.exp(btc_target_spread),
                "ETHUSDT-PERP": eth_base,
                "ETHUSDC-PERP": eth_base * np.exp(eth_spread),
                "SOLUSDT-PERP": sol_base,
                "SOLUSDC-PERP": sol_base * np.exp(sol_spread),
            },
            index=idx,
        )
        cfg = PremiumConfig(
            proxy_method="pw_rolling",
            pw_window="30min",
            pw_min_period_fraction=0.5,
            pw_rho_clip=0.95,
        )
        proxy_pairs = [("ETHUSDC-PERP", "ETHUSDT-PERP"), ("SOLUSDC-PERP", "SOLUSDT-PERP")]

        full, _ = build_premium_frame(
            matrix,
            cfg,
            proxy_pairs=proxy_pairs,
            freq="1min",
        )
        prefix, _ = build_premium_frame(
            matrix.iloc[:140],
            cfg,
            proxy_pairs=proxy_pairs,
            freq="1min",
        )

        np.testing.assert_allclose(
            full["stablecoin_proxy"].iloc[:140].to_numpy(dtype=float),
            prefix["stablecoin_proxy"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            full["p"].iloc[:140].to_numpy(dtype=float),
            prefix["p"].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_invalid_proxy_method_raises(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": [100.0, 100.2, 100.4, 100.3],
                "BTCUSDT-PERP": [100.1, 100.1, 100.2, 100.2],
                "ETHUSDC-PERP": [10.0, 10.1, 10.2, 10.3],
                "ETHUSDT-PERP": [10.0, 10.0, 10.1, 10.1],
            },
            index=idx,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported proxy_method"):
            build_premium_frame(
                matrix,
                PremiumConfig(proxy_method="unsupported"),
                proxy_pairs=[("ETHUSDC-PERP", "ETHUSDT-PERP")],
                freq="1min",
            )

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

    def test_robust_filter_dynamic_floor_is_causal(self) -> None:
        idx = pd.date_range("2024-01-01", periods=240, freq="1min", tz="UTC")
        calm = np.sin(np.linspace(0.0, 8.0, 120)) * 1e-4
        stressed = np.sin(np.linspace(0.0, 40.0, 120)) * 0.5
        premium = pd.Series(np.concatenate([calm, stressed]), index=idx, name="premium")

        full = robust_filter(
            premium,
            window_obs=20,
            z_threshold=3.0,
            sigma_floor=1e-6,
            min_period_fraction=0.25,
        )
        prefix = robust_filter(
            premium.iloc[:120],
            window_obs=20,
            z_threshold=3.0,
            sigma_floor=1e-6,
            min_period_fraction=0.25,
        )

        np.testing.assert_allclose(
            full["sigma_hat"].iloc[:120].to_numpy(dtype=float),
            prefix["sigma_hat"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            full["z_t"].iloc[:120].to_numpy(dtype=float),
            prefix["z_t"].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_single_bar_spike_sanitizer_is_causal(self) -> None:
        idx = pd.date_range("2024-08-05 13:58:00", periods=6, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": [51980.0, 51938.0, 53418.0, 53180.0, 52940.0, 52930.0],
                "BTCUSDT-PERP": [51968.0, 51928.0, 51928.0, 53170.0, 52936.0, 52920.0],
                "ETHUSDC-PERP": [2500.0, 2501.0, 2502.0, 2503.0, 2504.0, 2505.0],
                "ETHUSDT-PERP": [2500.2, 2501.2, 2502.2, 2503.2, 2504.2, 2505.2],
            },
            index=idx,
        )
        full, _ = sanitize_single_bar_spikes(
            matrix,
            jump_threshold_log=0.015,
            reversion_tolerance_log=0.003,
            counterpart_max_move_log=0.002,
        )
        prefix, _ = sanitize_single_bar_spikes(
            matrix.iloc[:5],
            jump_threshold_log=0.015,
            reversion_tolerance_log=0.003,
            counterpart_max_move_log=0.002,
        )

        pd.testing.assert_frame_equal(full.iloc[:5], prefix)


class ProxySmoothnessTests(unittest.TestCase):
    def test_proxy_diff_std_is_capped_by_target_pair_diff_std(self) -> None:
        rng = np.random.default_rng(1234)
        idx = pd.date_range("2024-01-01", periods=360, freq="1min", tz="UTC")
        n = len(idx)

        stable = np.cumsum(rng.normal(0.0, 1.5e-5, n))
        btc_base = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0, 6e-4, n)))
        eth_base = 2_000.0 * np.exp(np.cumsum(rng.normal(0.0, 8e-4, n)))
        sol_base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 11e-4, n)))
        bnb_base = 350.0 * np.exp(np.cumsum(rng.normal(0.0, 9e-4, n)))

        btc_spread = stable + rng.normal(0.0, 7e-5, n)
        eth_spread = stable + rng.normal(0.0, 3.5e-4, n)
        sol_spread = stable + rng.normal(0.0, 4.0e-4, n)
        bnb_spread = stable + rng.normal(0.0, 3.8e-4, n)

        matrix = pd.DataFrame(
            {
                "BTCUSDT-PERP": btc_base,
                "BTCUSDC-PERP": btc_base * np.exp(btc_spread),
                "ETHUSDT-PERP": eth_base,
                "ETHUSDC-PERP": eth_base * np.exp(eth_spread),
                "SOLUSDT-PERP": sol_base,
                "SOLUSDC-PERP": sol_base * np.exp(sol_spread),
                "BNBUSDT-PERP": bnb_base,
                "BNBUSDC-PERP": bnb_base * np.exp(bnb_spread),
            },
            index=idx,
        )

        frame, _ = build_premium_frame(
            matrix,
            PremiumConfig(proxy_method="median"),
            proxy_pairs=[
                ("ETHUSDC-PERP", "ETHUSDT-PERP"),
                ("SOLUSDC-PERP", "SOLUSDT-PERP"),
                ("BNBUSDC-PERP", "BNBUSDT-PERP"),
            ],
            freq="1min",
        )

        proxy_diff_std = float(frame["stablecoin_proxy"].diff().std(ddof=0))
        target_diff_std = float(frame["p_naive"].diff().std(ddof=0))
        self.assertTrue(np.isfinite(proxy_diff_std))
        self.assertTrue(np.isfinite(target_diff_std))
        self.assertLessEqual(proxy_diff_std, target_diff_std + 1e-12)


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

    def test_sharpe_reports_full_series_and_active_variants(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.2, -0.1, 0.25, -0.15, 0.3], index=idx, name="premium")
        decision = pd.Series(["Widen", "Trade", "Widen", "Trade", "Widen", "Widen"], index=idx, name="decision")
        m_t = pd.Series([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], index=idx, name="m_t")

        log, metrics = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="one_bar",
        )

        net = log["net_pnl"].astype(float)
        in_market = log["position"].shift(1).abs() > 0
        active = net.loc[in_market & net.notna()]

        full_std = float(net.std(ddof=0))
        active_std = float(active.std(ddof=0))
        expected_full = float(net.mean() / full_std) if full_std > 0 else 0.0
        expected_active = float(active.mean() / active_std) if active_std > 0 else 0.0
        annualization = float(np.sqrt(periods_per_year_from_freq("1min")))

        self.assertAlmostEqual(float(metrics["sharpe"]), expected_full, places=12)
        self.assertAlmostEqual(float(metrics["sharpe_active"]), expected_active, places=12)
        self.assertAlmostEqual(float(metrics["sharpe_full_annualized"]), expected_full * annualization, places=9)
        self.assertAlmostEqual(float(metrics["sharpe_active_annualized"]), expected_active * annualization, places=9)
        self.assertEqual(int(metrics["n_bars"]), len(log))
        self.assertEqual(int(metrics["n_active_bars"]), int((in_market & net.notna()).sum()))


class StatefulExitTests(unittest.TestCase):
    def test_max_holding_bars_enforces_exit_without_same_bar_reentry(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], index=idx, name="premium")
        decision = pd.Series(["Trade"] * len(idx), index=idx, name="decision")
        m_t = pd.Series([1.0] * len(idx), index=idx, name="m_t")

        log, _ = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="stateful",
            min_holding_bars=1,
            max_holding_bars=2,
        )

        self.assertEqual(float(log.loc[idx[2], "position"]), 0.0)
        self.assertEqual(float(log.loc[idx[5], "position"]), 0.0)
        self.assertEqual(int((log["position_event"] == "Exit").sum()), 2)

    def test_stateful_does_not_exit_on_widen_by_default(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx, name="premium")
        decision = pd.Series(["Trade", "Widen", "Widen", "Widen", "Widen"], index=idx, name="decision")
        m_t = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx, name="m_t")

        log, _ = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="stateful",
            exit_on_widen=False,
            exit_on_mean_reversion=False,
            min_holding_bars=3,
        )

        self.assertEqual(int((log["position_event"] == "Exit").sum()), 0)
        self.assertTrue(bool((log["position_sign"] != 0.0).all()))
        self.assertTrue(bool((log["position"] != 0.0).all()))

    def test_stateful_exit_on_widen_remains_available_for_legacy_mode(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx, name="premium")
        decision = pd.Series(["Trade", "Widen", "Widen", "Widen", "Widen"], index=idx, name="decision")
        m_t = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx, name="m_t")

        log, _ = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="stateful",
            exit_on_widen=True,
            exit_on_mean_reversion=False,
            min_holding_bars=3,
        )

        self.assertEqual(int((log["position_event"] == "Exit").sum()), 1)
        self.assertEqual(str(log.loc[idx[3], "position_event"]), "Exit")
        self.assertEqual(float(log.loc[idx[3], "position"]), 0.0)

    def test_stateful_position_size_persists_across_widen_and_updates_on_positive_signal(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx, name="premium")
        decision = pd.Series(["Trade", "Widen", "Widen", "Widen", "Widen"], index=idx, name="decision")
        m_t = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx, name="m_t")
        size_signal = pd.Series([0.4, 0.0, 0.8, np.nan, 0.0], index=idx, name="position_size")

        log, _ = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_size=size_signal,
            position_mode="stateful",
            exit_on_widen=False,
            exit_on_mean_reversion=False,
            min_holding_bars=3,
        )

        expected_size = pd.Series([0.4, 0.4, 0.8, 0.8, 0.8], index=idx)
        np.testing.assert_allclose(
            log["position_size"].to_numpy(dtype=float),
            expected_size.to_numpy(dtype=float),
        )
        invalid = (log["position_sign"] != 0.0) & (log["position_size"] == 0.0)
        self.assertEqual(int(invalid.sum()), 0)

    def test_stateful_riskoff_and_mean_reversion_exits_are_unchanged(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 0.1, 0.0, -0.1], index=idx, name="premium")

        riskoff_decision = pd.Series(["Trade", "Risk-off", "Widen", "Widen"], index=idx, name="decision")
        riskoff_m_t = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx, name="m_t")
        riskoff_log, _ = run_backtest(
            premium,
            riskoff_decision,
            riskoff_m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="stateful",
            exit_on_widen=False,
            exit_on_mean_reversion=False,
            min_holding_bars=1,
        )
        self.assertEqual(str(riskoff_log.loc[idx[1], "position_event"]), "Exit")
        self.assertEqual(float(riskoff_log.loc[idx[1], "position"]), 0.0)

        meanrev_decision = pd.Series(["Trade", "Widen", "Widen", "Widen"], index=idx, name="decision")
        meanrev_m_t = pd.Series([1.0, 1.0, -0.1, -0.1], index=idx, name="m_t")
        meanrev_log, _ = run_backtest(
            premium,
            meanrev_decision,
            meanrev_m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="stateful",
            exit_on_widen=False,
            exit_on_mean_reversion=True,
            min_holding_bars=1,
        )
        self.assertEqual(int((meanrev_log["position_event"] == "Exit").sum()), 1)
        self.assertEqual(str(meanrev_log.loc[idx[2], "position_event"]), "Exit")
        self.assertEqual(float(meanrev_log.loc[idx[2], "position"]), 0.0)


class HawkesAdaptiveThresholdTests(unittest.TestCase):
    def test_adaptive_hawkes_thresholds_generate_causal_signals(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        n_t = pd.Series([0.01, 0.03, 0.02, 0.05, 0.08, 0.06, 0.10, 0.12, 0.15, 0.20], index=idx, name="n_t")
        m_t = pd.Series([0.0] * len(idx), index=idx, name="m_t")
        T_t = pd.Series([1.0] * len(idx), index=idx, name="T_t")
        chi_t = pd.Series([0.0] * len(idx), index=idx, name="chi_t")
        sigma_hat = pd.Series([1.0] * len(idx), index=idx, name="sigma_hat")
        regime = pd.Series(["transient"] * len(idx), index=idx, name="regime")
        depeg_flag = pd.Series([False] * len(idx), index=idx, name="depeg_flag")
        cfg = StrategyConfig(
            entry_k=10.0,
            hawkes_threshold_mode="expanding",
            hawkes_threshold_min_periods=3,
            hawkes_widen_quantile=0.6,
            hawkes_risk_off_quantile=0.9,
        )

        out = build_decisions(
            m_t=m_t,
            T_t=T_t,
            chi_t=chi_t,
            sigma_hat=sigma_hat,
            regime=regime,
            depeg_flag=depeg_flag,
            n_t=n_t,
            cfg=cfg,
        )

        self.assertIn("hawkes_widen_threshold", out.columns)
        self.assertIn("hawkes_riskoff_threshold", out.columns)
        self.assertIn("hawkes_widen_signal", out.columns)
        self.assertIn("hawkes_riskoff_signal", out.columns)
        self.assertTrue(out["hawkes_widen_signal"].iloc[5:].any())
        self.assertTrue(out["hawkes_riskoff_signal"].iloc[7:].any())


class HawkesQualityTests(unittest.TestCase):
    def test_low_variance_single_refit_hawkes_fails_quality_gate(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        hawkes_frame = pd.DataFrame(
            {
                "n_t": pd.Series(0.10, index=idx),
                "hawkes_fit_ok": pd.Series(True, index=idx),
                "hawkes_refit_count": pd.Series(1.0, index=idx),
            },
            index=idx,
        )
        passed, reason, metrics = evaluate_hawkes_quality(hawkes_frame, HawkesConfig())
        self.assertFalse(passed)
        self.assertIn("insufficient_refits", reason)
        self.assertIn("low_n_unique", reason)
        self.assertIn("low_n_variance", reason)
        self.assertEqual(float(metrics["hawkes_refit_count"]), 1.0)

    def test_pipeline_disables_hawkes_decisions_when_quality_fails(self) -> None:
        idx = pd.date_range("2024-01-01", periods=600, freq="1min", tz="UTC")
        base = pd.Series(np.linspace(1.0, 1.02, len(idx)), index=idx)
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": 50000.0 * base,
                "BTCUSDT-PERP": 50000.0 * (base + 1e-6),
                "ETHUSDC-PERP": 2500.0 * base,
                "ETHUSDT-PERP": 2500.0 * (base + 1e-6),
                "SOLUSDC-PERP": 80.0 * base,
                "SOLUSDT-PERP": 80.0 * (base + 1e-6),
                "BNBUSDC-PERP": 400.0 * base,
                "BNBUSDT-PERP": 400.0 * (base + 1e-6),
            },
            index=idx,
        )
        config = load_config("configs/config.yaml")
        run_cfg = deepcopy(config)
        run_cfg.setdefault("hawkes", {})
        run_cfg["hawkes"]["enabled"] = True
        run_cfg["onchain"] = {"enabled": False}

        signal = run_pipeline(run_cfg, matrix)["signal_frame"]
        self.assertFalse(bool(signal["hawkes_quality_pass"].iloc[0]))
        self.assertFalse(bool(signal["hawkes_riskoff_signal"].astype(bool).any()))

    def test_hawkes_enabled_signal_frame_exports_to_parquet_without_duplicate_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=600, freq="1min", tz="UTC")
        base = pd.Series(np.linspace(1.0, 1.02, len(idx)), index=idx)
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": 50000.0 * base,
                "BTCUSDT-PERP": 50000.0 * (base + 1e-6),
                "ETHUSDC-PERP": 2500.0 * base,
                "ETHUSDT-PERP": 2500.0 * (base + 1e-6),
                "SOLUSDC-PERP": 80.0 * base,
                "SOLUSDT-PERP": 80.0 * (base + 1e-6),
                "BNBUSDC-PERP": 400.0 * base,
                "BNBUSDT-PERP": 400.0 * (base + 1e-6),
            },
            index=idx,
        )
        config = load_config("configs/config.yaml")
        run_cfg = deepcopy(config)
        run_cfg.setdefault("hawkes", {})
        run_cfg["hawkes"]["enabled"] = True
        run_cfg["onchain"] = {"enabled": False}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tables_dir = Path(tmp_dir) / "tables"
            figures_dir = Path(tmp_dir) / "figures"
            run_cfg["outputs"] = {"tables_dir": str(tables_dir), "figures_dir": str(figures_dir)}

            results = run_pipeline(run_cfg, matrix)
            signal = results["signal_frame"]
            self.assertFalse(bool(signal.columns.duplicated().any()))

            def _fake_to_parquet(frame: pd.DataFrame, path: Path | str, *args: object, **kwargs: object) -> None:
                duplicate_cols = frame.columns[frame.columns.duplicated()].tolist()
                if duplicate_cols:
                    raise ValueError(f"Duplicate column names found: {duplicate_cols}")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"PAR1")

            with patch.object(pd.DataFrame, "to_parquet", autospec=True, side_effect=_fake_to_parquet):
                exported = export_outputs(results, run_cfg)
            signal_path = Path(exported["signal_frame"])
            self.assertEqual(signal_path.suffix, ".parquet")
            self.assertTrue(signal_path.exists())
            self.assertFalse((tables_dir / "signal_frame.csv").exists())

    def test_hawkes_enabled_bybit_integration_preserves_parquet_exports(self) -> None:
        matrix_path = Path("data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv")
        if not matrix_path.exists():
            self.skipTest(f"Missing bundled bybit matrix: {matrix_path}")

        config = load_config("configs/config.yaml")
        run_cfg = deepcopy(config)
        run_cfg.setdefault("hawkes", {})
        run_cfg["hawkes"]["enabled"] = True
        run_cfg["onchain"] = {"enabled": False}
        run_cfg.setdefault("data", {})
        run_cfg["data"]["resample_rule"] = str(run_cfg["data"].get("resample_rule", "1min"))
        matrix = load_price_matrix(
            matrix_path,
            expected_freq=run_cfg["data"]["resample_rule"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tables_dir = Path(tmp_dir) / "tables"
            figures_dir = Path(tmp_dir) / "figures"
            run_cfg["outputs"] = {"tables_dir": str(tables_dir), "figures_dir": str(figures_dir)}

            results = run_pipeline(run_cfg, matrix)

            def _fake_to_parquet(frame: pd.DataFrame, path: Path | str, *args: object, **kwargs: object) -> None:
                duplicate_cols = frame.columns[frame.columns.duplicated()].tolist()
                if duplicate_cols:
                    raise ValueError(f"Duplicate column names found: {duplicate_cols}")
                index_names = list(frame.index.names) if isinstance(frame.index, pd.MultiIndex) else [frame.index.name]
                overlap = [name for name in index_names if name is not None and name in frame.columns]
                if overlap:
                    raise ValueError(f"Index/column name collision found: {overlap}")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"PAR1")

            with patch.object(pd.DataFrame, "to_parquet", autospec=True, side_effect=_fake_to_parquet):
                exported = export_outputs(results, run_cfg)

            signal_path = Path(exported["signal_frame"])
            proxy_path = Path(exported["proxy_components"])
            self.assertEqual(signal_path.suffix, ".parquet")
            self.assertEqual(proxy_path.suffix, ".parquet")
            self.assertTrue(signal_path.exists())
            self.assertTrue(proxy_path.exists())
            self.assertFalse((tables_dir / "signal_frame.csv").exists())
            self.assertFalse((tables_dir / "stablecoin_proxy_components.csv").exists())


class StrategyFallbackThresholdTests(unittest.TestCase):
    def test_missing_sigma_hat_uses_signal_scale_fallback(self) -> None:
        idx = pd.date_range("2024-01-01", periods=130, freq="1min", tz="UTC")
        m_t = pd.Series([0.01] * 129 + [0.08], index=idx, name="m_t")
        T_t = pd.Series(1.0, index=idx, name="T_t")
        chi_t = pd.Series(0.0, index=idx, name="chi_t")
        regime = pd.Series("transient", index=idx, name="regime")
        depeg_flag = pd.Series(False, index=idx, name="depeg_flag")
        cfg = StrategyConfig(
            entry_k=2.0,
            threshold_min_periods=10_000,
        )

        out = build_decisions(
            m_t=m_t,
            T_t=T_t,
            chi_t=chi_t,
            sigma_hat=None,
            regime=regime,
            depeg_flag=depeg_flag,
            n_t=None,
            cfg=cfg,
        )

        self.assertTrue(float(out["entry_threshold"].iloc[-1]) < 0.05)
        self.assertEqual(str(out["decision"].iloc[-1]), "Trade")


class StressSourceDecisionTests(unittest.TestCase):
    def test_stress_sources_drive_different_actions(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        m_t = pd.Series([0.05, 0.05, 0.05], index=idx, name="m_t")
        T_t = pd.Series(1.0, index=idx, name="T_t")
        chi_t = pd.Series(0.0, index=idx, name="chi_t")
        sigma_hat = pd.Series(0.01, index=idx, name="sigma_hat")
        regime = pd.Series(["stress", "stress", "stress"], index=idx, name="regime")
        depeg_flag = pd.Series([False, False, False], index=idx, name="depeg_flag")
        stablecoin_proxy = pd.Series([0.003, -0.003, 0.0], index=idx, name="stablecoin_proxy")
        cfg = StrategyConfig(entry_k=1.0, threshold_min_periods=10_000)

        out = build_decisions(
            m_t=m_t,
            T_t=T_t,
            chi_t=chi_t,
            sigma_hat=sigma_hat,
            regime=regime,
            depeg_flag=depeg_flag,
            stablecoin_proxy=stablecoin_proxy,
            cfg=cfg,
        )

        self.assertEqual(str(out.loc[idx[0], "stress_source"]), "usdc_depeg_stress")
        self.assertEqual(str(out.loc[idx[1], "stress_source"]), "usdt_backing_concern")
        self.assertEqual(str(out.loc[idx[2], "stress_source"]), "technical_flow_imbalance")
        self.assertEqual(str(out.loc[idx[0], "decision"]), "Widen")
        self.assertEqual(str(out.loc[idx[1], "decision"]), "Widen")
        self.assertEqual(str(out.loc[idx[2], "decision"]), "Widen")

    def test_depeg_flag_forces_riskoff_for_all_stress_sources(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        m_t = pd.Series([0.05, 0.05, 0.05], index=idx, name="m_t")
        T_t = pd.Series(1.0, index=idx, name="T_t")
        chi_t = pd.Series(0.0, index=idx, name="chi_t")
        sigma_hat = pd.Series(0.01, index=idx, name="sigma_hat")
        regime = pd.Series(["transient", "transient", "transient"], index=idx, name="regime")
        depeg_flag = pd.Series([True, True, True], index=idx, name="depeg_flag")
        stablecoin_proxy = pd.Series([0.003, -0.003, 0.0], index=idx, name="stablecoin_proxy")
        cfg = StrategyConfig(entry_k=1.0, threshold_min_periods=10_000)

        out = build_decisions(
            m_t=m_t,
            T_t=T_t,
            chi_t=chi_t,
            sigma_hat=sigma_hat,
            regime=regime,
            depeg_flag=depeg_flag,
            stablecoin_proxy=stablecoin_proxy,
            cfg=cfg,
        )

        self.assertTrue(bool((out["decision"] == "Risk-off").all()))
        self.assertTrue(bool((out["riskoff_flag"]).all()))
        self.assertTrue(bool((out["riskoff_reason"] == "depeg_flag").all()))


class ConfidenceSizingTests(unittest.TestCase):
    def test_confidence_sizing_is_bounded_and_event_sensitive(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        m_t = pd.Series([0.0, 0.02, 0.04, 0.08, 0.12], index=idx, name="m_t")
        T_t = pd.Series(1.0, index=idx, name="T_t")
        chi_t = pd.Series(0.0, index=idx, name="chi_t")
        sigma_hat = pd.Series(0.01, index=idx, name="sigma_hat")
        regime = pd.Series("transient", index=idx, name="regime")
        depeg_flag = pd.Series(False, index=idx, name="depeg_flag")
        event = pd.Series([False, False, True, False, False], index=idx, name="event")
        cfg = StrategyConfig(entry_k=2.0, threshold_min_periods=10_000)

        out = build_decisions(
            m_t=m_t,
            T_t=T_t,
            chi_t=chi_t,
            sigma_hat=sigma_hat,
            regime=regime,
            depeg_flag=depeg_flag,
            event=event,
            cfg=cfg,
        )

        size = pd.to_numeric(out["position_size"], errors="coerce")
        self.assertTrue(((size >= 0.0) & (size <= 1.0)).all())
        trade_mask = out["decision"] == "Trade"
        self.assertTrue(bool(trade_mask.any()))
        self.assertTrue(bool((size.loc[trade_mask] > 0.0).all()))
        self.assertLess(float(size.loc[idx[2]]), float(size.loc[idx[3]]))


class PositionSizingBacktestTests(unittest.TestCase):
    def test_position_size_scales_pnl_and_turnover(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        premium = pd.Series([0.0, 1.0, 0.0], index=idx, name="premium")
        decision = pd.Series(["Widen", "Trade", "Widen"], index=idx, name="decision")
        m_t = pd.Series([0.0, 1.0, 0.0], index=idx, name="m_t")
        half_size = pd.Series([0.0, 0.5, 0.0], index=idx, name="position_size")

        full_log, full_metrics = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_mode="one_bar",
        )
        half_log, half_metrics = run_backtest(
            premium,
            decision,
            m_t,
            freq="1min",
            cost_bps=0.0,
            position_size=half_size,
            position_mode="one_bar",
        )

        self.assertAlmostEqual(float(half_log["net_pnl"].sum()), 0.5 * float(full_log["net_pnl"].sum()), places=12)
        self.assertAlmostEqual(float(half_metrics["turnover"]), 0.5 * float(full_metrics["turnover"]), places=12)
        self.assertAlmostEqual(float(half_metrics["avg_active_position_size"]), 0.5, places=12)


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

    def test_loader_rejects_expected_resample_rule_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "matrix.csv"
            raw = pd.DataFrame(
                {
                    "timestamp_utc": [
                        "2024-01-01T00:00:00Z",
                        "2024-01-01T00:01:00Z",
                        "2024-01-01T00:02:00Z",
                    ],
                    "BTCUSDT-PERP": [100.0, 101.0, 102.0],
                }
            )
            raw.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "resample_rule"):
                load_price_matrix(path, expected_freq="5min")


class FrequencyValidationTests(unittest.TestCase):
    def test_run_pipeline_rejects_resample_rule_mismatch(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
        base = pd.Series(np.linspace(1.0, 1.01, len(idx)), index=idx)
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": 50_000.0 * base,
                "BTCUSDT-PERP": 50_000.0 * (base + 1e-6),
            },
            index=idx,
        )
        config = {
            "data": {"resample_rule": "5min"},
            "onchain": {"enabled": False},
            "premium": {"fail_on_missing_proxy": False},
        }

        with self.assertRaisesRegex(ValueError, "resample_rule"):
            run_pipeline(config, matrix)


class ExportDiagnosticsTests(unittest.TestCase):
    def test_safety_diagnostics_includes_frequency_consistency_fields(self) -> None:
        idx = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC", name="timestamp_utc")
        base = pd.Series(np.linspace(1.0, 1.01, len(idx)), index=idx)
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": 50_000.0 * base,
                "BTCUSDT-PERP": 50_000.0 * (base + 1e-6),
                "ETHUSDC-PERP": 2_500.0 * base,
                "ETHUSDT-PERP": 2_500.0 * (base + 1e-6),
            },
            index=idx,
        )
        config = load_config("configs/config.yaml")
        run_cfg = deepcopy(config)
        run_cfg["onchain"] = {"enabled": False}
        run_cfg.setdefault("data", {})
        run_cfg["data"]["resample_rule"] = "1min"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tables_dir = Path(tmp_dir) / "tables"
            figures_dir = Path(tmp_dir) / "figures"
            run_cfg["outputs"] = {"tables_dir": str(tables_dir), "figures_dir": str(figures_dir)}

            results = run_pipeline(run_cfg, matrix)
            export_outputs(results, run_cfg)

            safety_diag = pd.read_csv(tables_dir / "safety_diagnostics.csv")
            self.assertIn("configured_resample_rule", safety_diag.columns)
            self.assertIn("expected_index_delta", safety_diag.columns)
            self.assertIn("observed_index_delta", safety_diag.columns)
            self.assertIn("index_delta_matches_config", safety_diag.columns)
            self.assertEqual(str(safety_diag.loc[0, "configured_resample_rule"]), "1min")
            self.assertEqual(str(safety_diag.loc[0, "expected_index_delta"]), "0 days 00:01:00")
            self.assertEqual(str(safety_diag.loc[0, "observed_index_delta"]), "0 days 00:01:00")
            self.assertTrue(bool(safety_diag.loc[0, "index_delta_matches_config"]))


class DataIngestTimestampTests(unittest.TestCase):
    def test_read_market_file_parses_epoch_milliseconds_as_utc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw.csv"
            raw = pd.DataFrame(
                {
                    "ts": [1704067200000, 1704067260000],
                    "symbol": ["BTCUSDT-PERP", "BTCUSDT-PERP"],
                    "price": [42000.0, 42010.0],
                }
            )
            raw.to_csv(path, index=False)

            frame = read_market_file(path)

            self.assertEqual(str(frame.loc[0, "timestamp_utc"]), "2024-01-01 00:00:00+00:00")
            self.assertEqual(str(frame.loc[1, "timestamp_utc"]), "2024-01-01 00:01:00+00:00")


class OnchainColumnsTests(unittest.TestCase):
    def test_empty_onchain_frame_includes_usd_deviation_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        frame = empty_onchain_frame(idx)
        self.assertIn("onchain_usdc_minus_1", frame.columns)
        self.assertIn("onchain_usdt_minus_1", frame.columns)
        self.assertIn("onchain_log_usdc_dev", frame.columns)
        self.assertIn("onchain_log_usdt_dev", frame.columns)
        self.assertIn("onchain_source_timestamp_utc", frame.columns)
        self.assertIn("onchain_source_age_hours", frame.columns)
        self.assertIn("onchain_data_stale", frame.columns)
        self.assertIn("onchain_depeg_flag_effective", frame.columns)
        self.assertIn("onchain_guardrail_fail_closed", frame.columns)
        self.assertFalse(bool(frame["onchain_guardrail_fail_closed"].astype(bool).any()))

    def test_empty_onchain_frame_can_fail_closed(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        frame = empty_onchain_frame(idx, fail_closed=True)
        self.assertTrue(bool(frame["onchain_guardrail_fail_closed"].astype(bool).all()))
        self.assertTrue(bool(frame["onchain_depeg_flag_effective"].astype(bool).all()))


class BybitIntervalTests(unittest.TestCase):
    def test_daily_and_weekly_intervals_map_to_ms(self) -> None:
        self.assertEqual(bybit_interval_to_ms("D"), 86_400_000)
        self.assertEqual(bybit_interval_to_ms("W"), 604_800_000)


class OnchainCadenceTests(unittest.TestCase):
    def test_onchain_depeg_persistence_uses_source_daily_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "onchain.csv"
            daily = pd.DataFrame(
                {
                    "timestamp_utc": [
                        "2024-03-12T00:00:00Z",
                        "2024-03-13T00:00:00Z",
                    ],
                    "onchain_usdc_price": [1.0, 1.0],
                    "onchain_usdt_price": [1.01, 1.01],  # persistent daily excursion
                }
            )
            daily.to_csv(cache_path, index=False)

            idx = pd.date_range("2024-03-12", periods=2 * 24 * 60, freq="1min", tz="UTC")
            stablecoin_proxy = pd.Series(0.0, index=idx, name="stablecoin_proxy")
            cfg = OnchainConfig(
                cache_path=str(cache_path),
                depeg_delta_log=0.005,
                depeg_min_consecutive=5,  # 5 daily observations required
                cache_max_age_hours=24 * 365,
            )

            frame = build_onchain_validation_frame(index=idx, stablecoin_proxy=stablecoin_proxy, cfg=cfg)
            self.assertEqual(int(frame["onchain_depeg_flag"].sum()), 0)

    def test_onchain_intraday_values_are_lagged_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "onchain.csv"
            daily = pd.DataFrame(
                {
                    "timestamp_utc": [
                        "2024-03-12T00:00:00Z",
                        "2024-03-13T00:00:00Z",
                    ],
                    "onchain_usdc_price": [1.0, 0.9],
                    "onchain_usdt_price": [1.0, 1.0],
                }
            )
            daily.to_csv(cache_path, index=False)

            idx = pd.date_range("2024-03-12", periods=2 * 24 * 60, freq="1min", tz="UTC")
            stablecoin_proxy = pd.Series(0.0, index=idx, name="stablecoin_proxy")
            cfg = OnchainConfig(
                cache_path=str(cache_path),
                cache_max_age_hours=24 * 365,
                intraday_lag_days=1,
                depeg_delta_log=0.001,
                depeg_min_consecutive=1,
            )
            frame = build_onchain_validation_frame(index=idx, stablecoin_proxy=stablecoin_proxy, cfg=cfg)

            ts = pd.Timestamp("2024-03-13T12:00:00Z")
            self.assertAlmostEqual(float(frame.loc[ts, "onchain_usdc_price"]), 1.0, places=12)
            self.assertEqual(str(frame.loc[ts, "onchain_source_timestamp_utc"]), "2024-03-12 00:00:00+00:00")
            self.assertGreater(float(frame.loc[ts, "onchain_source_age_hours"]), 24.0)

    def test_onchain_stale_rows_are_marked_unavailable_and_excluded_from_effective_depeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "onchain.csv"
            daily = pd.DataFrame(
                {
                    "timestamp_utc": ["2024-03-10T00:00:00Z"],
                    "onchain_usdc_price": [1.0],
                    "onchain_usdt_price": [1.02],
                }
            )
            daily.to_csv(cache_path, index=False)

            idx = pd.date_range("2024-03-15", periods=12 * 60, freq="1min", tz="UTC")
            stablecoin_proxy = pd.Series(0.0, index=idx, name="stablecoin_proxy")
            cfg = OnchainConfig(
                cache_path=str(cache_path),
                cache_max_age_hours=24 * 365,
                intraday_lag_days=0,
                depeg_delta_log=0.001,
                depeg_min_consecutive=1,
                max_source_age_hours=12,
            )
            frame = build_onchain_validation_frame(index=idx, stablecoin_proxy=stablecoin_proxy, cfg=cfg)

            self.assertTrue(bool(frame["onchain_data_stale"].astype(bool).all()))
            self.assertFalse(bool(frame["onchain_data_available"].astype(bool).any()))
            self.assertFalse(bool(frame["onchain_depeg_flag_effective"].astype(bool).any()))


class OnchainFailSafePipelineTests(unittest.TestCase):
    def _build_reference_matrix(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=600, freq="1min", tz="UTC")
        base = pd.Series(np.linspace(1.0, 1.02, len(idx)), index=idx)
        return pd.DataFrame(
            {
                "BTCUSDC-PERP": 50000.0 * base,
                "BTCUSDT-PERP": 50000.0 * (base + 1e-6),
                "ETHUSDC-PERP": 2500.0 * base,
                "ETHUSDT-PERP": 2500.0 * (base + 1e-6),
                "SOLUSDC-PERP": 80.0 * base,
                "SOLUSDT-PERP": 80.0 * (base + 1e-6),
                "BNBUSDC-PERP": 400.0 * base,
                "BNBUSDT-PERP": 400.0 * (base + 1e-6),
            },
            index=idx,
        )

    def test_pipeline_fail_closes_when_onchain_validation_errors(self) -> None:
        matrix = self._build_reference_matrix()
        run_cfg = load_config("configs/config.yaml")
        run_cfg.setdefault("onchain", {})
        run_cfg["onchain"]["enabled"] = True
        run_cfg["onchain"]["fail_closed_on_error"] = True

        with patch("src.pipeline.build_onchain_validation_frame", side_effect=RuntimeError("provider down")):
            signal = run_pipeline(run_cfg, matrix)["signal_frame"]

        self.assertTrue(bool(signal["onchain_guardrail_fail_closed"].astype(bool).all()))
        self.assertTrue(bool(signal["onchain_depeg_flag_effective"].astype(bool).all()))
        self.assertTrue(bool(signal["depeg_flag"].astype(bool).all()))
        self.assertTrue(bool(signal["decision"].astype(str).eq("Risk-off").all()))

    def test_pipeline_can_opt_out_of_fail_closed_on_onchain_errors(self) -> None:
        matrix = self._build_reference_matrix()
        run_cfg = load_config("configs/config.yaml")
        run_cfg.setdefault("onchain", {})
        run_cfg["onchain"]["enabled"] = True
        run_cfg["onchain"]["fail_closed_on_error"] = False

        with patch("src.pipeline.build_onchain_validation_frame", side_effect=RuntimeError("provider down")):
            signal = run_pipeline(run_cfg, matrix)["signal_frame"]

        self.assertFalse(bool(signal["onchain_guardrail_fail_closed"].astype(bool).any()))
        self.assertFalse(bool(signal["onchain_depeg_flag_effective"].astype(bool).any()))


class DepegCadenceTests(unittest.TestCase):
    def test_depeg_min_consecutive_is_interpreted_in_minutes(self) -> None:
        idx = pd.date_range("2024-01-01", periods=310, freq="1s", tz="UTC")
        proxy = pd.Series(0.003, index=idx, name="stablecoin_proxy")
        proxy.iloc[300:] = 0.0

        flag = compute_depeg_flag(
            proxy,
            delta_log=0.002,
            min_consecutive=5,
            freq="1s",
        )

        self.assertFalse(bool(flag.iloc[298]))
        self.assertTrue(bool(flag.iloc[299]))
        self.assertFalse(bool(flag.iloc[300]))


class DuplicateColumnSafetyTests(unittest.TestCase):
    def test_prepare_frame_for_parquet_raises_on_duplicate_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        frame = pd.DataFrame(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            index=idx,
            columns=["dup", "dup"],
        )

        with self.assertRaisesRegex(ValueError, "Duplicate column names found"):
            _prepare_frame_for_parquet(frame, context="unit_test.parquet")


class GlitchFilterTests(unittest.TestCase):
    def test_glitch_filter_is_centered_on_robust_location(self) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="1min", tz="UTC")
        drift = np.linspace(0.0, 0.004, 39)
        log_ret = 0.20 + drift
        prices = np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
        raw = pd.DataFrame(
            {
                "timestamp_utc": idx,
                "symbol": "BTCUSDT-PERP",
                "price": prices,
            }
        )

        filtered = _remove_glitches(raw, sigma_threshold=2.0)
        # A smooth, shifted return process should not be aggressively removed.
        self.assertGreaterEqual(filtered.shape[0], 35)


class StablecoinSpikeSanitizerTests(unittest.TestCase):
    def test_single_bar_stale_spike_is_replaced_with_previous_value(self) -> None:
        idx = pd.date_range("2024-08-05 13:58:00", periods=5, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDC-PERP": [51980.0, 51938.0, 53418.0, 53180.0, 52940.0],
                "BTCUSDT-PERP": [51968.0, 51928.0, 51928.0, 53170.0, 52936.0],
                "ETHUSDC-PERP": [2500.0, 2501.0, 2502.0, 2503.0, 2504.0],
                "ETHUSDT-PERP": [2500.2, 2501.2, 2502.2, 2503.2, 2504.2],
            },
            index=idx,
        )
        cleaned, diagnostics = sanitize_single_bar_spikes(
            matrix,
            jump_threshold_log=0.015,
            reversion_tolerance_log=0.003,
            counterpart_max_move_log=0.002,
        )

        spike_ts = pd.Timestamp("2024-08-05 14:00:00Z")
        self.assertEqual(float(cleaned.loc[spike_ts, "BTCUSDC-PERP"]), float(matrix.shift(1).loc[spike_ts, "BTCUSDC-PERP"]))
        self.assertEqual(float(cleaned.loc[spike_ts, "BTCUSDT-PERP"]), float(matrix.loc[spike_ts, "BTCUSDT-PERP"]))
        self.assertFalse(diagnostics.empty)
        self.assertIn("BTCUSDC-PERP", diagnostics["symbol"].astype(str).tolist())

    def test_persistent_dislocation_is_not_reclassified_as_single_bar_spike(self) -> None:
        idx = pd.date_range("2023-03-11 00:00:00", periods=6, freq="1min", tz="UTC")
        matrix = pd.DataFrame(
            {
                "BTCUSDC-SPOT": [20000.0, 19000.0, 18950.0, 18890.0, 18920.0, 18870.0],
                "BTCUSDT-SPOT": [20000.0, 19990.0, 19995.0, 20010.0, 20005.0, 20000.0],
            },
            index=idx,
        )
        cleaned, diagnostics = sanitize_single_bar_spikes(
            matrix,
            jump_threshold_log=0.015,
            reversion_tolerance_log=0.003,
            counterpart_max_move_log=0.002,
        )

        pd.testing.assert_frame_equal(cleaned, matrix)
        self.assertTrue(diagnostics.empty)


class EpisodeSafetyRegressionTests(unittest.TestCase):
    def test_depeg_bars_are_always_riskoff_in_march_2023_episodes(self) -> None:
        config = load_config("configs/config.yaml")
        episodes = [
            Path("data/processed/episodes/bybit_usdc_depeg_2023/prices_matrix.csv"),
            Path("data/processed/episodes/okx_usdc_depeg_2023/prices_matrix.csv"),
        ]
        missing = [str(path) for path in episodes if not path.exists()]
        if missing:
            self.skipTest(f"Missing bundled episode matrices: {missing}")

        for path in episodes:
            with self.subTest(episode=path.parent.name):
                run_cfg = deepcopy(config)
                run_cfg.setdefault("data", {})
                run_cfg["data"]["price_matrix_path"] = str(path)
                signal = run_pipeline(run_cfg, load_price_matrix(path))["signal_frame"]
                depeg = signal["depeg_flag"].astype(bool)
                riskoff = signal["decision"].astype(str).eq("Risk-off")
                self.assertGreater(int(depeg.sum()), 0)
                self.assertEqual(int((depeg & ~riskoff).sum()), 0)


class ExecutionProxyTests(unittest.TestCase):
    def test_cross_quote_uses_indeterminate_band_on_normalized_delta(self) -> None:
        slippage = pd.DataFrame(
            [
                {
                    "episode": "ep",
                    "venue": "binance",
                    "market_type": "derivatives",
                    "root": "BTC",
                    "quote": "USDC",
                    "symbol": "BTCUSDC-PERP",
                    "impact_large_mean_bps": 15.0,
                    "impact_large_mean_excess_bps": 2.0,
                    "impact_large_mean_norm": 1.02,
                    "impact_all_mean_bps": 10.0,
                },
                {
                    "episode": "ep",
                    "venue": "binance",
                    "market_type": "derivatives",
                    "root": "BTC",
                    "quote": "USDT",
                    "symbol": "BTCUSDT-PERP",
                    "impact_large_mean_bps": 14.5,
                    "impact_large_mean_excess_bps": 1.8,
                    "impact_large_mean_norm": 1.00,
                    "impact_all_mean_bps": 9.9,
                },
            ]
        )
        out = build_cross_quote_comparison(slippage, norm_delta_tolerance=0.05)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(str(out.loc[0, "preferred_quote_on_large_norm"]), "indeterminate")
        self.assertAlmostEqual(float(out.loc[0, "impact_large_delta_norm_usdc_minus_usdt"]), 0.02, places=9)

    def test_venue_summary_reports_descriptive_deltas_not_preferred_counts(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "episode": "ep",
                    "venue": "binance",
                    "market_type": "derivatives",
                    "root": "BTC",
                    "impact_large_delta_usdc_minus_usdt_bps": -0.5,
                    "impact_large_delta_excess_usdc_minus_usdt_bps": -0.2,
                    "impact_large_delta_norm_usdc_minus_usdt": -0.1,
                    "preferred_quote_on_large_norm": "USDC",
                }
            ]
        )
        resilience = pd.DataFrame(
            [
                {
                    "venue": "binance",
                    "market_type": "derivatives",
                    "quote": "USDC",
                    "recovery_median_bars": 3.0,
                    "unrecovered_ratio": 0.0,
                },
                {
                    "venue": "binance",
                    "market_type": "derivatives",
                    "quote": "USDT",
                    "recovery_median_bars": 4.0,
                    "unrecovered_ratio": 0.1,
                },
            ]
        )
        out = build_venue_summary(comparison, resilience)
        self.assertIn("mean_delta_large_norm", out.columns)
        self.assertIn("mean_delta_large_excess_bps", out.columns)
        self.assertIn("n_indeterminate_norm", out.columns)
        self.assertNotIn("usdc_preferred_count", out.columns)

    def test_snapshot_trade_frame_and_slippage_metrics(self) -> None:
        trades = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    ["2024-03-12 00:00:05Z", "2024-03-12 00:00:10Z"],
                    utc=True,
                ),
                "price": [70000.0, 70010.0],
                "volume": [7.0, 21.4267],
                "trade_notional": [500000.0, 1500000.0],
                "aggressor_side": ["buy", "sell"],
                "symbol": ["BTCUSDT-PERP", "BTCUSDT-PERP"],
                "venue": ["binance", "binance"],
                "episode": ["ep", "ep"],
                "execution_source": ["trade_ticks", "trade_ticks"],
                "root": ["BTC", "BTC"],
                "quote": ["USDT", "USDT"],
                "suffix": ["PERP", "PERP"],
                "market_type": ["derivatives", "derivatives"],
            }
        )
        snapshots = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2024-03-12 00:00:00Z"], utc=True),
                "symbol": ["BTCUSDT-PERP"],
                "venue": ["binance"],
                "ask_notional_1pct": [1_000_000.0],
                "ask_notional_2pct": [2_000_000.0],
                "ask_notional_3pct": [3_000_000.0],
                "ask_notional_4pct": [4_000_000.0],
                "ask_notional_5pct": [5_000_000.0],
                "bid_notional_1pct": [1_000_000.0],
                "bid_notional_2pct": [2_000_000.0],
                "bid_notional_3pct": [3_000_000.0],
                "bid_notional_4pct": [4_000_000.0],
                "bid_notional_5pct": [5_000_000.0],
            }
        )

        aligned = build_snapshot_trade_frame(trades, snapshots, snapshot_match_tolerance_sec=120)
        self.assertEqual(aligned.shape[0], 2)
        self.assertAlmostEqual(float(aligned.loc[0, "book_walk_bps"]), 50.0, places=9)
        self.assertAlmostEqual(float(aligned.loc[1, "book_walk_bps"]), 150.0, places=9)
        self.assertAlmostEqual(float(aligned.loc[0, "dnl"]), 0.5, places=9)
        self.assertAlmostEqual(float(aligned.loc[1, "dnl"]), 1.5, places=9)
        self.assertAlmostEqual(float(aligned.loc[1, "queue_load"]), 1.5, places=9)

        slippage = build_snapshot_slippage(aligned)
        self.assertEqual(slippage.shape[0], 1)
        self.assertAlmostEqual(float(slippage.loc[0, "impact_all_mean_bps"]), 100.0, places=9)
        self.assertEqual(str(slippage.loc[0, "slippage_method"]), "orderbook_snapshot_bookwalk")

    def test_slippage_vs_size_builds_quantile_curve(self) -> None:
        ts = pd.date_range("2024-03-12 00:00:00", periods=30, freq="1s", tz="UTC")
        dnl = np.linspace(0.1, 3.0, 30)
        aligned = pd.DataFrame(
            {
                "timestamp_utc": ts,
                "episode": ["ep"] * len(ts),
                "venue": ["binance"] * len(ts),
                "market_type": ["derivatives"] * len(ts),
                "root": ["BTC"] * len(ts),
                "quote": ["USDT"] * len(ts),
                "symbol": ["BTCUSDT-PERP"] * len(ts),
                "snapshot_matched": [True] * len(ts),
                "dnl": dnl,
                "book_walk_bps": 100.0 * dnl,
                "queue_load": 0.5 * dnl,
            }
        )
        curve = build_slippage_vs_size(aligned, n_bins=5)
        self.assertFalse(curve.empty)
        self.assertEqual(int(curve["n_obs"].sum()), 30)
        self.assertLessEqual(int(curve["size_bin"].nunique()), 5)

    def test_l2_coverage_reports_missing_when_no_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            coverage = build_l2_coverage(["ep_a", "ep_b"], Path(tmp_dir))
            self.assertEqual(coverage.shape[0], 2)
            self.assertFalse(bool(coverage["l2_ready"].astype(bool).any()))

    def test_l2_coverage_detects_zip_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_root = Path(tmp_dir) / "ep_a"
            episode_root.mkdir(parents=True, exist_ok=True)
            (episode_root / "BTCUSDT-bookDepth-2024-08-05.zip").write_bytes(b"zip")
            (episode_root / "BTCUSDT-trades-2024-08-05.zip").write_bytes(b"zip")
            coverage = build_l2_coverage(["ep_a"], Path(tmp_dir))
            self.assertEqual(coverage.shape[0], 1)
            self.assertTrue(bool(coverage.loc[0, "l2_orderbook_available"]))
            self.assertTrue(bool(coverage.loc[0, "tick_trades_available"]))
            self.assertTrue(bool(coverage.loc[0, "l2_ready"]))

    def test_l2_coverage_detects_tar_gz_orderbook_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_root = Path(tmp_dir) / "ep_okx"
            episode_root.mkdir(parents=True, exist_ok=True)
            (episode_root / "BTC-USDT-SWAP-L2orderbook-400lv-2026-02-01.tar.gz").write_bytes(b"tgz")
            (episode_root / "allfuture-trades-2026-02-01.zip").write_bytes(b"zip")
            coverage = build_l2_coverage(["ep_okx"], Path(tmp_dir))
            self.assertEqual(coverage.shape[0], 1)
            self.assertTrue(bool(coverage.loc[0, "l2_orderbook_available"]))
            self.assertTrue(bool(coverage.loc[0, "tick_trades_available"]))
            self.assertTrue(bool(coverage.loc[0, "l2_ready"]))

    def test_okx_tick_loader_parses_zip_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "allfuture-trades-2024-05-01.zip"
            csv_text = (
                "instrument_name/inst,trade_id/id,side/s,size/qty,price/p,created_time/ts\n"
                "BTC-USDT-SWAP,1,buy,2,45000,1714521600000\n"
                "BTC-USDT-SWAP,2,sell,1,45010,1714521601000\n"
            )
            with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("allfuture-trades-2024-05-01.csv", csv_text)

            parsed = _load_okx_tick_file(path)
            self.assertFalse(parsed.empty)
            self.assertIn("symbol", parsed.columns)
            self.assertIn("venue", parsed.columns)
            self.assertEqual(str(parsed.loc[0, "symbol"]), "BTC-USDT-SWAP")
            self.assertEqual(str(parsed.loc[0, "venue"]), "okx")
            self.assertAlmostEqual(float(parsed.loc[0, "volume"]), 3.0, places=9)

    def test_symbol_parser_handles_hyphenated_contracts(self) -> None:
        parsed = _extract_root_quote_from_symbol("BTC-USDT-SWAP")
        self.assertEqual(parsed, ("BTC", "USDT"))
        parsed_usdc = _extract_root_quote_from_symbol("ETHUSDC-240329")
        self.assertEqual(parsed_usdc, ("ETH", "USDC"))


if __name__ == "__main__":
    unittest.main()
