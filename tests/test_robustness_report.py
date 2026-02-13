from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from src.ablation_core import FactorialVariant, build_factorial_variants, compute_core_frames
from src.pipeline import load_config
from src.robustness_report import (
    apply_latency_one_bar,
    build_variant_payload,
    build_walkforward_splits,
    compute_strict_verdict_table,
    scenario_costs,
)


def _make_synthetic_matrix(*, start: str, periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    n = len(idx)

    stable = np.cumsum(rng.normal(0.0, 2e-5, n))
    btc_base = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0, 6e-4, n)))
    eth_base = 2_000.0 * np.exp(np.cumsum(rng.normal(0.0, 7e-4, n)))
    sol_base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 9e-4, n)))
    bnb_base = 300.0 * np.exp(np.cumsum(rng.normal(0.0, 8e-4, n)))

    btc_spread = stable + rng.normal(0.0, 7e-5, n)
    eth_spread = stable + rng.normal(0.0, 6e-5, n)
    sol_spread = stable + rng.normal(0.0, 7e-5, n)
    bnb_spread = stable + rng.normal(0.0, 6e-5, n)

    return pd.DataFrame(
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


class RobustnessReportTests(unittest.TestCase):
    def test_factorial_variants_are_exhaustive_and_unique(self) -> None:
        variants = build_factorial_variants()
        self.assertEqual(len(variants), 16)
        ids = [v.variant_id for v in variants]
        self.assertEqual(len(ids), len(set(ids)))

    def test_walkforward_split_is_chronological(self) -> None:
        m1 = _make_synthetic_matrix(start="2024-01-01", periods=120, seed=1)
        m2 = _make_synthetic_matrix(start="2024-01-03", periods=120, seed=2)
        m3 = _make_synthetic_matrix(start="2024-01-05", periods=120, seed=3)
        items = [("ep1", m1), ("ep2", m2), ("ep3", m3)]

        splits = build_walkforward_splits(items, min_train_episodes=1)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0]["train_ids"], ["ep1"])
        self.assertEqual(splits[0]["test_id"], "ep2")
        self.assertEqual(splits[1]["train_ids"], ["ep1", "ep2"])
        self.assertEqual(splits[1]["test_id"], "ep3")

    def test_statmech_off_variant_forces_gating_effective_off(self) -> None:
        cfg = load_config("configs/config.yaml")
        cfg = deepcopy(cfg)
        cfg.setdefault("onchain", {})
        cfg["onchain"]["enabled"] = False
        cfg.setdefault("execution_unifier", {})
        cfg["execution_unifier"]["enabled"] = False

        matrix = _make_synthetic_matrix(start="2024-02-01", periods=180, seed=11)
        frames = compute_core_frames(cfg, matrix, premium_leg="debiased")
        variant = FactorialVariant(premium="debiased", gating=True, statmech=False, hawkes=False)

        payload = build_variant_payload(cfg, frames, variant)
        self.assertFalse(payload["statmech_effective_on"])
        self.assertFalse(payload["gating_effective_on"])
        self.assertEqual(payload["gating_effective_reason"], "requires_statmech")

    def test_latency_stress_shifts_decision_size_and_cost(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
        decision = pd.Series(["Trade", "Widen", "Trade", "Risk-off"], index=idx, name="decision")
        size = pd.Series([0.8, 0.2, 0.7, 0.0], index=idx, name="position_size")
        cost = pd.Series([1.0, 1.2, 1.3, 1.1], index=idx, name="dynamic_cost_bps")
        m_t = pd.Series([0.1, 0.2, -0.1, -0.2], index=idx, name="m_t")

        d2, s2, c2, _ = apply_latency_one_bar(decision, size, cost, m_t)
        self.assertEqual(d2.iloc[0], "Risk-off")
        self.assertEqual(d2.iloc[1], "Trade")
        self.assertAlmostEqual(float(s2.iloc[0]), 0.0)
        self.assertAlmostEqual(float(s2.iloc[2]), 0.2)
        self.assertAlmostEqual(float(c2.iloc[2]), 1.2)

    def test_stress_cost_multipliers(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        dynamic = pd.Series([2.0, 3.0], index=idx, name="dynamic_cost_bps")
        kwargs = {
            "index": idx,
            "base_cost_bps": 1.0,
            "spread_roundtrip_bps": None,
            "fees_roundtrip_bps": None,
        }
        fees = scenario_costs(dynamic, scenario="fees_x2", **kwargs)
        spread = scenario_costs(dynamic, scenario="spread_x2", **kwargs)
        liq = scenario_costs(dynamic, scenario="liquidity_half", **kwargs)
        comb = scenario_costs(dynamic, scenario="combined_worst", **kwargs)

        np.testing.assert_allclose(fees.to_numpy(dtype=float), np.array([2.6, 3.6]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(spread.to_numpy(dtype=float), np.array([2.4, 3.4]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(liq.to_numpy(dtype=float), np.array([3.0, 5.0]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(comb.to_numpy(dtype=float), np.array([4.0, 6.0]), rtol=1e-12, atol=1e-12)

    def test_strict_verdict_rule(self) -> None:
        rows = [
            {"split_id": 1, "test_episode": "ep2", "variant_id": "v", "scenario": "base", "sharpe": 0.1, "pnl_net": 0.2},
            {"split_id": 1, "test_episode": "ep2", "variant_id": "v", "scenario": "fees_x2", "sharpe": 0.1, "pnl_net": 0.1},
            {"split_id": 1, "test_episode": "ep2", "variant_id": "v", "scenario": "spread_x2", "sharpe": 0.1, "pnl_net": 0.1},
            {"split_id": 1, "test_episode": "ep2", "variant_id": "v", "scenario": "latency_1bar", "sharpe": 0.1, "pnl_net": 0.1},
            {"split_id": 1, "test_episode": "ep2", "variant_id": "v", "scenario": "liquidity_half", "sharpe": -0.1, "pnl_net": 0.1},
            {
                "split_id": 1,
                "test_episode": "ep2",
                "variant_id": "v",
                "scenario": "combined_worst",
                "sharpe": -0.1,
                "pnl_net": -0.1,
            },
        ]
        verdict = compute_strict_verdict_table(pd.DataFrame(rows))
        self.assertEqual(verdict.shape[0], 1)
        self.assertEqual(int(verdict.iloc[0]["single_pass_count"]), 3)
        self.assertTrue(bool(verdict.iloc[0]["singles_majority_pass"]))
        self.assertTrue(bool(verdict.iloc[0]["verdict_pass"]))

    def test_cli_smoke_generates_expected_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episodes_dir = tmp_path / "episodes"
            (episodes_dir / "ep1").mkdir(parents=True, exist_ok=True)
            (episodes_dir / "ep2").mkdir(parents=True, exist_ok=True)

            m1 = _make_synthetic_matrix(start="2024-03-01", periods=180, seed=21)
            m2 = _make_synthetic_matrix(start="2024-03-03", periods=180, seed=22)
            m1.to_csv(episodes_dir / "ep1" / "prices_matrix.csv", index=True, index_label="timestamp_utc")
            m2.to_csv(episodes_dir / "ep2" / "prices_matrix.csv", index=True, index_label="timestamp_utc")

            cfg = load_config("configs/config.yaml")
            cfg = deepcopy(cfg)
            cfg.setdefault("onchain", {})
            cfg["onchain"]["enabled"] = False
            cfg.setdefault("execution_unifier", {})
            cfg["execution_unifier"]["enabled"] = False

            config_path = tmp_path / "config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)

            out_dir = tmp_path / "robustness_out"
            cmd = [
                sys.executable,
                "-m",
                "src.robustness_report",
                "--config",
                str(config_path),
                "--episodes",
                str(episodes_dir / "*" / "prices_matrix.csv"),
                "--output-dir",
                str(out_dir),
                "--entry-k",
                "1.0",
                "--t-widen",
                "0.95",
                "--chi-widen",
                "0.95",
                "--stress",
                "0.9",
                "--recovery",
                "0.6",
                "--max-combos",
                "1",
                "--min-train-episodes",
                "1",
            ]
            subprocess.run(cmd, cwd=repo_root, check=True)

            expected = [
                out_dir / "walkforward_split_metrics.csv",
                out_dir / "ablation_factorial_oos.csv",
                out_dir / "stress_matrix_oos.csv",
                out_dir / "robustness_verdict.csv",
                out_dir / "robustness_summary.md",
            ]
            for path in expected:
                self.assertTrue(path.exists(), f"missing {path}")


if __name__ == "__main__":
    unittest.main()
