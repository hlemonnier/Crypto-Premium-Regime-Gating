from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from src import presentation_pack


def _write_episode_metrics(
    reports_root: Path,
    episode: str,
    *,
    naive_sharpe: float,
    gated_sharpe: float,
    naive_pnl: float,
    gated_pnl: float,
    naive_active_ratio: float = 0.10,
    gated_active_ratio: float = 0.10,
) -> None:
    tables_dir = reports_root / "episodes" / episode / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sharpe": [naive_sharpe, gated_sharpe],
            "pnl_net": [naive_pnl, gated_pnl],
            "max_drawdown": [-0.01, -0.01],
            "turnover": [1.0, 1.0],
            "flip_rate": [0.01, 0.01],
            "active_ratio": [naive_active_ratio, gated_active_ratio],
            "hit_rate": [0.50, 0.50],
            "n_bars": [1000, 1000],
            "n_active_bars": [100, 100],
            "horizon_days": [1.0, 1.0],
            "cost_bps_applied_mean": [1.0, 1.0],
        },
        index=["naive", "gated"],
    )
    frame.to_csv(tables_dir / "metrics.csv")


def _write_signal_proxy_and_coherence(reports_root: Path, episode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = reports_root / "episodes" / episode / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    signal = pd.DataFrame(
        {
            "onchain_proxy": [0.001, 0.002],
            "onchain_divergence": [0.001, 0.002],
            "onchain_depeg_flag": [0, 1],
            "depeg_flag": [0, 1],
            "onchain_usdc_minus_1": [0.0005, 0.0010],
            "onchain_usdt_minus_1": [0.0004, 0.0008],
            "decision": ["Trade", "Risk-off"],
            "stress_source": ["technical_flow_imbalance", "usdc_depeg_stress"],
            "confidence_score": [0.8, 0.0],
            "position_size": [0.7, 0.0],
        }
    )
    (tables_dir / "signal_frame.parquet").touch()

    proxy = pd.DataFrame({"ETHUSDC-PERP__ETHUSDT-PERP": [0.0]})
    (tables_dir / "stablecoin_proxy_components.parquet").touch()

    coherence = pd.DataFrame(
        [
            {
                "n_bars": 2,
                "event_bars": 1,
                "event_segment_count": 1,
                "event_segment_median_bars": 1.0,
                "regime_segment_median_bars": 2.0,
                "decision_segment_median_bars": 1.0,
                "decision_flip_rate": 0.5,
            }
        ]
    )
    coherence.to_csv(tables_dir / "coherence_diagnostics.csv", index=False)
    return signal, proxy


class PresentationPackClaimStatusTests(unittest.TestCase):
    def _run_pack(self, reports_root: Path, output_dir: Path, episodes: list[str]) -> None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "presentation_pack",
                "--reports-root",
                str(reports_root),
                "--output-dir",
                str(output_dir),
                "--episodes",
                *episodes,
            ]
            presentation_pack.main()
        finally:
            sys.argv = old_argv

    def test_claim_status_not_supported_when_robust_deltas_are_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports"
            out = Path(tmp) / "final"
            episodes = ["ep_a", "ep_b"]
            _write_episode_metrics(
                root,
                "ep_a",
                naive_sharpe=0.03,
                gated_sharpe=-0.01,
                naive_pnl=0.02,
                gated_pnl=-0.01,
                naive_active_ratio=0.20,
                gated_active_ratio=0.20,
            )
            _write_episode_metrics(
                root,
                "ep_b",
                naive_sharpe=0.02,
                gated_sharpe=-0.02,
                naive_pnl=0.01,
                gated_pnl=-0.02,
                naive_active_ratio=0.20,
                gated_active_ratio=0.20,
            )

            self._run_pack(root, out, episodes)

            claim = pd.read_csv(out / "claim_status.csv")
            self.assertEqual(claim.loc[0, "status"], "not_supported")
            self.assertEqual(claim.loc[0, "positioning"], "safety_risk_control")
            summary = (out / "executive_summary.md").read_text(encoding="utf-8")
            self.assertIn("Performance claim (`improved decision-making`): **not supported**", summary)

    def test_claim_status_supported_when_robust_deltas_are_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports"
            out = Path(tmp) / "final"
            episodes = ["ep_a", "ep_b"]
            _write_episode_metrics(
                root,
                "ep_a",
                naive_sharpe=0.01,
                gated_sharpe=0.03,
                naive_pnl=0.01,
                gated_pnl=0.02,
                naive_active_ratio=0.20,
                gated_active_ratio=0.20,
            )
            _write_episode_metrics(
                root,
                "ep_b",
                naive_sharpe=0.00,
                gated_sharpe=0.02,
                naive_pnl=0.00,
                gated_pnl=0.01,
                naive_active_ratio=0.20,
                gated_active_ratio=0.20,
            )

            self._run_pack(root, out, episodes)

            claim = pd.read_csv(out / "claim_status.csv")
            self.assertEqual(claim.loc[0, "status"], "supported")
            self.assertEqual(claim.loc[0, "positioning"], "performance_outperformance")
            summary = (out / "executive_summary.md").read_text(encoding="utf-8")
            self.assertIn("Performance claim (`improved decision-making`): **supported**", summary)

    def test_pack_reads_parquet_episode_tables_and_exports_coherence_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports"
            out = Path(tmp) / "final"
            episodes = ["ep_a"]
            _write_episode_metrics(
                root,
                "ep_a",
                naive_sharpe=0.01,
                gated_sharpe=0.02,
                naive_pnl=0.01,
                gated_pnl=0.02,
                naive_active_ratio=0.20,
                gated_active_ratio=0.20,
            )
            signal_frame, proxy_frame = _write_signal_proxy_and_coherence(root, "ep_a")

            def _fake_read_parquet(path, *args, **kwargs):
                p = Path(path)
                if p.name == "signal_frame.parquet":
                    return signal_frame.copy()
                if p.name == "stablecoin_proxy_components.parquet":
                    return proxy_frame.copy()
                raise FileNotFoundError(p)

            with patch.object(presentation_pack.pd, "read_parquet", side_effect=_fake_read_parquet):
                self._run_pack(root, out, episodes)

            summary = (out / "executive_summary.md").read_text(encoding="utf-8")
            self.assertIn("## On-Chain Validation Snapshot", summary)
            self.assertIn("## Proxy Coverage Notes", summary)
            self.assertIn("## Coherence Diagnostics", summary)
            self.assertTrue((out / "final_coherence_diagnostics.csv").exists())


if __name__ == "__main__":
    unittest.main()
