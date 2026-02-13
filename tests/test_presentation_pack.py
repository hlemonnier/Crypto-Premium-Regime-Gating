from __future__ import annotations

import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
