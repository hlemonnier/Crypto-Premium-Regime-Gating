from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

from src.pipeline import export_outputs, load_config, load_price_matrix, run_pipeline
from src.presentation_pack import DEFAULT_EPISODES


def _find_episode_matrix(episode_dir: Path) -> Path:
    parquet_path = episode_dir / "prices_matrix.parquet"
    csv_path = episode_dir / "prices_matrix.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No prices_matrix.(parquet|csv) found in {episode_dir}")


def _write_generation_summary(
    path: Path,
    *,
    generated: list[tuple[str, Path]],
    skipped: list[tuple[str, Path, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Generated episodes:\n")
        for episode, matrix_path in generated:
            handle.write(f"- {episode}: {matrix_path}\n")
        if skipped:
            handle.write("\nSkipped episodes:\n")
            for episode, matrix_path, reason in skipped:
                handle.write(f"- {episode}: {matrix_path}\n")
                handle.write(f"  reason: {reason}\n")


def _run_single_episode(
    *,
    episode: str,
    matrix_path: Path,
    base_config: dict[str, Any],
    reports_root: Path,
) -> None:
    run_cfg = deepcopy(base_config)
    run_cfg.setdefault("data", {})
    run_cfg["data"]["price_matrix_path"] = str(matrix_path)
    run_cfg.setdefault("outputs", {})
    run_cfg["outputs"]["tables_dir"] = str(reports_root / "episodes" / episode / "tables")
    run_cfg["outputs"]["figures_dir"] = str(reports_root / "episodes" / episode / "figures")

    data_cfg = run_cfg.get("data", {})
    matrix = load_price_matrix(
        matrix_path,
        sanitize_pair_spikes=bool(data_cfg.get("sanitize_single_bar_spikes", True)),
        single_bar_spike_jump_log=float(data_cfg.get("single_bar_spike_jump_log", 0.015)),
        single_bar_spike_reversion_log=float(data_cfg.get("single_bar_spike_reversion_log", 0.003)),
        single_bar_spike_counterpart_max_log=float(data_cfg.get("single_bar_spike_counterpart_max_log", 0.002)),
        single_bar_spike_min_cross_pairs=int(data_cfg.get("single_bar_spike_min_cross_pairs", 1)),
        expected_freq=str(data_cfg.get("resample_rule", "1min")),
    )
    results = run_pipeline(run_cfg, matrix)
    export_outputs(results, run_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate multi-episode submission outputs and final executive summary."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Pipeline config path.")
    parser.add_argument("--processed-root", default="data/processed/episodes", help="Processed episodes root.")
    parser.add_argument("--reports-root", default="reports", help="Reports root.")
    parser.add_argument("--final-output-dir", default="reports/final", help="Final output directory.")
    parser.add_argument("--l2-root", default="data/processed/orderbook", help="L2 input root for execution diagnostics.")
    parser.add_argument("--episodes", nargs="+", default=DEFAULT_EPISODES, help="Episode ids to regenerate.")
    parser.add_argument(
        "--skip-execution-quality",
        action="store_true",
        help="Skip execution_quality regeneration step.",
    )
    parser.add_argument(
        "--strict-l2-required",
        action="store_true",
        help="Pass strict L2 requirement when running execution_quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    processed_root = Path(args.processed_root)
    reports_root = Path(args.reports_root)
    final_output_dir = Path(args.final_output_dir)

    generated: list[tuple[str, Path]] = []
    skipped: list[tuple[str, Path, str]] = []

    for episode in args.episodes:
        episode_dir = processed_root / episode
        try:
            matrix_path = _find_episode_matrix(episode_dir)
            _run_single_episode(
                episode=episode,
                matrix_path=matrix_path,
                base_config=base_config,
                reports_root=reports_root,
            )
            generated.append((episode, matrix_path))
            print(f"[ok] episode={episode} matrix={matrix_path}")
        except Exception as exc:
            fallback_path = episode_dir / "prices_matrix.csv"
            skipped.append((episode, fallback_path, str(exc)))
            print(f"[skip] episode={episode} reason={exc}")

    summary_path = reports_root / "episodes" / "generation_summary.txt"
    _write_generation_summary(summary_path, generated=generated, skipped=skipped)
    print(f"- generation_summary: {summary_path}")

    if not generated:
        raise RuntimeError("No episodes were regenerated successfully.")

    selected_episodes = [episode for episode, _ in generated]

    if not args.skip_execution_quality:
        exec_cmd = [
            sys.executable,
            "-m",
            "src.execution_quality",
            "--output-dir",
            str(final_output_dir),
            "--processed-root",
            str(processed_root),
            "--l2-root",
            str(args.l2_root),
            "--episodes",
            *selected_episodes,
        ]
        if args.strict_l2_required:
            exec_cmd.append("--strict-l2-required")
        subprocess.run(exec_cmd, check=True)

    pack_cmd = [
        sys.executable,
        "-m",
        "src.presentation_pack",
        "--reports-root",
        str(reports_root),
        "--output-dir",
        str(final_output_dir),
        "--episodes",
        *selected_episodes,
    ]
    subprocess.run(pack_cmd, check=True)

    print("Submission regeneration completed.")
    print(f"- episodes_regenerated: {len(selected_episodes)}")
    print(f"- final_summary: {final_output_dir / 'executive_summary.md'}")


if __name__ == "__main__":
    main()
