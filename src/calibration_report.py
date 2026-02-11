from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.pipeline import load_config, load_price_matrix, run_pipeline


DEFAULT_BASELINE_PARAMS = {
    "strategy": {
        "entry_k": 2.0,
        "t_widen_quantile": 0.80,
        "chi_widen_quantile": 0.80,
    },
    "regimes": {
        "stress_quantile": 0.85,
        "recovery_quantile": 0.65,
    },
}


def _set_nested(config: dict[str, Any], updates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cfg = deepcopy(config)
    for section, values in updates.items():
        cfg.setdefault(section, {})
        for key, value in values.items():
            cfg[section][key] = value
    return cfg


def _derive_episode_name(path: Path) -> str:
    # .../episodes/<episode>/prices_matrix.csv
    if path.parent.name == "tables":
        return path.parent.parent.name
    if path.parent.name in {"processed", "data"}:
        return path.stem
    return path.parent.name


def resolve_episode_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(Path().glob(pattern)))
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final calibration report (baseline Notice params vs tuned params)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Current tuned config.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["data/processed/episodes/*2024_binance/prices_matrix.csv"],
        help="Episode matrix globs for calibration validation.",
    )
    parser.add_argument("--output-dir", default="reports/final", help="Calibration report output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuned_cfg = load_config(args.config)
    baseline_cfg = _set_nested(tuned_cfg, DEFAULT_BASELINE_PARAMS)

    files = resolve_episode_files(args.episodes)
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for matrix_path in files:
        matrix = load_price_matrix(matrix_path)
        episode = _derive_episode_name(matrix_path)

        try:
            tuned = run_pipeline(tuned_cfg, matrix)["metrics"]
            base = run_pipeline(baseline_cfg, matrix)["metrics"]
        except Exception as exc:
            reason = str(exc).replace("\n", " ")
            skipped_rows.append(
                {
                    "episode": episode,
                    "matrix_path": str(matrix_path),
                    "reason": reason,
                }
            )
            print(f"Skipping incompatible episode {episode}: {reason}")
            continue

        tuned_g = tuned.loc["gated"]
        base_g = base.loc["gated"]

        row = {
            "episode": episode,
            "matrix_path": str(matrix_path),
            "baseline_gated_sharpe_full_raw": float(base_g["sharpe"]),
            "tuned_gated_sharpe_full_raw": float(tuned_g["sharpe"]),
            "delta_gated_sharpe_full_raw": float(tuned_g["sharpe"] - base_g["sharpe"]),
            "baseline_gated_pnl_net": float(base_g["pnl_net"]),
            "tuned_gated_pnl_net": float(tuned_g["pnl_net"]),
            "delta_gated_pnl_net": float(tuned_g["pnl_net"] - base_g["pnl_net"]),
            "baseline_gated_active_ratio": float(base_g["active_ratio"]),
            "tuned_gated_active_ratio": float(tuned_g["active_ratio"]),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No compatible episodes were processed. "
            "Check symbol coverage (USDC/USDT target pair + proxy pairs) in selected matrices."
        )

    details = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    numeric = details.select_dtypes(include=["number"])
    aggregate = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "median": numeric.median(),
            "min": numeric.min(),
            "max": numeric.max(),
        }
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "calibration_details.csv"
    agg_path = out_dir / "calibration_aggregate.csv"
    md_path = out_dir / "calibration_report.md"
    skipped_path = out_dir / "calibration_skipped.csv"

    details.to_csv(details_path, index=False)
    aggregate.to_csv(agg_path, index=True)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)

    tuned_params = {
        "strategy.entry_k": tuned_cfg.get("strategy", {}).get("entry_k"),
        "strategy.t_widen_quantile": tuned_cfg.get("strategy", {}).get("t_widen_quantile"),
        "strategy.chi_widen_quantile": tuned_cfg.get("strategy", {}).get("chi_widen_quantile"),
        "regimes.stress_quantile": tuned_cfg.get("regimes", {}).get("stress_quantile"),
        "regimes.recovery_quantile": tuned_cfg.get("regimes", {}).get("recovery_quantile"),
    }

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Calibration Report\n\n")
        handle.write("Comparison: baseline Notice defaults vs tuned config on 2024 episodes.\n\n")
        handle.write("Metric naming: `*_sharpe_full_raw` denotes full-series, non-annualized Sharpe.\n\n")
        handle.write("## Tuned Parameters\n\n")
        for key, value in tuned_params.items():
            handle.write(f"- `{key}`: `{value}`\n")
        handle.write("\n## Episode Details\n\n")
        handle.write("```text\n")
        handle.write(details.to_string(index=False))
        handle.write("\n```\n\n")
        handle.write("## Aggregate Stats\n\n")
        handle.write("```text\n")
        handle.write(aggregate.to_string())
        handle.write("\n```\n")
        if skipped_rows:
            skipped_df = pd.DataFrame(skipped_rows).sort_values("episode").reset_index(drop=True)
            handle.write("\n## Skipped Episodes\n\n")
            handle.write("```text\n")
            handle.write(skipped_df.to_string(index=False))
            handle.write("\n```\n")

    print("Calibration report completed.")
    print(f"- details: {details_path}")
    print(f"- aggregate: {agg_path}")
    print(f"- markdown: {md_path}")
    if skipped_rows:
        print(f"- skipped: {skipped_path}")


if __name__ == "__main__":
    main()
