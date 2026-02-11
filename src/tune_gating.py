from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml

from src.pipeline import load_config, load_price_matrix, run_pipeline


@dataclass(frozen=True)
class CandidateGrid:
    entry_k: list[float]
    t_widen_quantile: list[float]
    chi_widen_quantile: list[float]
    stress_quantile: list[float]
    recovery_quantile: list[float]


def parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    out = [float(v) for v in vals]
    if not out:
        raise ValueError(f"No numeric values provided: {raw}")
    return out


def build_param_grid(grid: CandidateGrid) -> list[dict[str, float]]:
    combos: list[dict[str, float]] = []
    for entry_k, t_q, chi_q, stress_q, recovery_q in product(
        grid.entry_k,
        grid.t_widen_quantile,
        grid.chi_widen_quantile,
        grid.stress_quantile,
        grid.recovery_quantile,
    ):
        if recovery_q >= stress_q:
            continue
        combos.append(
            {
                "entry_k": float(entry_k),
                "t_widen_quantile": float(t_q),
                "chi_widen_quantile": float(chi_q),
                "stress_quantile": float(stress_q),
                "recovery_quantile": float(recovery_q),
            }
        )
    return combos


def evaluate_combo(
    *,
    base_config: dict,
    matrices: dict[str, pd.DataFrame],
    combo: dict[str, float],
    min_active_ratio: float,
) -> dict[str, float]:
    cfg = deepcopy(base_config)
    cfg.setdefault("strategy", {})
    cfg.setdefault("regimes", {})

    cfg["strategy"]["entry_k"] = combo["entry_k"]
    cfg["strategy"]["t_widen_quantile"] = combo["t_widen_quantile"]
    cfg["strategy"]["chi_widen_quantile"] = combo["chi_widen_quantile"]
    cfg["regimes"]["stress_quantile"] = combo["stress_quantile"]
    cfg["regimes"]["recovery_quantile"] = combo["recovery_quantile"]

    sharpes_full_raw: list[float] = []
    pnls: list[float] = []
    actives: list[float] = []
    turnovers: list[float] = []

    for _, matrix in matrices.items():
        results = run_pipeline(cfg, matrix)
        gated = results["metrics"].loc["gated"]
        sharpes_full_raw.append(float(gated["sharpe"]))
        pnls.append(float(gated["pnl_net"]))
        actives.append(float(gated["active_ratio"]))
        turnovers.append(float(gated["turnover"]))

    mean_sharpe_full_raw = float(np.mean(sharpes_full_raw))
    std_sharpe_full_raw = float(np.std(sharpes_full_raw))
    min_sharpe_full_raw = float(np.min(sharpes_full_raw))
    mean_pnl = float(np.mean(pnls))
    mean_active = float(np.mean(actives))
    mean_turnover = float(np.mean(turnovers))

    # Objective: improve full-series raw Sharpe while penalizing instability/dead strategies.
    score = mean_sharpe_full_raw - 0.25 * std_sharpe_full_raw + 25.0 * mean_pnl
    if mean_active < min_active_ratio:
        score -= 50.0

    return {
        **combo,
        "score": score,
        "mean_sharpe_full_raw": mean_sharpe_full_raw,
        "std_sharpe_full_raw": std_sharpe_full_raw,
        "min_sharpe_full_raw": min_sharpe_full_raw,
        "mean_pnl_net": mean_pnl,
        "mean_active_ratio": mean_active,
        "mean_turnover": mean_turnover,
    }


def resolve_episode_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        files.extend(matches)
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(
            f"No episode files matched patterns: {patterns}"
        )
    return deduped


def filter_compatible_matrices(
    base_config: dict,
    matrices: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    compatible: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for path, matrix in matrices.items():
        try:
            run_pipeline(base_config, matrix)
        except Exception as exc:
            skipped[path] = str(exc).replace("\n", " ")
            continue
        compatible[path] = matrix
    return compatible, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune regime gating parameters on 2024 episodes.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["data/processed/episodes/*2024*/prices_matrix.*"],
        help="Glob pattern(s) for episode matrices.",
    )
    parser.add_argument("--entry-k", default="0.5,0.75,1.0,1.25", help="Candidate list.")
    parser.add_argument("--t-widen", default="0.95,0.97,0.99", help="Candidate list.")
    parser.add_argument("--chi-widen", default="0.95,0.97,0.99", help="Candidate list.")
    parser.add_argument("--stress", default="0.9,0.95,0.99", help="Candidate list.")
    parser.add_argument("--recovery", default="0.6,0.8", help="Candidate list.")
    parser.add_argument("--min-active-ratio", type=float, default=0.002, help="Penalty threshold.")
    parser.add_argument("--max-combos", type=int, default=0, help="Randomly sample N combos (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for combo sampling.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Default: reports/tables/gating_tuning_<timestamp>.csv",
    )
    parser.add_argument("--top", type=int, default=15, help="How many top rows to print.")
    parser.add_argument("--apply", action="store_true", help="Apply best combo to config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    base_config = load_config(config_path)

    files = resolve_episode_files(args.episodes)
    matrices: dict[str, pd.DataFrame] = {str(path): load_price_matrix(path) for path in files}
    print("Loaded episodes:")
    for path, matrix in matrices.items():
        print(f"- {path}: {matrix.shape}")

    matrices, skipped = filter_compatible_matrices(base_config, matrices)
    if skipped:
        print("Skipped incompatible episodes:")
        for path, reason in skipped.items():
            print(f"- {path}: {reason}")
    if not matrices:
        raise RuntimeError("No compatible episodes available after symbol coverage checks.")
    print("Episodes kept for tuning:")
    for path, matrix in matrices.items():
        print(f"- {path}: {matrix.shape}")

    grid = CandidateGrid(
        entry_k=parse_float_list(args.entry_k),
        t_widen_quantile=parse_float_list(args.t_widen),
        chi_widen_quantile=parse_float_list(args.chi_widen),
        stress_quantile=parse_float_list(args.stress),
        recovery_quantile=parse_float_list(args.recovery),
    )
    combos = build_param_grid(grid)
    if args.max_combos and args.max_combos > 0 and len(combos) > args.max_combos:
        rnd = random.Random(args.seed)
        combos = rnd.sample(combos, args.max_combos)
    print(f"Evaluating {len(combos)} parameter combinations...")

    rows: list[dict[str, float]] = []
    for i, combo in enumerate(combos, start=1):
        row = evaluate_combo(
            base_config=base_config,
            matrices=matrices,
            combo=combo,
            min_active_ratio=args.min_active_ratio,
        )
        rows.append(row)
        if i % 25 == 0 or i == len(combos):
            print(f"- progress: {i}/{len(combos)}")

    table = pd.DataFrame(rows).sort_values(["score", "mean_sharpe_full_raw"], ascending=False).reset_index(drop=True)
    print("Top combinations:")
    print(table.head(args.top).to_string(index=False))

    out_path = (
        Path(args.output)
        if args.output
        else Path("reports/tables")
        / f"gating_tuning_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Tuning results saved to: {out_path}")

    if args.apply:
        best = table.iloc[0].to_dict()
        updated = deepcopy(base_config)
        updated.setdefault("strategy", {})
        updated.setdefault("regimes", {})
        updated["strategy"]["entry_k"] = float(best["entry_k"])
        updated["strategy"]["t_widen_quantile"] = float(best["t_widen_quantile"])
        updated["strategy"]["chi_widen_quantile"] = float(best["chi_widen_quantile"])
        updated["regimes"]["stress_quantile"] = float(best["stress_quantile"])
        updated["regimes"]["recovery_quantile"] = float(best["recovery_quantile"])
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(updated, handle, sort_keys=False)
        print(f"Applied best combo to config: {config_path}")


if __name__ == "__main__":
    main()
