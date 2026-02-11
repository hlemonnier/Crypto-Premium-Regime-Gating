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


def _apply_combo(base_config: dict, combo: dict[str, float]) -> dict:
    cfg = deepcopy(base_config)
    cfg.setdefault("strategy", {})
    cfg.setdefault("regimes", {})
    cfg["strategy"]["entry_k"] = combo["entry_k"]
    cfg["strategy"]["t_widen_quantile"] = combo["t_widen_quantile"]
    cfg["strategy"]["chi_widen_quantile"] = combo["chi_widen_quantile"]
    cfg["regimes"]["stress_quantile"] = combo["stress_quantile"]
    cfg["regimes"]["recovery_quantile"] = combo["recovery_quantile"]
    return cfg


def _safe_stat(values: list[float], fn, default: float = float("nan")) -> float:
    if not values:
        return default
    return float(fn(values))


def evaluate_dataset(
    *,
    base_config: dict,
    matrices: dict[str, pd.DataFrame],
    combo: dict[str, float],
    min_active_ratio: float,
    prefix: str,
) -> dict[str, float]:
    if not matrices:
        return {
            f"{prefix}_n_episodes": 0.0,
            f"{prefix}_score": float("nan"),
            f"{prefix}_mean_sharpe_full_raw": float("nan"),
            f"{prefix}_std_sharpe_full_raw": float("nan"),
            f"{prefix}_min_sharpe_full_raw": float("nan"),
            f"{prefix}_mean_pnl_net": float("nan"),
            f"{prefix}_mean_active_ratio": float("nan"),
            f"{prefix}_mean_turnover": float("nan"),
        }

    cfg = _apply_combo(base_config, combo)

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

    mean_sharpe_full_raw = _safe_stat(sharpes_full_raw, np.mean)
    std_sharpe_full_raw = _safe_stat(sharpes_full_raw, np.std)
    min_sharpe_full_raw = _safe_stat(sharpes_full_raw, np.min)
    mean_pnl = _safe_stat(pnls, np.mean)
    mean_active = _safe_stat(actives, np.mean)
    mean_turnover = _safe_stat(turnovers, np.mean)

    # Objective: improve full-series raw Sharpe while penalizing instability/dead strategies.
    score = mean_sharpe_full_raw - 0.25 * std_sharpe_full_raw + 25.0 * mean_pnl
    if mean_active < min_active_ratio:
        score -= 50.0

    return {
        f"{prefix}_n_episodes": float(len(matrices)),
        f"{prefix}_score": float(score),
        f"{prefix}_mean_sharpe_full_raw": float(mean_sharpe_full_raw),
        f"{prefix}_std_sharpe_full_raw": float(std_sharpe_full_raw),
        f"{prefix}_min_sharpe_full_raw": float(min_sharpe_full_raw),
        f"{prefix}_mean_pnl_net": float(mean_pnl),
        f"{prefix}_mean_active_ratio": float(mean_active),
        f"{prefix}_mean_turnover": float(mean_turnover),
    }


def evaluate_combo(
    *,
    base_config: dict,
    train_matrices: dict[str, pd.DataFrame],
    oos_matrices: dict[str, pd.DataFrame],
    combo: dict[str, float],
    min_active_ratio: float,
    oos_weight: float,
) -> dict[str, float]:
    train = evaluate_dataset(
        base_config=base_config,
        matrices=train_matrices,
        combo=combo,
        min_active_ratio=min_active_ratio,
        prefix="train",
    )
    oos = evaluate_dataset(
        base_config=base_config,
        matrices=oos_matrices,
        combo=combo,
        min_active_ratio=min_active_ratio,
        prefix="oos",
    )

    selection_score = float(train["train_score"])
    if oos_matrices and np.isfinite(oos["oos_score"]):
        w = float(np.clip(oos_weight, 0.0, 1.0))
        selection_score = (1.0 - w) * float(train["train_score"]) + w * float(oos["oos_score"])

    return {
        **combo,
        "selection_score": float(selection_score),
        **train,
        **oos,
    }


def resolve_episode_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        files.extend(matches)
    deduped = sorted(set(files))
    if not deduped:
        raise FileNotFoundError(f"No episode files matched patterns: {patterns}")
    return deduped


def _matrix_start_end(matrix: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if matrix.empty:
        ts = pd.Timestamp("1970-01-01", tz="UTC")
        return ts, ts
    idx = matrix.index
    return idx.min(), idx.max()


def _sort_matrix_items(matrices: dict[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    return sorted(matrices.items(), key=lambda item: (_matrix_start_end(item[1])[0], item[0]))


def split_train_oos_by_time(
    matrices: dict[str, pd.DataFrame],
    *,
    holdout_count: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if len(matrices) <= 1 or holdout_count <= 0:
        return dict(matrices), {}

    items = _sort_matrix_items(matrices)
    holdout = min(max(1, int(holdout_count)), len(items) - 1)
    train_items = items[:-holdout]
    oos_items = items[-holdout:]
    return dict(train_items), dict(oos_items)


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


def _print_matrix_block(title: str, matrices: dict[str, pd.DataFrame]) -> None:
    print(title)
    if not matrices:
        print("- (none)")
        return
    for path, matrix in _sort_matrix_items(matrices):
        start, end = _matrix_start_end(matrix)
        print(f"- {path}: {matrix.shape}, start={start}, end={end}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune regime gating parameters with explicit OOS evaluation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["data/processed/episodes/*2024*/prices_matrix.*"],
        help="Glob pattern(s) for episode matrices (used when train/test globs are not provided).",
    )
    parser.add_argument(
        "--train-episodes",
        nargs="+",
        default=None,
        help="Optional explicit train episode globs. If set, --test-episodes must also be set.",
    )
    parser.add_argument(
        "--test-episodes",
        nargs="+",
        default=None,
        help="Optional explicit OOS episode globs. If set, --train-episodes must also be set.",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=2,
        help="Chronological OOS holdout count when explicit train/test globs are not provided.",
    )
    parser.add_argument(
        "--min-train-episodes",
        type=int,
        default=2,
        help="Minimum number of train episodes required to accept the split.",
    )
    parser.add_argument(
        "--min-oos-episodes",
        type=int,
        default=2,
        help="Minimum number of OOS episodes required to accept the split.",
    )
    parser.add_argument(
        "--oos-weight",
        type=float,
        default=0.5,
        help="Weight of OOS score in final selection_score (0=train only, 1=OOS only).",
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


def _load_matrices(files: list[Path]) -> dict[str, pd.DataFrame]:
    return {str(path): load_price_matrix(path) for path in files}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    base_config = load_config(config_path)

    explicit_split = (args.train_episodes is not None) or (args.test_episodes is not None)
    if explicit_split and (args.train_episodes is None or args.test_episodes is None):
        raise ValueError("--train-episodes and --test-episodes must be provided together.")

    if explicit_split:
        train_files = resolve_episode_files(args.train_episodes)
        oos_files = resolve_episode_files(args.test_episodes)
        all_files = sorted(set(train_files + oos_files))
        all_matrices = _load_matrices(all_files)
        train_keys = {str(p) for p in train_files}
        oos_keys = {str(p) for p in oos_files}
    else:
        files = resolve_episode_files(args.episodes)
        all_matrices = _load_matrices(files)
        train_keys = set(all_matrices.keys())
        oos_keys: set[str] = set()

    _print_matrix_block("Loaded episodes:", all_matrices)

    compatible, skipped = filter_compatible_matrices(base_config, all_matrices)
    if skipped:
        print("Skipped incompatible episodes:")
        for path, reason in skipped.items():
            print(f"- {path}: {reason}")
    if not compatible:
        raise RuntimeError("No compatible episodes available after symbol coverage checks.")

    if explicit_split:
        train_matrices = {k: v for k, v in compatible.items() if k in train_keys}
        oos_matrices = {k: v for k, v in compatible.items() if k in oos_keys}
    else:
        train_matrices, oos_matrices = split_train_oos_by_time(
            compatible,
            holdout_count=args.holdout_count,
        )

    if not train_matrices:
        raise RuntimeError("Training split is empty after compatibility filtering.")
    if len(train_matrices) < max(1, int(args.min_train_episodes)):
        raise RuntimeError(
            "Training split is too small for robust selection: "
            f"got {len(train_matrices)} episodes, require >= {int(args.min_train_episodes)}."
        )
    if len(oos_matrices) < max(1, int(args.min_oos_episodes)):
        raise RuntimeError(
            "OOS split is too small for robust validation: "
            f"got {len(oos_matrices)} episodes, require >= {int(args.min_oos_episodes)}. "
            "Add more episodes or pass explicit --train-episodes/--test-episodes."
        )

    _print_matrix_block("Train episodes:", train_matrices)
    _print_matrix_block("OOS episodes:", oos_matrices)

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
            train_matrices=train_matrices,
            oos_matrices=oos_matrices,
            combo=combo,
            min_active_ratio=args.min_active_ratio,
            oos_weight=args.oos_weight,
        )
        rows.append(row)
        if i % 25 == 0 or i == len(combos):
            print(f"- progress: {i}/{len(combos)}")

    table = pd.DataFrame(rows)
    sort_cols = ["selection_score", "train_mean_sharpe_full_raw"]
    if oos_matrices:
        sort_cols.insert(1, "oos_mean_sharpe_full_raw")
    table = table.sort_values(sort_cols, ascending=False).reset_index(drop=True)

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
