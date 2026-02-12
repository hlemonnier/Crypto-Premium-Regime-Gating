from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 140
    figsize_timeline: tuple[float, float] = (14.0, 7.0)
    figsize_panel: tuple[float, float] = (14.0, 5.0)
    figsize_phase: tuple[float, float] = (8.0, 6.0)


DECISION_COLORS = {
    "Trade": "#1f77b4",
    "Widen": "#ff7f0e",
    "Risk-off": "#d62728",
}


def _hawkes_quality_pass(frame: pd.DataFrame) -> bool:
    if "hawkes_quality_pass" not in frame.columns:
        return False
    quality = frame["hawkes_quality_pass"].fillna(False).astype(bool)
    return bool(quality.any())


def plot_figure_1_timeline(frame: pd.DataFrame, output_path: str | Path, cfg: PlotConfig) -> Path:
    required = {"p_naive", "p_smooth", "regime", "event"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Figure 1 requires columns: {sorted(required)}. Missing: {sorted(missing)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg.figsize_timeline, dpi=cfg.dpi)
    x = frame.index
    ax.plot(x, frame["p_naive"], label="p_naive", linewidth=1.0, alpha=0.65, color="#808080")
    ax.plot(x, frame["p_smooth"], label="p_smooth", linewidth=1.4, color="#003f5c")

    stress = frame["regime"].eq("stress").fillna(False).to_numpy()
    y_min = float(np.nanmin(np.concatenate([frame["p_naive"].to_numpy(), frame["p_smooth"].to_numpy()])))
    y_max = float(np.nanmax(np.concatenate([frame["p_naive"].to_numpy(), frame["p_smooth"].to_numpy()])))
    ax.fill_between(x, y_min, y_max, where=stress, color="#d62728", alpha=0.08, label="stress regime")

    event_idx = frame.index[frame["event"].fillna(False)]
    if not event_idx.empty:
        ax.scatter(
            event_idx,
            frame.loc[event_idx, "p_smooth"],
            s=16,
            color="#bc5090",
            label="events |z|>u",
            zorder=3,
        )

    ax.set_title("Figure 1 - Premium Timeline, Regimes, and Events")
    ax.set_xlabel("timestamp_utc")
    ax.set_ylabel("log premium")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_figure_2_panel(frame: pd.DataFrame, output_path: str | Path, cfg: PlotConfig) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg.figsize_panel, dpi=cfg.dpi)
    x = frame.index
    hawkes_mode = _hawkes_quality_pass(frame) and "n_t" in frame.columns and frame["n_t"].notna().any()
    if hawkes_mode:
        ax.plot(x, frame["n_t"], color="#003f5c", label="n(t)")
        widen_thr = (
            pd.to_numeric(frame.get("hawkes_widen_threshold"), errors="coerce")
            if "hawkes_widen_threshold" in frame.columns
            else None
        )
        risk_thr = (
            pd.to_numeric(frame.get("hawkes_riskoff_threshold"), errors="coerce")
            if "hawkes_riskoff_threshold" in frame.columns
            else None
        )
        if widen_thr is not None and widen_thr.notna().any():
            ax.plot(x, widen_thr, color="#ff7f0e", linestyle="--", linewidth=1.2, label="Widen threshold")
        else:
            ax.axhline(0.70, color="#ff7f0e", linestyle="--", linewidth=1.2, label="Widen threshold")
        if risk_thr is not None and risk_thr.notna().any():
            ax.plot(x, risk_thr, color="#d62728", linestyle="--", linewidth=1.2, label="Risk-off threshold")
        else:
            ax.axhline(0.85, color="#d62728", linestyle="--", linewidth=1.2, label="Risk-off threshold")
        ax.set_ylabel("branching ratio n(t)")
        ax.set_title("Figure 2 - Hawkes Branching Ratio")
    else:
        required = {"T_t", "chi_t"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Figure 2 baseline requires columns: {sorted(required)}")
        ax.plot(x, frame["T_t"], color="#003f5c", label="T_t")
        ax2 = ax.twinx()
        ax2.plot(x, frame["chi_t"], color="#bc5090", label="chi_t", alpha=0.8)
        ax2.set_ylabel("chi_t")
        ax.set_ylabel("T_t")
        ax.set_title("Figure 2 - Temperature and Susceptibility")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")

    ax.grid(alpha=0.2)
    ax.set_xlabel("timestamp_utc")
    if hawkes_mode:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_figure_3_phase_space(
    frame: pd.DataFrame,
    output_path: str | Path,
    cfg: PlotConfig,
    *,
    entry_k: float = 2.0,
) -> Path:
    required = {"T_t", "decision"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Figure 3 requires columns: {sorted(required)}. Missing: {sorted(missing)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=cfg.figsize_phase, dpi=cfg.dpi)

    hawkes_mode = _hawkes_quality_pass(frame) and "n_t" in frame.columns and frame["n_t"].notna().any()
    if hawkes_mode:
        y_col = "n_t"
        ax.set_ylabel("n_t")
        ax.set_title("Figure 3 - Phase Space (T_t, n_t)")
        widen_thr = (
            pd.to_numeric(frame.get("hawkes_widen_threshold"), errors="coerce")
            if "hawkes_widen_threshold" in frame.columns
            else None
        )
        risk_thr = (
            pd.to_numeric(frame.get("hawkes_riskoff_threshold"), errors="coerce")
            if "hawkes_riskoff_threshold" in frame.columns
            else None
        )
        if widen_thr is not None and widen_thr.notna().any():
            ax.plot(
                frame["T_t"],
                widen_thr,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Widen threshold",
            )
        else:
            ax.axhline(0.70, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.8)
        if risk_thr is not None and risk_thr.notna().any():
            ax.plot(
                frame["T_t"],
                risk_thr,
                color="#d62728",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Risk-off threshold",
            )
        else:
            ax.axhline(0.85, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
    else:
        if "m_t" not in frame.columns:
            raise KeyError("Figure 3 baseline requires m_t when Hawkes is not used.")
        y_col = "abs_m_t"
        frame = frame.copy()
        frame[y_col] = frame["m_t"].abs()
        if "entry_threshold" in frame.columns and frame["entry_threshold"].notna().any():
            frame["entry_boundary"] = pd.to_numeric(frame["entry_threshold"], errors="coerce")
            boundary_label = "entry_threshold"
        elif "sigma_hat" in frame.columns and frame["sigma_hat"].notna().any():
            frame["entry_boundary"] = entry_k * frame["T_t"] * frame["sigma_hat"]
            boundary_label = "|m|=k*T*sigma_hat"
        else:
            frame["entry_boundary"] = entry_k * frame["T_t"]
            boundary_label = "|m|=k*T"
        ax.set_ylabel("|m_t|")
        ax.set_title("Figure 3 - Phase Space (T_t, |m_t|)")
        boundary = frame[["T_t", "entry_boundary"]].dropna().sort_values("T_t")
        if not boundary.empty:
            ax.plot(
                boundary["T_t"],
                boundary["entry_boundary"],
                color="#003f5c",
                linestyle="--",
                linewidth=1.0,
                label=boundary_label,
            )

    for decision, color in DECISION_COLORS.items():
        subset = frame[frame["decision"] == decision]
        if subset.empty:
            continue
        ax.scatter(
            subset["T_t"],
            subset[y_col],
            s=14,
            alpha=0.55,
            label=decision,
            color=color,
        )

    ax.set_xlabel("T_t")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
