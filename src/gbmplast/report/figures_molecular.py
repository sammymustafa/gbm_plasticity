"""Module C figures: paired effects, state transitions, composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.report.theme import (
    CATEGORICAL,
    DIVERGING,
    ROLE,
    STATUS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    figure_context,
    series_color,
)


def paired_slope_plot(
    paired: pd.DataFrame,
    programme: str,
    out_path: Path,
    *,
    title: str = "",
    label: str | None = None,
) -> Path | None:
    """One line per patient from primary to recurrent.

    Two x positions and one line per patient: colour carries direction
    (increase / decrease), which is a polarity, so it uses the diverging
    poles rather than categorical hues.
    """
    a_col, b_col = f"{programme}__primary", f"{programme}__recurrent"
    if a_col not in paired.columns or b_col not in paired.columns:
        return None
    a = pd.to_numeric(paired[a_col], errors="coerce")
    b = pd.to_numeric(paired[b_col], errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]

    up, down = DIVERGING[2], DIVERGING[0]
    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(4.0, 5.0))
        for x0, x1 in zip(a, b):
            ax.plot([0, 1], [x0, x1],
                    color=(up if x1 > x0 else down), alpha=0.55, linewidth=1.3,
                    zorder=2)
        ax.scatter(np.zeros(len(a)), a, s=22, color=TEXT_SECONDARY,
                   alpha=0.6, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter(np.ones(len(b)), b, s=22, color=TEXT_SECONDARY,
                   alpha=0.6, zorder=3, edgecolor="white", linewidth=0.6)
        ax.plot([0, 1], [a.mean(), b.mean()], color="#0b0b0b", linewidth=2.6,
                marker="o", markersize=7, zorder=4)
        n_up = int((b.to_numpy() > a.to_numpy()).sum())
        ax.annotate(f"{n_up}/{len(a)} increased", xy=(0.5, 1.01),
                    xycoords="axes fraction", ha="center", fontsize=8,
                    color=TEXT_MUTED)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["primary", "recurrent"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(label or programme.replace("_", " "))
        ax.set_title(title or programme.replace("_", " "))
        ax.grid(axis="x", visible=False)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def effect_forest(effects: pd.DataFrame, out_path: Path, *,
                  title: str = "", top_n: int = 18) -> Path | None:
    """Paired mean difference with bootstrap CI, ordered by effect size."""
    tested = effects[effects.get("status") == "tested"].copy()
    if tested.empty:
        return None
    tested = tested.reindex(
        tested["mean_delta"].abs().sort_values(ascending=False).index).head(top_n)
    tested = tested.sort_values("mean_delta")

    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(7.2, 0.40 * len(tested) + 1.8))
        ax.axvline(0, color=ROLE["reference_line"], linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=1)
        for i, (_, row) in enumerate(tested.iterrows()):
            significant = (np.isfinite(row.get("wilcoxon_fdr", np.nan))
                           and row["wilcoxon_fdr"] < 0.05)
            color = (DIVERGING[2] if row["mean_delta"] > 0 else DIVERGING[0])
            ax.plot([row["delta_ci_low"], row["delta_ci_high"]], [i, i],
                    color=color, linewidth=2.0,
                    alpha=1.0 if significant else 0.45, zorder=3)
            ax.scatter([row["mean_delta"]], [i], s=46, color=color,
                       edgecolor="white", linewidth=1.0,
                       alpha=1.0 if significant else 0.45, zorder=4)
            if significant:
                ax.annotate("*", xy=(row["delta_ci_high"], i),
                            xytext=(5, -3), textcoords="offset points",
                            fontsize=11, color=color)
        ax.set_yticks(range(len(tested)))
        ax.set_yticklabels([p.replace("_", " ") for p in tested["programme"]],
                           fontsize=8)
        ax.set_xlabel("paired mean difference (recurrent - primary)")
        ax.set_title(title or "Within-patient programme change, "
                              "with patient bootstrap 95% CI")
        ax.grid(axis="y", visible=False)
        ax.annotate("* FDR < 0.05 (Wilcoxon, BH across programmes)",
                    xy=(0.0, -0.085), xycoords="axes fraction",
                    fontsize=7.5, color=TEXT_MUTED)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def state_transition_heatmap(matrix: pd.DataFrame, out_path: Path, *,
                             title: str = "") -> Path | None:
    """Counts of primary -> recurrent dominant state. Sequential, one hue."""
    if matrix is None or matrix.empty:
        return None
    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        data = matrix.to_numpy(dtype=float)
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, fontsize=9)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=9)
        ax.set_xlabel("recurrent dominant state")
        ax.set_ylabel("primary dominant state")
        threshold = data.max() * 0.6 if data.size else 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.annotate(f"{int(data[i, j])}", xy=(j, i), ha="center",
                            va="center", fontsize=10,
                            color="white" if data[i, j] > threshold else "#0b0b0b")
        ax.set_title(title or "Dominant malignant state, primary -> recurrent")
        ax.grid(visible=False)
        fig.colorbar(im, ax=ax, label="patients", fraction=0.046, pad=0.04)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def composition_plot(scores: pd.DataFrame, out_path: Path, *,
                     title: str = "") -> Path | None:
    """Malignant-state weighting per sample, split by primary/recurrent."""
    weight_cols = [c for c in scores.columns if c.startswith("state_weight_")]
    if not weight_cols or "sample_role" not in scores.columns:
        return None
    roles = ["primary", "first_recurrence"]
    present = [r for r in roles if (scores["sample_role"] == r).any()]
    if not present:
        return None

    with figure_context() as plt:
        fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.2),
                                 squeeze=False, sharey=True)
        for k, role in enumerate(present):
            ax = axes[0][k]
            sub = scores[scores["sample_role"] == role]
            order = sub[weight_cols].idxmax(axis=1).rank(method="first")
            sub = sub.loc[order.sort_values().index]
            bottom = np.zeros(len(sub))
            for j, col in enumerate(weight_cols):
                values = pd.to_numeric(sub[col], errors="coerce").fillna(0).to_numpy()
                ax.bar(range(len(sub)), values, bottom=bottom, width=0.86,
                       color=series_color(j),
                       label=col.replace("state_weight_", ""),
                       edgecolor="#fcfcfb", linewidth=0.6)
                bottom = bottom + values
            ax.set_title(f"{role.replace('_', ' ')}  (n = {len(sub)})",
                         fontsize=10)
            ax.set_xlabel("sample")
            ax.set_xticks([])
            if k == 0:
                ax.set_ylabel("relative state weighting")
            ax.set_ylim(0, 1)
            ax.grid(axis="x", visible=False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(title or "Relative weighting of malignant-state "
                              "signatures (not a cell-type deconvolution)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plasticity_figures(paired: pd.DataFrame, effects: pd.DataFrame,
                       pairs: pd.DataFrame, matrix: pd.DataFrame,
                       scores: pd.DataFrame, figdir: Path) -> list[Path]:
    figdir = Path(figdir)
    written: list[Path] = []
    path = effect_forest(effects, figdir / "paired_effects_forest.png")
    if path:
        written.append(path)
    path = state_transition_heatmap(matrix, figdir / "state_transitions.png")
    if path:
        written.append(path)
    path = composition_plot(scores, figdir / "state_composition.png")
    if path:
        written.append(path)

    # Slope plots for the four malignant states plus antigen presentation.
    for programme in ("state_MES", "state_NPC", "state_AC", "state_OPC",
                      "HLA_I", "IFN_GAMMA", "STEMNESS"):
        path = paired_slope_plot(
            paired, programme, figdir / f"paired__{programme}.png",
            title=programme.replace("state_", "").replace("_", " "))
        if path:
            written.append(path)
    return written
