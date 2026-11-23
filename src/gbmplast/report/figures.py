"""Figures: ROC, calibration, trajectories, cohort flow, paired effects.

Every figure obeys the project theme: one y-axis per panel, a legend
whenever two or more series are drawn, direct labels where the series count
allows, recessive grid and axes.  Accompanying CSVs are written next to each
figure so the numbers are readable without the picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.report.theme import (
    CATEGORICAL,
    GRID,
    ROLE,
    STATUS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    figure_context,
    series_color,
)


# --------------------------------------------------------------------------
# Module A / B shared: discrimination and calibration
# --------------------------------------------------------------------------

def roc_curves(results: list, out_path: Path, *, title: str = "") -> Path | None:
    """One ROC curve per model, chance line in neutral grey."""
    from sklearn.metrics import roc_curve

    usable = _curve_models(results, "auroc")
    if not usable:
        return None

    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.3))
        ax.plot([0, 1], [0, 1], color=ROLE["reference_line"], linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("chance", xy=(0.62, 0.58), color=TEXT_MUTED, fontsize=7.5,
                    rotation=38, rotation_mode="anchor")
        for i, r in enumerate(usable):
            p = r.predictions.dropna(subset=["y_prob", "y_true"])
            if p["y_true"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(p["y_true"], p["y_prob"])
            color = series_color(i)
            label = f"{_pretty(r.model)}  AUROC {r.metrics['auroc']:.3f}"
            ax.plot(fpr, tpr, color=color, label=label, zorder=3 + i)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title or "Discrimination (out-of-fold, patient-grouped CV)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        _legend_below(fig, ax, n=len(usable))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def pr_curves(results: list, out_path: Path, *, title: str = "") -> Path | None:
    """Precision-recall, with the prevalence floor drawn explicitly."""
    from sklearn.metrics import precision_recall_curve

    usable = _curve_models(results, "auprc")
    if not usable:
        return None
    prevalence = usable[0].metrics.get("prevalence", np.nan)

    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.3))
        if np.isfinite(prevalence):
            ax.axhline(prevalence, color=ROLE["reference_line"], linewidth=1.2,
                       linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"prevalence {prevalence:.2f}", xy=(0.02, prevalence),
                        xytext=(0.02, prevalence + 0.03), color=TEXT_MUTED,
                        fontsize=7.5)
        for i, r in enumerate(usable):
            p = r.predictions.dropna(subset=["y_prob", "y_true"])
            if p["y_true"].nunique() < 2:
                continue
            precision, recall, _ = precision_recall_curve(p["y_true"], p["y_prob"])
            ax.plot(recall, precision, color=series_color(i),
                    label=f"{_pretty(r.model)}  AUPRC {r.metrics['auprc']:.3f}",
                    zorder=3 + i)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision-recall (out-of-fold)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        _legend_below(fig, ax, n=len(usable))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def calibration_plot(results: list, out_path: Path, *,
                     title: str = "") -> Path | None:
    """Observed vs predicted, with bin sizes annotated (the relief rule)."""
    usable = [r for r in _curve_models(results, "auroc")
              if r.calibration is not None and not r.calibration.empty]
    if not usable:
        return None

    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.3))
        ax.plot([0, 1], [0, 1], color=ROLE["reference_line"], linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("perfect calibration", xy=(0.52, 0.49), color=TEXT_MUTED,
                    fontsize=7.5, rotation=38, rotation_mode="anchor")
        for i, r in enumerate(usable):
            t = r.calibration
            color = series_color(i)
            slope = r.metrics.get("calibration_slope", np.nan)
            label = _pretty(r.model)
            if np.isfinite(slope):
                label += f"  slope {slope:.2f}"
            ax.plot(t["mean_predicted"], t["observed_frequency"], color=color,
                    marker="o", markersize=5, markeredgecolor="white",
                    markeredgewidth=1.2, label=label, zorder=3 + i)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title or "Calibration (out-of-fold, quantile bins)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        _legend_below(fig, ax, n=len(usable))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def metric_comparison(results: list, out_path: Path, *, metric: str = "auroc",
                      title: str = "") -> Path | None:
    """Horizontal bars with bootstrap CIs and a value label on every bar."""
    rows = []
    for r in results:
        value = r.metrics.get(metric)
        if value is None or not np.isfinite(value):
            continue
        low = high = np.nan
        if r.metric_ci is not None:
            sel = r.metric_ci[r.metric_ci["metric"] == metric]
            if len(sel):
                low = float(sel["ci_low"].iloc[0])
                high = float(sel["ci_high"].iloc[0])
        rows.append({"model": _pretty(r.model), "value": float(value),
                     "low": low, "high": high,
                     "is_baseline": "baseline" in r.model})
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("value")

    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(frame) + 1.6))
        y = np.arange(len(frame))
        for i, (_, row) in enumerate(frame.iterrows()):
            color = (ROLE["baseline_model"] if row["is_baseline"]
                     else series_color(0))
            ax.barh(i, row["value"], height=0.58, color=color, zorder=3)
            if np.isfinite(row["low"]) and np.isfinite(row["high"]):
                ax.plot([row["low"], row["high"]], [i, i], color=TEXT_SECONDARY,
                        linewidth=1.4, zorder=4)
            ax.annotate(f"{row['value']:.3f}", xy=(row["value"], i),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["model"], fontsize=8)
        ax.set_xlabel(metric.upper())
        ax.set_title(title or f"{metric.upper()} with patient-level bootstrap 95% CI")
        ax.grid(axis="y", visible=False)
        if metric in {"auroc", "auprc"}:
            ax.set_xlim(0, 1.0)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Module A: trajectories and cohort flow
# --------------------------------------------------------------------------

def patient_trajectories(
    table: pd.DataFrame,
    out_path: Path,
    *,
    subregions: tuple[str, ...] = ("enhancing", "non_enhancing"),
    n_patients: int = 12,
    title: str = "",
) -> Path | None:
    """Small multiples: volume over time per patient, RANO PD marked.

    Small multiples rather than many overlaid lines, so the series count per
    panel stays at the number of subregions.
    """
    needed = [f"cur_{s}__volume_mm3" for s in subregions]
    if not set(needed + ["time_since_t0_days", "patient_id"]).issubset(table.columns):
        return None
    counts = table.groupby("patient_id").size().sort_values(ascending=False)
    chosen = list(counts.index[:n_patients])
    if not chosen:
        return None

    ncols = 4
    nrows = int(np.ceil(len(chosen) / ncols))
    with figure_context() as plt:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                                 squeeze=False)
        for idx, pid in enumerate(chosen):
            ax = axes[idx // ncols][idx % ncols]
            sub = table[table["patient_id"] == pid].sort_values("time_since_t0_days")
            t = pd.to_numeric(sub["time_since_t0_days"], errors="coerce") / 30.44
            for j, subregion in enumerate(subregions):
                v = pd.to_numeric(sub[f"cur_{subregion}__volume_mm3"],
                                  errors="coerce") / 1000.0
                ax.plot(t, v, color=ROLE.get(subregion, series_color(j)),
                        marker="o", markersize=3.5, linewidth=1.8,
                        label=subregion.replace("_", "-"))
            pd_rows = sub[sub.get("next_rano_progression") == 1.0]
            if len(pd_rows):
                ax.scatter(
                    pd.to_numeric(pd_rows["time_since_t0_days"],
                                  errors="coerce") / 30.44,
                    np.zeros(len(pd_rows)),
                    marker="^", s=26, color=STATUS["critical"], zorder=5,
                    label="next visit PD")
            ax.set_title(str(pid).split("::")[-1], fontsize=8.5)
            ax.tick_params(labelsize=7)
            if idx % ncols == 0:
                ax.set_ylabel("volume (mL)", fontsize=8)
            if idx // ncols == nrows - 1:
                ax.set_xlabel("months since surgery", fontsize=8)
        for idx in range(len(chosen), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_axis_off()

        handles, labels = axes[0][0].get_legend_handles_labels()
        seen: dict = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        fig.legend(seen.values(), seen.keys(), loc="lower center",
                   ncol=len(seen), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(title or "Tumour subregion volume trajectories "
                              "(HD-GLIO-AUTO segmentation)", fontsize=11)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def cohort_flow_diagram(flow_frame: pd.DataFrame, out_path: Path, *,
                        title: str = "") -> Path | None:
    """A counts-and-exclusions diagram -- not a chart, a labelled flow."""
    if flow_frame is None or flow_frame.empty:
        return None
    steps = flow_frame[flow_frame["kind"] == "step"]
    exclusions = flow_frame[flow_frame["kind"] == "exclusion"]
    if steps.empty:
        return None

    n_rows = len(steps) + len(exclusions)
    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(8.2, 0.52 * n_rows + 1.2))
        ax.set_axis_off()
        y = 0.0
        for _, row in steps.iterrows():
            ax.add_patch(plt.Rectangle((0.03, y), 0.52, 0.42,
                                       facecolor="#eef4fc",
                                       edgecolor=series_color(0),
                                       linewidth=1.2, zorder=2))
            ax.text(0.05, y + 0.21, f"{row['label']}", va="center",
                    fontsize=9, color="#0b0b0b", zorder=3)
            ax.text(0.53, y + 0.21, f"n = {int(row['n']):,}", va="center",
                    ha="right", fontsize=9, fontweight="semibold",
                    color=series_color(0), zorder=3)
            y -= 0.62
        y_ex = -0.1
        for _, row in exclusions.iterrows():
            ax.text(0.60, y_ex, f"- excluded: {row['label']}", va="center",
                    fontsize=8, color=TEXT_SECONDARY)
            ax.text(0.985, y_ex, f"n = {int(row['n']):,}", va="center",
                    ha="right", fontsize=8, color=STATUS["serious"])
            y_ex -= 0.42
        ax.set_xlim(0, 1)
        ax.set_ylim(min(y, y_ex) - 0.2, 0.52)
        ax.set_title(title or "Analysis cohort flow", loc="left", fontsize=11)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def observed_vs_predicted(result, out_path: Path, *,
                          title: str = "") -> Path | None:
    """Regression diagnostic: one series, so no legend box."""
    if result.task != "regression":
        return None
    p = result.predictions.dropna(subset=["y_true", "y_pred"])
    if len(p) < 5:
        return None
    with figure_context() as plt:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        lo = float(min(p["y_true"].min(), p["y_pred"].min()))
        hi = float(max(p["y_true"].max(), p["y_pred"].max()))
        ax.plot([lo, hi], [lo, hi], color=ROLE["reference_line"],
                linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.scatter(p["y_pred"], p["y_true"], s=22, color=series_color(0),
                   alpha=0.75, edgecolor="white", linewidth=0.7, zorder=3)
        mae = result.metrics.get("mae", np.nan)
        ax.set_xlabel("Predicted relative change")
        ax.set_ylabel("Observed relative change")
        ax.set_title(title or f"{_pretty(result.model)}  "
                              f"(MAE {mae:.3f}, n = {len(p)})")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def next_visit_figures(results: list, eligible: pd.DataFrame, figdir: Path,
                       *, outcome: str) -> list[Path]:
    figdir = Path(figdir)
    written: list[Path] = []
    if outcome == "rano":
        for fn, name, metric in (
            (roc_curves, "roc_curves.png", None),
            (pr_curves, "precision_recall.png", None),
            (calibration_plot, "calibration.png", None),
        ):
            path = fn(results, figdir / name)
            if path:
                written.append(path)
        for metric in ("auroc", "auprc"):
            path = metric_comparison(results, figdir / f"{metric}_comparison.png",
                                     metric=metric)
            if path:
                written.append(path)
    else:
        path = metric_comparison(results, figdir / "mae_comparison.png",
                                 metric="mae",
                                 title="MAE with patient-level bootstrap 95% CI "
                                       "(lower is better)")
        if path:
            written.append(path)
        best = min((r for r in results if r.task == "regression"),
                   key=lambda r: r.metrics.get("mae", np.inf), default=None)
        if best is not None:
            path = observed_vs_predicted(best, figdir / "observed_vs_predicted.png")
            if path:
                written.append(path)

    path = patient_trajectories(eligible, figdir / "patient_trajectories.png")
    if path:
        written.append(path)
    return written


#: Curve plots carry at most this many series. Past it, the lines stop being
#: separable even with a validated palette, so the weakest models are dropped
#: from the figure (every model stays in the metrics table).
MAX_CURVE_SERIES = 6


def _curve_models(results: list, metric: str) -> list:
    """Models worth drawing a curve for, best first.

    Models whose discrimination is undefined by construction (constant
    predictions within every fold, e.g. a prevalence baseline) are excluded:
    plotting their pooled curve would invite a reading the number does not
    support. They remain in the metrics table, flagged.
    """
    usable = [r for r in results
              if r.task == "classification"
              and r.metrics.get("discrimination_defined") is not False
              and np.isfinite(r.metrics.get(metric, np.nan))]
    usable = sorted(usable, key=lambda r: -r.metrics[metric])
    return usable[:min(MAX_CURVE_SERIES, len(CATEGORICAL))]


def _legend_below(fig, ax, *, n: int) -> None:
    """Legend under the panel, so it never sits on top of the data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=1 if n <= 4 else 2, frameon=False,
              handlelength=1.8, columnspacing=1.4, borderaxespad=0.0)
    fig.subplots_adjust(bottom=0.30)


def _pretty(name: str) -> str:
    return (name.replace("__", " + ").replace("_", " ")
            .replace("gradient boosting", "GBM")
            .replace("elastic net", "elastic-net"))
