#!/usr/bin/env python
"""Add explanatory callouts to the pipeline's figures.

The figures themselves are produced by the analysis code; this script only
overlays numbered callouts so a reader who has not run the pipeline can tell
what they are looking at. Nothing here changes a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "annotated"
INK = "#0b0b0b"
ACCENT = "#2a78d6"
CRIT = "#d33c3c"
SURFACE = "#fcfcfb"


def annotate(src: Path, dest: Path, title: str, notes: list[tuple[float, float, str]],
             *, note_width: float = 0.30, fontsize: float = 8.4) -> Path:
    """Place numbered markers on the figure and a numbered key beside it."""
    img = mpimg.imread(src)
    h, w = img.shape[0], img.shape[1]
    aspect = h / w
    fig_w = 13.0
    # The image occupies (1 - note_width) of the figure's width, so the
    # figure height that preserves its aspect is that width times the
    # aspect -- not the full width, which left a band of empty page.
    image_w_in = fig_w * (1 - note_width)
    fig = plt.figure(figsize=(fig_w, image_w_in * aspect), dpi=110,
                     facecolor=SURFACE)
    ax = fig.add_axes([0.0, 0.0, 1 - note_width, 1.0])
    ax.imshow(img)
    ax.set_axis_off()

    key = fig.add_axes([1 - note_width + 0.012, 0.0, note_width - 0.02, 1.0])
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.text(0.0, 0.975, title, fontsize=10.5, fontweight="semibold",
             color=INK, va="top", wrap=True)

    y = 0.90
    line_h = 0.030 if aspect > 0.6 else 0.055
    for i, (fx, fy, text) in enumerate(notes, 1):
        ax.annotate(str(i), xy=(fx * w, fy * h), xycoords="data",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.34", fc=ACCENT, ec="white",
                              lw=1.4))
        wrapped = _wrap(text, 46)
        key.text(0.0, y, f"{i}", fontsize=9, fontweight="bold", color="white",
                 va="top", ha="center",
                 bbox=dict(boxstyle="circle,pad=0.3", fc=ACCENT, ec="none"))
        key.text(0.065, y, wrapped, fontsize=fontsize, color=INK, va="top")
        y -= line_h * (1.6 + wrapped.count("\n"))

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return dest


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def main() -> int:
    made = []

    # 1. LUMIERE trajectories
    src = ROOT / "figures/lumiere/next_visit_rano/patient_trajectories.png"
    if src.exists():
        made.append(annotate(
            src, OUT / "01_lumiere_trajectories.png",
            "What one LUMIERE patient looks like",
            [(0.075, 0.13, "One panel per patient. The x-axis is months since "
                           "the index surgery, not calendar time."),
             (0.14, 0.20, "Orange = non-enhancing/oedema volume in mL, measured "
                          "by the automated segmentation at every study."),
             (0.14, 0.295, "Blue = enhancing tumour. It sits near zero after a "
                           "gross total resection and rises at recurrence - "
                           "that rise is what the model is asked to anticipate."),
             (0.165, 0.325, "Red triangles mark transitions where the NEXT "
                            "study was rated progressive disease by the "
                            "neuroradiologist. These are the labels."),
             (0.52, 0.52, "Trajectories differ wildly in shape and in how "
                          "often patients were scanned - which is why every "
                          "split is grouped by patient.")]))

    # 2. ROC
    src = ROOT / "figures/lumiere/next_visit_rano/roc_curves.png"
    if src.exists():
        made.append(annotate(
            src, OUT / "02_lumiere_roc.png",
            "Why the naive baseline wins",
            [(0.30, 0.22, "Blue is the naive rule: 'predict PD next visit if "
                          "this visit is PD'. It is the highest curve."),
             (0.46, 0.34, "The combined radiomics + clinical models track it "
                          "but never clearly beat it."),
             (0.62, 0.62, "The grey diagonal is chance. Clinical-only "
                          "(dark green) sits on it."),
             (0.045, 0.885, "Legend carries each model's AUROC, so the figure "
                          "is readable without the table.")],
            note_width=0.36))

    # 3. Cohort flow
    src = ROOT / "figures/lumiere/next_visit_rano/cohort_flow.png"
    if src.exists():
        made.append(annotate(
            src, OUT / "03_lumiere_cohort_flow.png",
            "How 477 transitions become 385 analysable ones",
            [(0.30, 0.17, "Every consecutive pair of dated studies from one "
                          "patient is a candidate transition."),
             (0.30, 0.37, "Only pairs whose SECOND study was rated by the "
                          "expert can have an outcome."),
             (0.75, 0.56, "The exclusion is stated with its count, not "
                          "silently applied."),
             (0.30, 0.80, "247 of 385 are progression events - a 64% rate "
                          "that reflects which studies were chosen for "
                          "rating, not routine surveillance.")],
            note_width=0.34))

    # 4. UCSD cohort flow
    src = ROOT / "figures/ucsd_ptgbm/radiogenomics_tumor_vs_treatment/cohort_flow.png"
    if src.exists():
        made.append(annotate(
            src, OUT / "04_ucsd_cohort_flow.png",
            "UCSD: every timepoint is labelled",
            [(0.27, 0.13, "243 post-treatment MRI timepoints from 178 "
                          "patients. Some patients contribute up to 4."),
             (0.27, 0.38, "Unusually, no timepoint is dropped for want of a "
                          "label - all 243 were adjudicated."),
             (0.27, 0.63, "192 show residual or recurrent tumour..."),
             (0.27, 0.77, "...and 51 are treatment-related change. That 79/21 "
                          "split is enriched by design and is not a "
                          "real-world prevalence.")],
            note_width=0.34))

    # 5. Calibration
    src = ROOT / "figures/lumiere/next_visit_rano/calibration.png"
    if src.exists():
        made.append(annotate(
            src, OUT / "05_lumiere_calibration.png",
            "Discrimination is not enough: are the probabilities honest?",
            [(0.52, 0.50, "The dashed diagonal is perfect calibration: of the "
                          "cases given a 70% risk, 70% should progress."),
             (0.33, 0.38, "A point above the line means the model "
                          "under-predicted risk in that bin; below means it "
                          "over-predicted."),
             (0.50, 0.86, "The legend reports each model's calibration slope. "
                          "1.0 is ideal; far from 1.0 means the probabilities "
                          "are too extreme or too timid.")],
            note_width=0.36))

    for path in made:
        print(path.relative_to(ROOT))
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
