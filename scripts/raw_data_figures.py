#!/usr/bin/env python
"""Show what the raw data actually looks like, from the real files.

Nothing here is schematic: every panel is rendered from a file that was
downloaded, and every number is read from that file's header.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "annotated"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
QUIET = "#52514e"
SEG_COLORS = {1: (1.00, 0.35, 0.20), 2: (0.20, 0.85, 0.95)}


def window(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    a, b = np.percentile(finite, [lo, hi])
    return np.zeros_like(img) if b <= a else np.clip((img - a) / (b - a), 0, 1)


def mri_panel(study: Path, out: Path) -> Path | None:
    """The four sequences of one real study, plus the segmentation overlay."""
    reg = study / "HD-GLIO-AUTO-segmentation" / "registered"
    mask_path = reg / "segmentation.nii.gz"
    if not mask_path.exists():
        return None
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj)
    idx = int(np.argmax((mask > 0).sum(axis=(0, 1))))

    sequences = [("CT1", "post-contrast T1"), ("T1", "pre-contrast T1"),
                 ("T2", "T2-weighted"), ("FLAIR", "T2-FLAIR")]
    panels = []
    for key, label in sequences:
        path = reg / f"{key}_r2s_bet_reg.nii.gz"
        if path.exists():
            panels.append((key, label, np.asarray(
                nib.load(str(path)).dataobj, dtype=np.float32)))
    if not panels:
        return None

    n = len(panels) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.9), dpi=130,
                             facecolor=SURFACE)
    for ax, (key, label, vol) in zip(axes, panels):
        ax.imshow(window(np.rot90(vol[:, :, idx])), cmap="gray",
                  interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(f"{key}\n{label}", fontsize=9.5, color=INK)

    ax = axes[-1]
    base = window(np.rot90(panels[0][2][:, :, idx]))
    msk = np.rot90(mask[:, :, idx])
    rgb = np.dstack([base] * 3)
    for value, color in SEG_COLORS.items():
        sel = msk == value
        for c in range(3):
            rgb[..., c][sel] = 0.45 * rgb[..., c][sel] + 0.55 * color[c]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title("automated segmentation\n(overlaid on CT1)", fontsize=9.5,
                 color=INK)

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=SEG_COLORS[1], label="label 1: non-enhancing"),
        mpatches.Patch(color=SEG_COLORS[2], label="label 2: enhancing")],
        loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5,
        frameon=False, ncol=1)

    zooms = mask_img.header.get_zooms()[:3]
    fig.suptitle(
        f"One LUMIERE study: {study.parent.name} / {study.name}  ·  "
        f"{mask_img.shape[0]}×{mask_img.shape[1]}×{mask_img.shape[2]} voxels at "
        f"{zooms[0]:.2f}×{zooms[1]:.2f}×{zooms[2]:.2f} mm  ·  axial slice {idx}",
        fontsize=10.5, color=INK, y=1.02)
    fig.text(0.5, -0.04,
             "All five volumes share one voxel grid, which is what makes a "
             "mask drawn on one of them valid on the others.",
             ha="center", fontsize=8.5, color=QUIET)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def native_vs_registered(study: Path, out: Path) -> Path | None:
    """Why registration matters: native sequences do not share a grid."""
    rows = []
    for key in ("CT1", "T1", "T2", "FLAIR"):
        native = study / f"{key}.nii.gz"
        reg = study / "HD-GLIO-AUTO-segmentation" / "registered" / f"{key}_r2s_bet_reg.nii.gz"
        if not (native.exists() and reg.exists()):
            continue
        a, b = nib.load(str(native)), nib.load(str(reg))
        rows.append((key,
                     f"{a.shape[0]}×{a.shape[1]}×{a.shape[2]}",
                     "×".join(f"{z:.2f}" for z in a.header.get_zooms()[:3]),
                     "".join(nib.aff2axcodes(a.affine)),
                     f"{b.shape[0]}×{b.shape[1]}×{b.shape[2]}",
                     "×".join(f"{z:.2f}" for z in b.header.get_zooms()[:3]),
                     "".join(nib.aff2axcodes(b.affine))))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.0), dpi=130,
                           facecolor=SURFACE)
    ax.set_axis_off()
    headers = ["sequence", "native shape", "native mm", "native orient",
               "registered shape", "registered mm", "registered orient"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e4e3de")
        if r == 0:
            cell.set_facecolor("#eef4fc")
            cell.set_text_props(fontweight="semibold", color=INK)
        elif c >= 4:
            cell.set_facecolor("#f6faf6")
    ax.set_title(
        f"The same study in native and registered space  ·  "
        f"{study.parent.name} / {study.name}",
        fontsize=11, color=INK, pad=18)
    fig.text(0.5, 0.02,
             "Native sequences differ in matrix size, voxel size and even "
             "orientation — a mask from one would not line up on another. "
             "The registered copies all share one grid.",
             ha="center", fontsize=8.5, color=QUIET)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    root = ROOT / "data/raw/lumiere/images/Imaging"
    if not root.exists():
        print("no LUMIERE images downloaded")
        return 1
    # Prefer the study with the most registered sequences locally present and
    # visible enhancing tumour, so the panel shows the real protocol rather
    # than whatever happened to be downloaded first.
    best = None
    for patient in sorted(root.iterdir()):
        if not patient.is_dir():
            continue
        for candidate in sorted(patient.iterdir()):
            reg = candidate / "HD-GLIO-AUTO-segmentation" / "registered"
            seg = reg / "segmentation.nii.gz"
            if not seg.exists():
                continue
            mask = np.asarray(nib.load(str(seg)).dataobj)
            enhancing = int((mask == 2).sum())
            if enhancing < 2000:
                continue
            n_seq = sum((reg / f"{k}_r2s_bet_reg.nii.gz").exists()
                        for k in ("CT1", "T1", "T2", "FLAIR"))
            score = (n_seq, enhancing)
            if best is None or score > best[0]:
                best = (score, candidate)
    study = best[1] if best else None
    if study is None:
        print("no suitable study found")
        return 1

    for path in (mri_panel(study, OUT / "10_raw_mri_study.png"),
                 native_vs_registered(study, OUT / "11_native_vs_registered.png")):
        if path:
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
