"""Visual QC: mask overlays and longitudinal montages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUBREGION_COLORS = {
    1: (1.00, 0.35, 0.20),   # HD-GLIO-AUTO label 1: non-enhancing
    2: (0.20, 0.85, 0.95),   # HD-GLIO-AUTO label 2: contrast-enhancing
    3: (0.95, 0.85, 0.25),   # DeepBraTumIA label 3: edema
    4: (0.75, 0.40, 0.95),
}


def _largest_lesion_slice(mask: np.ndarray) -> int:
    per_slice = (mask > 0).sum(axis=(0, 1))
    return int(np.argmax(per_slice)) if per_slice.max() > 0 else mask.shape[2] // 2


def _window(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def overlay_figure(
    image_path: Path,
    mask_path: Path,
    out_path: Path,
    *,
    title: str = "",
    labels: dict[int, str] | None = None,
    alpha: float = 0.45,
) -> Path:
    """Axial slice at maximum lesion area, with the mask overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import nibabel as nib

    image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(mask_path)).dataobj)
    index = _largest_lesion_slice(mask)

    base = _window(np.rot90(image[:, :, index]))
    msk = np.rot90(mask[:, :, index])

    rgb = np.dstack([base] * 3)
    for value, color in SUBREGION_COLORS.items():
        sel = msk == value
        if not sel.any():
            continue
        for channel in range(3):
            rgb[..., channel][sel] = (
                (1 - alpha) * rgb[..., channel][sel] + alpha * color[channel])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"{title}\nslice {index}", fontsize=9)
    present = [v for v in SUBREGION_COLORS if (msk == v).any()]
    if present:
        handles = [mpatches.Patch(color=SUBREGION_COLORS[v],
                                  label=(labels or {}).get(v, f"label {v}"))
                   for v in present]
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.7)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def patient_trajectory_montage(
    study_dirs: list[Path],
    out_path: Path,
    *,
    sequence: str = "CT1",
    patient: str = "",
    labels: dict[int, str] | None = None,
    alpha: float = 0.45,
) -> Path | None:
    """One panel per study for a patient: the visual form of a trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    panels = []
    for study_dir in study_dirs:
        registered = study_dir / "HD-GLIO-AUTO-segmentation" / "registered"
        image_path = registered / f"{sequence}_r2s_bet_reg.nii.gz"
        mask_path = registered / "segmentation.nii.gz"
        if not (image_path.exists() and mask_path.exists()):
            continue
        image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(str(mask_path)).dataobj)
        index = _largest_lesion_slice(mask)
        voxel = float(np.prod(nib.load(str(mask_path)).header.get_zooms()[:3]))
        volumes = {int(v): float((mask == v).sum()) * voxel / 1000.0
                   for v in np.unique(mask) if int(v) != 0}
        panels.append((study_dir.name, _window(np.rot90(image[:, :, index])),
                       np.rot90(mask[:, :, index]), volumes))
    if not panels:
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), dpi=130)
    if n == 1:
        axes = [axes]
    for ax, (name, base, msk, volumes) in zip(axes, panels):
        rgb = np.dstack([base] * 3)
        for value, color in SUBREGION_COLORS.items():
            sel = msk == value
            if not sel.any():
                continue
            for channel in range(3):
                rgb[..., channel][sel] = (
                    (1 - alpha) * rgb[..., channel][sel] + alpha * color[channel])
        ax.imshow(rgb, interpolation="nearest")
        ax.set_axis_off()
        caption = ", ".join(
            f"{(labels or {}).get(v, v)}: {vol:.1f} mL"
            for v, vol in sorted(volumes.items()))
        ax.set_title(f"{name}\n{caption}", fontsize=7)
    fig.suptitle(f"{patient} - {sequence} with HD-GLIO-AUTO segmentation",
                 fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def qc_montage_for_studies(imaging_root: Path, outdir: Path, *,
                           limit: int = 6) -> list[Path]:
    """Write one trajectory montage per locally-present patient."""
    from gbmplast.adapters.lumiere import SEGMENTATION_LABELS

    labels = SEGMENTATION_LABELS["HD-GLIO-AUTO"]
    written: list[Path] = []
    for patient_dir in sorted(Path(imaging_root).iterdir()):
        if not patient_dir.is_dir():
            continue
        studies = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        path = patient_trajectory_montage(
            studies, Path(outdir) / f"{patient_dir.name}_trajectory.png",
            patient=patient_dir.name, labels=labels)
        if path:
            written.append(path)
        if limit and len(written) >= limit:
            break
    return written
