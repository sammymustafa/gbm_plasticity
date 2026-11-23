"""Image loading and geometry description.

Public GBM cohorts ship images that have *already* been preprocessed in
different ways (LUMIERE: HD-BET skull-stripped, with a separately registered
copy; UPENN-GBM: SRI24-registered and N4-corrected; UCSD-PTGBM: resampled to
a 1 mm MNI grid).  This module therefore *describes* geometry and leaves
re-processing to an explicit, opt-in step -- redoing registration or
resampling on already-standardised data degrades it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class ImageGeometry:
    path: str
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    orientation: str
    dtype: str
    affine_determinant: float
    is_isotropic: bool
    n_voxels: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = "x".join(str(s) for s in self.shape)
        d["spacing"] = "x".join(f"{s:.4f}" for s in self.spacing)
        return d


def load(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI without reading voxel data into memory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def geometry(path: Path, *, isotropic_tol: float = 0.05) -> ImageGeometry:
    img = load(path)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    spread = (max(zooms) - min(zooms)) / max(zooms) if max(zooms) > 0 else 0.0
    return ImageGeometry(
        path=str(path),
        shape=tuple(int(s) for s in img.shape[:3]),
        spacing=zooms,
        orientation="".join(nib.aff2axcodes(img.affine)),
        dtype=str(img.get_data_dtype()),
        affine_determinant=float(np.linalg.det(img.affine[:3, :3])),
        is_isotropic=spread <= isotropic_tol,
        n_voxels=int(np.prod(img.shape[:3])),
    )


def same_grid(a: Path, b: Path, *, atol: float = 1e-3) -> tuple[bool, list[str]]:
    """Do two images share a voxel grid? Returns (ok, reasons for mismatch)."""
    ia, ib = load(a), load(b)
    reasons: list[str] = []
    if ia.shape[:3] != ib.shape[:3]:
        reasons.append(f"shape {ia.shape[:3]} != {ib.shape[:3]}")
    if not np.allclose(ia.affine, ib.affine, atol=atol):
        delta = float(np.abs(np.asarray(ia.affine) - np.asarray(ib.affine)).max())
        reasons.append(f"affine differs (max abs delta {delta:.4g})")
    return (not reasons, reasons)


def voxel_volume_mm3(path: Path) -> float:
    img = load(path)
    return float(np.prod(img.header.get_zooms()[:3]))


def label_volumes(mask_path: Path, *,
                  labels: dict[int, str] | None = None) -> dict[str, float]:
    """Volume in mm^3 per label value present in a mask.

    ``labels`` maps the integer label to a canonical subregion name; values
    absent from the mapping are reported under ``label_<n>`` rather than
    being dropped or guessed.
    """
    img = load(mask_path)
    data = np.asarray(img.dataobj)
    voxel = float(np.prod(img.header.get_zooms()[:3]))
    out: dict[str, float] = {}
    for value in np.unique(data):
        value = int(value)
        if value == 0:
            continue
        name = (labels or {}).get(value, f"label_{value}")
        out[name] = float((data == value).sum()) * voxel
    return out


def label_voxel_counts(mask_path: Path) -> dict[int, int]:
    data = np.asarray(load(mask_path).dataobj)
    return {int(v): int((data == v).sum()) for v in np.unique(data) if int(v) != 0}


def reorient_to_ras(path: Path, out_path: Path) -> Path:
    """Explicit, opt-in canonical reorientation (RAS+).

    Not used by default: the cohorts' own preprocessing is preserved, and
    PyRadiomics is orientation-invariant for the features used here as long
    as image and mask share a grid.
    """
    img = load(path)
    canonical = nib.as_closest_canonical(img)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(canonical, str(out_path))
    return out_path
