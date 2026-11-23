"""Visual and numeric QC for image/mask pairs.

Checks performed per study
--------------------------
1. every expected file is present and loadable
2. orientation and voxel spacing are reported, and flagged when a sequence
   is markedly anisotropic
3. image and mask share a voxel grid (shape + affine) -- the single most
   common cause of silently wrong radiomics
4. mask labels are a subset of the cohort's documented label set
5. the mask is non-empty, lies inside the field of view, and its voxel
   counts are reported so they can be reconciled with published features
6. intensity sanity: non-constant, finite, and non-degenerate inside the mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.imaging import io


@dataclass
class StudyQC:
    cohort: str
    patient: str
    study: str
    sequence: str
    status: str = "pass"
    findings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.status = "fail"
        self.findings.append(message)

    def warn(self, message: str) -> None:
        if self.status == "pass":
            self.status = "warn"
        self.findings.append(message)

    def as_row(self) -> dict:
        return {
            "cohort": self.cohort, "patient": self.patient,
            "study": self.study, "sequence": self.sequence,
            "status": self.status,
            "n_findings": len(self.findings),
            "findings": "; ".join(self.findings),
            **self.metrics,
        }


def qc_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    *,
    cohort: str,
    patient: str,
    study: str,
    sequence: str,
    expected_labels: dict[int, str] | None = None,
    anisotropy_threshold: float = 3.0,
) -> StudyQC:
    qc = StudyQC(cohort=cohort, patient=patient, study=study, sequence=sequence)

    for label, path in (("image", image_path), ("mask", mask_path)):
        if not Path(path).exists():
            qc.fail(f"{label} missing: {Path(path).name}")
    if qc.status == "fail":
        return qc

    try:
        img_geom = io.geometry(image_path)
        mask_geom = io.geometry(mask_path)
    except Exception as exc:
        qc.fail(f"load failed: {exc}")
        return qc

    qc.metrics.update({
        "image_shape": "x".join(map(str, img_geom.shape)),
        "image_spacing": "x".join(f"{s:.3f}" for s in img_geom.spacing),
        "image_orientation": img_geom.orientation,
        "image_dtype": img_geom.dtype,
        "mask_shape": "x".join(map(str, mask_geom.shape)),
        "mask_orientation": mask_geom.orientation,
        "voxel_volume_mm3": round(float(np.prod(img_geom.spacing)), 5),
    })

    # 2. anisotropy
    spacing = np.asarray(img_geom.spacing, dtype=float)
    if spacing.min() > 0:
        ratio = float(spacing.max() / spacing.min())
        qc.metrics["anisotropy_ratio"] = round(ratio, 3)
        if ratio >= anisotropy_threshold:
            qc.warn(f"anisotropic voxels (ratio {ratio:.2f}); texture features "
                    "are not comparable across differently-sampled studies")

    # 3. shared grid
    ok, reasons = io.same_grid(image_path, mask_path)
    qc.metrics["grid_match"] = ok
    if not ok:
        qc.fail("image/mask grid mismatch: " + "; ".join(reasons))
        return qc

    if img_geom.orientation != mask_geom.orientation:
        qc.fail(f"orientation mismatch image={img_geom.orientation} "
                f"mask={mask_geom.orientation}")

    # 4 + 5. labels and mask extent
    import nibabel as nib
    mask = np.asarray(nib.load(str(mask_path)).dataobj)
    values = sorted({int(v) for v in np.unique(mask)} - {0})
    qc.metrics["mask_labels"] = ",".join(map(str, values))
    if expected_labels is not None:
        unexpected = [v for v in values if v not in expected_labels]
        if unexpected:
            qc.fail(f"unexpected mask labels {unexpected}; documented labels "
                    f"are {sorted(expected_labels)}")
        missing = [v for v in expected_labels if v not in values]
        if missing:
            qc.warn(f"documented labels absent from this mask: {missing} "
                    "(legitimate when a subregion is not present)")

    voxel_volume = float(np.prod(img_geom.spacing))
    for value in values:
        count = int((mask == value).sum())
        name = (expected_labels or {}).get(value, f"label_{value}")
        qc.metrics[f"n_voxels_{name}"] = count
        qc.metrics[f"volume_mm3_{name}"] = round(count * voxel_volume, 2)

    if not values:
        qc.warn("mask is empty (no tumour labels); features cannot be computed")
        return qc

    coords = np.argwhere(mask > 0)
    touching = []
    for axis in range(3):
        if coords[:, axis].min() == 0:
            touching.append(f"axis{axis}-min")
        if coords[:, axis].max() == mask.shape[axis] - 1:
            touching.append(f"axis{axis}-max")
    if touching:
        qc.warn(f"mask touches the field-of-view boundary ({', '.join(touching)}); "
                "the lesion may be clipped")

    # 6. intensities
    image = np.asarray(nib.load(str(image_path)).dataobj, dtype=np.float64)
    if not np.isfinite(image).all():
        qc.fail("image contains non-finite values")
        return qc
    inside = image[mask > 0]
    qc.metrics.update({
        "image_min": round(float(image.min()), 4),
        "image_max": round(float(image.max()), 4),
        "mask_intensity_mean": round(float(inside.mean()), 4),
        "mask_intensity_std": round(float(inside.std()), 4),
    })
    if float(inside.std()) == 0.0:
        qc.warn("constant intensity inside the mask; texture features "
                "will be degenerate")
    if float(image.std()) == 0.0:
        qc.fail("constant image")
    return qc


# --------------------------------------------------------------------------
# cohort drivers
# --------------------------------------------------------------------------

def run_cohort_qc(adapter, *, limit: int = 10,
                  make_figures: bool = True) -> pd.DataFrame | None:
    """Dispatch to a cohort-specific QC walk over locally present images."""
    if adapter.key == "lumiere":
        return _lumiere_qc(adapter, limit=limit, make_figures=make_figures)
    return None


def _lumiere_qc(adapter, *, limit: int, make_figures: bool) -> pd.DataFrame | None:
    from gbmplast.adapters.lumiere import SEGMENTATION_LABELS

    root = adapter.raw_dir / "images" / "Imaging"
    if not root.exists():
        return None

    expected = SEGMENTATION_LABELS["HD-GLIO-AUTO"]
    rows: list[dict] = []
    studies = 0
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            seg = (study_dir / "HD-GLIO-AUTO-segmentation" / "registered"
                   / "segmentation.nii.gz")
            if not seg.exists():
                continue
            # The released features were computed in the registered space, so
            # QC the registered pair; that is what reproduces them.
            for sequence in ("CT1", "FLAIR", "T1", "T2"):
                image = (study_dir / "HD-GLIO-AUTO-segmentation" / "registered"
                         / f"{sequence}_r2s_bet_reg.nii.gz")
                if not image.exists():
                    continue
                qc = qc_image_mask_pair(
                    image, seg, cohort="LUMIERE",
                    patient=patient_dir.name, study=study_dir.name,
                    sequence=sequence, expected_labels=expected)
                rows.append(qc.as_row())
            # Also QC the native-space pair, whose geometry differs per sequence.
            for sequence in ("CT1", "FLAIR", "T1", "T2"):
                native_img = study_dir / f"{sequence}.nii.gz"
                native_seg = (study_dir / "HD-GLIO-AUTO-segmentation" / "native"
                              / f"segmentation_{sequence}_origspace.nii.gz")
                if not (native_img.exists() and native_seg.exists()):
                    continue
                qc = qc_image_mask_pair(
                    native_img, native_seg, cohort="LUMIERE",
                    patient=patient_dir.name, study=study_dir.name,
                    sequence=f"{sequence}_native", expected_labels=expected)
                rows.append(qc.as_row())
            studies += 1
            if limit and studies >= limit:
                break
        if limit and studies >= limit:
            break

    if not rows:
        return None
    table = pd.DataFrame(rows)

    if make_figures:
        try:
            from gbmplast.imaging.viz import qc_montage_for_studies
            figdir = adapter.config.paths.figures / adapter.key / "imaging_qc"
            qc_montage_for_studies(root, figdir, limit=min(limit, 6))
        except Exception as exc:         # figures are never load-bearing
            print(f"   (montage skipped: {exc})")
    return table
