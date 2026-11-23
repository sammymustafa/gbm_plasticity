"""Radiomics extraction with PyRadiomics.

Design notes
------------
* The cohorts already publish feature tables (LUMIERE: PyRadiomics;
  UPENN-GBM: CaPTk).  Those are the primary source for modelling, because
  they cover every patient without transferring tens of gigabytes.  This
  module exists to (a) *reproduce* a sample of the published features as a
  correctness check and (b) extract features the cohorts did not publish,
  including from a future institutional cohort.
* Settings mirror the LUMIERE release where possible (``original`` image type,
  fixed bin width) so extracted and published values are comparable; the
  exact settings are recorded in every output row.
* No resampling by default: LUMIERE's registered space is already consistent
  within a study, and resampling anisotropic clinical data silently changes
  texture features. Pass ``resample_spacing`` to opt in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SETTINGS: dict = {
    "binWidth": 25,
    "label": 1,
    "interpolator": "sitkBSpline",
    "resampledPixelSpacing": None,     # no resampling unless requested
    "normalize": False,
    "removeOutliers": None,
    "force2D": False,
    "geometryTolerance": 1e-4,
}

#: The exact settings LUMIERE used for its published feature tables, read
#: from ``diagnostics_Configuration_Settings`` in
#: ``LUMIERE-pyradiomics-*-features.csv`` rather than guessed.  Use this
#: preset whenever extracted values must be comparable with the published
#: ones: ``normalize``/``normalizeScale`` rescale intensities and
#: ``binWidth``/``voxelArrayShift`` set the discretisation, so an extraction
#: with different values is not comparable even on identical images.
#:
#: Published with PyRadiomics v3.0.1 / numpy 1.21.5 / SimpleITK 2.0.2.
LUMIERE_PUBLISHED_SETTINGS: dict = {
    "minimumROIDimensions": 2,
    "minimumROISize": None,
    "normalize": True,
    "normalizeScale": 100,
    "removeOutliers": None,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "preCrop": True,
    "padDistance": 5,
    "distances": [1],
    "force2D": False,
    "force2Ddimension": 0,
    "resegmentRange": None,
    "label": 1,
    "additionalInfo": True,
    "correctMask": True,
    "geometryTolerance": 1e-05,
    "binWidth": 5,
    "voxelArrayShift": 300,
}

SETTINGS_PRESETS: dict[str, dict] = {
    "default": DEFAULT_SETTINGS,
    "lumiere_published": LUMIERE_PUBLISHED_SETTINGS,
}


def settings_preset(name: str) -> dict:
    if name not in SETTINGS_PRESETS:
        raise ValueError(f"unknown settings preset {name!r}; "
                         f"known: {sorted(SETTINGS_PRESETS)}")
    return dict(SETTINGS_PRESETS[name])

DEFAULT_FEATURE_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm",
                           "gldm", "ngtdm")


def build_extractor(settings: dict | None = None,
                    feature_classes: tuple[str, ...] = DEFAULT_FEATURE_CLASSES,
                    resample_spacing: list[float] | None = None,
                    preset: str = "default"):
    """Construct a configured PyRadiomics extractor."""
    from radiomics import featureextractor

    merged = settings_preset(preset)
    if resample_spacing:
        merged["resampledPixelSpacing"] = list(resample_spacing)
    if settings:
        merged.update(settings)
    extractor = featureextractor.RadiomicsFeatureExtractor(**merged)
    extractor.disableAllFeatures()
    for name in feature_classes:
        extractor.enableFeatureClassByName(name)
    extractor._gbmplast_preset = preset
    return extractor


def extract_one(
    image_path: Path,
    mask_path: Path,
    *,
    label: int,
    extractor=None,
    settings: dict | None = None,
    resample_spacing: list[float] | None = None,
    preset: str = "default",
) -> dict:
    """Extract features for a single (image, mask, label).

    Returns a dict of feature name -> value plus ``diagnostics_*`` entries,
    or a row carrying ``extraction_error`` when PyRadiomics refuses (for
    example when the label is absent, which is a legitimate outcome).
    """
    import SimpleITK as sitk

    extractor = extractor or build_extractor(
        settings, resample_spacing=resample_spacing, preset=preset)
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    try:
        result = extractor.execute(image, mask, label=int(label))
    except Exception as exc:
        return {"extraction_error": f"{type(exc).__name__}: {exc}"}
    out: dict = {}
    for key, value in result.items():
        if isinstance(value, (np.ndarray,)) and value.size == 1:
            out[key] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def extract_study(
    image_paths: dict[str, Path],
    mask_path: Path,
    *,
    labels: dict[int, str],
    identifiers: dict,
    extractor=None,
    resample_spacing: list[float] | None = None,
) -> pd.DataFrame:
    """Extract for every (sequence, label) of one study."""
    extractor = extractor or build_extractor(resample_spacing=resample_spacing)
    rows = []
    for sequence, image_path in image_paths.items():
        if not Path(image_path).exists():
            continue
        for value, subregion in labels.items():
            features = extract_one(image_path, mask_path, label=value,
                                   extractor=extractor)
            rows.append({
                **identifiers,
                "sequence": sequence,
                "label": value,
                "subregion": subregion,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "resampled": bool(resample_spacing),
                "settings_preset": getattr(extractor, "_gbmplast_preset",
                                           "unspecified"),
                **features,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def extract_cohort_features(adapter, *, limit: int = 10,
                            sequences: tuple[str, ...] | None = None,
                            resample_spacing: list[float] | None = None
                            ) -> pd.DataFrame | None:
    if adapter.key == "lumiere":
        return extract_lumiere(adapter, limit=limit, sequences=sequences,
                               resample_spacing=resample_spacing)
    return None


def extract_lumiere(adapter, *, limit: int = 10,
                    sequences: tuple[str, ...] | None = None,
                    resample_spacing: list[float] | None = None,
                    preset: str = "lumiere_published",
                    ) -> pd.DataFrame | None:
    """Extract from locally-present LUMIERE studies in the registered space."""
    from gbmplast.adapters.lumiere import COHORT, SEGMENTATION_LABELS
    from gbmplast.schema import make_key, make_patient_id

    root = adapter.raw_dir / "images" / "Imaging"
    if not root.exists():
        return None
    sequences = sequences or ("CT1", "FLAIR", "T1", "T2")
    labels = SEGMENTATION_LABELS["HD-GLIO-AUTO"]
    # Default to the cohort's own published settings so locally extracted
    # features are directly comparable with the released tables.
    extractor = build_extractor(resample_spacing=resample_spacing,
                                preset=preset)

    frames: list[pd.DataFrame] = []
    count = 0
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            registered = study_dir / "HD-GLIO-AUTO-segmentation" / "registered"
            mask = registered / "segmentation.nii.gz"
            if not mask.exists():
                continue
            images = {s: registered / f"{s}_r2s_bet_reg.nii.gz"
                      for s in sequences}
            images = {s: p for s, p in images.items() if p.exists()}
            if not images:
                continue
            frames.append(extract_study(
                images, mask, labels=labels,
                identifiers={
                    "cohort": COHORT,
                    "patient_id": make_patient_id(COHORT, patient_dir.name),
                    "visit_id": make_key(COHORT, patient_dir.name, study_dir.name),
                    "local_patient_id": patient_dir.name,
                    "local_visit_id": study_dir.name,
                    "segmentation": "HD-GLIO-AUTO",
                    "space": "registered",
                },
                extractor=extractor))
            count += 1
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compare_with_published(
    extracted: pd.DataFrame,
    published: pd.DataFrame,
    *,
    features: list[str] | None = None,
    key_cols: tuple[str, ...] = ("visit_id", "sequence", "subregion"),
) -> pd.DataFrame:
    """Reconcile locally extracted features against the cohort's published ones.

    This is a correctness check on the extraction setup, not an analysis.
    Reports, per feature, the number of comparable rows, the median relative
    difference and the maximum absolute relative difference.
    """
    features = features or [
        "original_shape_VoxelVolume", "original_shape_MeshVolume",
        "original_shape_SurfaceArea", "original_shape_SurfaceVolumeRatio",
        "original_firstorder_Mean", "original_firstorder_Entropy",
    ]
    left = extracted.copy()
    right = published.copy()
    merged = left.merge(right, on=list(key_cols), suffixes=("_ext", "_pub"))
    rows = []
    for feature in features:
        a, b = f"{feature}_ext", f"{feature}_pub"
        if a not in merged.columns or b not in merged.columns:
            rows.append({"feature": feature, "n": 0,
                         "note": "absent from one of the tables"})
            continue
        x = pd.to_numeric(merged[a], errors="coerce")
        y = pd.to_numeric(merged[b], errors="coerce")
        ok = x.notna() & y.notna() & (y.abs() > 0)
        if not ok.any():
            rows.append({"feature": feature, "n": 0,
                         "note": "no comparable rows"})
            continue
        rel = ((x[ok] - y[ok]) / y[ok]).abs()
        rows.append({
            "feature": feature,
            "n": int(ok.sum()),
            "median_rel_diff": round(float(rel.median()), 6),
            "p95_rel_diff": round(float(rel.quantile(0.95)), 6),
            "max_rel_diff": round(float(rel.max()), 6),
            "n_within_1pct": int((rel <= 0.01).sum()),
            "note": "",
        })
    return pd.DataFrame(rows)
