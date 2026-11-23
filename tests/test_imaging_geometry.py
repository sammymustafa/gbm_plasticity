"""Masking geometry: orientation, spacing, grid identity, label mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from gbmplast.imaging import io, qc  # noqa: E402


def _write(tmp_path: Path, name: str, array: np.ndarray,
           affine: np.ndarray, dtype=np.float32) -> Path:
    img = nib.Nifti1Image(array.astype(dtype), affine)
    path = tmp_path / name
    nib.save(img, str(path))
    return path


def _affine(spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = spacing
    return a


@pytest.fixture
def image_and_mask(tmp_path):
    rng = np.random.default_rng(5)
    image = rng.normal(100, 15, (24, 24, 12))
    mask = np.zeros((24, 24, 12), dtype=np.uint8)
    mask[6:14, 6:14, 3:8] = 1
    mask[9:12, 9:12, 4:6] = 2
    a = _affine()
    return (_write(tmp_path, "img.nii.gz", image, a),
            _write(tmp_path, "mask.nii.gz", mask, a, dtype=np.uint8))


def test_geometry_reports_shape_spacing_orientation(image_and_mask):
    image_path, _ = image_and_mask
    g = io.geometry(image_path)
    assert g.shape == (24, 24, 12)
    assert g.spacing == (1.0, 1.0, 1.0)
    assert g.is_isotropic
    assert len(g.orientation) == 3


def test_same_grid_detects_shape_and_affine_mismatch(tmp_path):
    a = _write(tmp_path, "a.nii.gz", np.zeros((10, 10, 5)), _affine())
    b = _write(tmp_path, "b.nii.gz", np.zeros((10, 10, 6)), _affine())
    c = _write(tmp_path, "c.nii.gz", np.zeros((10, 10, 5)),
               _affine((2.0, 2.0, 2.0)))
    assert io.same_grid(a, a)[0]
    ok, reasons = io.same_grid(a, b)
    assert not ok and any("shape" in r for r in reasons)
    ok, reasons = io.same_grid(a, c)
    assert not ok and any("affine" in r for r in reasons)


def test_label_volumes_use_voxel_spacing(tmp_path):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1             # 8 voxels
    path = _write(tmp_path, "m.nii.gz", mask, _affine((2.0, 2.0, 2.0)),
                  dtype=np.uint8)
    volumes = io.label_volumes(path, labels={1: "enhancing"})
    assert volumes["enhancing"] == pytest.approx(8 * 8.0)   # 8 vox x 8 mm^3


def test_qc_passes_a_well_formed_pair(image_and_mask):
    image_path, mask_path = image_and_mask
    result = qc.qc_image_mask_pair(
        image_path, mask_path, cohort="T", patient="p", study="s",
        sequence="CT1", expected_labels={1: "non_enhancing", 2: "enhancing"})
    assert result.status == "pass", result.findings
    assert result.metrics["grid_match"] is True
    assert result.metrics["n_voxels_enhancing"] > 0


def test_qc_fails_on_grid_mismatch(tmp_path, image_and_mask):
    image_path, _ = image_and_mask
    wrong = np.zeros((12, 12, 6), dtype=np.uint8)
    wrong[2:5, 2:5, 1:3] = 1
    wrong_path = _write(tmp_path, "wrong.nii.gz", wrong,
                        _affine((2.0, 2.0, 2.0)), dtype=np.uint8)
    result = qc.qc_image_mask_pair(
        image_path, wrong_path, cohort="T", patient="p", study="s",
        sequence="CT1", expected_labels={1: "a"})
    assert result.status == "fail"
    assert any("grid mismatch" in f for f in result.findings)


def test_qc_flags_undocumented_labels(image_and_mask, tmp_path):
    image_path, _ = image_and_mask
    mask = np.zeros((24, 24, 12), dtype=np.uint8)
    mask[5:9, 5:9, 3:6] = 9          # not in the documented label set
    path = _write(tmp_path, "odd.nii.gz", mask, _affine(), dtype=np.uint8)
    result = qc.qc_image_mask_pair(
        image_path, path, cohort="T", patient="p", study="s", sequence="CT1",
        expected_labels={1: "non_enhancing", 2: "enhancing"})
    assert result.status == "fail"
    assert any("unexpected mask labels" in f for f in result.findings)


def test_qc_warns_on_anisotropy(tmp_path):
    rng = np.random.default_rng(7)
    image = rng.normal(100, 10, (20, 20, 6))
    mask = np.zeros((20, 20, 6), dtype=np.uint8)
    mask[5:10, 5:10, 2:4] = 1
    a = _affine((0.5, 0.5, 5.0))         # 10:1
    image_path = _write(tmp_path, "i.nii.gz", image, a)
    mask_path = _write(tmp_path, "m.nii.gz", mask, a, dtype=np.uint8)
    result = qc.qc_image_mask_pair(image_path, mask_path, cohort="T",
                                   patient="p", study="s", sequence="CT1",
                                   expected_labels={1: "a"})
    assert result.status == "warn"
    assert any("anisotropic" in f for f in result.findings)
    assert result.metrics["anisotropy_ratio"] == pytest.approx(10.0)


def test_qc_warns_when_the_mask_touches_the_boundary(tmp_path):
    rng = np.random.default_rng(9)
    image = rng.normal(100, 10, (12, 12, 6))
    mask = np.zeros((12, 12, 6), dtype=np.uint8)
    mask[0:4, 0:4, 0:2] = 1              # touches three minima
    a = _affine()
    result = qc.qc_image_mask_pair(
        _write(tmp_path, "i.nii.gz", image, a),
        _write(tmp_path, "m.nii.gz", mask, a, dtype=np.uint8),
        cohort="T", patient="p", study="s", sequence="CT1",
        expected_labels={1: "a"})
    assert any("field-of-view boundary" in f for f in result.findings)


def test_qc_warns_on_an_empty_mask(tmp_path):
    rng = np.random.default_rng(13)
    a = _affine()
    result = qc.qc_image_mask_pair(
        _write(tmp_path, "i.nii.gz", rng.normal(size=(8, 8, 4)), a),
        _write(tmp_path, "m.nii.gz", np.zeros((8, 8, 4), dtype=np.uint8), a,
               dtype=np.uint8),
        cohort="T", patient="p", study="s", sequence="CT1",
        expected_labels={1: "a"})
    assert any("mask is empty" in f for f in result.findings)


@pytest.mark.requires_data
def test_real_lumiere_masks_match_their_images(config):
    """Registered LUMIERE images and masks must share a grid."""
    from tests.conftest import require_dataset

    adapter = require_dataset(config, "lumiere")
    root = adapter.raw_dir / "images" / "Imaging"
    if not root.exists():
        pytest.skip("no LUMIERE images downloaded")
    checked = 0
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            registered = study_dir / "HD-GLIO-AUTO-segmentation" / "registered"
            mask = registered / "segmentation.nii.gz"
            image = registered / "CT1_r2s_bet_reg.nii.gz"
            if not (mask.exists() and image.exists()):
                continue
            ok, reasons = io.same_grid(image, mask)
            assert ok, f"{study_dir}: {reasons}"
            labels = set(io.label_voxel_counts(mask))
            assert labels <= {1, 2}, f"{study_dir}: unexpected labels {labels}"
            checked += 1
            if checked >= 12:
                return
    if checked == 0:
        pytest.skip("no registered LUMIERE pairs found")
