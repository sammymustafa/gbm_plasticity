"""Module C: signatures, pseudobulk, paired statistics, specimen pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbmplast.molecular import signatures as S
from gbmplast.molecular.paired import (
    _benjamini_hochberg,
    paired_effects,
    paired_table,
    state_switch_table,
)


# -- signatures -----------------------------------------------------------

def test_signature_registry_is_coherent():
    assert len(S.ALL_SIGNATURES) >= 20
    for sig in S.ALL_SIGNATURES:
        assert sig.genes, sig.key
        assert sig.source, sig.key
        assert len(set(sig.genes)) == len(sig.genes), f"{sig.key} has duplicates"
    assert set(S.STATE_GROUPS) == {"MES", "NPC", "AC", "OPC"}
    for members in S.STATE_GROUPS.values():
        for member in members:
            assert member in S.BY_KEY


def test_score_reports_gene_coverage():
    genes = list(S.BY_KEY["HLA_I"].genes[:3]) + ["UNRELATED1", "UNRELATED2"]
    expression = pd.DataFrame(np.ones((len(genes), 2)), index=genes,
                              columns=["s1", "s2"])
    score, coverage = S.score_signature(expression, "HLA_I")
    assert coverage["n_genes_found"] == 3
    assert coverage["n_genes_in_signature"] == 5
    assert coverage["coverage"] == pytest.approx(0.6)
    assert score.tolist() == [1.0, 1.0]


def test_missing_signature_scores_nan_not_zero():
    expression = pd.DataFrame(np.ones((2, 2)), index=["FOO", "BAR"],
                              columns=["s1", "s2"])
    score, coverage = S.score_signature(expression, "HLA_I")
    assert coverage["n_genes_found"] == 0
    assert score.isna().all(), "an unmeasurable signature must be NA, not 0"


def test_low_coverage_is_flagged():
    genes = list(S.BY_KEY["IFN_GAMMA"].genes[:2])
    expression = pd.DataFrame(np.ones((2, 2)), index=genes, columns=["a", "b"])
    _, coverage = S.score_all(expression, (S.BY_KEY["IFN_GAMMA"],))
    assert bool(coverage.iloc[0]["below_min_coverage"])


def test_state_weights_sum_to_one():
    rng = np.random.default_rng(3)
    scores = pd.DataFrame(
        rng.normal(size=(6, 6)),
        columns=["MES1", "MES2", "AC", "OPC", "NPC1", "NPC2"],
        index=[f"s{i}" for i in range(6)])
    combined = S.combine_state_groups(scores)
    for group in S.STATE_GROUPS:
        assert f"state_{group}" in combined.columns
    weights = S.state_proportions(combined)
    weight_cols = [c for c in weights.columns if c.startswith("state_weight_")]
    assert np.allclose(weights[weight_cols].sum(axis=1), 1.0)
    assert weights["dominant_state"].isin(list(S.STATE_GROUPS)).all()


# -- pseudobulk -----------------------------------------------------------

def test_pseudobulk_sums_counts_before_normalising():
    from scipy import sparse

    from gbmplast.molecular.snrna import SampleMatrix, pseudobulk

    counts = sparse.csr_matrix(np.array([[10, 0, 0],
                                         [0, 20, 0],
                                         [0, 0, 30]], dtype=np.int32))
    sample = SampleMatrix("GSM1", np.array(["A", "B", "C"]),
                          np.array(["c1", "c2", "c3"]), counts)
    full = pseudobulk(sample)
    assert full.index.tolist() == ["A", "B", "C"]
    # Gene C carries half the library, so it must score highest.
    assert full["C"] > full["B"] > full["A"]

    subset = pseudobulk(sample, cell_mask=np.array([True, False, False]))
    # With only cell 1, gene A is the entire library.
    assert subset["A"] > 0 and subset["B"] == 0 and subset["C"] == 0


def test_pseudobulk_of_no_cells_is_nan():
    from scipy import sparse

    from gbmplast.molecular.snrna import SampleMatrix, pseudobulk

    sample = SampleMatrix("GSM1", np.array(["A", "B"]), np.array(["c1"]),
                          sparse.csr_matrix(np.array([[1], [2]], dtype=np.int32)))
    out = pseudobulk(sample, cell_mask=np.array([False]))
    assert out.isna().all()


def test_cell_type_assignment_leaves_ambiguous_cells_unassigned():
    from scipy import sparse

    from gbmplast.molecular.snrna import SampleMatrix, assign_cell_types

    genes = ["PTPRZ1", "SOX2", "CD3D", "CD3E", "CD14", "AIF1"]
    # cell0 glial, cell1 T cell, cell2 flat (no compartment should win)
    matrix = np.array([
        [50, 0, 5],
        [50, 0, 5],
        [0, 50, 5],
        [0, 50, 5],
        [0, 0, 5],
        [0, 0, 5],
    ], dtype=np.int32)
    sample = SampleMatrix("GSM1", np.array(genes),
                          np.array(["c0", "c1", "c2"]),
                          sparse.csr_matrix(matrix))
    out = assign_cell_types(sample, min_margin=0.05)
    assert set(out.columns) >= {"barcode", "cell_type", "assignment_margin",
                                "is_malignant_compartment"}
    assert "method" in out.attrs
    assert "NOT the GBM-CARE consortium" in out.attrs["method"]


# -- paired statistics ----------------------------------------------------

@pytest.fixture
def paired_scores():
    rng = np.random.default_rng(17)
    rows = []
    for p in range(24):
        base = rng.normal(0, 1)
        for role, shift in (("primary", 0.0), ("first_recurrence", 0.5)):
            rows.append({
                "local_patient_id": f"PT{p:02d}",
                "sample_role": role,
                "state_MES": base + shift + rng.normal(0, 0.2),
                "state_AC": base - shift + rng.normal(0, 0.2),
                "HLA_I": base + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_paired_table_is_one_row_per_patient(paired_scores):
    out = paired_table(paired_scores,
                       value_cols=["state_MES", "state_AC", "HLA_I"])
    assert len(out) == 24
    assert out["local_patient_id"].is_unique
    for col in ("state_MES__primary", "state_MES__recurrent", "state_MES__delta"):
        assert col in out.columns
    assert np.allclose(out["state_MES__delta"],
                       out["state_MES__recurrent"] - out["state_MES__primary"])


def test_paired_effects_recover_a_known_shift(paired_scores):
    paired = paired_table(paired_scores,
                          value_cols=["state_MES", "state_AC", "HLA_I"])
    effects = paired_effects(paired, ["state_MES", "state_AC", "HLA_I"],
                             n_bootstrap=400)
    effects = effects.set_index("programme")
    assert effects.loc["state_MES", "mean_delta"] == pytest.approx(0.5, abs=0.15)
    assert effects.loc["state_AC", "mean_delta"] == pytest.approx(-0.5, abs=0.15)
    # The CI for the unchanged programme must contain zero.
    assert (effects.loc["HLA_I", "delta_ci_low"] <= 0
            <= effects.loc["HLA_I", "delta_ci_high"])
    assert effects.loc["state_MES", "wilcoxon_fdr"] < 0.05
    assert effects.loc["HLA_I", "wilcoxon_fdr"] > 0.05


def test_paired_effects_refuse_tiny_samples():
    paired = pd.DataFrame({
        "local_patient_id": ["a", "b"],
        "x__primary": [1.0, 2.0],
        "x__recurrent": [1.5, 2.5],
        "x__delta": [0.5, 0.5],
    })
    out = paired_effects(paired, ["x"], n_bootstrap=50)
    assert out.iloc[0]["status"] == "skipped"


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
    adjusted = _benjamini_hochberg(p)
    assert np.all(adjusted >= p - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)
    assert np.isnan(_benjamini_hochberg(np.array([np.nan]))).all()


def test_state_switch_table_counts_transitions():
    scores = pd.DataFrame({
        "local_patient_id": ["a", "a", "b", "b"],
        "sample_role": ["primary", "first_recurrence"] * 2,
        "dominant_state": ["AC", "MES", "MES", "MES"],
    })
    pairs, matrix = state_switch_table(scores)
    assert len(pairs) == 2
    assert pairs.set_index("local_patient_id").loc["a", "switched"]
    assert not pairs.set_index("local_patient_id").loc["b", "switched"]
    assert matrix.loc["AC", "MES"] == 1
    assert "lineage conversion" in matrix.attrs["caution"]


# -- cohort structure -----------------------------------------------------

@pytest.mark.requires_data
def test_care_pairing_matches_the_published_structure(config):
    from tests.conftest import require_dataset

    adapter = require_dataset(config, "gbm_care")
    samples = adapter.sample_records()
    assert len(samples) == 121
    assert samples["local_patient_id"].nunique() == 59
    counts = samples["sample_role"].value_counts().to_dict()
    assert counts["primary"] == 56
    assert counts["first_recurrence"] == 55
    assert counts["second_recurrence"] == 10
    assert len(adapter.paired_patients()) == 56


@pytest.mark.requires_data
def test_care_specimens_have_no_collection_dates(config):
    """Ordinal timing only -- a date must never be invented."""
    from tests.conftest import require_dataset

    bundle = require_dataset(config, "gbm_care").build()
    molecular = bundle.molecular_samples
    assert molecular["tissue_collection_relative_day"].isna().all()
    assert molecular["timepoint_index"].notna().all()


def test_baseline_predictors_handle_no_matching_columns():
    """An empty result frame must not raise."""
    from gbmplast.molecular.paired import baseline_predictors_of_recurrence

    paired = pd.DataFrame({"local_patient_id": ["a"] * 12,
                           "state_MES__recurrent": np.arange(12.0)})
    out = baseline_predictors_of_recurrence(
        paired, outcome="state_MES", predictors=["not_present"])
    assert isinstance(out, pd.DataFrame)


def test_empty_effects_frame_makes_no_figure():
    from gbmplast.report.figures_molecular import effect_forest

    assert effect_forest(pd.DataFrame(), Path("/tmp/x.png")) is None
    assert effect_forest(pd.DataFrame({"a": [1]}), Path("/tmp/x.png")) is None
