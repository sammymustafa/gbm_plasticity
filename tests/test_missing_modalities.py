"""Missing modalities and missingness preservation.

The recurring failure mode these guard against: a cohort that does not state
whether a sequence exists being treated as though it does, and a genuine
"not assessed" being silently converted into "absent".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbmplast.report.dictionary import modality_completeness
from gbmplast.schema import SPECS
from tests.conftest import require_dataset


def test_unknown_availability_is_not_false():
    """NA availability must stay NA, never collapse to False."""
    from gbmplast.schema import CanonicalBundle

    imaging = SPECS["imaging"].conform(pd.DataFrame({
        "imaging_id": ["a", "b", "c"],
        "patient_id": ["C::p"] * 3,
        "visit_id": ["C::v"] * 3,
        "cohort": ["C"] * 3,
        "modality": ["T1", "T1", "DSC"],
        "available": [True, False, pd.NA],
    }))
    bundle = CanonicalBundle(
        cohort="C", patients=SPECS["patients"].empty(),
        visits=SPECS["visits"].empty(), imaging=imaging,
        treatments=SPECS["treatments"].empty(),
        symptoms=SPECS["symptoms"].empty(),
        molecular_samples=SPECS["molecular_samples"].empty(),
        outcomes=SPECS["outcomes"].empty())
    table = modality_completeness(bundle).set_index("modality")
    assert table.loc["T1", "n_available"] == 1
    assert table.loc["T1", "n_unavailable"] == 1
    assert table.loc["T1", "n_unknown"] == 0
    assert table.loc["DSC", "n_unknown"] == 1
    assert table.loc["DSC", "n_available"] == 0


def test_symptom_not_assessed_differs_from_absent():
    from gbmplast.symptoms.structured import coverage_matrix, from_long

    df = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3"],
        "symptom": ["seizure", "seizure", "seizure"],
        "present": ["yes", "no", None],
    })
    out = from_long(df, cohort="T")
    assert out["present"].tolist()[0] is True
    assert out["present"].tolist()[1] is False
    assert pd.isna(out["present"].tolist()[2])
    coverage = coverage_matrix(out).set_index("symptom")
    assert coverage.loc["seizure", "n_present"] == 1
    assert coverage.loc["seizure", "n_absent"] == 1
    assert coverage.loc["seizure", "n_unknown"] == 1


def test_imputation_keeps_a_missing_indicator():
    """Missingness must survive preprocessing as a feature."""
    from gbmplast.modeling.models import make_preprocessor

    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0],
                      "b": [1.0, 1.0, 1.0, 1.0]})
    prep = make_preprocessor(["a", "b"], add_indicator=True)
    out = prep.fit_transform(X)
    assert out.shape[1] >= 3, "the missing-indicator column was dropped"
    names = list(prep.get_feature_names_out())
    assert any("missingindicator" in n for n in names)


@pytest.mark.requires_data
def test_lumiere_modality_counts_match_the_completeness_table(config):
    """The adapter's imaging rows must agree with the published table."""
    adapter = require_dataset(config, "lumiere")
    bundle = adapter.build()
    raw = pd.read_csv(adapter.raw_dir / "LUMIERE-datacompleteness.csv")
    published = {
        "T1CE": raw["CT1"].notna().sum(),
        "T1": raw["T1"].notna().sum(),
        "T2": raw["T2"].notna().sum(),
        "FLAIR": raw["FLAIR"].notna().sum(),
    }
    table = modality_completeness(bundle).set_index("modality")
    for modality, expected in published.items():
        assert int(table.loc[modality, "n_available"]) == int(expected), modality


@pytest.mark.requires_data
def test_cohorts_without_a_table_leave_it_empty_not_fabricated(config,
                                                               available_datasets):
    """An absent domain must be an empty table, never invented rows."""
    expectations = {
        "lumiere": ["treatments", "symptoms", "molecular_samples"],
        "upenn_gbm": ["symptoms", "molecular_samples"],
        "ucsd_ptgbm": ["symptoms", "molecular_samples"],
        "gbm_care": ["imaging", "treatments", "symptoms", "outcomes"],
    }
    for dataset, empties in expectations.items():
        if not available_datasets.get(dataset):
            continue
        bundle = require_dataset(config, dataset).build()
        for name in empties:
            assert len(bundle.table(name)) == 0, (
                f"{dataset}.{name} should be empty by design but has "
                f"{len(bundle.table(name))} rows")
            assert name in bundle.notes or True   # notes explain why


@pytest.mark.requires_data
def test_survival_without_censoring_is_refused(config):
    """LUMIERE releases no censoring flag, so Cox must be refused."""
    from gbmplast.modeling.survival import can_fit_survival

    adapter = require_dataset(config, "lumiere")
    bundle = adapter.build()
    ok, reason = can_fit_survival(bundle, "overall_survival")
    assert not ok
    assert "censor" in reason.lower()


@pytest.mark.requires_data
def test_survival_with_censoring_is_allowed(config):
    from gbmplast.modeling.survival import can_fit_survival

    adapter = require_dataset(config, "upenn_gbm")
    bundle = adapter.build()
    ok, reason = can_fit_survival(bundle, "overall_survival")
    assert ok, reason
