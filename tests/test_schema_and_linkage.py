"""Patient-ID linkage, namespacing and canonical-schema conformance."""

from __future__ import annotations

import pandas as pd
import pytest

from gbmplast.schema import (
    CANONICAL_TABLES,
    SPECS,
    make_key,
    make_patient_id,
)
from tests.conftest import require_dataset


def test_patient_ids_are_namespaced():
    assert make_patient_id("LUMIERE", "Patient-001") == "LUMIERE::Patient-001"
    assert make_key("A", "b", "c") == "A::b::c"


def test_conform_adds_missing_columns_and_keeps_extras():
    spec = SPECS["visits"]
    df = pd.DataFrame({"visit_id": ["v"], "patient_id": ["p"],
                       "cohort_specific_thing": [1]})
    out = spec.conform(df)
    for column in spec.column_names:
        assert column in out.columns
    assert "cohort_specific_thing" in out.columns
    assert out.columns[0] == "visit_id"


def test_empty_frame_has_declared_dtypes():
    out = SPECS["outcomes"].empty()
    assert out.empty
    assert set(SPECS["outcomes"].column_names).issubset(out.columns)


@pytest.mark.requires_data
@pytest.mark.parametrize("dataset",
                         ["lumiere", "upenn_gbm", "ucsd_ptgbm", "gbm_care"])
def test_cohort_builds_and_validates(config, dataset):
    adapter = require_dataset(config, dataset)
    bundle = adapter.build()
    from gbmplast.validate import validate_bundle

    report = validate_bundle(bundle)
    assert report.ok, [f.message for f in report.errors]


@pytest.mark.requires_data
@pytest.mark.parametrize("dataset",
                         ["lumiere", "upenn_gbm", "ucsd_ptgbm", "gbm_care"])
def test_every_table_is_namespaced_to_its_cohort(config, dataset):
    adapter = require_dataset(config, dataset)
    bundle = adapter.build()
    cohort = adapter.describe().key
    for name in CANONICAL_TABLES:
        table = bundle.table(name)
        if table.empty or "patient_id" not in table.columns:
            continue
        ids = table["patient_id"].dropna().astype(str)
        assert ids.str.startswith(f"{cohort}::").all(), (
            f"{dataset}.{name} has ids outside the {cohort} namespace")


@pytest.mark.requires_data
def test_cross_cohort_join_yields_nothing(config, available_datasets):
    """The structural guarantee: cohorts cannot be joined by accident."""
    from gbmplast.adapters import get_adapter

    present = [k for k in ("lumiere", "upenn_gbm", "ucsd_ptgbm", "gbm_care")
               if available_datasets.get(k)]
    if len(present) < 2:
        pytest.skip("need at least two cohorts present")
    bundles = {k: get_adapter(k, config).build() for k in present}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            left = set(bundles[a].patients["patient_id"])
            right = set(bundles[b].patients["patient_id"])
            assert not (left & right), (
                f"{a} and {b} share patient ids; a cross-cohort join would "
                "silently succeed")


@pytest.mark.requires_data
@pytest.mark.parametrize("dataset",
                         ["lumiere", "upenn_gbm", "ucsd_ptgbm", "gbm_care"])
def test_foreign_keys_resolve(config, dataset):
    adapter = require_dataset(config, dataset)
    bundle = adapter.build()
    patients = set(bundle.patients["patient_id"])
    visits = set(bundle.visits["visit_id"])
    for name in ("visits", "imaging", "molecular_samples", "outcomes",
                 "treatments"):
        table = bundle.table(name)
        if table.empty:
            continue
        assert set(table["patient_id"].dropna()).issubset(patients)
        if name in {"imaging"} and "visit_id" in table.columns:
            assert set(table["visit_id"].dropna()).issubset(visits)
