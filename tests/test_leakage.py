"""Leakage: grouped splits, future information, outcome-derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbmplast.longitudinal.build import assert_no_future_features
from gbmplast.modeling.splits import (
    assert_group_disjoint,
    feasible_n_splits,
    patient_bootstrap_indices,
    split_indices,
)


def test_future_feature_guard_rejects_next_columns():
    with pytest.raises(ValueError, match="future information"):
        assert_no_future_features(["cur_enhancing__volume_mm3",
                                   "next_rano_progression"])
    with pytest.raises(ValueError, match="future information"):
        assert_no_future_features(["outcome_time_days"])
    assert_no_future_features(["cur_x", "prev_y", "slope_z"])  # no raise


def test_grouped_split_never_shares_a_patient(synthetic_longitudinal):
    df = synthetic_longitudinal
    folds = split_indices(df, df["next_rano_progression"], df["patient_id"],
                          n_splits=4, seed=1)
    assert_group_disjoint(df["patient_id"], folds)
    for train_idx, test_idx in folds:
        assert not (set(df["patient_id"].iloc[train_idx])
                    & set(df["patient_id"].iloc[test_idx]))


def test_group_disjoint_detects_a_broken_split():
    groups = pd.Series(["a", "a", "b", "b"])
    bad = [(np.array([0, 2]), np.array([1, 3]))]   # 'a' and 'b' on both sides
    with pytest.raises(AssertionError, match="both train and test"):
        assert_group_disjoint(groups, bad)


def test_every_row_is_tested_exactly_once(synthetic_longitudinal):
    df = synthetic_longitudinal
    folds = split_indices(df, df["next_rano_progression"], df["patient_id"],
                          n_splits=4, seed=1)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested) == list(range(len(df)))


def test_fold_count_shrinks_when_groups_are_scarce():
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    y = pd.Series([0, 0, 1, 1, 0, 1])
    assert feasible_n_splits(groups, y, 5) <= 3
    assert feasible_n_splits(groups, y, 5) >= 2


def test_patient_bootstrap_resamples_patients_not_rows():
    groups = pd.Series(np.repeat(["p1", "p2", "p3"], 4))
    draws = patient_bootstrap_indices(groups, n_bootstrap=50, seed=3)
    for idx in draws:
        picked = groups.iloc[idx]
        counts = picked.value_counts()
        # Every patient appears as a whole block of 4 rows, or not at all.
        assert set(counts.to_numpy()) <= {4, 8, 12}


def test_evaluator_refuses_a_future_feature(synthetic_longitudinal):
    from gbmplast.modeling.evaluate import evaluate_classifier
    from gbmplast.modeling.models import elastic_net_classifier

    with pytest.raises(ValueError, match="future information"):
        evaluate_classifier(
            synthetic_longitudinal,
            feature_cols=["cur_enhancing__volume_mm3",
                          "next_enhancing__volume_mm3"],
            outcome_col="next_rano_progression",
            model_name="leaky", estimator=elastic_net_classifier(),
            n_splits=3, n_bootstrap=0)


def test_preprocessing_is_fitted_inside_the_fold(synthetic_longitudinal):
    """An imputer fitted on the whole dataset would leak test statistics."""
    from sklearn.pipeline import Pipeline

    from gbmplast.modeling.models import build_pipeline, elastic_net_classifier

    df = synthetic_longitudinal.copy()
    pipeline = build_pipeline(["cur_enhancing__volume_mm3"],
                              elastic_net_classifier())
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["prep", "model"]
    # The preprocessor must be unfitted before use.
    with pytest.raises(Exception):
        pipeline.named_steps["prep"].transform(df[["cur_enhancing__volume_mm3"]])


def test_baselines_do_not_retain_state_across_folds():
    from gbmplast.modeling.evaluate import clone_baseline
    from gbmplast.modeling.models import LastObservationClassifier

    original = LastObservationClassifier("index_rano_status")
    original.fit(pd.DataFrame({"index_rano_status": ["PD", "SD"]}), [1, 0])
    assert original.rates_
    fresh = clone_baseline(original)
    assert fresh.rates_ == {}
    assert fresh.column == "index_rano_status"


@pytest.mark.requires_data
def test_longitudinal_predictors_exclude_next_visit_columns(config):
    from gbmplast.longitudinal.build import build_longitudinal
    from tests.conftest import require_dataset

    adapter = require_dataset(config, "lumiere")
    result = build_longitudinal(adapter, config)
    for group in ("clinical", "imaging", "dynamics", "history"):
        for column in result.feature_groups.get(group, []):
            assert not column.startswith("next_")
            assert not column.startswith("outcome_")


@pytest.mark.requires_data
def test_index_features_do_not_depend_on_the_next_visit(config):
    """Perturb ONLY each patient's final visit; predictors must not move.

    A patient's final dated visit is never an index visit -- it can only
    ever be the *next* visit of the last transition. So if any predictor
    changed when its volume is altered, information would be flowing
    backwards from the outcome into the features.
    """
    from gbmplast.longitudinal.build import build_transitions, visit_imaging_summary
    from tests.conftest import require_dataset

    adapter = require_dataset(config, "lumiere")
    bundle = adapter.build()
    radiomics = adapter.load_radiomics("HD-GLIO-AUTO")
    summary = visit_imaging_summary(radiomics,
                                    subregions=("enhancing", "non_enhancing"),
                                    primary_sequence="CT1")
    base, _, groups = build_transitions(
        bundle.visits, summary, bundle.patients,
        subregions=("enhancing", "non_enhancing"))
    assert not base.empty

    # The last dated visit of each patient.
    visits = bundle.visits.copy()
    visits["_t"] = pd.to_numeric(visits["relative_time_days"], errors="coerce")
    final = (visits[visits["_t"].notna()]
             .sort_values("_t").groupby("patient_id").tail(1)["visit_id"])
    final_set = set(final)

    perturbed_summary = summary.copy()
    mask = perturbed_summary["visit_id"].isin(final_set)
    assert mask.any()
    for column in ("enhancing__volume_mm3", "non_enhancing__volume_mm3"):
        perturbed_summary.loc[mask, column] = (
            pd.to_numeric(perturbed_summary.loc[mask, column],
                          errors="coerce") * 7.0 + 5000.0)

    perturbed, _, _ = build_transitions(
        bundle.visits, perturbed_summary, bundle.patients,
        subregions=("enhancing", "non_enhancing"))

    predictors = [c for g in groups.values() for c in g]
    assert predictors
    assert all(not c.startswith(("next_", "outcome_")) for c in predictors)

    key = ["patient_id", "index_visit_id"]
    a = base.set_index(key).sort_index()
    b = perturbed.set_index(key).sort_index()
    assert a.index.equals(b.index), "perturbation changed the transition set"

    numeric = [c for c in predictors
               if c in a.columns and pd.api.types.is_numeric_dtype(a[c])]
    assert numeric
    for column in numeric:
        left = pd.to_numeric(a[column], errors="coerce")
        right = pd.to_numeric(b[column], errors="coerce")
        assert left.equals(right) or np.allclose(
            left.fillna(-999.0), right.fillna(-999.0), rtol=1e-9, atol=1e-9), (
            f"predictor {column!r} changed when only the FINAL visit's "
            "volume was perturbed -- information is leaking backwards")

    # Sanity: the outcome columns DID move, so the perturbation was real.
    moved = False
    for column in ("next_enhancing__volume_mm3", "next_log_ratio_enhancing"):
        if column in a.columns:
            left = pd.to_numeric(a[column], errors="coerce").fillna(-999.0)
            right = pd.to_numeric(b[column], errors="coerce").fillna(-999.0)
            moved = moved or not np.allclose(left, right, rtol=1e-9, atol=1e-9)
    assert moved, "the perturbation did not change any outcome; test is inert"
