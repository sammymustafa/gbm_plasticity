"""Nested, patient-grouped cross-validation with leakage checks.

Shape of an evaluation
----------------------
Outer loop  -- patient-grouped (stratified where the outcome is binary);
               produces out-of-fold predictions for every row.
Inner loop  -- patient-grouped within the training fold only; selects
               features and hyperparameters.

Everything that learns from data -- imputation statistics, scaling,
univariate feature selection, hyperparameters -- is fitted inside the
training fold.  Nothing is fitted on the full dataset before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline

from gbmplast.longitudinal.build import assert_no_future_features
from gbmplast.modeling import metrics as M
from gbmplast.modeling.models import (
    build_pipeline,
    make_preprocessor,
)
from gbmplast.modeling.splits import (
    assert_group_disjoint,
    feasible_n_splits,
    fold_summary,
    split_indices,
)


@dataclass
class EvalResult:
    model: str
    task: str
    n_rows: int
    n_patients: int
    predictions: pd.DataFrame
    metrics: dict
    metric_ci: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    folds: pd.DataFrame | None = None
    selected_features: pd.DataFrame | None = None
    best_params: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def metric_row(self) -> dict:
        row = {"model": self.model, "task": self.task,
               "n_rows": self.n_rows, "n_patients": self.n_patients}
        row.update({k: v for k, v in self.metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating))})
        return row


# --------------------------------------------------------------------------
# feature-selection helper (fitted inside the fold)
# --------------------------------------------------------------------------

def _max_features(n_train: int, n_events: int | None, *,
                  min_per_feature: int = 5, cap: int = 25) -> int:
    """Cap feature count by events-per-variable, not by wishful thinking."""
    budget = (n_events if n_events is not None else n_train) // min_per_feature
    return int(max(2, min(cap, budget)))


def build_selection_pipeline(
    numeric_cols: list[str],
    estimator,
    *,
    task: str,
    k: int,
    add_indicator: bool = True,
) -> Pipeline:
    score_fn = f_classif if task == "classification" else f_regression
    return Pipeline([
        ("prep", make_preprocessor(numeric_cols, add_indicator=add_indicator)),
        ("select", SelectKBest(score_func=score_fn, k=min(k, len(numeric_cols)))),
        ("model", estimator),
    ])


# --------------------------------------------------------------------------
# main evaluators
# --------------------------------------------------------------------------

def evaluate_classifier(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    outcome_col: str,
    group_col: str = "patient_id",
    model_name: str,
    estimator=None,
    baseline=None,
    param_grid: dict | None = None,
    n_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20251123,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 777,
    select_features: bool = True,
    min_events_per_feature: int = 5,
    max_features: int = 25,
) -> EvalResult:
    """Out-of-fold evaluation of one classifier (or a fitted-per-fold baseline)."""
    assert_no_future_features(feature_cols)

    work = data.dropna(subset=[outcome_col, group_col]).reset_index(drop=True)
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    groups = work[group_col]
    notes: list[str] = []

    if baseline is not None:
        X = work
    else:
        missing = [c for c in feature_cols if c not in work.columns]
        if missing:
            raise KeyError(f"features absent from the table: {missing[:5]}")
        X = work[feature_cols]
        all_nan = [c for c in feature_cols if X[c].isna().all()]
        if all_nan:
            notes.append(f"{len(all_nan)} features are entirely missing and "
                         "were dropped")
            feature_cols = [c for c in feature_cols if c not in all_nan]
            X = work[feature_cols]

    folds = split_indices(X, y, groups, n_splits=n_splits, seed=seed,
                          stratify=True)
    assert_group_disjoint(groups, folds)
    notes.append(f"{len(folds)} outer folds, grouped by {group_col}")

    oof = np.full(len(work), np.nan, dtype=float)
    selected_rows: list[dict] = []
    best_params: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            notes.append(f"fold {fold_i}: single-class training set, skipped")
            continue

        if baseline is not None:
            model = clone_baseline(baseline)
            model.fit(X.iloc[train_idx], y_train)
            oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
            continue

        n_events = int(min(y_train.sum(), len(y_train) - y_train.sum()))
        k = _max_features(len(train_idx), n_events,
                          min_per_feature=min_events_per_feature,
                          cap=max_features)
        pipeline = (build_selection_pipeline(feature_cols, clone(estimator),
                                             task="classification", k=k)
                    if select_features else
                    build_pipeline(feature_cols, clone(estimator)))

        if param_grid:
            inner_n = feasible_n_splits(groups.iloc[train_idx], y_train,
                                        inner_splits)
            inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True,
                                         random_state=seed)
            search = GridSearchCV(pipeline, param_grid, scoring="roc_auc",
                                  cv=inner, n_jobs=1, refit=True,
                                  error_score=np.nan)
            search.fit(X.iloc[train_idx], y_train,
                       groups=groups.iloc[train_idx])
            fitted = search.best_estimator_
            best_params.append({"fold": fold_i, **search.best_params_})
        else:
            fitted = pipeline.fit(X.iloc[train_idx], y_train)

        oof[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        selected_rows += _record_selected(fitted, feature_cols, fold_i)

    predictions = pd.DataFrame({
        group_col: groups,
        "y_true": y.to_numpy(dtype=float),
        "y_prob": oof,
    })
    for col in ("index_visit_id", "next_visit_id", "local_patient_id"):
        if col in work.columns:
            predictions[col] = work[col].to_numpy()

    ok = np.isfinite(oof)
    point = M.binary_metrics(y[ok], oof[ok])
    point.update(M.calibration_metrics(y[ok], oof[ok]))

    # A predictor that is CONSTANT WITHIN each fold (e.g. a prevalence
    # baseline) has no discrimination by construction, but pooling
    # out-of-fold predictions across folds with different training
    # prevalences produces a handful of distinct values and therefore an
    # AUROC that drifts off 0.5. That number reflects between-fold
    # prevalence variation, not discrimination, so say so rather than
    # letting a reader interpret 0.43 as "worse than chance".
    n_distinct_overall = int(pd.Series(oof[ok]).round(10).nunique())
    constant_within_folds = True
    for _, test_idx in folds:
        vals = pd.Series(oof[test_idx]).dropna().round(10)
        if vals.nunique() > 1:
            constant_within_folds = False
            break
    if constant_within_folds and n_distinct_overall <= len(folds):
        point["discrimination_defined"] = False
        notes.append(
            "predictions are constant within each fold, so this model has no "
            "discrimination by construction; the pooled AUROC/calibration "
            "slope reflect between-fold prevalence variation only and should "
            "be read as 'undefined (~0.5)'.")
    else:
        point["discrimination_defined"] = True
    ci = (M.bootstrap_ci(y[ok], oof[ok], groups[ok],
                         metric_fn=M.binary_metrics,
                         metric_keys=("auroc", "auprc", "brier",
                                      "sensitivity", "specificity"),
                         n_bootstrap=n_bootstrap, seed=bootstrap_seed)
          if ok.sum() >= 20 else None)
    calibration = M.calibration_table(y[ok], oof[ok])

    return EvalResult(
        model=model_name, task="classification",
        n_rows=int(ok.sum()), n_patients=int(groups[ok].nunique()),
        predictions=predictions, metrics=point, metric_ci=ci,
        calibration=calibration,
        folds=fold_summary(y, groups, folds),
        selected_features=(pd.DataFrame(selected_rows)
                           if selected_rows else None),
        best_params=best_params, notes=notes,
    )


def evaluate_regressor(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    outcome_col: str,
    group_col: str = "patient_id",
    model_name: str,
    estimator=None,
    baseline=None,
    param_grid: dict | None = None,
    n_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20251123,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 777,
    select_features: bool = True,
    max_features: int = 25,
) -> EvalResult:
    assert_no_future_features(feature_cols)

    work = data.dropna(subset=[outcome_col, group_col]).reset_index(drop=True)
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work[y.notna()].reset_index(drop=True)
    y = y[y.notna()].reset_index(drop=True)
    groups = work[group_col]
    notes: list[str] = []

    if baseline is not None:
        X = work
    else:
        X = work[feature_cols]
        all_nan = [c for c in feature_cols if X[c].isna().all()]
        if all_nan:
            notes.append(f"{len(all_nan)} features entirely missing, dropped")
            feature_cols = [c for c in feature_cols if c not in all_nan]
            X = work[feature_cols]

    folds = split_indices(X, y, groups, n_splits=n_splits, seed=seed,
                          stratify=False)
    assert_group_disjoint(groups, folds)
    notes.append(f"{len(folds)} outer folds, grouped by {group_col}")

    oof = np.full(len(work), np.nan, dtype=float)
    selected_rows: list[dict] = []
    best_params: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if baseline is not None:
            model = clone_baseline(baseline)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[test_idx] = model.predict(X.iloc[test_idx])
            continue

        k = _max_features(len(train_idx), None, min_per_feature=10,
                          cap=max_features)
        pipeline = (build_selection_pipeline(feature_cols, clone(estimator),
                                             task="regression", k=k)
                    if select_features else
                    build_pipeline(feature_cols, clone(estimator)))
        if param_grid:
            inner_n = feasible_n_splits(groups.iloc[train_idx], None,
                                        inner_splits)
            inner = GroupKFold(n_splits=inner_n)
            search = GridSearchCV(pipeline, param_grid,
                                  scoring="neg_mean_absolute_error",
                                  cv=inner, n_jobs=1, refit=True,
                                  error_score=np.nan)
            search.fit(X.iloc[train_idx], y.iloc[train_idx],
                       groups=groups.iloc[train_idx])
            fitted = search.best_estimator_
            best_params.append({"fold": fold_i, **search.best_params_})
        else:
            fitted = pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = fitted.predict(X.iloc[test_idx])
        selected_rows += _record_selected(fitted, feature_cols, fold_i)

    predictions = pd.DataFrame({
        group_col: groups, "y_true": y.to_numpy(dtype=float), "y_pred": oof})
    for col in ("index_visit_id", "next_visit_id"):
        if col in work.columns:
            predictions[col] = work[col].to_numpy()

    ok = np.isfinite(oof)
    point = M.regression_metrics(y[ok], oof[ok])
    ci = (M.bootstrap_ci(y[ok], oof[ok], groups[ok],
                         metric_fn=M.regression_metrics,
                         metric_keys=("mae", "rmse", "r2", "spearman"),
                         n_bootstrap=n_bootstrap, seed=bootstrap_seed)
          if ok.sum() >= 20 else None)

    return EvalResult(
        model=model_name, task="regression",
        n_rows=int(ok.sum()), n_patients=int(groups[ok].nunique()),
        predictions=predictions, metrics=point, metric_ci=ci,
        folds=fold_summary(y, groups, folds),
        selected_features=(pd.DataFrame(selected_rows)
                           if selected_rows else None),
        best_params=best_params, notes=notes,
    )


# --------------------------------------------------------------------------

def clone_baseline(baseline):
    """Baselines are plain objects; re-instantiate so no fold state leaks."""
    return baseline.__class__(**{k: v for k, v in vars(baseline).items()
                                 if not k.endswith("_")})


def _record_selected(fitted, feature_cols: list[str],
                     fold_i: int) -> list[dict]:
    """Which features survived in-fold selection, with their coefficients."""
    rows: list[dict] = []
    try:
        prep = fitted.named_steps["prep"]
        names = list(prep.get_feature_names_out())
    except Exception:
        names = list(feature_cols)
    mask = None
    if "select" in getattr(fitted, "named_steps", {}):
        try:
            mask = fitted.named_steps["select"].get_support()
        except Exception:
            mask = None
    chosen = [n for n, keep in zip(names, mask)] if mask is None else \
             [n for n, keep in zip(names, mask) if keep]
    model = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else None
    coefs = None
    if model is not None:
        if hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
        elif hasattr(model, "feature_importances_"):
            coefs = np.ravel(model.feature_importances_)
    for i, name in enumerate(chosen):
        rows.append({
            "fold": fold_i, "feature": name,
            "weight": (float(coefs[i]) if coefs is not None and i < len(coefs)
                       else np.nan),
        })
    return rows


def summarise_selection(selected: pd.DataFrame | None,
                        n_folds: int) -> pd.DataFrame:
    """Selection stability: how often a feature survived, and its mean weight."""
    if selected is None or selected.empty:
        return pd.DataFrame()
    out = (selected.groupby("feature")
           .agg(n_folds_selected=("fold", "nunique"),
                mean_weight=("weight", "mean"),
                mean_abs_weight=("weight", lambda s: float(np.nanmean(np.abs(s)))))
           .reset_index())
    out["selection_frequency"] = (out["n_folds_selected"] / max(1, n_folds)).round(3)
    return out.sort_values(["selection_frequency", "mean_abs_weight"],
                           ascending=False).reset_index(drop=True)
