"""Feature attribution -- computed only where it is interpretable.

Two deliberate restrictions:

1. **Attribution is computed out-of-fold.** A SHAP or permutation score from
   a model refitted on all the data describes a model nobody evaluated.
   Here the model is refitted per fold and attributions are averaged across
   folds, with their spread reported, so instability is visible.

2. **Attribution is refused when it would mislead.** With many correlated
   radiomic features and few patients, per-feature importances are not
   stable, and presenting a ranked list implies a precision that is not
   there.  :func:`attribution_is_interpretable` states the condition and the
   drivers refuse rather than print a league table.

Whatever is produced describes **what the model used**, not what drives the
biology.  That sentence is attached to every output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

CAUTION = (
    "Feature attributions describe what the fitted model used, not a causal "
    "or biological mechanism. With correlated features, attribution is "
    "shared arbitrarily among them."
)


def attribution_is_interpretable(
    n_rows: int,
    n_patients: int,
    n_features: int,
    *,
    min_rows_per_feature: int = 10,
    min_patients: int = 40,
) -> tuple[bool, str]:
    """Would a per-feature ranking mean anything at this sample size?"""
    if n_patients < min_patients:
        return False, (
            f"{n_patients} patients is too few for stable per-feature "
            f"attribution (minimum {min_patients}); refusing to print a "
            "ranked list that would not replicate.")
    ratio = n_rows / max(1, n_features)
    if ratio < min_rows_per_feature:
        return False, (
            f"{n_rows} rows for {n_features} features is "
            f"{ratio:.1f} rows per feature, below the {min_rows_per_feature} "
            "needed for attribution to be stable. Use the selection-frequency "
            "table instead, which reports how often a feature survived "
            "in-fold selection.")
    return True, (f"{n_rows} rows, {n_patients} patients, {n_features} "
                  f"features ({ratio:.1f} rows per feature)")


def permutation_importance_cv(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    outcome_col: str,
    estimator,
    group_col: str = "patient_id",
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 20251123,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Permutation importance computed on each held-out fold, then pooled."""
    from gbmplast.modeling.models import build_pipeline
    from gbmplast.modeling.splits import assert_group_disjoint, split_indices

    work = data.dropna(subset=[outcome_col, group_col]).reset_index(drop=True)
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    X = work[feature_cols]
    groups = work[group_col]

    folds = split_indices(X, y, groups, n_splits=n_splits, seed=seed,
                          stratify=scoring == "roc_auc")
    assert_group_disjoint(groups, folds)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if y.iloc[train_idx].nunique() < 2:
            continue
        pipeline = build_pipeline(feature_cols, clone(estimator))
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        try:
            result = permutation_importance(
                pipeline, X.iloc[test_idx], y.iloc[test_idx],
                n_repeats=n_repeats, random_state=seed + fold_i,
                scoring=scoring, n_jobs=1)
        except Exception:
            continue
        for name, mean, sd in zip(feature_cols, result.importances_mean,
                                  result.importances_std):
            rows.append({"fold": fold_i, "feature": name,
                         "importance": float(mean), "importance_sd": float(sd)})
    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows)
    out = (long.groupby("feature")
           .agg(mean_importance=("importance", "mean"),
                sd_across_folds=("importance", "std"),
                min_importance=("importance", "min"),
                max_importance=("importance", "max"),
                n_folds=("fold", "nunique"))
           .reset_index())
    # A feature whose importance changes sign across folds is not a finding.
    out["sign_stable"] = (out["min_importance"] * out["max_importance"]) > 0
    out["scoring"] = scoring
    out = out.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    out.attrs["caution"] = CAUTION
    return out


def shap_values_cv(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    outcome_col: str,
    estimator,
    group_col: str = "patient_id",
    n_splits: int = 5,
    seed: int = 20251123,
    max_display: int = 25,
) -> pd.DataFrame:
    """Mean |SHAP| per feature, pooled over held-out folds.

    Tree models only -- a KernelExplainer on a linear model adds nothing over
    its coefficients and costs a great deal.
    """
    try:
        import shap
    except ImportError:
        return pd.DataFrame([{"status": "skipped",
                              "reason": "shap is not installed"}])

    from gbmplast.modeling.models import build_pipeline
    from gbmplast.modeling.splits import assert_group_disjoint, split_indices

    work = data.dropna(subset=[outcome_col, group_col]).reset_index(drop=True)
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    X = work[feature_cols]
    groups = work[group_col]

    folds = split_indices(X, y, groups, n_splits=n_splits, seed=seed)
    assert_group_disjoint(groups, folds)

    collected = []
    for train_idx, test_idx in folds:
        if y.iloc[train_idx].nunique() < 2:
            continue
        pipeline = build_pipeline(feature_cols, clone(estimator))
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        model = pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return pd.DataFrame([{
                "status": "skipped",
                "reason": ("SHAP here is restricted to tree models; for the "
                           "linear model read its coefficients and the "
                           "selection-frequency table instead.")}])
        transformed = pipeline.named_steps["prep"].transform(X.iloc[test_idx])
        names = list(pipeline.named_steps["prep"].get_feature_names_out())
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            collected.append(pd.DataFrame(
                np.abs(values), columns=names).mean().rename("mean_abs_shap"))
        except Exception as exc:
            return pd.DataFrame([{"status": "failed", "reason": str(exc)[:160]}])

    if not collected:
        return pd.DataFrame()
    frame = pd.concat(collected, axis=1)
    out = pd.DataFrame({
        "feature": frame.index,
        "mean_abs_shap": frame.mean(axis=1).to_numpy(),
        "sd_across_folds": frame.std(axis=1).to_numpy(),
        "n_folds": frame.notna().sum(axis=1).to_numpy(),
    }).sort_values("mean_abs_shap", ascending=False).head(max_display)
    out.attrs["caution"] = CAUTION
    return out.reset_index(drop=True)


def explain_if_valid(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    outcome_col: str,
    estimator,
    group_col: str = "patient_id",
    seed: int = 20251123,
    scoring: str = "roc_auc",
) -> tuple[pd.DataFrame | None, str]:
    """Attribute, or say clearly why not."""
    work = data.dropna(subset=[outcome_col, group_col])
    ok, reason = attribution_is_interpretable(
        len(work), work[group_col].nunique(), len(feature_cols))
    if not ok:
        return None, reason
    importance = permutation_importance_cv(
        work, feature_cols=feature_cols, outcome_col=outcome_col,
        estimator=estimator, group_col=group_col, seed=seed, scoring=scoring)
    return importance, reason
