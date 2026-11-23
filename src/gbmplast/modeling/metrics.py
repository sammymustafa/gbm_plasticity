"""Metrics with patient-level bootstrap confidence intervals.

Discrimination alone is not enough for a clinical prediction question, so
every binary evaluation reports AUROC *and* AUPRC (which is sensitive to the
class balance that matters here), a Brier score, and calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from gbmplast.modeling.splits import patient_bootstrap_indices


# --------------------------------------------------------------------------
# point estimates
# --------------------------------------------------------------------------

def binary_metrics(y_true, y_prob, *, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    out: dict = {"n": int(len(y_true)), "n_positive": int(y_true.sum()),
                 "prevalence": float(y_true.mean()) if len(y_true) else np.nan}
    if len(np.unique(y_true)) < 2:
        out.update({"auroc": np.nan, "auprc": np.nan, "brier": np.nan,
                    "note": "single class present; discrimination undefined"})
        return out
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    out.update({
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        # Baseline Brier for a model that always predicts prevalence; the
        # comparison is what makes a Brier score interpretable.
        "brier_baseline": float(np.mean((y_true - y_true.mean()) ** 2)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "balanced_accuracy": np.nanmean([
            tp / (tp + fn) if (tp + fn) else np.nan,
            tn / (tn + fp) if (tn + fp) else np.nan]),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
    })
    out["brier_skill_score"] = (
        1.0 - out["brier"] / out["brier_baseline"]
        if out["brier_baseline"] > 0 else np.nan)
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "mae": np.nan, "rmse": np.nan,
                "r2": np.nan}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "medae": float(np.median(np.abs(y_true - y_pred))),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        # MAE of the mean predictor: the reference any model must beat.
        "mae_baseline_mean": float(np.mean(np.abs(y_true - y_true.mean()))),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred),
                                                 method="spearman")),
    }


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibration_table(y_true, y_prob, *, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    if len(y_true) == 0:
        return pd.DataFrame()
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 3:
        edges = np.array([y_prob.min(), y_prob.mean(), y_prob.max() + 1e-9])
    bins = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0,
                   len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = bins == b
        if not sel.any():
            continue
        rows.append({
            "bin": b,
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "n": int(sel.sum()),
            "mean_predicted": float(y_prob[sel].mean()),
            "observed_frequency": float(y_true[sel].mean()),
        })
    return pd.DataFrame(rows)


def calibration_metrics(y_true, y_prob, *, n_bins: int = 10) -> dict:
    """Calibration slope/intercept plus expected calibration error."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    out: dict = {}
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    if not table.empty:
        weights = table["n"] / table["n"].sum()
        out["ece"] = float((weights * (table["mean_predicted"]
                                       - table["observed_frequency"]).abs()).sum())
        out["mce"] = float((table["mean_predicted"]
                            - table["observed_frequency"]).abs().max())
    # Calibration-in-the-large and slope via logistic recalibration.
    if len(np.unique(y_true)) == 2:
        eps = 1e-6
        p = np.clip(y_prob, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(penalty=None, solver="lbfgs",
                                    max_iter=1000)
            lr.fit(logit.reshape(-1, 1), y_true)
            out["calibration_slope"] = float(lr.coef_[0][0])
            out["calibration_intercept"] = float(lr.intercept_[0])
        except Exception:
            out["calibration_slope"] = np.nan
            out["calibration_intercept"] = np.nan
        out["mean_predicted"] = float(y_prob.mean())
        out["observed_prevalence"] = float(y_true.mean())
        out["calibration_in_the_large"] = float(y_prob.mean() - y_true.mean())
    return out


# --------------------------------------------------------------------------
# bootstrap CIs
# --------------------------------------------------------------------------

def bootstrap_ci(
    y_true,
    y_pred,
    groups,
    *,
    metric_fn,
    metric_keys: tuple[str, ...],
    n_bootstrap: int = 2000,
    seed: int = 777,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Patient-level (cluster) bootstrap percentile CIs for chosen metrics."""
    y_true = pd.Series(np.asarray(y_true, dtype=float)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred, dtype=float)).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    point = metric_fn(y_true, y_pred)
    replicate_indices = patient_bootstrap_indices(
        groups, n_bootstrap=n_bootstrap, seed=seed)
    draws: dict[str, list[float]] = {k: [] for k in metric_keys}
    for idx in replicate_indices:
        try:
            values = metric_fn(y_true.iloc[idx], y_pred.iloc[idx])
        except Exception:
            continue
        for key in metric_keys:
            value = values.get(key, np.nan)
            if value is not None and np.isfinite(value):
                draws[key].append(float(value))

    rows = []
    for key in metric_keys:
        samples = np.asarray(draws[key], dtype=float)
        rows.append({
            "metric": key,
            "estimate": point.get(key, np.nan),
            "ci_low": (float(np.quantile(samples, alpha / 2))
                       if samples.size >= 20 else np.nan),
            "ci_high": (float(np.quantile(samples, 1 - alpha / 2))
                        if samples.size >= 20 else np.nan),
            "n_bootstrap_valid": int(samples.size),
            "n_patients": int(groups.nunique()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# model comparison
# --------------------------------------------------------------------------

def paired_bootstrap_difference(
    y_true,
    pred_a,
    pred_b,
    groups,
    *,
    metric_fn,
    metric_key: str,
    n_bootstrap: int = 2000,
    seed: int = 777,
    alpha: float = 0.05,
) -> dict:
    """CI for metric(a) - metric(b) on the *same* bootstrap patient samples.

    Pairing the replicates is what makes the comparison of two models on one
    cohort meaningful; independent bootstraps would overstate the spread.
    """
    y_true = pd.Series(np.asarray(y_true, dtype=float)).reset_index(drop=True)
    pred_a = pd.Series(np.asarray(pred_a, dtype=float)).reset_index(drop=True)
    pred_b = pd.Series(np.asarray(pred_b, dtype=float)).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    observed = (metric_fn(y_true, pred_a).get(metric_key, np.nan)
                - metric_fn(y_true, pred_b).get(metric_key, np.nan))
    diffs = []
    for idx in patient_bootstrap_indices(groups, n_bootstrap=n_bootstrap,
                                        seed=seed):
        try:
            a = metric_fn(y_true.iloc[idx], pred_a.iloc[idx]).get(metric_key)
            b = metric_fn(y_true.iloc[idx], pred_b.iloc[idx]).get(metric_key)
        except Exception:
            continue
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        diffs.append(float(a - b))
    samples = np.asarray(diffs, dtype=float)
    return {
        "metric": metric_key,
        "difference": float(observed) if np.isfinite(observed) else np.nan,
        "ci_low": (float(np.quantile(samples, alpha / 2))
                   if samples.size >= 20 else np.nan),
        "ci_high": (float(np.quantile(samples, 1 - alpha / 2))
                    if samples.size >= 20 else np.nan),
        # Two-sided bootstrap p-value for H0: difference = 0.
        "p_two_sided": (float(2 * min((samples <= 0).mean(),
                                      (samples >= 0).mean()))
                        if samples.size >= 20 else np.nan),
        "n_bootstrap_valid": int(samples.size),
    }
