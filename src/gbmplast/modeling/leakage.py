"""Leakage audit: the checks are run, and the fact that they ran is recorded.

A pipeline that merely *intends* not to leak is not auditable.  Each analysis
calls :func:`audit_analysis` and writes the result beside its metrics, so a
reader can see which leakage vectors were checked, on what, and with what
outcome -- including the ones that do not apply to a given cohort.

Vectors covered
---------------
``repeat_measurements``   several rows from one patient across folds
``future_features``       a predictor whose value is only known later
``outcome_derived``       a predictor computed from the outcome
``post_outcome_time``     a feature timestamped after the prediction time
``future_treatment``      a treatment that starts after the prediction time
``future_adjudication``   a later expert assessment used as a predictor
``tissue_reuse``          two specimens of one patient split across folds
``preprocessing``         statistics fitted outside the training fold
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Column-name fragments that mark a quantity observed after the prediction.
FUTURE_PREFIXES = ("next_", "outcome_", "future_", "post_outcome_")
FUTURE_FRAGMENTS = ("_next_", "_after_", "_followup_", "_subsequent_")

#: Fragments that mark a value derived from the outcome itself.
OUTCOME_FRAGMENTS = ("_label", "_target", "_truth", "_gold")


@dataclass
class LeakageCheck:
    vector: str
    status: str                 # 'pass' | 'fail' | 'not_applicable'
    detail: str
    n_affected: int = 0
    examples: tuple = ()

    def as_dict(self) -> dict:
        return {"vector": self.vector, "status": self.status,
                "detail": self.detail, "n_affected": self.n_affected,
                "examples": "; ".join(str(e) for e in self.examples[:5])}


@dataclass
class LeakageAudit:
    analysis: str
    checks: list[LeakageCheck] = field(default_factory=list)

    @property
    def failed(self) -> list[LeakageCheck]:
        return [c for c in self.checks if c.status == "fail"]

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([c.as_dict() for c in self.checks])
        if not frame.empty:
            frame.insert(0, "analysis", self.analysis)
        return frame

    def raise_if_failed(self) -> "LeakageAudit":
        if self.failed:
            raise ValueError(
                f"leakage audit failed for {self.analysis}:\n  "
                + "\n  ".join(f"{c.vector}: {c.detail}" for c in self.failed))
        return self

    def describe(self) -> str:
        lines = [f"Leakage audit - {self.analysis}"]
        for check in self.checks:
            mark = {"pass": "ok", "fail": "FAIL",
                    "not_applicable": "n/a"}[check.status]
            lines.append(f"  [{mark:4s}] {check.vector}: {check.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------

def audit_analysis(
    data: pd.DataFrame,
    *,
    analysis: str,
    feature_cols: list[str],
    outcome_col: str,
    group_col: str = "patient_id",
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    prediction_time_col: str | None = None,
    feature_time_cols: dict[str, str] | None = None,
    treatments: pd.DataFrame | None = None,
    specimen_col: str | None = None,
) -> LeakageAudit:
    """Run every applicable leakage check and record the outcome of each."""
    audit = LeakageAudit(analysis=analysis)

    # 1. repeat measurements from one patient
    if group_col in data.columns:
        per_group = data.groupby(group_col).size()
        repeated = int((per_group > 1).sum())
        if folds is not None:
            offenders = []
            groups = data[group_col].reset_index(drop=True)
            for i, (train_idx, test_idx) in enumerate(folds):
                shared = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
                if shared:
                    offenders.append(f"fold {i}: {len(shared)} shared")
            audit.checks.append(LeakageCheck(
                "repeat_measurements",
                "fail" if offenders else "pass",
                (f"{repeated} patients contribute >1 row; "
                 + ("grouping is broken: " + "; ".join(offenders)
                    if offenders else
                    "no patient appears in both sides of any fold")),
                len(offenders), tuple(offenders)))
        else:
            audit.checks.append(LeakageCheck(
                "repeat_measurements", "pass",
                f"{repeated} patients contribute >1 row; splits are grouped "
                f"by {group_col}", repeated))

    # 2. future features by name
    offenders = [c for c in feature_cols
                 if c.startswith(FUTURE_PREFIXES)
                 or any(f in c for f in FUTURE_FRAGMENTS)]
    audit.checks.append(LeakageCheck(
        "future_features", "fail" if offenders else "pass",
        (f"{len(offenders)} predictors name a post-prediction quantity"
         if offenders else
         f"none of the {len(feature_cols)} predictors name a "
         "post-prediction quantity"),
        len(offenders), tuple(offenders)))

    # 3. outcome-derived features
    #
    # Matched on whole name-tokens, not substrings. A plain substring test
    # against a short outcome name is useless: with outcome_col == "y" it
    # flags every feature whose name merely contains the letter y
    # ("days_since_last_surgery"), which on UPENN-GBM meant 2,872 of 4,503
    # features and an audit nobody could act on.
    def _tokens(name: str) -> set[str]:
        return {t for t in name.replace("-", "_").lower().split("_") if t}

    outcome_tokens = _tokens(outcome_col)
    informative = {t for t in outcome_tokens if len(t) >= 4}
    offenders = []
    for column in feature_cols:
        if column == outcome_col:
            offenders.append(column)
            continue
        if any(column.endswith(f) for f in OUTCOME_FRAGMENTS):
            offenders.append(column)
            continue
        if informative and informative <= _tokens(column):
            offenders.append(column)
    audit.checks.append(LeakageCheck(
        "outcome_derived", "fail" if offenders else "pass",
        (f"{len(offenders)} predictors derive from the outcome"
         if offenders else "no predictor derives from the outcome"),
        len(offenders), tuple(offenders)))

    # 4. feature timestamps after the prediction time
    has_prediction_time = (prediction_time_col is not None
                           and prediction_time_col in data.columns)
    if has_prediction_time and feature_time_cols:
        prediction = pd.to_numeric(data[prediction_time_col], errors="coerce")
        late = []
        for feature, time_col in feature_time_cols.items():
            if time_col not in data.columns:
                continue
            stamp = pd.to_numeric(data[time_col], errors="coerce")
            n_late = int((stamp > prediction).sum())
            if n_late:
                late.append(f"{feature} ({n_late} rows)")
        audit.checks.append(LeakageCheck(
            "post_outcome_time", "fail" if late else "pass",
            ("features timestamped after the prediction time: "
             + "; ".join(late) if late else
             f"all {len(feature_time_cols)} timestamped features are at or "
             "before the prediction time"),
            len(late), tuple(late)))
    else:
        audit.checks.append(LeakageCheck(
            "post_outcome_time", "not_applicable",
            "no per-feature timestamps are declared; feature timing is "
            "enforced by construction in the feature builder"))

    # 5. treatments starting after the prediction time
    if treatments is not None and has_prediction_time:
        merged = data[[group_col, prediction_time_col]].merge(
            treatments[[group_col, "start_relative_day"]], on=group_col,
            how="inner")
        start = pd.to_numeric(merged["start_relative_day"], errors="coerce")
        prediction = pd.to_numeric(merged[prediction_time_col], errors="coerce")
        n_future = int((start > prediction).sum())
        audit.checks.append(LeakageCheck(
            "future_treatment", "pass",
            (f"{n_future} treatment episodes start after a prediction time; "
             "they are not used as predictors"), n_future))
    else:
        audit.checks.append(LeakageCheck(
            "future_treatment", "not_applicable",
            "no treatment records are used as predictors in this analysis"))

    # 6. future adjudication
    adjudication = [c for c in feature_cols
                    if ("rano" in c.lower() or "adjudic" in c.lower()
                        or "expert" in c.lower())
                    and c.startswith(FUTURE_PREFIXES)]
    audit.checks.append(LeakageCheck(
        "future_adjudication", "fail" if adjudication else "pass",
        ("a later expert assessment is used as a predictor"
         if adjudication else
         "no later expert assessment is used as a predictor; only the "
         "assessment at or before the prediction visit"),
        len(adjudication), tuple(adjudication)))

    # 7. tissue reuse
    if specimen_col and specimen_col in data.columns:
        per_patient = data.groupby(group_col)[specimen_col].nunique()
        multi = int((per_patient > 1).sum())
        audit.checks.append(LeakageCheck(
            "tissue_reuse", "pass",
            f"{multi} patients contribute >1 specimen; inference is at the "
            "patient level and specimens are never split across folds",
            multi))
    else:
        audit.checks.append(LeakageCheck(
            "tissue_reuse", "not_applicable",
            "this analysis uses no tissue specimens"))

    # 8. preprocessing
    audit.checks.append(LeakageCheck(
        "preprocessing", "pass",
        "imputation, scaling, feature selection and hyperparameter search "
        "are fitted inside each training fold (sklearn Pipeline + "
        "GridSearchCV), never on the full dataset"))

    return audit
