"""Structured-symptom ingestion for the institutional cohort.

Accepts a long or wide table of symptom assessments and emits canonical
``symptoms`` rows.  Absence of a row means *not assessed* (NA), which is
distinct from an assessed-and-absent row (``present = False``).  The
distinction is preserved end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.schema import SYMPTOMS, make_key, make_patient_id
from gbmplast.symptoms.schema import SYMPTOM_KEYS, TERMS_BY_KEY

_TRUE = {"1", "true", "yes", "y", "present", "t"}
_FALSE = {"0", "false", "no", "n", "absent", "f"}


def parse_presence(value: object) -> object:
    """Map a cell to True / False / pd.NA. Unrecognised values stay NA."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    # Numeric 1/0 (including 1.0/0.0 from a float column) is the commonest
    # structured encoding; handle it numerically so '1.0' is not missed.
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return pd.NA
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    try:                      # '1.0' / '0.0' arriving as text
        numeric = float(text)
    except ValueError:
        return pd.NA
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    return pd.NA


def from_long(
    df: pd.DataFrame,
    *,
    cohort: str,
    patient_col: str = "patient_id",
    symptom_col: str = "symptom",
    present_col: str = "present",
    visit_col: str | None = "visit_id",
    severity_col: str | None = None,
    severity_scale: str | None = None,
    assessment_day_col: str | None = "assessment_relative_day",
    onset_day_col: str | None = "onset_relative_day",
    resolution_day_col: str | None = "resolution_relative_day",
    strict_vocabulary: bool = True,
) -> pd.DataFrame:
    """Canonical ``symptoms`` rows from a long (one row per observation) table."""
    rows = []
    unknown_terms: set[str] = set()
    for i, r in df.reset_index(drop=True).iterrows():
        symptom = str(r[symptom_col]).strip()
        if symptom not in SYMPTOM_KEYS:
            unknown_terms.add(symptom)
            if strict_vocabulary:
                continue
        local_pt = str(r[patient_col]).strip()
        pid = (local_pt if "::" in local_pt
               else make_patient_id(cohort, local_pt))
        visit = (str(r[visit_col]).strip()
                 if visit_col and visit_col in df.columns
                 and not pd.isna(r.get(visit_col)) else pd.NA)
        rows.append({
            "symptom_id": make_key(cohort, local_pt, symptom, str(i)),
            "patient_id": pid,
            "cohort": cohort,
            "visit_id": visit,
            "symptom": symptom,
            "present": parse_presence(r[present_col]),
            "severity": (_opt(r, severity_col)),
            "severity_scale": severity_scale or pd.NA,
            "assessment_relative_day": _opt_int(r, assessment_day_col),
            "onset_relative_day": _opt_int(r, onset_day_col),
            "resolution_relative_day": _opt_int(r, resolution_day_col),
            "source": "structured",
            "negated": pd.NA,
            "temporality": "current",
            "validated_by_human": pd.NA,
        })
    out = SYMPTOMS.conform(pd.DataFrame(rows))
    if unknown_terms:
        action = "dropped" if strict_vocabulary else "kept"
        out.attrs["unknown_symptom_terms"] = sorted(unknown_terms)
        out.attrs["unknown_symptom_action"] = action
    return out


def from_wide(
    df: pd.DataFrame,
    *,
    cohort: str,
    patient_col: str = "patient_id",
    visit_col: str | None = "visit_id",
    symptom_columns: dict[str, str] | None = None,
    assessment_day_col: str | None = "assessment_relative_day",
) -> pd.DataFrame:
    """Canonical rows from a wide table with one column per symptom.

    ``symptom_columns`` maps dataframe column -> canonical symptom key.  When
    omitted, columns whose names match a canonical key are used.
    """
    mapping = symptom_columns or {
        c: c for c in df.columns if c in SYMPTOM_KEYS
    }
    if not mapping:
        raise ValueError(
            "No symptom columns found. Pass symptom_columns={'col': "
            f"'canonical_key'}}; canonical keys are {list(SYMPTOM_KEYS)}")
    long_rows = []
    for _, r in df.iterrows():
        for col, key in mapping.items():
            long_rows.append({
                patient_col: r[patient_col],
                "symptom": key,
                "present": r[col],
                **({visit_col: r[visit_col]} if visit_col and visit_col in df.columns else {}),
                **({assessment_day_col: r[assessment_day_col]}
                   if assessment_day_col and assessment_day_col in df.columns else {}),
            })
    return from_long(pd.DataFrame(long_rows), cohort=cohort,
                     patient_col=patient_col, visit_col=visit_col,
                     assessment_day_col=assessment_day_col)


def coverage_matrix(symptoms: pd.DataFrame) -> pd.DataFrame:
    """Per-symptom assessed / present / absent / not-assessed counts."""
    if symptoms.empty:
        return pd.DataFrame(columns=["symptom", "n_rows", "n_present",
                                     "n_absent", "n_unknown", "pct_assessed"])
    rows = []
    for key in SYMPTOM_KEYS:
        sub = symptoms[symptoms["symptom"] == key]
        present = (sub["present"] == True).sum()      # noqa: E712
        absent = (sub["present"] == False).sum()      # noqa: E712
        unknown = int(sub["present"].isna().sum())
        rows.append({
            "symptom": key,
            "label": TERMS_BY_KEY[key].label,
            "n_rows": len(sub),
            "n_present": int(present),
            "n_absent": int(absent),
            "n_unknown": unknown,
            "pct_assessed": (round(100 * (present + absent) / len(sub), 1)
                             if len(sub) else 0.0),
        })
    return pd.DataFrame(rows)


def _opt(row: pd.Series, col: str | None) -> object:
    if not col or col not in row.index or pd.isna(row[col]):
        return pd.NA
    return row[col]


def _opt_int(row: pd.Series, col: str | None) -> object:
    v = _opt(row, col)
    if v is pd.NA:
        return pd.NA
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return pd.NA
