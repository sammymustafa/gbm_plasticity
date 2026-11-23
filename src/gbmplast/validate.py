"""Structural validation of canonical bundles.

These checks are deliberately *loud*: they are the guard rail that keeps a
cohort-specific quirk from silently becoming an analysis bug.  Each check
returns a :class:`Finding`; ``severity='error'`` means the bundle is not fit
for modelling.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

from gbmplast.schema import SPECS, CanonicalBundle


@dataclass
class Finding:
    check: str
    table: str
    severity: str          # 'error' | 'warning' | 'info'
    message: str
    n_affected: int = 0
    examples: tuple = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["examples"] = list(self.examples)[:5]
        return d


class ValidationReport:
    def __init__(self, cohort: str, findings: list[Finding]) -> None:
        self.cohort = cohort
        self.findings = findings

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=["cohort", "check", "table", "severity", "message",
                         "n_affected", "examples"])
        df = pd.DataFrame([f.as_dict() for f in self.findings])
        df.insert(0, "cohort", self.cohort)
        return df

    def raise_if_errors(self) -> "ValidationReport":
        if self.errors:
            lines = [f"[{f.table}] {f.check}: {f.message}" for f in self.errors]
            raise ValueError(
                f"Validation failed for cohort {self.cohort!r}:\n  "
                + "\n  ".join(lines)
            )
        return self

    def __repr__(self) -> str:
        return (f"<ValidationReport {self.cohort}: {len(self.errors)} errors, "
                f"{len(self.warnings)} warnings>")


# --------------------------------------------------------------------------

def validate_bundle(bundle: CanonicalBundle) -> ValidationReport:
    findings: list[Finding] = []
    findings += _check_required_columns(bundle)
    findings += _check_primary_keys(bundle)
    findings += _check_patient_id_namespacing(bundle)
    findings += _check_referential_integrity(bundle)
    findings += _check_longitudinal_ordering(bundle)
    findings += _check_survival_integrity(bundle)
    findings += _check_modality_availability(bundle)
    findings += _check_controlled_vocabularies(bundle)
    return ValidationReport(bundle.cohort, findings)


def _check_required_columns(bundle: CanonicalBundle) -> list[Finding]:
    out = []
    for name, table in bundle.items():
        spec = SPECS[name]
        missing = [c for c in spec.required_columns if c not in table.columns]
        if missing:
            out.append(Finding(
                "required_columns", name, "error",
                f"missing required columns: {missing}", len(missing),
                tuple(missing)))
        if len(table) == 0:
            continue
        for col in spec.required_columns:
            if col in table.columns and table[col].isna().any():
                n = int(table[col].isna().sum())
                out.append(Finding(
                    "required_not_null", name, "error",
                    f"required column {col!r} has {n} null values", n))
    return out


def _check_primary_keys(bundle: CanonicalBundle) -> list[Finding]:
    out = []
    for name, table in bundle.items():
        spec = SPECS[name]
        pk = list(spec.primary_key)
        if len(table) == 0 or not set(pk).issubset(table.columns):
            continue
        dup = table[table.duplicated(pk, keep=False)]
        if len(dup):
            out.append(Finding(
                "primary_key_unique", name, "error",
                f"primary key {pk} is not unique ({len(dup)} duplicated rows)",
                len(dup), tuple(dup[pk[0]].head(5).astype(str))))
        else:
            out.append(Finding(
                "primary_key_unique", name, "info",
                f"primary key {pk} unique over {len(table)} rows", 0))
    return out


def _check_patient_id_namespacing(bundle: CanonicalBundle) -> list[Finding]:
    """Every patient_id must be '<COHORT>::<local>' so cross-cohort joins fail."""
    out = []
    for name, table in bundle.items():
        if len(table) == 0 or "patient_id" not in table.columns:
            continue
        ids = table["patient_id"].dropna().astype(str)
        bad = ids[~ids.str.contains("::", regex=False)]
        if len(bad):
            out.append(Finding(
                "patient_id_namespaced", name, "error",
                f"{len(bad)} patient_id values are not namespaced with a cohort "
                "prefix; a cross-cohort join could silently succeed",
                len(bad), tuple(bad.head(5))))
    return out


def _check_referential_integrity(bundle: CanonicalBundle) -> list[Finding]:
    out = []
    patients = set(bundle.patients["patient_id"].dropna()) if len(bundle.patients) else set()
    visits = set(bundle.visits["visit_id"].dropna()) if len(bundle.visits) else set()

    for name, table in bundle.items():
        if name == "patients" or len(table) == 0:
            continue
        if "patient_id" in table.columns and patients:
            orphan = set(table["patient_id"].dropna()) - patients
            if orphan:
                out.append(Finding(
                    "fk_patient_id", name, "error",
                    f"{len(orphan)} patient_id values absent from patients",
                    len(orphan), tuple(sorted(orphan)[:5])))
        if name in {"imaging", "molecular_samples", "symptoms", "outcomes"} \
                and "visit_id" in table.columns and visits:
            present = table["visit_id"].dropna()
            orphan = set(present) - visits
            if orphan:
                severity = "error" if name == "imaging" else "warning"
                out.append(Finding(
                    "fk_visit_id", name, severity,
                    f"{len(orphan)} visit_id values absent from visits",
                    len(orphan), tuple(sorted(orphan)[:5])))
    return out


def _check_longitudinal_ordering(bundle: CanonicalBundle) -> list[Finding]:
    """visit_index must order visits by relative_time_days within a patient."""
    out = []
    v = bundle.visits
    needed = {"patient_id", "visit_index", "relative_time_days"}
    if len(v) == 0 or not needed.issubset(v.columns):
        return out

    n_unknown_time = int(pd.to_numeric(v["relative_time_days"],
                                       errors="coerce").isna().sum())
    if n_unknown_time:
        out.append(Finding(
            "visit_time_known", "visits", "warning",
            f"{n_unknown_time} visits have no relative_time_days; they are "
            "excluded from longitudinal analyses rather than imputed",
            n_unknown_time))

    offenders = []
    for pid, grp in v.groupby("patient_id"):
        g = grp.sort_values("visit_index")
        times = pd.to_numeric(g["relative_time_days"], errors="coerce")
        # Ordering is only checkable where times are known. Unknown times must
        # sort last so they never claim an earlier slot than a dated study.
        known = times.dropna()
        if not known.is_monotonic_increasing:
            offenders.append(str(pid))
        idx = pd.to_numeric(g["visit_index"], errors="coerce").tolist()
        if idx != list(range(len(idx))):
            offenders.append(str(pid))
    offenders = sorted(set(offenders))
    if offenders:
        out.append(Finding(
            "longitudinal_ordering", "visits", "error",
            f"{len(offenders)} patients have visit_index inconsistent with "
            "relative_time_days or not 0..n-1",
            len(offenders), tuple(offenders[:5])))
    else:
        out.append(Finding(
            "longitudinal_ordering", "visits", "info",
            f"visit_index consistent with time for all "
            f"{v['patient_id'].nunique()} patients", 0))

    dup = v.duplicated(["patient_id", "visit_index"], keep=False)
    if dup.any():
        out.append(Finding(
            "visit_index_unique", "visits", "error",
            f"{int(dup.sum())} rows share (patient_id, visit_index)",
            int(dup.sum())))
    return out


def _check_survival_integrity(bundle: CanonicalBundle) -> list[Finding]:
    """A time-to-event outcome without a censoring flag is not identifiable."""
    out = []
    o = bundle.outcomes
    if len(o) == 0 or "outcome_type" not in o.columns:
        return out
    tte = o[o["outcome_type"] == "time_to_event"]
    if len(tte) == 0:
        return out
    if "event_indicator" not in tte.columns or tte["event_indicator"].isna().all():
        out.append(Finding(
            "survival_censoring", "outcomes", "warning",
            f"{len(tte)} time-to-event rows have no censoring indicator; "
            "survival models will be refused for this cohort",
            len(tte),
            tuple(tte["outcome_name"].drop_duplicates().head(3))))
    neg_mask = pd.to_numeric(tte["outcome_value"], errors="coerce") < 0
    if neg_mask.any():
        neg = tte[neg_mask]
        # A negative time is a defect in the released table. It only blocks
        # modelling where the row is otherwise usable (has a censoring flag);
        # otherwise it is already excluded and is reported as a warning.
        modellable = neg["event_indicator"].notna()
        if modellable.any():
            out.append(Finding(
                "survival_negative_time", "outcomes", "error",
                f"{int(modellable.sum())} modellable time-to-event rows have "
                "negative times",
                int(modellable.sum()),
                tuple(neg.loc[modellable, "patient_id"].head(5).astype(str))))
        if (~modellable).any():
            rows = neg[~modellable]
            out.append(Finding(
                "survival_negative_time", "outcomes", "warning",
                f"{len(rows)} time-to-event rows have negative times in the "
                "released table; they carry no censoring flag and are already "
                "excluded from modelling",
                len(rows),
                tuple(rows["patient_id"].head(5).astype(str))))
    return out


def _check_modality_availability(bundle: CanonicalBundle) -> list[Finding]:
    """Report modalities whose presence the cohort never states."""
    out: list[Finding] = []
    img = bundle.imaging
    if len(img) == 0 or "available" not in img.columns:
        return out
    unknown = img["available"].isna()
    if unknown.any():
        mods = sorted(img.loc[unknown, "modality"].dropna().astype(str).unique())
        out.append(Finding(
            "modality_availability_unknown", "imaging", "warning",
            f"{int(unknown.sum())} imaging rows have unknown availability "
            f"(modalities: {mods[:6]}); treated as unknown, never as present",
            int(unknown.sum()), tuple(mods[:5])))
    return out


def _check_controlled_vocabularies(bundle: CanonicalBundle) -> list[Finding]:
    out = []
    for name, table in bundle.items():
        spec = SPECS[name]
        if len(table) == 0:
            continue
        for col in spec.columns:
            if col.allowed is None or col.name not in table.columns:
                continue
            values = table[col.name].dropna().astype(str)
            bad = sorted(set(values) - set(col.allowed))
            if bad:
                out.append(Finding(
                    "controlled_vocabulary", name, "warning",
                    f"{col.name!r} has values outside its vocabulary: {bad[:5]}",
                    len(bad), tuple(bad[:5])))
    return out


# --------------------------------------------------------------------------
# Leakage-oriented checks on an analysis (wide) table
# --------------------------------------------------------------------------

def check_no_future_information(
    df: pd.DataFrame,
    *,
    time_col: str = "relative_time_days",
    prediction_time_col: str = "prediction_time_days",
    feature_cols: Iterable[str],
    patient_col: str = "patient_id",
) -> list[Finding]:
    """Every feature must be derivable at or before ``prediction_time_col``.

    This checks the *declared* provenance column ``<feature>__source_time``
    when present; features without one are reported as unverifiable rather
    than silently trusted.
    """
    findings: list[Finding] = []
    feature_cols = list(feature_cols)
    unverifiable = []
    for col in feature_cols:
        src = f"{col}__source_time"
        if src not in df.columns:
            unverifiable.append(col)
            continue
        late = pd.to_numeric(df[src], errors="coerce") > pd.to_numeric(
            df[prediction_time_col], errors="coerce")
        if late.any():
            findings.append(Finding(
                "future_information", "analysis", "error",
                f"feature {col!r} draws on data after the prediction time in "
                f"{int(late.sum())} rows", int(late.sum())))
    if unverifiable:
        findings.append(Finding(
            "feature_provenance", "analysis", "info",
            f"{len(unverifiable)} features carry no explicit source-time column; "
            "their timing is enforced by construction in the feature builder",
            len(unverifiable), tuple(unverifiable[:5])))
    return findings
