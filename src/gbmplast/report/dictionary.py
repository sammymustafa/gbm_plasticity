"""Data dictionaries, availability matrices and the cross-cohort audit.

Everything here is descriptive: it reports what a cohort contains, including
what it does *not* contain.  No value is filled in, and "unknown" is kept
distinct from "absent" throughout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters.base import DatasetAdapter
from gbmplast.schema import CANONICAL_TABLES, SPECS, CanonicalBundle


# --------------------------------------------------------------------------
# Canonical-table dictionaries
# --------------------------------------------------------------------------

def canonical_dictionary() -> pd.DataFrame:
    """The project's own schema, as a table (cohort-independent)."""
    rows = []
    for name, spec in SPECS.items():
        for col in spec.columns:
            rows.append({
                "table": name,
                "primary_key": ", ".join(spec.primary_key),
                "column": col.name,
                "dtype": col.dtype,
                "required": col.required,
                "allowed_values": ", ".join(col.allowed) if col.allowed else "",
                "description": col.description,
            })
    return pd.DataFrame(rows)


def bundle_dictionary(bundle: CanonicalBundle) -> pd.DataFrame:
    """Column-level missingness for the canonical tables of one cohort."""
    rows = []
    for name, table in bundle.items():
        spec = SPECS[name]
        canonical = set(spec.column_names)
        for col in table.columns:
            series = table[col]
            nonnull = series.dropna()
            rows.append({
                "cohort": bundle.cohort,
                "table": name,
                "column": col,
                "is_canonical": col in canonical,
                "dtype": str(series.dtype),
                "n_rows": len(series),
                "n_present": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "pct_missing": (round(100 * series.isna().mean(), 2)
                                if len(series) else np.nan),
                "n_unique": int(nonnull.nunique()) if len(nonnull) else 0,
                "example": _example(nonnull),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Availability / missingness matrices
# --------------------------------------------------------------------------

def missingness_matrix(bundle: CanonicalBundle, table: str) -> pd.DataFrame:
    """Per-column percent-missing for one table, wide (one row)."""
    df = bundle.table(table)
    if df.empty:
        return pd.DataFrame([{"cohort": bundle.cohort, "table": table,
                              "n_rows": 0}])
    pct = (df.isna().mean() * 100).round(2).to_dict()
    return pd.DataFrame([{"cohort": bundle.cohort, "table": table,
                          "n_rows": len(df), **pct}])


def modality_completeness(bundle: CanonicalBundle) -> pd.DataFrame:
    """Per-modality availability, keeping 'unknown' separate from 'absent'."""
    img = bundle.imaging
    if img.empty:
        return pd.DataFrame(columns=["cohort", "modality", "n_rows",
                                     "n_available", "n_unavailable",
                                     "n_unknown", "pct_available",
                                     "n_visits_with_modality"])
    rows = []
    for modality, grp in img.groupby("modality"):
        available = (grp["available"] == True)       # noqa: E712
        unavailable = (grp["available"] == False)    # noqa: E712
        unknown = grp["available"].isna()
        rows.append({
            "cohort": bundle.cohort,
            "modality": modality,
            "n_rows": len(grp),
            "n_available": int(available.sum()),
            "n_unavailable": int(unavailable.sum()),
            "n_unknown": int(unknown.sum()),
            "pct_available": round(100 * available.mean(), 2),
            "n_visits_with_modality": int(
                grp.loc[available, "visit_id"].nunique()),
            "n_patients_with_modality": int(
                grp.loc[available, "patient_id"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("modality").reset_index(drop=True)


def visit_modality_matrix(bundle: CanonicalBundle) -> pd.DataFrame:
    """One row per visit, one boolean column per modality."""
    img = bundle.imaging
    if img.empty:
        return pd.DataFrame()
    wide = (img.assign(flag=img["available"])
            .pivot_table(index=["patient_id", "visit_id"], columns="modality",
                         values="flag", aggfunc="max"))
    return wide.reset_index()


# --------------------------------------------------------------------------
# Cohort-level audit
# --------------------------------------------------------------------------

def id_uniqueness(bundle: CanonicalBundle) -> pd.DataFrame:
    rows = []
    for name, table in bundle.items():
        spec = SPECS[name]
        pk = list(spec.primary_key)
        if table.empty or not set(pk).issubset(table.columns):
            rows.append({"cohort": bundle.cohort, "table": name,
                         "primary_key": ", ".join(pk), "n_rows": len(table),
                         "n_unique": 0, "n_duplicated": 0, "unique": True,
                         "n_null_key": 0})
            continue
        n_dup = int(table.duplicated(pk).sum())
        rows.append({
            "cohort": bundle.cohort, "table": name,
            "primary_key": ", ".join(pk),
            "n_rows": len(table),
            "n_unique": int(table[pk].drop_duplicates().shape[0]),
            "n_duplicated": n_dup,
            "unique": n_dup == 0,
            "n_null_key": int(table[pk[0]].isna().sum()),
        })
    return pd.DataFrame(rows)


def longitudinal_coverage(bundle: CanonicalBundle,
                          min_studies: int = 2) -> pd.DataFrame:
    """How many patients have enough repeat imaging for trajectory modelling."""
    visits = bundle.visits
    if visits.empty:
        return pd.DataFrame([{"cohort": bundle.cohort, "n_patients": 0}])
    imaged = visits[visits.get("has_imaging", pd.Series(dtype=bool)) == True]  # noqa: E712
    per_patient_all = visits.groupby("patient_id").size()
    per_patient_img = (imaged.groupby("patient_id").size()
                       if len(imaged) else pd.Series(dtype=int))
    dated = visits[pd.to_numeric(visits["relative_time_days"],
                                 errors="coerce").notna()]
    per_patient_dated = (dated.groupby("patient_id").size()
                         if len(dated) else pd.Series(dtype=int))
    return pd.DataFrame([{
        "cohort": bundle.cohort,
        "n_patients": int(visits["patient_id"].nunique()),
        "n_visits": len(visits),
        "n_imaged_visits": int(len(imaged)),
        "n_dated_visits": int(len(dated)),
        "median_visits_per_patient": float(per_patient_all.median()),
        "max_visits_per_patient": int(per_patient_all.max()),
        f"n_patients_with_{min_studies}plus_visits":
            int((per_patient_all >= min_studies).sum()),
        f"n_patients_with_{min_studies}plus_imaged_studies":
            int((per_patient_img >= min_studies).sum()) if len(per_patient_img) else 0,
        f"n_patients_with_{min_studies}plus_dated_visits":
            int((per_patient_dated >= min_studies).sum()) if len(per_patient_dated) else 0,
        "n_patients_with_rano":
            int(visits.loc[visits.get("rano_status").notna(), "patient_id"].nunique())
            if "rano_status" in visits.columns else 0,
    }])


def molecular_pairing(bundle: CanonicalBundle) -> pd.DataFrame:
    """Paired-specimen counts -- the basis of any plasticity analysis."""
    mol = bundle.molecular_samples
    if mol.empty:
        return pd.DataFrame([{"cohort": bundle.cohort, "n_specimens": 0,
                              "n_patients_with_specimens": 0,
                              "n_patients_with_paired_specimens": 0}])
    per_patient = mol.groupby("patient_id").size()
    roles = (mol.groupby("patient_id")["sample_role"]
             .apply(lambda s: set(s.dropna())))
    paired = roles.apply(
        lambda r: ("primary" in r) and bool(
            r & {"first_recurrence", "second_recurrence"}
            | {x for x in r if str(x).startswith("recurrence")}))
    rows = {
        "cohort": bundle.cohort,
        "n_specimens": len(mol),
        "n_patients_with_specimens": int(mol["patient_id"].nunique()),
        "n_patients_with_2plus_specimens": int((per_patient >= 2).sum()),
        "n_patients_with_paired_primary_recurrent": int(paired.sum()),
        "n_with_collection_day": int(
            mol["tissue_collection_relative_day"].notna().sum()),
        "assays": ", ".join(sorted(mol["assay"].dropna().unique())),
    }
    for role, n in mol["sample_role"].value_counts().items():
        rows[f"n_role_{role}"] = int(n)
    return pd.DataFrame([rows])


def outcome_summary(bundle: CanonicalBundle) -> pd.DataFrame:
    """Per-outcome counts, events, censoring status and adjudication basis."""
    out = bundle.outcomes
    if out.empty:
        return pd.DataFrame(columns=["cohort", "outcome_name", "outcome_type",
                                     "n_rows", "n_patients"])
    rows = []
    for (name, otype), grp in out.groupby(["outcome_name", "outcome_type"],
                                          dropna=False):
        record = {
            "cohort": bundle.cohort,
            "outcome_name": name,
            "outcome_type": otype,
            "n_rows": len(grp),
            "n_patients": int(grp["patient_id"].nunique()),
            "n_value_present": int(grp["outcome_value"].notna().sum()),
            "pct_missing_value": round(100 * grp["outcome_value"].isna().mean(), 2),
        }
        if otype == "time_to_event":
            ev = grp["event_indicator"]
            record.update({
                "n_events": int((ev == True).sum()),      # noqa: E712
                "n_censored": int((ev == False).sum()),   # noqa: E712
                "n_censoring_unknown": int(ev.isna().sum()),
                "modellable": bool(ev.notna().any()),
                "modelling_note": (
                    "usable for penalised Cox"
                    if ev.notna().any() else
                    "NOT usable: no censoring indicator released"),
            })
        if otype in {"binary", "categorical"}:
            counts = grp["outcome_label"].value_counts(dropna=False).to_dict()
            record["label_counts"] = "; ".join(
                f"{k}={v}" for k, v in counts.items())
            pos = grp["outcome_value"] == 1.0
            record["n_positive"] = int(pos.sum())
            record["positive_rate"] = (round(float(pos.mean()), 4)
                                       if grp["outcome_value"].notna().any()
                                       else np.nan)
        asc = grp["ascertainment"].value_counts(dropna=False).to_dict()
        record["ascertainment"] = "; ".join(f"{k}={v}" for k, v in asc.items())
        rows.append(record)
    return pd.DataFrame(rows)


def cohort_overview(adapter: DatasetAdapter,
                    bundle: CanonicalBundle | None = None) -> pd.DataFrame:
    """One row describing a cohort: what it is, what it has, what it lacks."""
    desc = adapter.describe()
    buildable, missing = adapter.is_buildable()
    record = {
        "dataset": adapter.key,
        "cohort": desc.key,
        "name": desc.name,
        "url": desc.url,
        "requires_registration": desc.requires_registration,
        "license": desc.license,
        "modalities": ", ".join(desc.modalities),
        "n_patients_published": desc.n_patients_published,
        "n_timepoints_published": desc.n_timepoints_published,
        "t0_definition": desc.t0_definition,
        "raw_present": buildable,
        "missing_required_files": "; ".join(missing),
        "known_limitations": " | ".join(desc.known_limitations),
    }
    if bundle is not None:
        for name, table in bundle.items():
            record[f"n_{name}"] = len(table)
        record["n_patients_local"] = int(bundle.patients["patient_id"].nunique()) \
            if len(bundle.patients) else 0
    return pd.DataFrame([record])


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_audit(
    adapter: DatasetAdapter,
    bundle: CanonicalBundle,
    outdir: Path,
) -> dict[str, Path]:
    """Write every audit artefact for one cohort; returns written paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artefacts = {
        "cohort_overview": cohort_overview(adapter, bundle),
        "source_availability": adapter.availability(),
        "source_data_dictionary": adapter.data_dictionary(),
        "canonical_data_dictionary": bundle_dictionary(bundle),
        "id_uniqueness": id_uniqueness(bundle),
        "modality_completeness": modality_completeness(bundle),
        "longitudinal_coverage": longitudinal_coverage(bundle),
        "molecular_pairing": molecular_pairing(bundle),
        "outcome_summary": outcome_summary(bundle),
    }
    written: dict[str, Path] = {}
    for name, frame in artefacts.items():
        path = outdir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path

    matrix = visit_modality_matrix(bundle)
    if not matrix.empty:
        path = outdir / "visit_modality_matrix.csv"
        matrix.to_csv(path, index=False)
        written["visit_modality_matrix"] = path

    notes = pd.DataFrame(
        [{"key": k, "note": v} for k, v in sorted(bundle.notes.items())])
    if not notes.empty:
        path = outdir / "cohort_notes.csv"
        notes.to_csv(path, index=False)
        written["cohort_notes"] = path
    return written


def _example(series: pd.Series) -> object:
    if len(series) == 0:
        return pd.NA
    text = str(series.iloc[0])
    return text if len(text) <= 60 else text[:57] + "..."
