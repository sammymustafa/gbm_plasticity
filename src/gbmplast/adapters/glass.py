"""GLASS adapter (optional extension) -- a working interface over gated data.

GLASS (Glioma Longitudinal AnalySiS) publishes longitudinal molecular
profiles of matched initial/recurrent gliomas through Synapse
(``synapse.org/glass``).  Access needs a Synapse account and acceptance of
the consortium's terms, so nothing can be fetched from an unauthenticated
environment.

This adapter is deliberately *not* a placeholder that raises
``NotImplementedError``: it implements the full
:class:`~gbmplast.adapters.base.DatasetAdapter` contract against a
documented, expected file layout.  Point ``cohorts.glass.raw_subdir`` at a
directory holding the downloaded tables and ``build()`` produces canonical
tables exactly like every other cohort.  Until then, ``availability()``
reports what is missing and ``acquisition_instructions()`` says how to get
it; the dependent analysis is skipped rather than faked.

Expected files (names as distributed by GLASS; override in config if the
consortium renames them)
------------------------------------------------------------------------
``clinical_surgeries.tsv``   one row per surgery: ``sample_barcode``,
                             ``case_barcode``, ``surgery_number``,
                             ``surgical_interval_mo``, ``histology``,
                             ``grade``, ``idh_status``,
                             ``codel_status``, ``mgmt_methylation``,
                             ``treatment_tmz``, ``treatment_radiotherapy``
``clinical_cases.tsv``       one row per case: ``case_barcode``, ``case_sex``,
                             ``case_age_diagnosis_years``,
                             ``case_overall_survival_mo``, ``case_vital_status``
``analysis_rnaseq_tpm.tsv``  genes x aliquots expression matrix (optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters.base import (
    CohortDescription,
    DatasetAdapter,
    SourceFile,
    register,
)
from gbmplast.schema import CanonicalBundle, make_key, make_patient_id

COHORT = "GLASS"

_NA = {"", "na", "n/a", "nan", "unknown", "not available", "null"}


def _clean(v: object) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).strip()
    return None if t.lower() in _NA else t


def _num(v: object) -> float | None:
    t = _clean(v)
    if t is None:
        return None
    try:
        return float(t)
    except ValueError:
        return None


@register
class GlassAdapter(DatasetAdapter):
    key = "glass"

    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=COHORT,
            name="GLASS: Glioma Longitudinal AnalySiS consortium",
            url="https://www.synapse.org/glass",
            access=(
                "Synapse account + acceptance of consortium terms required. "
                "Not retrievable from an unauthenticated environment."
            ),
            license="See GLASS consortium terms of use on Synapse.",
            citation=(
                "GLASS Consortium. Glioma through the looking GLASS: molecular "
                "evolution of diffuse gliomas and the Glioma Longitudinal "
                "Analysis consortium. Neuro Oncol (2018); and Varn et al., "
                "Cell (2022)."
            ),
            modalities=("bulk_RNA_seq", "WES", "WGS", "methylation"),
            n_patients_published=None,
            t0_definition="First surgery; GLASS releases surgical intervals in months.",
            known_limitations=(
                "Gated: requires Synapse credentials, so this repository ships "
                "the adapter and acquisition instructions, not the data.",
                "Surgical intervals are released in months, a coarser "
                "resolution than the day-level canonical schema.",
                "Mixed glioma histologies and grades: a GBM-only analysis must "
                "filter on histology/grade explicitly.",
            ),
            requires_registration=True,
            notes={"optional": "Module C extension; not required for the "
                               "primary GBM-CARE plasticity analysis."},
        )

    def source_files(self) -> list[SourceFile]:
        cfg = self.config.get("cohorts.glass.files", {}) or {}
        return [
            SourceFile("surgeries", cfg.get("surgeries", "clinical_surgeries.tsv"),
                       "One row per surgery with interval, histology and markers.",
                       url="https://www.synapse.org/glass", manual=True,
                       note="Synapse account + consortium terms required."),
            SourceFile("cases", cfg.get("cases", "clinical_cases.tsv"),
                       "One row per case with demographics and survival.",
                       url="https://www.synapse.org/glass", manual=True,
                       note="Synapse account + consortium terms required."),
            SourceFile("rnaseq", cfg.get("rnaseq", "analysis_rnaseq_tpm.tsv"),
                       "Gene x aliquot expression matrix.", required=False,
                       url="https://www.synapse.org/glass", manual=True,
                       note="Synapse account + consortium terms required."),
        ]

    def acquisition_instructions(self) -> str:
        return (
            "GLASS acquisition (manual, one time):\n"
            "  1. Create a Synapse account at https://www.synapse.org/register\n"
            "  2. Open https://www.synapse.org/glass and accept the GLASS "
            "conditions for use.\n"
            "  3. Download the clinical tables (clinical_surgeries, "
            "clinical_cases) and, if wanted, the RNA-seq TPM matrix.\n"
            "  4. Place them in:  "
            f"{self.raw_dir}\n"
            "  5. Re-run:  gbmplast canonicalize --dataset glass\n"
            "\nThe synapseclient package can automate step 3 once "
            "authenticated:\n"
            "  pip install synapseclient && synapse login && "
            "synapse get -r <synID>\n"
            "This repository intentionally stores no Synapse credentials."
        )

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()
        files = {s.key: s.resolve(self.raw_dir) for s in self.source_files()}
        surgeries = _read_table(files["surgeries"])
        cases = _read_table(files["cases"])
        surgeries.columns = [c.strip().lower() for c in surgeries.columns]
        cases.columns = [c.strip().lower() for c in cases.columns]

        _require_columns(surgeries, ["case_barcode", "surgery_number"], "surgeries")
        _require_columns(cases, ["case_barcode"], "cases")

        patients = self._build_patients(cases, surgeries)
        visits = self._build_visits(surgeries)
        molecular = self._build_molecular(surgeries)
        treatments = self._build_treatments(surgeries)
        outcomes = self._build_outcomes(cases)

        bundle = self.empty_bundle(patients=patients, visits=visits,
                                   molecular_samples=molecular,
                                   treatments=treatments, outcomes=outcomes)
        bundle.notes.update({
            "imaging": "GLASS releases no imaging in this distribution.",
            "resolution": "Surgical intervals are months; converted to days as "
                          "interval_months * 30.44 and flagged as month precision.",
        })
        return bundle

    def _build_patients(self, cases: pd.DataFrame,
                        surgeries: pd.DataFrame) -> pd.DataFrame:
        first = (surgeries.sort_values("surgery_number")
                 .groupby("case_barcode").first().reset_index())
        merged = cases.merge(first, on="case_barcode", how="left",
                             suffixes=("", "_surg"))
        rows = []
        for _, r in merged.iterrows():
            local = _clean(r["case_barcode"])
            if local is None:
                continue
            rows.append({
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_patient_id": local,
                "sex": (_clean(r.get("case_sex")) or "").lower() or pd.NA,
                "age_at_diagnosis_years": _num(r.get("case_age_diagnosis_years")),
                "idh_status": _map_idh(_clean(r.get("idh_status"))),
                "mgmt_status": _map_mgmt(_clean(r.get("mgmt_methylation"))),
                "who_grade": _clean(r.get("grade")),
                "histology": _clean(r.get("histology")),
                "codeletion_1p19q": _clean(r.get("codel_status")),
                "t0_definition": "first surgery (surgery_number == 1)",
            })
        return pd.DataFrame(rows)

    def _build_visits(self, surgeries: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in surgeries.iterrows():
            local = _clean(r["case_barcode"])
            n = _num(r["surgery_number"])
            if local is None or n is None:
                continue
            months = _num(r.get("surgical_interval_mo"))
            rows.append({
                "visit_id": make_key(COHORT, local, f"S{int(n)}"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_visit_id": f"{local}_S{int(n)}",
                "local_patient_id": local,
                "timepoint_index": int(n),
                "relative_time_days": (int(round(months * 30.44))
                                       if months is not None else pd.NA),
                "relative_time_days_precision": "month",
                "treatment_phase": ("pre_operative" if int(n) == 1
                                    else "recurrence"),
                "sample_role": ("primary" if int(n) == 1
                                else f"recurrence_{int(n) - 1}"),
                "has_imaging": False,
                "has_molecular": True,
            })
        visits = pd.DataFrame(rows)
        if visits.empty:
            return visits
        visits = visits.sort_values(["patient_id", "timepoint_index"]).reset_index(drop=True)
        visits["visit_index"] = visits.groupby("patient_id").cumcount()
        return visits

    def _build_molecular(self, surgeries: pd.DataFrame) -> pd.DataFrame:
        rows = []
        counts = surgeries.groupby("case_barcode").size().to_dict()
        for _, r in surgeries.iterrows():
            local = _clean(r["case_barcode"])
            n = _num(r["surgery_number"])
            if local is None or n is None:
                continue
            barcode = _clean(r.get("sample_barcode")) or f"{local}_S{int(n)}"
            months = _num(r.get("surgical_interval_mo"))
            rows.append({
                "specimen_id": make_key(COHORT, barcode),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "visit_id": make_key(COHORT, local, f"S{int(n)}"),
                "local_specimen_id": barcode,
                "assay": "bulk_RNA_seq",
                "tissue_collection_relative_day": (int(round(months * 30.44))
                                                   if months is not None else pd.NA),
                "timepoint_index": int(n),
                "sample_role": ("primary" if int(n) == 1
                                else f"recurrence_{int(n) - 1}"),
                "treatment_phase": ("pre_operative" if int(n) == 1 else "recurrence"),
                "is_paired": counts.get(r["case_barcode"], 0) >= 2,
            })
        return pd.DataFrame(rows)

    def _build_treatments(self, surgeries: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in surgeries.iterrows():
            local = _clean(r["case_barcode"])
            n = _num(r["surgery_number"])
            if local is None or n is None:
                continue
            months = _num(r.get("surgical_interval_mo"))
            start = int(round(months * 30.44)) if months is not None else pd.NA
            for col, ttype, name in (
                ("treatment_tmz", "chemotherapy", "temozolomide"),
                ("treatment_radiotherapy", "radiotherapy", "radiotherapy"),
            ):
                flag = _clean(r.get(col))
                if flag is None:
                    continue
                if str(flag).strip().lower() not in {"1", "true", "yes", "y"}:
                    continue
                rows.append({
                    "treatment_id": make_key(COHORT, local, f"S{int(n)}", name),
                    "patient_id": make_patient_id(COHORT, local),
                    "cohort": COHORT,
                    "treatment_type": ttype,
                    "treatment_name": name,
                    "start_relative_day": start,
                    "treatment_phase": "adjuvant",
                    "is_inferred": True,
                })
        return pd.DataFrame(rows)

    def _build_outcomes(self, cases: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in cases.iterrows():
            local = _clean(r["case_barcode"])
            if local is None:
                continue
            months = _num(r.get("case_overall_survival_mo"))
            vital = (_clean(r.get("case_vital_status")) or "").lower()
            event = {"dead": True, "deceased": True,
                     "alive": False, "living": False}.get(vital)
            rows.append({
                "outcome_id": make_key(COHORT, local, "overall_survival"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "outcome_name": "overall_survival",
                "outcome_type": "time_to_event",
                "outcome_value": (months * 30.44) if months is not None else pd.NA,
                "outcome_unit": "days",
                "event_indicator": event if event is not None else pd.NA,
                "ascertainment": "registry",
                "notes": "Converted from months (month precision).",
            })
        return pd.DataFrame(rows)


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False, dtype=str)


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{COHORT}] {label} table is missing required columns {missing}. "
            f"Found: {list(df.columns)[:15]}. GLASS occasionally renames "
            "columns between data releases; set cohorts.glass.files in the "
            "config to point at the right files, and check the release notes."
        )


def _map_idh(v: str | None) -> object:
    if v is None:
        return pd.NA
    low = v.lower()
    if "wt" in low or "wildtype" in low or "wild-type" in low:
        return "wildtype"
    if "mut" in low:
        return "mutant"
    return "unknown"


def _map_mgmt(v: str | None) -> object:
    if v is None:
        return pd.NA
    low = v.lower()
    if low.startswith("unmethyl") or low in {"0", "false"}:
        return "unmethylated"
    if low.startswith("methyl") or low in {"1", "true"}:
        return "methylated"
    return "unknown"
