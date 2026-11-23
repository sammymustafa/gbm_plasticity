"""Canonical cross-cohort table schemas.

Every dataset adapter emits the same seven tables.  A cohort that cannot
populate a column leaves it *absent as NA* -- missingness is preserved and
never imputed at the adapter layer.

Design rules
------------
1. ``patient_id`` is globally unique because every adapter prefixes the
   cohort key (``LUMIERE::Patient-001``).  This makes an accidental
   cross-cohort join impossible to perform silently: the ids simply never
   match.
2. All times are *relative* integers in days from a cohort-defined anchor
   (``t0_definition`` in the cohort config), because none of the public
   cohorts release calendar dates.  ``relative_time_days`` may be negative
   (pre-operative imaging).
3. Columns suffixed ``_source`` record provenance; columns suffixed
   ``_is_imputed`` are never written by adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------

#: Where a visit sits relative to standard-of-care treatment.
TREATMENT_PHASE = (
    "pre_operative",
    "post_operative",          # after resection, before chemoradiation
    "chemoradiation",
    "adjuvant",                # adjuvant temozolomide / maintenance
    "recurrence",
    "unknown",
)

#: RANO-style response categories plus the two surgical anchors used by LUMIERE.
RANO_STATUS = ("CR", "PR", "SD", "PD", "Pre-Op", "Post-Op", "unknown")

#: Imaging modalities in the canonical ``imaging`` table.
MODALITY = (
    "T1",       # pre-contrast T1
    "T1CE",     # post-contrast T1  (LUMIERE 'CT1')
    "T2",
    "FLAIR",
    "DWI",
    "DTI",
    "DSC",      # dynamic susceptibility contrast perfusion
    "DCE",
    "ASL",
    "RSI",      # restriction spectrum imaging (UCSD)
    "SEG",      # segmentation label map
    "other",
)

#: Tumour subregions.  Mapping from cohort-specific label integers to these
#: names lives in each adapter, never in analysis code.
SUBREGION = ("enhancing", "non_enhancing", "necrosis", "edema", "whole_tumor")

#: Assay families for molecular specimens.
ASSAY = (
    "snRNA_seq",
    "scRNA_seq",
    "bulk_RNA_seq",
    "bulk_DNA_seq",
    "WES",
    "WGS",
    "methylation",
    "IHC",
    "targeted_panel",
    "other",
)

#: Outcome ascertainment basis -- critical for UCSD where some labels are
#: histopathology-proven and others are consensus or follow-up based.
ASCERTAINMENT = (
    "histopathology",
    "clinical_consensus",
    "radiologic_followup",
    "expert_read",
    "registry",
    "unknown",
)

TableName = Literal[
    "patients", "visits", "imaging", "treatments",
    "symptoms", "molecular_samples", "outcomes",
]

CANONICAL_TABLES: tuple[str, ...] = (
    "patients", "visits", "imaging", "treatments",
    "symptoms", "molecular_samples", "outcomes",
)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool
    description: str
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TableSpec:
    name: str
    primary_key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    description: str = ""

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def empty(self) -> pd.DataFrame:
        """An empty frame with the full canonical column set and dtypes."""
        data = {}
        for c in self.columns:
            data[c.name] = pd.Series(dtype=_pandas_dtype(c.dtype))
        return pd.DataFrame(data)

    def conform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing optional columns as NA and order columns canonically.

        Extra (cohort-specific) columns are *kept*, appended after the
        canonical block, so no information is discarded.
        """
        out = df.copy()
        for c in self.columns:
            if c.name not in out.columns:
                out[c.name] = pd.Series([pd.NA] * len(out), index=out.index,
                                        dtype=_pandas_dtype(c.dtype))
        canonical = self.column_names
        extra = [c for c in out.columns if c not in canonical]
        return out[canonical + extra]


def _pandas_dtype(dtype: str) -> str:
    return {
        "str": "string",
        "int": "Int64",
        "float": "Float64",
        "bool": "boolean",
    }.get(dtype, "object")


def _c(name: str, dtype: str, required: bool, desc: str,
       allowed: tuple[str, ...] | None = None) -> ColumnSpec:
    return ColumnSpec(name=name, dtype=dtype, required=required,
                      description=desc, allowed=allowed)


# --------------------------------------------------------------------------
# Table definitions
# --------------------------------------------------------------------------

PATIENTS = TableSpec(
    name="patients",
    primary_key=("patient_id",),
    description="One row per patient. Demographics and baseline molecular markers.",
    columns=(
        _c("patient_id", "str", True, "Globally unique, '<COHORT>::<local id>'."),
        _c("cohort", "str", True, "Cohort key, e.g. 'LUMIERE'."),
        _c("local_patient_id", "str", True, "Identifier as published by the cohort."),
        _c("sex", "str", False, "'male' | 'female' | NA."),
        _c("age_at_diagnosis_years", "float", False, "Age at index surgery/diagnosis."),
        _c("idh_status", "str", False, "'wildtype' | 'mutant' | NA."),
        _c("idh_method", "str", False, "Assay used to call IDH."),
        _c("mgmt_status", "str", False, "'methylated' | 'unmethylated' | NA."),
        _c("mgmt_value", "float", False, "Quantitative MGMT methylation where released."),
        _c("mgmt_method", "str", False, "Assay used to call MGMT."),
        _c("who_grade", "str", False, "WHO grade as published."),
        _c("histology", "str", False, "Histologic diagnosis as published."),
        _c("t0_definition", "str", True, "What relative day 0 means for this patient."),
        _c("site", "str", False, "Contributing centre, where released."),
    ),
)

VISITS = TableSpec(
    name="visits",
    primary_key=("visit_id",),
    description="One row per clinical/imaging timepoint. The longitudinal spine.",
    columns=(
        _c("visit_id", "str", True, "Globally unique visit key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("local_visit_id", "str", True, "Timepoint label as published (e.g. 'week-044')."),
        _c("relative_time_days", "int", False,
           "Days from the patient's t0 anchor. May be negative. NA when the "
           "cohort releases a study whose offset it does not know (e.g. 8 "
           "UPENN-GBM recurrence scans); such visits are excluded from "
           "longitudinal analyses rather than imputed."),
        _c("relative_time_days_precision", "str", False,
           "'day' | 'week' | 'month' -- LUMIERE timepoints are week-resolution."),
        _c("visit_index", "int", True, "0-based ordinal of the visit within the patient."),
        _c("treatment_phase", "str", False, "Phase at this visit.", TREATMENT_PHASE),
        _c("rano_status", "str", False, "Expert RANO rating if adjudicated.", RANO_STATUS),
        _c("rano_rationale", "str", False, "Free-text rationale as published."),
        _c("rano_lt_3_months", "bool", False,
           "Cohort flag: interval to prior study < 3 months."),
        _c("rano_nonmeasurable", "bool", False, "Cohort flag: non-measurable lesions only."),
        _c("has_imaging", "bool", False, "Any imaging series attached to this visit."),
        _c("has_molecular", "bool", False, "Any specimen attached to this visit."),
    ),
)

IMAGING = TableSpec(
    name="imaging",
    primary_key=("imaging_id",),
    description="One row per (visit, modality) image series or derived map.",
    columns=(
        _c("imaging_id", "str", True, "Globally unique series key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("visit_id", "str", True, "FK -> visits.visit_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("modality", "str", True, "Canonical modality.", MODALITY),
        _c("local_modality", "str", False, "Modality label as published (e.g. 'CT1')."),
        _c("acquisition_date_relative", "int", False,
           "Days from t0; equals the visit time when only visit-level timing exists."),
        _c("available", "bool", False,
           "Whether the cohort states this series exists. NA means the cohort "
           "documents the acquisition protocol but publishes no per-scan "
           "availability flag (e.g. UCSD-PTGBM) -- unknown, not absent."),
        _c("relative_path", "str", False, "Path within the cohort's raw tree."),
        _c("series_uid", "str", False, "DICOM SeriesInstanceUID where released."),
        _c("manufacturer", "str", False, ""),
        _c("model", "str", False, ""),
        _c("field_strength_t", "float", False, ""),
        _c("slice_thickness_mm", "float", False, ""),
        _c("in_plane_spacing_mm", "str", False, "Row/col spacing as published."),
        _c("rows", "int", False, ""),
        _c("columns", "int", False, ""),
        _c("slice_count", "int", False, ""),
        _c("echo_time_ms", "float", False, ""),
        _c("repetition_time_ms", "float", False, ""),
        _c("inversion_time_ms", "float", False, ""),
        _c("flip_angle_deg", "float", False, ""),
        _c("is_preprocessed", "bool", False,
           "True when the cohort already skull-stripped/registered this series."),
        _c("preprocessing_note", "str", False, "What the cohort already did."),
        _c("segmentation_source", "str", False,
           "Tool or reader that produced an associated segmentation."),
    ),
)

TREATMENTS = TableSpec(
    name="treatments",
    primary_key=("treatment_id",),
    description="One row per treatment episode.",
    columns=(
        _c("treatment_id", "str", True, "Globally unique treatment key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("treatment_type", "str", True,
           "'surgery' | 'radiotherapy' | 'chemotherapy' | 'immunotherapy' | "
           "'steroid' | 'device' | 'other'."),
        _c("treatment_name", "str", False, "Agent/procedure as published."),
        _c("start_relative_day", "int", False, "Days from t0."),
        _c("end_relative_day", "int", False, "Days from t0."),
        _c("treatment_phase", "str", False, "Phase this episode defines.", TREATMENT_PHASE),
        _c("dose", "float", False, ""),
        _c("dose_unit", "str", False, ""),
        _c("extent_of_resection", "str", False, "'GTR' | 'STR' | 'biopsy' | NA."),
        _c("is_inferred", "bool", False,
           "True when the episode is inferred from another field rather than "
           "released as a treatment record."),
    ),
)

SYMPTOMS = TableSpec(
    name="symptoms",
    primary_key=("symptom_id",),
    description=(
        "One row per (patient, symptom, assessment). Empty for every public "
        "cohort used here -- populated only by the institutional adapter."
    ),
    columns=(
        _c("symptom_id", "str", True, "Globally unique symptom-observation key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("visit_id", "str", False, "FK -> visits.visit_id when visit-anchored."),
        _c("symptom", "str", True, "Controlled term, see gbmplast.symptoms.schema."),
        _c("present", "bool", False, "True/False/NA. NA means not assessed."),
        _c("severity", "str", False, "Cohort-defined severity grade."),
        _c("assessment_relative_day", "int", False, "Days from t0 of the assessment."),
        _c("onset_relative_day", "int", False, "Days from t0 of symptom onset."),
        _c("resolution_relative_day", "int", False, "Days from t0 of resolution."),
        _c("source", "str", True, "'structured' | 'note_extraction' | 'manual_review'."),
        _c("provenance", "str", False,
           "For extracted values: document id + character offsets."),
        _c("negated", "bool", False, "Extraction flag: term appeared negated."),
        _c("temporality", "str", False, "'current' | 'historical' | 'hypothetical'."),
        _c("validated_by_human", "bool", False, "Manual-validation hook."),
    ),
)

MOLECULAR_SAMPLES = TableSpec(
    name="molecular_samples",
    primary_key=("specimen_id",),
    description="One row per assayed specimen.",
    columns=(
        _c("specimen_id", "str", True, "Globally unique specimen key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("visit_id", "str", False, "FK -> visits.visit_id when linkable."),
        _c("local_specimen_id", "str", True, "Specimen id as published."),
        _c("assay", "str", True, "Assay family.", ASSAY),
        _c("platform", "str", False, "Sequencing/array platform."),
        _c("tissue_collection_relative_day", "int", False,
           "Days from t0 at which tissue was taken. NA when the cohort "
           "releases only ordinal timepoints."),
        _c("timepoint_index", "int", False,
           "Ordinal specimen index within the patient (1 = first resection)."),
        _c("treatment_phase", "str", False, "Phase at collection.", TREATMENT_PHASE),
        _c("sample_role", "str", False,
           "'primary' | 'first_recurrence' | 'second_recurrence' | 'other'."),
        _c("is_paired", "bool", False,
           "True when this patient has >=2 specimens in this cohort."),
        _c("accession", "str", False, "Public accession (e.g. GSM id)."),
        _c("file_name", "str", False, "Released data file for this specimen."),
        _c("n_cells", "int", False, "Nuclei/cells passing the cohort's own QC."),
    ),
)

OUTCOMES = TableSpec(
    name="outcomes",
    primary_key=("outcome_id",),
    description=(
        "One row per (patient, outcome). Survival rows carry an explicit "
        "event indicator; when a cohort releases times *without* a censoring "
        "flag, event_indicator stays NA and survival modelling is refused "
        "downstream rather than assumed."
    ),
    columns=(
        _c("outcome_id", "str", True, "Globally unique outcome key."),
        _c("patient_id", "str", True, "FK -> patients.patient_id."),
        _c("cohort", "str", True, "Cohort key."),
        _c("visit_id", "str", False, "FK -> visits.visit_id for visit-level outcomes."),
        _c("outcome_name", "str", True,
           "'overall_survival' | 'progression_free_survival' | "
           "'rano_progression' | 'tumor_vs_treatment_change' | ..."),
        _c("outcome_type", "str", True, "'time_to_event' | 'binary' | 'categorical' | 'continuous'."),
        _c("outcome_value", "float", False, "Numeric value (days for time_to_event)."),
        _c("outcome_label", "str", False, "Categorical/binary label as published."),
        _c("event_indicator", "bool", False,
           "time_to_event only: True = event observed, False = censored, "
           "NA = cohort released no censoring information."),
        _c("ascertainment", "str", False, "How the outcome was established.", ASCERTAINMENT),
        _c("ascertainment_relative_day", "int", False,
           "Days from t0 at which the outcome was ascertained."),
        _c("followup_relative_day", "int", False, "Last known follow-up, days from t0."),
        _c("notes", "str", False, "Cohort caveats for this row."),
    ),
)

SPECS: dict[str, TableSpec] = {
    "patients": PATIENTS,
    "visits": VISITS,
    "imaging": IMAGING,
    "treatments": TREATMENTS,
    "symptoms": SYMPTOMS,
    "molecular_samples": MOLECULAR_SAMPLES,
    "outcomes": OUTCOMES,
}


@dataclass
class CanonicalBundle:
    """The seven canonical tables for one cohort."""

    cohort: str
    patients: pd.DataFrame
    visits: pd.DataFrame
    imaging: pd.DataFrame
    treatments: pd.DataFrame
    symptoms: pd.DataFrame
    molecular_samples: pd.DataFrame
    outcomes: pd.DataFrame
    notes: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        if name not in CANONICAL_TABLES:
            raise KeyError(f"{name!r} is not a canonical table: {CANONICAL_TABLES}")
        return getattr(self, name)

    def items(self):
        for name in CANONICAL_TABLES:
            yield name, self.table(name)

    def summary(self) -> pd.DataFrame:
        rows = [{"table": n, "n_rows": len(t), "n_cols": t.shape[1]}
                for n, t in self.items()]
        return pd.DataFrame(rows)


def make_patient_id(cohort: str, local_id: str) -> str:
    """Namespace a local id so cross-cohort joins cannot silently succeed."""
    return f"{cohort}::{local_id}"


def make_key(cohort: str, *parts: str) -> str:
    return "::".join([cohort, *[str(p) for p in parts]])
