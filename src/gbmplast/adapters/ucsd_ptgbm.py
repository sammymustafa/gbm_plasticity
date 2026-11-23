"""UCSD-PTGBM adapter (post-treatment high-grade glioma).

Source
------
Gagnon L, et al. *The University of California San Diego annotated
post-treatment high-grade glioma multimodal MRI dataset (UCSD-PTGBM).*
TCIA (2025/2026).  Collection page:
https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/

Verified structure
------------------
``UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx``
    Sheet ``Cinical info`` (sic): 243 rows keyed ``UCSD-PTGBM-NNNN_TT``
    -- 178 patients, 1-4 post-treatment timepoints each.  Sheet
    ``Data Dictionary`` documents every column; this adapter follows it
    rather than guessing.

    All times are released as **days from that scan's acquisition** and may
    be negative (event before the scan).  The adapter re-anchors them to
    the patient's initial surgery/diagnosis so that every cohort shares one
    convention:  ``day_from_t0(X) = days_from_acq(X) - days_from_acq(surgery)``.

``details_of_negative_cases_TCIA.xlsx``
    The 51 timepoints judged to show **no** residual/recurrent tumour, with
    the basis of that judgement:

    ===========================  ===  ==========================
    Dx                           n    ascertainment
    ===========================  ===  ==========================
    NS (non-specific)            29   radiologic_followup
    PsP (clinically confirmed)    9   clinical_consensus
    Radiation necrosis            7   unknown  (basis not released)
    PsP (bx proven)               6   histopathology
    ===========================  ===  ==========================

    The remaining ``243 - 51 = 192`` timepoints are tumour-positive, which
    matches the collection abstract exactly.

Outcome-label integrity
-----------------------
The abstract states that 45 of the 192 tumour-positive timepoints are
pathology-proven, but **the released tables carry no column identifying
which ones**.  The adapter therefore assigns an adjudication tier to every
*negative* timepoint and leaves the positive timepoints' ascertainment as
``unknown``.  Analyses that need a histopathology-only comparison can use
the negative tiers; they cannot construct a pathology-only positive class
from what is released, and this module refuses to invent one.
"""

from __future__ import annotations

import re
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

COHORT = "UCSD-PTGBM"
CLINICAL_SHEET = "Cinical info"          # spelling as published
DICTIONARY_SHEET = "Data Dictionary"
NEGATIVE_SHEET = "all_negative_cases_TCIA"

_ID_RE = re.compile(r"^(?P<base>UCSD-PTGBM-\d+)_(?P<tp>\d+)$")

_NA_STRINGS = {"unknown", "not available", "na", "n/a", "nan", "", "-", "none"}

#: Negative-case Dx string -> (canonical label, ascertainment tier).
NEGATIVE_DX = {
    "PsP (bx proven)":            ("pseudoprogression", "histopathology"),
    "PsP (clinically confirmed)": ("pseudoprogression", "clinical_consensus"),
    "Radiation necrosis":         ("radiation_necrosis", "unknown"),
    "NS":                         ("nonspecific_treatment_change",
                                   "radiologic_followup"),
}

#: Column holding days from acquisition to each clinical event.
DAYS_FROM_ACQ = {
    "initial_surgery": "Days from Acquisition to Date of initial surgery, "
                       "treatment or diagnosis",
    "last_surgery_prior": "Days from Acquisition to Date of last surgery prior "
                          "to scan if different",
    "last_followup": "Days from Acquisition to Date of last follow-up",
    "death": "Days from Acquisition to Date of death",
    "first_radiation": "Days from Acquistion to Date of first radiation",
    "last_radiation_prior": "Days from Acquistion to Date of last radiation "
                            "prior to scan",
    "first_chemo": "Days from Acquisition to Date of 1st chemo start",
    "last_avastin_prior": "Days from Acquisition to Date of last Avastin "
                          "treatment prior to scan",
    "other_treatment": "Days from Acquisition to Other treatment dates",
}


def _clean(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _NA_STRINGS else text


def _num(value: object) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_scan_id(scan_id: str) -> tuple[str, int]:
    m = _ID_RE.match(str(scan_id).strip())
    if not m:
        raise ValueError(f"Unparseable UCSD-PTGBM id: {scan_id!r}")
    return m.group("base"), int(m.group("tp"))


@register
class UcsdPtgbmAdapter(DatasetAdapter):
    key = "ucsd_ptgbm"

    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=COHORT,
            name="UCSD-PTGBM: annotated post-treatment high-grade glioma multimodal MRI",
            url="https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/",
            access=(
                "Clinical tables are direct public downloads. The imaging "
                "(33.1 GB NIfTI) requires the IBM Aspera Connect plugin and is "
                "not retrievable from a headless environment without it."
            ),
            license="CC BY 4.0.",
            citation=(
                "Gagnon L, Gupta D, Mastorakos G, White N, Goodwill V, "
                "McDonald C, Beaumont T, Conlin C, Seibert T, Nguyen U, "
                "Hattangadi-Gluth J, Kesari S, Schulte J, Piccioni D, "
                "Schmainda K, Farid N, Dale A, Rudie J. The University of "
                "California San Diego annotated post-treatment high-grade "
                "glioma multimodal MRI dataset (UCSD-PTGBM). TCIA (2025)."
            ),
            modalities=("T1", "T1CE", "FLAIR", "RSI", "DSC", "ASL", "SEG"),
            n_patients_published=178,
            n_timepoints_published=243,
            t0_definition=(
                "Initial surgery / first treatment / diagnosis date. Released "
                "offsets are days from each scan's acquisition and are "
                "re-anchored by the adapter."
            ),
            known_limitations=(
                "Images need IBM Aspera Connect; only the clinical tables are "
                "retrievable programmatically here.",
                "Pathology confirmation is identifiable only for the NEGATIVE "
                "class; the released tables do not flag which tumour-positive "
                "timepoints are biopsy-proven.",
                "Overall survival is populated for 86/243 timepoints and PFS "
                "for 87; initial-surgery offset is unknown for 45.",
                "MGMT is missing for 94 timepoints and IDH for 75.",
                "The collection abstract and the v3 table disagree slightly on "
                "the negative-class breakdown; the adapter follows the table.",
            ),
            requires_registration=False,
            notes={
                "version": "v3 (2026-03-13) corrected mis-assigned patient IDs "
                           "in earlier versions; use v3 only.",
            },
        )

    def source_files(self) -> list[SourceFile]:
        base = "https://www.cancerimagingarchive.net/wp-content/uploads/"
        return [
            SourceFile("clinical",
                       "UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx",
                       "Per-timepoint clinical, molecular and treatment table "
                       "plus the official data dictionary.",
                       url=base + "UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx"),
            SourceFile("negative_cases", "details_of_negative_cases_TCIA.xlsx",
                       "The 51 timepoints without residual/recurrent tumour and "
                       "the basis of each judgement.",
                       url=base + "details_of_negative_cases_TCIA.xlsx"),
            SourceFile("bvec", "UCSD-PTGBM-bval_bvec_20260109.txt",
                       "Diffusion b-values/vectors for the RSI acquisition.",
                       required=False,
                       url=base + "UCSD-PTGBM-bval_bvec_20260109.txt"),
        ]

    # -- raw readers ------------------------------------------------------
    def _clinical(self) -> pd.DataFrame:
        path = self.raw_dir / "UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx"
        df = pd.read_excel(path, sheet_name=CLINICAL_SHEET)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def data_dictionary_official(self) -> pd.DataFrame:
        """The cohort's own data dictionary sheet, de-duplicated."""
        path = self.raw_dir / "UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx"
        dd = pd.read_excel(path, sheet_name=DICTIONARY_SHEET)
        dd = dd.dropna(how="all").drop_duplicates()
        dd.columns = ["column", "description"]
        return dd.reset_index(drop=True)

    def _negatives(self) -> pd.DataFrame:
        path = self.raw_dir / "details_of_negative_cases_TCIA.xlsx"
        df = pd.read_excel(path, sheet_name=NEGATIVE_SHEET)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()
        clinical = self._clinical()
        negatives = self._negatives()

        clinical = clinical.copy()
        parsed = clinical["ID"].map(split_scan_id)
        clinical["local_patient_id"] = [p for p, _ in parsed]
        clinical["timepoint_index"] = [t for _, t in parsed]
        # Offset of the scan from t0 (initial surgery): negate the released
        # "days from acquisition to surgery".
        surg = clinical[DAYS_FROM_ACQ["initial_surgery"]].map(_num)
        clinical["_days_acq_to_surgery"] = surg
        clinical["scan_day_from_t0"] = surg.map(
            lambda d: -d if d is not None and not pd.isna(d) else np.nan)

        neg_map = self._negative_map(negatives)

        patients = self._build_patients(clinical)
        visits = self._build_visits(clinical, neg_map)
        imaging = self._build_imaging(clinical, visits)
        treatments = self._build_treatments(clinical)
        outcomes = self._build_outcomes(clinical, neg_map)

        bundle = self.empty_bundle(
            patients=patients, visits=visits, imaging=imaging,
            treatments=treatments, outcomes=outcomes,
        )
        bundle.notes.update({
            "symptoms": "UCSD-PTGBM releases no symptom data.",
            "molecular_samples": (
                "MGMT/IDH/1p19q/ATRX are released per timepoint without "
                "specimen ids or collection dates, so molecular_samples is "
                "empty by design and markers live on 'patients'."
            ),
            "outcome_adjudication": (
                "Negative timepoints carry an explicit adjudication tier "
                "(histopathology / clinical_consensus / radiologic_followup / "
                "unknown). Positive timepoints carry 'unknown' because the "
                "released tables do not flag which are biopsy-proven."
            ),
            "time_anchor": (
                "All released offsets are days-from-acquisition; the adapter "
                "re-anchors them to initial surgery. Where the surgery offset "
                "is unknown (45 timepoints) derived day columns stay NA."
            ),
        })
        return bundle

    def _negative_map(self, negatives: pd.DataFrame) -> dict[str, dict]:
        out: dict[str, dict] = {}
        unknown = sorted(set(negatives["Dx"].dropna().astype(str)) - set(NEGATIVE_DX))
        if unknown:
            raise ValueError(
                f"[{COHORT}] unrecognised negative-case Dx values {unknown}; "
                "refusing to guess their adjudication tier."
            )
        for _, r in negatives.iterrows():
            scan_id = _clean(r["ID"])
            dx = _clean(r["Dx"])
            if scan_id is None or dx is None:
                continue
            label, tier = NEGATIVE_DX[dx]
            out[scan_id] = {"label": label, "ascertainment": tier, "dx_raw": dx}
        return out

    def _build_patients(self, clinical: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for local, grp in clinical.groupby("local_patient_id"):
            r = grp.sort_values("timepoint_index").iloc[0]
            age = _num(r["Patient's Age"])
            scan_day = r["scan_day_from_t0"]
            rows.append({
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_patient_id": local,
                "sex": (_clean(r["Sex at birth"]) or "").lower() or pd.NA,
                "race": _clean(r["Race"]),
                "ethnicity": _clean(r["Ethnicity"]),
                # Age is released at time of scan; convert to diagnosis age only
                # when the scan's offset from surgery is known.
                "age_at_diagnosis_years": (
                    age - scan_day / 365.25
                    if age is not None and not pd.isna(scan_day) else pd.NA
                ),
                "age_at_first_scan_years": age,
                "idh_status": _map_idh(_clean(r["IDH"])),
                "idh_status_raw": _clean(r["IDH"]),
                "mgmt_status": _map_mgmt(_clean(r["MGMT"])),
                "mgmt_status_raw": _clean(r["MGMT"]),
                "codeletion_1p19q": _clean(r["1p19q"]),
                "atrx_status": _clean(r["ATRX"]),
                "who_grade": _clean(r["Grade"]),
                "histology": _clean(r["Primary Diagnosis"]),
                "who_2021_diagnosis": _clean(r["WHO 2021 Diagnosis"]),
                "t0_definition": "initial surgery / first treatment / diagnosis",
                "n_timepoints": int(len(grp)),
                "site": "University of California San Diego",
            })
        return pd.DataFrame(rows)

    def _build_visits(self, clinical: pd.DataFrame,
                      neg_map: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for _, r in clinical.iterrows():
            scan_id = r["ID"]
            local = r["local_patient_id"]
            day = r["scan_day_from_t0"]
            neg = neg_map.get(scan_id)
            rows.append({
                "visit_id": make_key(COHORT, scan_id),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_visit_id": scan_id,
                "local_patient_id": local,
                "timepoint_index": int(r["timepoint_index"]),
                "relative_time_days": (int(day) if not pd.isna(day) else pd.NA),
                "relative_time_days_precision": "day",
                # Every released scan is post-treatment by construction.
                "treatment_phase": "post_operative" if neg else "recurrence",
                "tumor_status": ("treatment_related_change" if neg
                                 else "tumor_positive"),
                "tumor_status_detail": neg["label"] if neg else "residual_or_recurrent_tumor",
                "tumor_status_dx_raw": neg["dx_raw"] if neg else pd.NA,
                "tumor_status_ascertainment": (neg["ascertainment"] if neg
                                               else "unknown"),
                "days_acq_to_last_surgery": _num(r[DAYS_FROM_ACQ["last_surgery_prior"]]),
                "days_acq_to_last_radiation": _num(r[DAYS_FROM_ACQ["last_radiation_prior"]]),
                "days_acq_to_last_avastin": _num(r[DAYS_FROM_ACQ["last_avastin_prior"]]),
                "field_strength_t": _num(r["Magnetic Field Strength"]),
                "manufacturer": _clean(r["Manufacturer"]),
                "model": _clean(r["Manufacturer's Model Name"]),
                "brats_id": _clean(r["BraTS ID"]),
                "has_imaging": True,
                "has_molecular": False,
            })
        visits = pd.DataFrame(rows)
        visits = visits.sort_values(
            ["patient_id", "timepoint_index"]).reset_index(drop=True)
        visits["visit_index"] = visits.groupby("patient_id").cumcount()
        return visits

    def _build_imaging(self, clinical: pd.DataFrame,
                       visits: pd.DataFrame) -> pd.DataFrame:
        """Modality rows reflect the published acquisition protocol.

        The clinical table carries no per-scan modality-availability flags, so
        ``available`` is NA (unknown) rather than True: the protocol says these
        sequences were acquired, but the tables do not confirm it per scan.
        """
        times = visits.set_index("local_visit_id")["relative_time_days"].to_dict()
        protocol = [
            ("T1", "T1", "3D IR-SPGR pre-gadolinium"),
            ("T1CE", "T1_post", "3D IR-SPGR post-gadolinium"),
            ("FLAIR", "FLAIR", "3D T2-FLAIR"),
            ("RSI", "RSI", "Restriction spectrum imaging cellularity map "
                           "(multishell, b=0/500/1500/4000)"),
            ("DSC", "DSC", "DSC perfusion, leakage-corrected CBV"),
            ("ASL", "ASL", "Arterial spin labelling (GE in-line analysis)"),
            ("SEG", "SEG", "Neuroradiologist-approved voxelwise tumour segmentation"),
        ]
        rows = []
        for _, r in clinical.iterrows():
            scan_id = r["ID"]
            local = r["local_patient_id"]
            for modality, local_mod, note in protocol:
                rows.append({
                    "imaging_id": make_key(COHORT, scan_id, local_mod),
                    "patient_id": make_patient_id(COHORT, local),
                    "visit_id": make_key(COHORT, scan_id),
                    "cohort": COHORT,
                    "modality": modality,
                    "local_modality": local_mod,
                    "acquisition_date_relative": times.get(scan_id, pd.NA),
                    # Unknown per scan: protocol-level, not table-confirmed.
                    "available": pd.NA,
                    "relative_path": f"{scan_id}/{local_mod}.nii.gz",
                    "field_strength_t": _num(r["Magnetic Field Strength"]),
                    "manufacturer": _clean(r["Manufacturer"]),
                    "model": _clean(r["Manufacturer's Model Name"]),
                    "is_preprocessed": True,
                    "preprocessing_note": (
                        f"{note}. All modalities registered and resampled to a "
                        "256x256x256 1 mm isotropic MNI atlas; FLAIR-derived "
                        "nnU-Net brain mask applied."
                    ),
                    "segmentation_source": ("neuroradiologist_approved"
                                            if modality == "SEG" else pd.NA),
                })
        return pd.DataFrame(rows)

    def _build_treatments(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """Treatment episodes re-anchored from days-from-acquisition to days-from-t0."""
        rows = []
        for local, grp in clinical.groupby("local_patient_id"):
            r = grp.sort_values("timepoint_index").iloc[0]
            pid = make_patient_id(COHORT, local)
            d_surg = r["_days_acq_to_surgery"]

            def to_t0(column_key: str, row=r) -> float | None:
                """days-from-acquisition -> days-from-initial-surgery."""
                d = _num(row[DAYS_FROM_ACQ[column_key]])
                if d is None or d_surg is None or pd.isna(d_surg):
                    return None
                return d - d_surg

            if _clean(r["Surgery"]) == "Yes":
                rows.append({
                    "treatment_id": make_key(COHORT, local, "surgery"),
                    "patient_id": pid, "cohort": COHORT,
                    "treatment_type": "surgery",
                    "treatment_name": "resection",
                    "start_relative_day": 0,
                    "treatment_phase": "post_operative",
                    "extent_of_resection": _clean(r["Surgery extend"]),
                    "n_episodes": _num(r["Number of surgeries"]),
                    "is_inferred": False,
                })
            if _clean(r["Radiation"]) == "Yes":
                rows.append({
                    "treatment_id": make_key(COHORT, local, "radiotherapy"),
                    "patient_id": pid, "cohort": COHORT,
                    "treatment_type": "radiotherapy",
                    "treatment_name": "radiation",
                    "start_relative_day": _as_int(to_t0("first_radiation")),
                    "end_relative_day": _as_int(to_t0("last_radiation_prior")),
                    "treatment_phase": "chemoradiation",
                    "n_episodes": _num(r["Number of radiation courses"]),
                    "is_inferred": False,
                })
            chemo = _clean(r["1st Chemo type"])
            if chemo:
                rows.append({
                    "treatment_id": make_key(COHORT, local, "chemotherapy"),
                    "patient_id": pid, "cohort": COHORT,
                    "treatment_type": "chemotherapy",
                    "treatment_name": chemo,
                    "start_relative_day": _as_int(to_t0("first_chemo")),
                    "treatment_phase": "chemoradiation",
                    "is_inferred": False,
                })
            if (_clean(r["Avastin® (bevacizumab)"]) or "").lower() in {"yes", "y"}:
                rows.append({
                    "treatment_id": make_key(COHORT, local, "bevacizumab"),
                    "patient_id": pid, "cohort": COHORT,
                    "treatment_type": "chemotherapy",
                    "treatment_name": "bevacizumab",
                    "end_relative_day": _as_int(to_t0("last_avastin_prior")),
                    "treatment_phase": "recurrence",
                    "is_inferred": False,
                })
            if _clean(r["Other treatment prior to scan"]) == "Yes":
                rows.append({
                    "treatment_id": make_key(COHORT, local, "other"),
                    "patient_id": pid, "cohort": COHORT,
                    "treatment_type": "other",
                    "treatment_name": "other (unspecified)",
                    "start_relative_day": _as_int(to_t0("other_treatment")),
                    "treatment_phase": "unknown",
                    "is_inferred": False,
                })
        return pd.DataFrame(rows)

    def _build_outcomes(self, clinical: pd.DataFrame,
                        neg_map: dict[str, dict]) -> pd.DataFrame:
        rows = []
        # Patient-level survival (OS/PFS are defined from initial surgery).
        for local, grp in clinical.groupby("local_patient_id"):
            r = grp.sort_values("timepoint_index").iloc[0]
            pid = make_patient_id(COHORT, local)
            d_surg = r["_days_acq_to_surgery"]
            d_death = _num(r[DAYS_FROM_ACQ["death"]])
            d_fu = _num(r[DAYS_FROM_ACQ["last_followup"]])
            event = d_death is not None
            os_days = _num(r["Overall survival"])
            if os_days is None and event and d_surg is not None and not pd.isna(d_surg):
                os_days = d_death - d_surg
            if os_days is None and not event and d_fu is not None \
                    and d_surg is not None and not pd.isna(d_surg):
                os_days = d_fu - d_surg
            rows.append({
                "outcome_id": make_key(COHORT, local, "overall_survival"),
                "patient_id": pid, "cohort": COHORT,
                "outcome_name": "overall_survival",
                "outcome_type": "time_to_event",
                "outcome_value": os_days if os_days is not None else pd.NA,
                "outcome_unit": "days",
                "event_indicator": event,
                "ascertainment": "registry",
                "followup_relative_day": _as_int(
                    (d_fu - d_surg) if d_fu is not None and d_surg is not None
                    and not pd.isna(d_surg) else None),
                "notes": ("Days between initial surgery and death (per the "
                          "cohort data dictionary); censored at last follow-up."),
            })
            pfs = _num(r["Progression Free survival"])
            if pfs is not None:
                rows.append({
                    "outcome_id": make_key(COHORT, local, "pfs"),
                    "patient_id": pid, "cohort": COHORT,
                    "outcome_name": "progression_free_survival",
                    "outcome_type": "time_to_event",
                    "outcome_value": pfs,
                    "outcome_unit": "days",
                    # The table gives a PFS time but no progression-event flag.
                    "event_indicator": pd.NA,
                    "ascertainment": "unknown",
                    "notes": ("Days between initial surgery and first sign of "
                              "progression. No progression-event indicator is "
                              "released, so this row is not usable for "
                              "time-to-event modelling."),
                })

        # Timepoint-level tumour vs treatment-related change (Module B target).
        for _, r in clinical.iterrows():
            scan_id = r["ID"]
            neg = neg_map.get(scan_id)
            label = "treatment_related_change" if neg else "tumor_positive"
            rows.append({
                "outcome_id": make_key(COHORT, scan_id, "tumor_vs_treatment"),
                "patient_id": make_patient_id(COHORT, r["local_patient_id"]),
                "cohort": COHORT,
                "visit_id": make_key(COHORT, scan_id),
                "outcome_name": "tumor_vs_treatment_change",
                "outcome_type": "binary",
                "outcome_value": float(label == "tumor_positive"),
                "outcome_label": label,
                "ascertainment": neg["ascertainment"] if neg else "unknown",
                "ascertainment_relative_day": (
                    int(r["scan_day_from_t0"])
                    if not pd.isna(r["scan_day_from_t0"]) else pd.NA),
                "outcome_detail": neg["label"] if neg else "residual_or_recurrent_tumor",
                "notes": (
                    "Positive class = residual/recurrent tumour. The released "
                    "tables do not flag which positives are biopsy-proven, so "
                    "their ascertainment is 'unknown'."
                ),
            })
        return pd.DataFrame(rows)


# -- helpers ---------------------------------------------------------------

def _as_int(value: float | None) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    return int(value)


def _map_idh(value: str | None) -> object:
    if value is None:
        return pd.NA
    low = value.lower()
    if low.startswith("wild"):
        return "wildtype"
    if low.startswith("mut"):
        return "mutant"
    return "unknown"


def _map_mgmt(value: str | None) -> object:
    if value is None:
        return pd.NA
    low = value.lower()
    if low.startswith("unmethyl"):
        return "unmethylated"
    if low.startswith("methyl"):
        return "methylated"
    return "unknown"
