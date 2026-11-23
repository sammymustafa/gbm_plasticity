"""UPENN-GBM adapter.

Source
------
Bakas S, et al. *The University of Pennsylvania glioblastoma (UPenn-GBM)
cohort: advanced MRI, clinical, genomics, & radiomics.* Scientific Data
(2022).  TCIA collection ``UPENN-GBM``.

Verified structure (from the downloaded tables, not assumed)
------------------------------------------------------------
``UPENN-GBM_clinical_info_v2.1.csv`` -- 671 rows keyed by scan id
``UPENN-GBM-XXXXX_SS`` where the suffix encodes the scan, not a visit date:

* ``_11`` (611 rows) -- baseline **pre-operative** scan;
  ``Time_since_baseline_preop == 0`` for every one of them.
* ``_21`` (60 rows)  -- a later **recurrence** scan;
  ``Time_since_baseline_preop`` is its offset in days from the baseline.

630 distinct patients; 41 have both scans, 19 contribute a ``_21`` scan only.

Survival is genuinely censored, unlike LUMIERE:

* ``Survival_Status == 'Deceased'`` (644) -> event observed, time taken from
  ``Survival_from_surgery_days_UPDATED``.
* ``'Alive'`` (17), ``'Lost to Follow-up'`` (7) and
  ``'Deceased - uncertain date of death'`` (3) -> censored, time taken from
  ``Survival_Censor``.

So ``outcomes.event_indicator`` is populated here and penalised Cox models
are admissible for this cohort.

``PsP_TP_score``
    Present for the 60 ``_21`` scans on a 1-6 scale.  The TCIA collection
    page does not publish the direction of the scale (whether 6 means
    "true progression" or "pseudoprogression").  It is therefore carried
    through as an annotated covariate and is **not** used as a modelling
    outcome; the pseudoprogression analysis uses UCSD-PTGBM, whose labels
    are explicit.
"""

from __future__ import annotations

import io
import re
import zipfile
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

COHORT = "UPENN-GBM"

_ID_RE = re.compile(r"^(?P<base>UPENN-GBM-\d+)_(?P<suffix>\d+)$")

#: Scan-id suffix -> (role, treatment phase).
SUFFIX_ROLE = {
    "11": ("baseline_preoperative", "pre_operative"),
    "21": ("recurrence", "recurrence"),
}

_NA_STRINGS = {"not available", "not applicable", "na", "n/a", "nan", "", "-",
               "unknown", "none"}

#: Survival_Status -> event observed?
SURVIVAL_EVENT = {
    "deceased": True,
    "alive": False,
    "lost to follow-up": False,
    "deceased - uncertain date of death": False,
}

#: CaPTk modality/map keys found in radiomic_features_CaPTk.zip.
CAPTK_MAPS = ("T1", "T1GD", "T2", "FLAIR",
              "DTI_AD", "DTI_FA", "DTI_RD", "DTI_TR",
              "DSC_ap-rCBV", "DSC_PH", "DSC_PSR")

#: CaPTk region code -> canonical subregion.
CAPTK_REGION = {"ET": "enhancing", "ED": "edema", "NC": "necrosis"}

#: Segmentation flavour embedded in the CaPTk file names.
CAPTK_SEGMENTATIONS = {"automatic": "automaticsegm", "corrected": "segm"}


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


def split_scan_id(scan_id: str) -> tuple[str, str]:
    m = _ID_RE.match(str(scan_id).strip())
    if not m:
        raise ValueError(f"Unparseable UPENN-GBM scan id: {scan_id!r}")
    return m.group("base"), m.group("suffix")


@register
class UpennGbmAdapter(DatasetAdapter):
    key = "upenn_gbm"

    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=COHORT,
            name="UPENN-GBM: advanced MRI, clinical, genomics and radiomics",
            url="https://www.cancerimagingarchive.net/collection/upenn-gbm/",
            access="Public (TCIA). Images downloadable via NBIA; the clinical "
                   "and radiomic tables are direct downloads.",
            license="CC BY 4.0 (see collection page).",
            citation=(
                "Bakas S, et al. The University of Pennsylvania glioblastoma "
                "(UPenn-GBM) cohort: advanced MRI, clinical, genomics, & "
                "radiomics. Sci Data 9, 453 (2022)."
            ),
            modalities=("T1", "T1CE", "T2", "FLAIR", "DTI", "DSC", "SEG"),
            n_patients_published=630,
            n_timepoints_published=671,
            t0_definition="Baseline pre-operative scan (scan suffix _11) = day 0.",
            known_limitations=(
                "Cross-sectional by design: only 41 of 630 patients have a "
                "second (recurrence) scan, so this cohort does not support "
                "longitudinal trajectory modelling.",
                "MGMT is available for 291 of 671 scans; IDH1 for 565.",
                "KPS is available for only 75 scans.",
                "PsP_TP_score direction is not documented on the collection "
                "page; it is carried as a covariate, not used as a label.",
                "Radiomic features are CaPTk-derived, not PyRadiomics, and are "
                "not directly comparable to the LUMIERE feature set.",
            ),
            requires_registration=False,
        )

    def source_files(self) -> list[SourceFile]:
        base = "https://www.cancerimagingarchive.net/wp-content/uploads/"
        return [
            SourceFile("clinical", "UPENN-GBM_clinical_info_v2.1.csv",
                       "Demographics, survival with censoring, IDH1, MGMT, KPS, "
                       "extent of resection, PsP/TP score.",
                       url=base + "UPENN-GBM_clinical_info_v2.1.csv"),
            SourceFile("availability", "UPENN-GBM_data_availability.csv",
                       "Per-scan availability of each imaging and clinical element.",
                       url=base + "UPENN-GBM_data_availability.csv"),
            SourceFile("acquisition", "UPENN-GBM_acquisition.csv",
                       "MRI acquisition parameters.",
                       url=base + "UPENN-GBM_acquisition.csv"),
            SourceFile("captk_zip", "radiomic_features_CaPTk.zip",
                       "CaPTk radiomic features: 11 maps x 3 regions x 2 "
                       "segmentation flavours.",
                       url=base + "radiomic_features_CaPTk.zip"),
            SourceFile("feature_list", "UPENN-GBM_CaPTk_radiomic_features_list.csv",
                       "Definition of each CaPTk feature.", required=False,
                       url=base + "UPENN-GBM_CaPTk_radiomic_features_list.csv"),
            SourceFile("fe_params", "UPENN-GBM_CaPTk_fe_params.csv",
                       "CaPTk feature-extraction parameters.", required=False,
                       url=base + "UPENN-GBM_CaPTk_fe_params.csv"),
        ]

    # -- raw readers ------------------------------------------------------
    def _read_csv(self, key: str) -> pd.DataFrame:
        sf = {s.key: s for s in self.source_files()}[key]
        path = sf.resolve(self.raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"[{COHORT}] missing {path}")
        return pd.read_csv(path, low_memory=False, dtype=str)

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()
        clinical = self._read_csv("clinical")
        availability = self._read_csv("availability")

        clinical = clinical.copy()
        parsed = clinical["ID"].map(split_scan_id)
        clinical["local_patient_id"] = [p for p, _ in parsed]
        clinical["scan_suffix"] = [s for _, s in parsed]
        clinical["t_since_baseline"] = clinical["Time_since_baseline_preop"].map(_num)

        patients = self._build_patients(clinical)
        visits = self._build_visits(clinical)
        imaging = self._build_imaging(availability, visits)
        treatments = self._build_treatments(clinical)
        outcomes = self._build_outcomes(clinical, visits)

        bundle = self.empty_bundle(
            patients=patients, visits=visits, imaging=imaging,
            treatments=treatments, outcomes=outcomes,
        )
        bundle.notes.update({
            "symptoms": "UPENN-GBM releases no symptom data (KPS only, on patients).",
            "molecular_samples": (
                "IDH1/MGMT are released at patient level with no specimen ids "
                "or collection times, so molecular_samples is empty by design "
                "and the markers live on 'patients'."
            ),
            "survival": (
                "Genuine censoring: Survival_Status distinguishes Deceased from "
                "Alive / Lost to Follow-up / uncertain-date, so Cox models are "
                "admissible."
            ),
            "longitudinal": (
                "Only 41/630 patients have two scans; this adapter deliberately "
                "does not expose a longitudinal trajectory interface."
            ),
        })
        return bundle

    def _build_patients(self, clinical: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for local, grp in clinical.groupby("local_patient_id"):
            # Prefer the baseline pre-operative row for patient-level facts.
            base = grp[grp["scan_suffix"] == "11"]
            r = (base.iloc[0] if len(base) else
                 grp.sort_values("t_since_baseline", na_position="last").iloc[0])
            rows.append({
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_patient_id": local,
                "sex": {"M": "male", "F": "female"}.get(_clean(r["Gender"]) or "", pd.NA),
                "age_at_diagnosis_years": _num(r["Age_at_scan_years"]),
                "idh_status": _map_idh(_clean(r["IDH1"])),
                "idh_status_raw": _clean(r["IDH1"]),
                "mgmt_status": _map_mgmt(_clean(r["MGMT"])),
                "mgmt_status_raw": _clean(r["MGMT"]),
                "kps": _num(r["KPS"]),
                "who_grade": "4",
                "histology": "glioblastoma",
                "t0_definition": (
                    "baseline pre-operative scan (_11)" if len(base)
                    else "earliest released scan (no baseline pre-operative scan)"
                ),
                "has_baseline_preop_scan": bool(len(base)),
                "n_scans": int(len(grp)),
                "site": "University of Pennsylvania",
            })
        return pd.DataFrame(rows)

    def _build_visits(self, clinical: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in clinical.iterrows():
            local = r["local_patient_id"]
            suffix = r["scan_suffix"]
            role, phase = SUFFIX_ROLE.get(suffix, ("other", "unknown"))
            t = r["t_since_baseline"]
            rows.append({
                "visit_id": make_key(COHORT, r["ID"]),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_visit_id": r["ID"],
                "local_patient_id": local,
                "scan_role": role,
                "scan_suffix": suffix,
                "relative_time_days": (int(t) if t is not None and not pd.isna(t)
                                       else pd.NA),
                "relative_time_days_precision": "day",
                "treatment_phase": phase,
                "age_at_scan_years": _num(r["Age_at_scan_years"]),
                "psp_tp_score": _num(r["PsP_TP_score"]),
                "psp_tp_score_note": (
                    "1-6 reader score released without a documented direction; "
                    "not used as an outcome label."
                    if _num(r["PsP_TP_score"]) is not None else pd.NA
                ),
                "has_imaging": True,
                "has_molecular": False,
            })
        visits = pd.DataFrame(rows)
        visits = visits.sort_values(
            ["patient_id", "relative_time_days", "local_visit_id"],
            na_position="last").reset_index(drop=True)
        visits["visit_index"] = visits.groupby("patient_id").cumcount()
        return visits

    def _build_imaging(self, availability: pd.DataFrame,
                       visits: pd.DataFrame) -> pd.DataFrame:
        times = visits.set_index("local_visit_id")["relative_time_days"].to_dict()
        # Structural imaging is released as four co-registered sequences.
        structural = {"T1": "T1", "T1GD": "T1CE", "T2": "T2", "FLAIR": "FLAIR"}
        rows = []
        for _, r in availability.iterrows():
            scan_id = _clean(r["ID"])
            if scan_id is None:
                continue
            local, _ = split_scan_id(scan_id)
            visit_id = make_key(COHORT, scan_id)
            pid = make_patient_id(COHORT, local)
            t = times.get(scan_id, pd.NA)
            struct_ok = _clean(r["Structural imaging"]) == "available"
            for local_mod, modality in structural.items():
                rows.append(_imaging_row(
                    pid, visit_id, scan_id, local_mod, modality, t, struct_ok,
                    "Co-registered to SRI24 atlas, skull-stripped, "
                    "N4 bias-corrected (CaPTk preprocessing)."))
            for col, local_mod, modality in (
                ("DTI imaging", "DTI", "DTI"),
                ("DSC imaging", "DSC", "DSC"),
            ):
                rows.append(_imaging_row(
                    pid, visit_id, scan_id, local_mod, modality, t,
                    _clean(r[col]) == "available",
                    "Derived maps released (DTI: AD/FA/RD/TR; DSC: ap-rCBV/PH/PSR)."))
            for col, tool in (
                ("Automatic Tumor Segmentation", "automatic"),
                ("Corrected Tumor Segmentation", "corrected"),
            ):
                row = _imaging_row(
                    pid, visit_id, scan_id, tool, "SEG", t,
                    _clean(r[col]) == "available",
                    "Labels: 1=necrotic core (NC), 2=peritumoral edema (ED), "
                    "4=enhancing tumour (ET) in the BraTS convention.")
                row["segmentation_source"] = tool
                rows.append(row)
        return pd.DataFrame(rows)

    def _build_treatments(self, clinical: pd.DataFrame) -> pd.DataFrame:
        """Only the extent-of-resection flag is released; it is marked inferred."""
        rows = []
        for local, grp in clinical.groupby("local_patient_id"):
            base = grp[grp["scan_suffix"] == "11"]
            if not len(base):
                continue
            r = base.iloc[0]
            gtr = _clean(r["GTR_over90percent"])
            if gtr is None:
                continue
            rows.append({
                "treatment_id": make_key(COHORT, local, "surgery"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "treatment_type": "surgery",
                "treatment_name": "resection",
                "start_relative_day": 0,
                "treatment_phase": "post_operative",
                "extent_of_resection": {"Y": "GTR_over_90pct",
                                        "N": "less_than_90pct"}.get(gtr, pd.NA),
                "is_inferred": True,
            })
        return pd.DataFrame(rows)

    def _build_outcomes(self, clinical: pd.DataFrame,
                        visits: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for local, grp in clinical.groupby("local_patient_id"):
            base = grp[grp["scan_suffix"] == "11"]
            anchor = (base.iloc[0] if len(base)
                      else grp.sort_values("t_since_baseline",
                                           na_position="last").iloc[0])
            status = (_clean(anchor["Survival_Status"]) or "").lower()
            event = SURVIVAL_EVENT.get(status)
            if event is True:
                time = _num(anchor["Survival_from_surgery_days_UPDATED"])
            elif event is False:
                time = _num(anchor["Survival_Censor"])
            else:
                time = None
            rows.append({
                "outcome_id": make_key(COHORT, local, "overall_survival"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "visit_id": make_key(COHORT, anchor["ID"]),
                "outcome_name": "overall_survival",
                "outcome_type": "time_to_event",
                "outcome_value": time if time is not None else pd.NA,
                "outcome_unit": "days",
                "outcome_label": _clean(anchor["Survival_Status"]),
                "event_indicator": event if event is not None else pd.NA,
                "ascertainment": "registry",
                "followup_relative_day": time if time is not None else pd.NA,
                "notes": (
                    "Time measured from the anchor scan; censored rows take "
                    "their time from Survival_Censor."
                ),
            })

        # Molecular-marker outcomes used as prediction targets in Module B.
        for _, r in clinical.iterrows():
            local = r["local_patient_id"]
            mgmt = _map_mgmt(_clean(r["MGMT"]))
            if mgmt in {"methylated", "unmethylated"}:
                rows.append({
                    "outcome_id": make_key(COHORT, r["ID"], "mgmt"),
                    "patient_id": make_patient_id(COHORT, local),
                    "cohort": COHORT,
                    "visit_id": make_key(COHORT, r["ID"]),
                    "outcome_name": "mgmt_methylation",
                    "outcome_type": "binary",
                    "outcome_value": float(mgmt == "methylated"),
                    "outcome_label": mgmt,
                    "ascertainment": "histopathology",
                    "notes": "Clinical MGMT promoter methylation assay.",
                })
            idh = _map_idh(_clean(r["IDH1"]))
            if idh in {"wildtype", "mutant"}:
                rows.append({
                    "outcome_id": make_key(COHORT, r["ID"], "idh"),
                    "patient_id": make_patient_id(COHORT, local),
                    "cohort": COHORT,
                    "visit_id": make_key(COHORT, r["ID"]),
                    "outcome_name": "idh_mutation",
                    "outcome_type": "binary",
                    "outcome_value": float(idh == "mutant"),
                    "outcome_label": idh,
                    "ascertainment": "histopathology",
                    "notes": "IDH1 status as released (NOS/NEC treated as unknown).",
                })
        return pd.DataFrame(rows)

    # -- radiomics --------------------------------------------------------
    def load_captk_features(
        self,
        segmentation: str = "automatic",
        maps: tuple[str, ...] | None = None,
        regions: tuple[str, ...] = ("ET", "ED", "NC"),
    ) -> pd.DataFrame:
        """Wide CaPTk feature table, one row per scan id.

        Parameters
        ----------
        segmentation
            ``'automatic'`` (611 scans) or ``'corrected'`` (232 scans).
        maps
            Subset of :data:`CAPTK_MAPS`; ``None`` uses all 11.
        regions
            Subset of ET/ED/NC.
        """
        if segmentation not in CAPTK_SEGMENTATIONS:
            raise ValueError(f"segmentation must be one of {list(CAPTK_SEGMENTATIONS)}")
        token = CAPTK_SEGMENTATIONS[segmentation]
        maps = tuple(maps) if maps else CAPTK_MAPS

        sf = {s.key: s for s in self.source_files()}["captk_zip"]
        path = sf.resolve(self.raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"[{COHORT}] missing {path}")

        merged: pd.DataFrame | None = None
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for map_name in maps:
                for region in regions:
                    fname = f"Radiomic_Features_CaPTk_{token}_{map_name}_{region}.csv"
                    if fname not in names:
                        # 'automaticsegm' files are named ..._automaticsegm_...
                        alt = f"Radiomic_Features_CaPTk_{token}_{map_name}_{region}.csv"
                        if alt not in names:
                            continue
                        fname = alt
                    with z.open(fname) as fh:
                        part = pd.read_csv(io.BytesIO(fh.read()))
                    part = part.rename(columns={"SubjectID": "local_visit_id"})
                    part = part.drop_duplicates("local_visit_id")
                    merged = part if merged is None else merged.merge(
                        part, on="local_visit_id", how="outer")
        if merged is None:
            raise FileNotFoundError(
                f"[{COHORT}] no CaPTk files matched segmentation={segmentation!r}, "
                f"maps={maps}, regions={regions}")

        merged["cohort"] = COHORT
        merged["segmentation"] = segmentation
        merged["local_patient_id"] = merged["local_visit_id"].map(
            lambda s: split_scan_id(s)[0])
        merged["patient_id"] = merged["local_patient_id"].map(
            lambda p: make_patient_id(COHORT, p))
        merged["visit_id"] = merged["local_visit_id"].map(
            lambda s: make_key(COHORT, s))
        front = ["patient_id", "visit_id", "local_patient_id", "local_visit_id",
                 "cohort", "segmentation"]
        rest = [c for c in merged.columns if c not in front]
        return merged[front + rest]

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        skip = {"patient_id", "visit_id", "local_patient_id", "local_visit_id",
                "cohort", "segmentation"}
        return [c for c in df.columns if c not in skip]


# -- helpers ---------------------------------------------------------------

def _imaging_row(pid: str, visit_id: str, scan_id: str, local_mod: str,
                 modality: str, t, available: bool, note: str) -> dict:
    return {
        "imaging_id": make_key(COHORT, scan_id, local_mod),
        "patient_id": pid,
        "visit_id": visit_id,
        "cohort": COHORT,
        "modality": modality,
        "local_modality": local_mod,
        "acquisition_date_relative": t,
        "available": available,
        "relative_path": f"{scan_id}/{local_mod}",
        "is_preprocessed": True,
        "preprocessing_note": note,
    }


def _map_idh(value: str | None) -> object:
    if value is None:
        return pd.NA
    low = value.lower()
    if low.startswith("wild"):
        return "wildtype"
    if low.startswith("mut"):
        return "mutant"
    if "nos" in low or "nec" in low:
        return "indeterminate"
    return "unknown"


def _map_mgmt(value: str | None) -> object:
    if value is None:
        return pd.NA
    low = value.lower()
    if low.startswith("unmethyl"):
        return "unmethylated"
    if low.startswith("methyl"):
        return "methylated"
    if low.startswith("indeter"):
        return "indeterminate"
    return "unknown"
