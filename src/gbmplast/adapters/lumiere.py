"""LUMIERE adapter.

Source
------
Suter et al., *The LUMIERE dataset: Longitudinal Glioblastoma MRI with
expert RANO evaluation*, Scientific Data (2022). doi:10.1038/s41597-022-01881-7
Figshare collection 5904905.

What the cohort actually releases (verified against the downloaded tables,
not assumed):

* ``LUMIERE-Demographics_Pathology.csv``  -- 91 patients; survival in **weeks**
  with *no censoring indicator*; IDH/MGMT with literal ``na`` strings.
* ``LUMIERE-ExpertRating-v202211.csv``    -- 616 rating rows over 613 unique
  (patient, timepoint) keys: three timepoints carry two ratings each.
* ``LUMIERE-datacompleteness.csv``        -- 638 studies x 4 sequences + 2
  segmentation tools, marked with ``x``.
* ``LUMIERE-MRinfo.csv``                  -- DICOM acquisition parameters per
  (patient, timepoint, sequence).
* ``LUMIERE-pyradiomics-{hdglioauto,deepbratumia}-features.csv`` -- 107
  PyRadiomics features per (patient, timepoint, sequence, label).

Decisions that would otherwise be silent assumptions
----------------------------------------------------
``t0``
    Timepoints are published as ``week-NNN`` relative to the index surgery,
    so ``relative_time_days = 7 * NNN`` at **week precision**.  The optional
    ``-K`` suffix distinguishes several studies within one week; it orders
    them but does not refine the day estimate.

``visits``
    The union of the completeness table and the rating table.  17 rated
    timepoints have no row in the completeness table (the README states a
    rating could be made from a subset of sequences), and dropping them
    would silently discard expert labels.

``rano_status``
    Where a timepoint carries two ratings they are *both* preserved in the
    auxiliary ``rano_assessments`` table.  ``visits.rano_status`` resolves
    them by the documented precedence PD > PR > CR > SD > Post-Op > Pre-Op
    and sets ``rano_conflict = True``.  ``Post-Op/PD`` is likewise resolved
    to ``PD`` with ``rano_postop = True``.

``idh_status``
    ``IDH1 neg, Sequencing required`` is an **indeterminate** immunostain, not
    a wild-type call, and is mapped to ``indeterminate`` -- never to
    ``wildtype``.

``overall survival``
    The cohort releases a survival time with no vital-status/censoring
    column, so ``outcomes.event_indicator`` is NA for every patient and
    :mod:`gbmplast.modeling.survival` refuses to fit a Cox model on it.
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

COHORT = "LUMIERE"

_TIMEPOINT_RE = re.compile(r"^week-(\d+)(?:-(\d+))?$")

#: Cohort sequence label -> canonical modality.
SEQUENCE_TO_MODALITY = {
    "CT1": "T1CE",
    "T1": "T1",
    "T2": "T2",
    "FLAIR": "FLAIR",
}

#: Segmentation tool -> {label integer: canonical subregion}.
#: Verified from the ``Label``/``Label name`` pairs in the released
#: PyRadiomics tables, not from the tools' documentation.
SEGMENTATION_LABELS = {
    "HD-GLIO-AUTO": {1: "non_enhancing", 2: "enhancing"},
    "DeepBraTumIA": {1: "necrosis", 2: "enhancing", 3: "edema"},
}

#: Label-name spelling used in the released feature tables -> canonical.
LABEL_NAME_TO_SUBREGION = {
    "Non-enhancing": "non_enhancing",
    "Contrast-enhancing": "enhancing",
    "Necrosis": "necrosis",
    "Edema": "edema",
}

#: Deterministic precedence when one study carries several expert ratings.
RANO_PRECEDENCE = ["PD", "PR", "CR", "SD", "Post-Op", "Pre-Op"]

_NA_STRINGS = {"na", "n/a", "nan", "", "none", "unknown"}


def _clean(value: object) -> str | None:
    """Normalise a cell to a stripped string, or None for the cohort's NA."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _NA_STRINGS else text


def parse_timepoint(timepoint: str) -> tuple[int, int]:
    """``'week-044'`` -> ``(44, 0)``; ``'week-000-2'`` -> ``(0, 2)``.

    Returns (week, within_week_index). Raises on an unparseable label rather
    than guessing.
    """
    match = _TIMEPOINT_RE.match(str(timepoint).strip())
    if not match:
        raise ValueError(f"Unparseable LUMIERE timepoint: {timepoint!r}")
    week = int(match.group(1))
    sub = int(match.group(2)) if match.group(2) else 0
    return week, sub


def _map_idh(value: str | None) -> tuple[str | None, str | None]:
    """Return (canonical status, raw string).

    'IDH1 neg, Sequencing required' is an inconclusive immunostain and must
    not be collapsed into 'wildtype'.
    """
    if value is None:
        return None, None
    low = value.lower()
    if low in {"wt", "wildtype", "wild type"}:
        return "wildtype", value
    if "mut" in low:
        return "mutant", value
    if "sequencing required" in low or "neg" in low:
        return "indeterminate", value
    return "unknown", value


def _map_mgmt(value: str | None) -> str | None:
    if value is None:
        return None
    low = value.lower()
    if low.startswith("not methyl") or low == "unmethylated":
        return "unmethylated"
    if low.startswith("methyl"):
        return "methylated"
    return "unknown"


def _parse_percent(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


@register
class LumiereAdapter(DatasetAdapter):
    key = "lumiere"

    # -- static description ----------------------------------------------
    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=COHORT,
            name="LUMIERE: Longitudinal Glioblastoma MRI with expert RANO evaluation",
            url="https://figshare.com/collections/"
                "The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905",
            access="Public download from Figshare. Provided for non-commercial use.",
            license="Non-commercial use (see LUMIERE-readme.pdf).",
            citation=(
                "Suter Y, et al. The LUMIERE dataset: Longitudinal Glioblastoma "
                "MRI with expert RANO evaluation. Sci Data 9, 768 (2022). "
                "doi:10.1038/s41597-022-01881-7"
            ),
            modalities=("T1", "T1CE", "T2", "FLAIR", "SEG"),
            n_patients_published=91,
            n_timepoints_published=638,
            t0_definition="Week 0 = index surgery; timepoints are week-resolution.",
            known_limitations=(
                "Overall survival is released in weeks with NO censoring "
                "indicator, so time-to-event modelling is not identifiable.",
                "Timepoint resolution is weeks, not days.",
                "No treatment records, no symptoms, no tissue specimens.",
                "Segmentations are automated (DeepBraTumIA, HD-GLIO-AUTO), "
                "not manually curated.",
                "3 of 613 rated studies carry two conflicting expert ratings.",
                "IDH is indeterminate ('sequencing required') for 10 patients.",
            ),
            requires_registration=False,
            notes={
                "imaging_zip": (
                    "Imaging-v202211.zip is 32.5 GB as a single archive. The "
                    "acquisition layer uses HTTP range requests to extract "
                    "individual patients without downloading the whole file."
                ),
            },
        )

    def source_files(self) -> list[SourceFile]:
        base = "https://ndownloader.figshare.com/files/"
        return [
            SourceFile("demographics", "LUMIERE-Demographics_Pathology.csv",
                       "Sex, age, IDH, MGMT, survival time (weeks).",
                       url=base + "37575637"),
            SourceFile("expert_rating", "LUMIERE-ExpertRating-v202211.csv",
                       "Expert RANO rating and rationale per study.",
                       url=base + "38369741"),
            SourceFile("completeness", "LUMIERE-datacompleteness.csv",
                       "Per-study availability of each sequence and segmentation.",
                       url=base + "37575523"),
            SourceFile("mrinfo", "LUMIERE-MRinfo.csv",
                       "DICOM acquisition parameters per sequence.",
                       url=base + "37691442"),
            SourceFile("radiomics_hdglio",
                       "LUMIERE-pyradiomics-hdglioauto-features.csv",
                       "PyRadiomics features on HD-GLIO-AUTO segmentations.",
                       url=base + "37680966"),
            SourceFile("radiomics_deepbratumia",
                       "LUMIERE-pyradiomics-deepbratumia-features.csv",
                       "PyRadiomics features on DeepBraTumIA segmentations.",
                       url=base + "37681080"),
            SourceFile("readme", "LUMIERE-readme.pdf",
                       "Official file-layout documentation.",
                       required=False, url=base + "37983597"),
            SourceFile("imaging_zip", "Imaging-v202211.zip",
                       "NIfTI images + segmentations (32.5 GB single archive).",
                       required=False, url=base + "38249697",
                       large=True, approx_bytes=32_565_470_000,
                       note="Optional: only needed for image-level QC and "
                            "de-novo feature extraction. Use "
                            "'gbmplast acquire --dataset lumiere --images "
                            "--n-patients 5' for a development subset, which "
                            "transfers megabytes via HTTP range requests."),
        ]

    # -- raw readers ------------------------------------------------------
    def _read(self, key: str) -> pd.DataFrame:
        sf = {s.key: s for s in self.source_files()}[key]
        path = sf.resolve(self.raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"[{COHORT}] missing {path}")
        return pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=True)

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()

        demographics = self._read("demographics")
        rating = self._read("expert_rating")
        completeness = self._read("completeness")
        mrinfo = self._read("mrinfo")

        patients = self._build_patients(demographics)
        rano = self._build_rano_assessments(rating)
        visits = self._build_visits(completeness, rano)
        imaging = self._build_imaging(completeness, mrinfo, visits)
        outcomes = self._build_outcomes(demographics, visits)

        bundle = self.empty_bundle(
            patients=patients,
            visits=visits,
            imaging=imaging,
            outcomes=outcomes,
        )
        bundle.notes.update({
            "treatments": (
                "LUMIERE releases no treatment records. The table is empty by "
                "design; treatment_phase on visits is inferred only from the "
                "Pre-Op/Post-Op expert ratings and is 'unknown' elsewhere."
            ),
            "symptoms": "LUMIERE releases no clinical symptoms. Table empty by design.",
            "molecular_samples": (
                "LUMIERE releases patient-level IDH/MGMT but no specimen-level "
                "records (no specimen ids, no collection times). Table empty by "
                "design; the markers live on 'patients'."
            ),
            "survival": (
                "Overall survival has no censoring indicator; event_indicator "
                "is NA for all 91 patients."
            ),
        })
        # auxiliary, non-canonical table kept alongside the bundle
        self._rano_assessments = rano
        return bundle

    # -- components -------------------------------------------------------
    def _build_patients(self, demographics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in demographics.iterrows():
            local = _clean(r["Patient"])
            idh_status, idh_raw = _map_idh(_clean(r["IDH (WT: wild type)"]))
            rows.append({
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_patient_id": local,
                "sex": (_clean(r["Sex"]) or "").lower() or pd.NA,
                "age_at_diagnosis_years": _to_float(_clean(r["Age at surgery (years)"])),
                "idh_status": idh_status if idh_status else pd.NA,
                "idh_status_raw": idh_raw,
                "idh_method": _clean(r["IDH method"]),
                "mgmt_status": _map_mgmt(_clean(r["MGMT qualitative"])),
                "mgmt_value": _parse_percent(_clean(r["MGMT quantitative"])),
                "mgmt_method": pd.NA,
                "who_grade": pd.NA,
                "histology": "glioblastoma",
                "t0_definition": "index surgery (week 0)",
                "site": "Inselspital Bern",
            })
        return pd.DataFrame(rows)

    def _build_rano_assessments(self, rating: pd.DataFrame) -> pd.DataFrame:
        rating_col = _find_col(rating, "Rating (according to RANO")
        rationale_col = _find_col(rating, "Rating rationale")
        rows = []
        for idx, r in rating.iterrows():
            local_pt = _clean(r["Patient"])
            local_tp = _clean(r["Date"])
            if local_pt is None or local_tp is None:
                continue
            week, sub = parse_timepoint(local_tp)
            raw = _clean(r[rating_col])
            status, postop = _normalise_rano(raw)
            rows.append({
                "cohort": COHORT,
                "patient_id": make_patient_id(COHORT, local_pt),
                "visit_id": make_key(COHORT, local_pt, local_tp),
                "local_patient_id": local_pt,
                "local_visit_id": local_tp,
                "row_index": int(idx),
                "week": week,
                "within_week_index": sub,
                "rano_status_raw": raw,
                "rano_status": status,
                "rano_postop": postop,
                "rano_rationale": _clean(r[rationale_col]),
                "rano_lt_3_months": _clean(r.get("LessThan3Months")) is not None,
                "rano_nonmeasurable": _clean(r.get("NonMeasurableLesions")) is not None,
            })
        return pd.DataFrame(rows)

    def _build_visits(self, completeness: pd.DataFrame,
                      rano: pd.DataFrame) -> pd.DataFrame:
        # Union of imaged studies and rated studies -- neither is a superset.
        keys = set()
        for _, r in completeness.iterrows():
            keys.add((_clean(r["Patient"]), _clean(r["Timepoint"])))
        imaged = set(keys)
        rated = {(r.local_patient_id, r.local_visit_id) for r in rano.itertuples()}
        keys |= rated

        resolved = _resolve_rano(rano)

        rows = []
        for local_pt, local_tp in sorted(keys):
            week, sub = parse_timepoint(local_tp)
            visit_id = make_key(COHORT, local_pt, local_tp)
            res = resolved.get(visit_id, {})
            rows.append({
                "visit_id": visit_id,
                "patient_id": make_patient_id(COHORT, local_pt),
                "cohort": COHORT,
                "local_visit_id": local_tp,
                "local_patient_id": local_pt,
                "relative_time_days": week * 7,
                "relative_time_weeks": week,
                "within_week_index": sub,
                "relative_time_days_precision": "week",
                "treatment_phase": res.get("treatment_phase", "unknown"),
                "rano_status": res.get("rano_status", pd.NA),
                "rano_status_raw": res.get("rano_status_raw", pd.NA),
                "rano_rationale": res.get("rano_rationale", pd.NA),
                "rano_conflict": res.get("rano_conflict", False),
                "rano_n_assessments": res.get("n", 0),
                "rano_postop": res.get("rano_postop", False),
                "rano_lt_3_months": res.get("rano_lt_3_months", pd.NA),
                "rano_nonmeasurable": res.get("rano_nonmeasurable", pd.NA),
                "has_imaging": (local_pt, local_tp) in imaged,
                "has_expert_rating": visit_id in resolved,
                "has_molecular": False,
            })
        visits = pd.DataFrame(rows)
        visits = visits.sort_values(
            ["patient_id", "relative_time_weeks", "within_week_index"]
        ).reset_index(drop=True)
        visits["visit_index"] = visits.groupby("patient_id").cumcount()
        return visits

    def _build_imaging(self, completeness: pd.DataFrame, mrinfo: pd.DataFrame,
                       visits: pd.DataFrame) -> pd.DataFrame:
        times = visits.set_index("visit_id")["relative_time_days"].to_dict()

        params: dict[tuple[str, str, str], pd.Series] = {}
        for _, r in mrinfo.iterrows():
            key = (_clean(r["Patient"]), _clean(r["Timepoint"]), _clean(r["Sequence"]))
            params[key] = r

        rows = []
        for _, r in completeness.iterrows():
            local_pt = _clean(r["Patient"])
            local_tp = _clean(r["Timepoint"])
            visit_id = make_key(COHORT, local_pt, local_tp)
            for seq, modality in SEQUENCE_TO_MODALITY.items():
                available = _clean(r.get(seq)) is not None
                p = params.get((local_pt, local_tp, seq))
                rows.append({
                    "imaging_id": make_key(COHORT, local_pt, local_tp, seq),
                    "patient_id": make_patient_id(COHORT, local_pt),
                    "visit_id": visit_id,
                    "cohort": COHORT,
                    "modality": modality,
                    "local_modality": seq,
                    "acquisition_date_relative": times.get(visit_id, pd.NA),
                    "available": available,
                    "relative_path": f"Imaging/{local_pt}/{local_tp}/{seq}.nii.gz",
                    "manufacturer": _clean(p["Manufacturer"]) if p is not None else pd.NA,
                    "model": _clean(p["Model"]) if p is not None else pd.NA,
                    "field_strength_t": _to_float(_clean(p["Field strength"])) if p is not None else pd.NA,
                    "slice_thickness_mm": _to_float(_clean(p["Slice thickness"])) if p is not None else pd.NA,
                    "in_plane_spacing_mm": _clean(p["Voxel size"]) if p is not None else pd.NA,
                    "rows": _to_int(_clean(p["Rows"])) if p is not None else pd.NA,
                    "columns": _to_int(_clean(p["Columns"])) if p is not None else pd.NA,
                    "slice_count": _to_int(_clean(p["Slice count"])) if p is not None else pd.NA,
                    "echo_time_ms": _to_float(_clean(p["Echo"])) if p is not None else pd.NA,
                    "repetition_time_ms": _to_float(_clean(p["Repetition"])) if p is not None else pd.NA,
                    "inversion_time_ms": _to_float(_clean(p["Inversion"])) if p is not None else pd.NA,
                    "flip_angle_deg": _to_float(_clean(p["Flip"])) if p is not None else pd.NA,
                    "is_preprocessed": True,
                    "preprocessing_note": "Converted from DICOM and skull-stripped with HD-BET; "
                                          "native space (not registered, not resampled).",
                    "segmentation_source": pd.NA,
                })
            for tool in ("DeepBraTumIA", "HD-GLIO-AUTO"):
                rows.append({
                    "imaging_id": make_key(COHORT, local_pt, local_tp, tool),
                    "patient_id": make_patient_id(COHORT, local_pt),
                    "visit_id": visit_id,
                    "cohort": COHORT,
                    "modality": "SEG",
                    "local_modality": tool,
                    "acquisition_date_relative": times.get(visit_id, pd.NA),
                    "available": _clean(r.get(tool)) is not None,
                    "relative_path": f"Imaging/{local_pt}/{local_tp}/{tool}-segmentation",
                    "is_preprocessed": True,
                    "preprocessing_note": (
                        "Automated segmentation; released in native space and in a "
                        "co-registered space (MNI atlas for DeepBraTumIA, per-case "
                        "reference image for HD-GLIO-AUTO)."
                    ),
                    "segmentation_source": tool,
                })
        return pd.DataFrame(rows)

    def _build_outcomes(self, demographics: pd.DataFrame,
                        visits: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in demographics.iterrows():
            local = _clean(r["Patient"])
            weeks = _to_float(_clean(r["Survival time (weeks)"]))
            rows.append({
                "outcome_id": make_key(COHORT, local, "overall_survival"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "outcome_name": "overall_survival",
                "outcome_type": "time_to_event",
                "outcome_value": weeks * 7.0 if weeks is not None else pd.NA,
                "outcome_unit": "days",
                "event_indicator": pd.NA,   # cohort releases no censoring flag
                "ascertainment": "unknown",
                "notes": (
                    "Survival time published in weeks without a vital-status or "
                    "censoring column; event_indicator is therefore NA and "
                    "time-to-event models are refused on this cohort."
                ),
            })

        # Visit-level RANO progression outcome (expert-adjudicated).
        rated = visits[visits["has_expert_rating"]]
        for _, v in rated.iterrows():
            status = v["rano_status"]
            if status is pd.NA or (isinstance(status, float) and np.isnan(status)):
                continue
            if status in {"Pre-Op", "Post-Op"}:
                label, value = status, pd.NA
            else:
                label, value = status, float(status == "PD")
            rows.append({
                "outcome_id": make_key(COHORT, v["local_patient_id"],
                                       v["local_visit_id"], "rano"),
                "patient_id": v["patient_id"],
                "cohort": COHORT,
                "visit_id": v["visit_id"],
                "outcome_name": "rano_progression",
                "outcome_type": "binary",
                "outcome_value": value,
                "outcome_label": label,
                "ascertainment": "expert_read",
                "ascertainment_relative_day": v["relative_time_days"],
                "notes": "Expert neuroradiologist RANO rating; week-resolution timing.",
            })
        return pd.DataFrame(rows)

    # -- feature tables (not canonical; used by the longitudinal module) --
    def load_radiomics(self, segmentation: str = "HD-GLIO-AUTO") -> pd.DataFrame:
        """Long-format PyRadiomics table for one segmentation tool.

        Columns: identifiers + ``subregion`` + ``sequence`` + feature columns.
        Shape features are identical across sequences by construction (one
        mask per study), which the longitudinal builder relies on.
        """
        key = {"HD-GLIO-AUTO": "radiomics_hdglio",
               "DeepBraTumIA": "radiomics_deepbratumia"}[segmentation]
        sf = {s.key: s for s in self.source_files()}[key]
        path = sf.resolve(self.raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"[{COHORT}] missing {path}")
        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns={"Time point": "local_visit_id",
                                "Patient": "local_patient_id",
                                "Sequence": "sequence"})
        df["cohort"] = COHORT
        df["segmentation"] = segmentation
        df["patient_id"] = df["local_patient_id"].map(
            lambda p: make_patient_id(COHORT, p))
        df["visit_id"] = [make_key(COHORT, p, t) for p, t
                          in zip(df["local_patient_id"], df["local_visit_id"])]
        df["subregion"] = df["Label name"].map(LABEL_NAME_TO_SUBREGION)
        unmapped = df.loc[df["subregion"].isna(), "Label name"].unique()
        if len(unmapped):
            raise ValueError(
                f"[{COHORT}] unmapped segmentation label names {list(unmapped)} "
                "- refusing to guess a subregion mapping."
            )
        weeks = df["local_visit_id"].map(lambda t: parse_timepoint(t)[0])
        subs = df["local_visit_id"].map(lambda t: parse_timepoint(t)[1])
        df["relative_time_days"] = weeks * 7
        df["within_week_index"] = subs
        return df

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        """PyRadiomics feature columns (excludes diagnostics and identifiers)."""
        skip_prefix = ("diagnostics_",)
        skip = {"local_patient_id", "local_visit_id", "Image", "Mask", "Label name",
                "Label", "sequence", "Reader", "cohort", "segmentation",
                "patient_id", "visit_id", "subregion", "relative_time_days",
                "within_week_index"}
        return [c for c in df.columns
                if c not in skip and not c.startswith(skip_prefix)]

    @property
    def rano_assessments(self) -> pd.DataFrame:
        """All expert ratings, including the duplicated/conflicting rows."""
        if not hasattr(self, "_rano_assessments"):
            self._rano_assessments = self._build_rano_assessments(
                self._read("expert_rating"))
        return self._rano_assessments


# -- helpers ---------------------------------------------------------------

def _find_col(df: pd.DataFrame, prefix: str) -> str:
    for c in df.columns:
        if c.startswith(prefix):
            return c
    raise KeyError(f"No column starting with {prefix!r}; got {list(df.columns)}")


def _normalise_rano(raw: str | None) -> tuple[object, bool]:
    """Map a raw rating cell to (canonical status, was_post_operative)."""
    if raw is None:
        return pd.NA, False
    text = raw.strip()
    if "/" in text:  # e.g. 'Post-Op/PD'
        parts = [p.strip() for p in text.split("/")]
        postop = any(p.lower() == "post-op" for p in parts)
        others = [p for p in parts if p.lower() != "post-op"]
        if others:
            return _canonical_rano(others[0]), postop
        return "Post-Op", True
    canonical = _canonical_rano(text)
    return canonical, canonical == "Post-Op"


def _canonical_rano(text: str) -> object:
    lookup = {"pd": "PD", "sd": "SD", "pr": "PR", "cr": "CR",
              "pre-op": "Pre-Op", "post-op": "Post-Op"}
    return lookup.get(text.strip().lower(), pd.NA)


def _resolve_rano(rano: pd.DataFrame) -> dict[str, dict]:
    """Collapse possibly-multiple ratings per visit by documented precedence."""
    out: dict[str, dict] = {}
    if rano.empty:
        return out
    for visit_id, grp in rano.groupby("visit_id"):
        statuses = [s for s in grp["rano_status"].tolist() if isinstance(s, str)]
        chosen = pd.NA
        for candidate in RANO_PRECEDENCE:
            if candidate in statuses:
                chosen = candidate
                break
        rationales = [r for r in grp["rano_rationale"].tolist() if isinstance(r, str)]
        out[visit_id] = {
            "rano_status": chosen,
            "rano_status_raw": " | ".join(
                sorted({str(s) for s in grp["rano_status_raw"] if isinstance(s, str)})),
            "rano_rationale": " | ".join(rationales) if rationales else pd.NA,
            "rano_conflict": len(set(statuses)) > 1,
            "n": int(len(grp)),
            "rano_postop": bool(grp["rano_postop"].any()),
            "rano_lt_3_months": bool(grp["rano_lt_3_months"].any()),
            "rano_nonmeasurable": bool(grp["rano_nonmeasurable"].any()),
            "treatment_phase": _phase_from_rano(chosen, statuses),
        }
    return out


def _phase_from_rano(chosen: object, statuses: list[str]) -> str:
    if "Pre-Op" in statuses:
        return "pre_operative"
    if "Post-Op" in statuses:
        return "post_operative"
    return "unknown"


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None
