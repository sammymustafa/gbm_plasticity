"""GBM-CARE adapter (GEO GSE274546, the 10X single-nucleus arm).

Source
------
The GBM CARE (Cellular Analysis of Resistance and Evolution) consortium,
*The Multi-Layered Transcriptional Architecture of Glioblastoma Ecosystems*.
GEO series GSE274546; PubMed 40346362 / 40346361.  A companion controlled
repository exists on Synapse (``synapse.org/care_glioblastoma``) and holds
the consortium's own cell-state annotations; it requires an account and a
data-use agreement and is handled by :class:`GlassAdapter`-style stubs in
:mod:`gbmplast.adapters.synapse_gated`.

Verified structure
------------------
121 samples / 59 patients, parsed from the GEO SOFT records:

* ``!Sample_title``            -- ``Patient N, Timepoint M, Chromium gene expression``
* ``!Sample_source_name_ch1``  -- ``GBM single-nucleus, <SITE><NN>, <phase>``
  where phase is ``Primary`` (56), ``1st Recurrent`` (55) or
  ``2nd Recurrent`` (10).
* one supplementary per sample: ``GSM*_P<N>T<M>_umi_counts.RDS.gz``
  (a dense gene x nucleus UMI matrix, ~20-50 MB gzipped).

**56 patients contribute a matched Primary + 1st Recurrent pair** and three
contribute a third timepoint -- this pairing is the whole basis of Module C.

Seven contributing sites are encoded in the patient code prefix:
TK (30 samples), FR (28), SN (20), SM (14), NL (13), DU (12), MD (4).

What is *not* released here
---------------------------
* No calendar or relative collection dates -- only an ordinal timepoint.
  ``tissue_collection_relative_day`` is therefore NA for every specimen and
  ``timepoint_index``/``sample_role`` carry the ordering.
* No per-patient clinical annotation, treatment records, or survival in this
  GEO series ("raw data are not available due to patient privacy concerns").
* No consortium cell-type/cell-state labels; those live on Synapse.
  :mod:`gbmplast.molecular` therefore derives cell types from canonical
  markers and scores *published* state signatures, and labels every such
  column as derived-here rather than consortium-published.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.acquire.geo import parse_soft, sample_supplementary_urls
from gbmplast.adapters.base import (
    CohortDescription,
    DatasetAdapter,
    SourceFile,
    register,
)
from gbmplast.schema import CanonicalBundle, make_key, make_patient_id

COHORT = "GBM-CARE"
SERIES = "GSE274546"

_SOURCE_RE = re.compile(
    r"GBM single-nucleus,\s*(?P<patient>[A-Za-z]+\d+)\s*,\s*(?P<phase>.+?)\s*$")
_TITLE_RE = re.compile(r"Patient\s+(?P<pnum>\d+),\s*Timepoint\s+(?P<tp>\d+)")

#: Released phase string -> (canonical sample role, treatment phase).
PHASE_MAP = {
    "primary": ("primary", "pre_operative"),
    "1st recurrent": ("first_recurrence", "recurrence"),
    "2nd recurrent": ("second_recurrence", "recurrence"),
}

#: Patient-code prefix -> contributing site, as named in the series protocols.
SITE_MAP = {
    "DU": "Duke University",
    "MD": "MD Anderson Cancer Center",
    "TK": "Tokyo University Hospital",
    "SM": "St. Michael's Hospital",
    "FR": "Pitie-Salpetriere Hospital",
    "NL": "NorLux",
    "SN": "Seoul National University",
}


@register
class GbmCareAdapter(DatasetAdapter):
    key = "gbm_care"

    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=COHORT,
            name="GBM-CARE: single-nucleus RNA-seq of paired primary/recurrent GBM",
            url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274546",
            access=(
                "GEO processed matrices are public. Raw sequencing is withheld "
                "for patient privacy. Consortium cell-state annotations and "
                "clinical annotation are on Synapse "
                "(synapse.org/care_glioblastoma) behind an account + DUA."
            ),
            license="See GEO/Synapse terms; Synapse portion requires a DUA.",
            citation=(
                "GBM CARE consortium. The Multi-Layered Transcriptional "
                "Architecture of Glioblastoma Ecosystems. GEO GSE274546; "
                "PMID 40346362, 40346361."
            ),
            modalities=("snRNA_seq",),
            n_patients_published=59,
            n_timepoints_published=121,
            t0_definition=(
                "Ordinal only: timepoint 1 = primary resection. No collection "
                "dates are released, so relative days are NA."
            ),
            known_limitations=(
                "No specimen collection dates: only an ordinal timepoint.",
                "No clinical covariates, treatment records or survival in this "
                "GEO series.",
                "No consortium cell-type or cell-state labels in GEO; this "
                "pipeline derives cell types from canonical markers and scores "
                "published signatures, which is not the same as the "
                "consortium's published annotation.",
                "Matrices are dense gene x nucleus; each sample needs ~1.4 GB "
                "of RAM when parsed, so pseudobulk is computed one sample at a "
                "time and the dense array is discarded immediately.",
            ),
            requires_registration=False,
            notes={
                "synapse": (
                    "To use the consortium's own malignant-state annotations, "
                    "register at synapse.org, accept the CARE data-use "
                    "agreement, and point cohorts.gbm_care.synapse_dir at the "
                    "downloaded folder. The molecular module will then prefer "
                    "published labels over marker-derived ones."
                ),
            },
        )

    def source_files(self) -> list[SourceFile]:
        return [
            SourceFile("samples_soft", f"{SERIES}_samples.soft.txt",
                       "GEO SOFT records for all 121 samples.",
                       url=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
                            f"acc={SERIES}&targ=gsm&form=text&view=brief")),
            SourceFile("series_soft", f"{SERIES}_series.soft.txt",
                       "GEO SOFT record for the series.", required=False,
                       url=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
                            f"acc={SERIES}&targ=self&form=text&view=full")),
        ]

    # -- parsing ----------------------------------------------------------
    def sample_records(self) -> pd.DataFrame:
        """One row per GEO sample with the parsed patient/phase structure."""
        path = self.raw_dir / f"{SERIES}_samples.soft.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"[{COHORT}] missing {path}. Run: gbmplast acquire --dataset gbm_care")
        rows = []
        for rec in parse_soft(path):
            source = str(rec.get("Sample_source_name_ch1", ""))
            m = _SOURCE_RE.match(source)
            if not m:
                raise ValueError(
                    f"[{COHORT}] unparseable source_name {source!r} for "
                    f"{rec['gsm']} - refusing to guess the patient/phase.")
            local_patient = m.group("patient")
            phase_raw = m.group("phase")
            key = phase_raw.strip().lower()
            if key not in PHASE_MAP:
                raise ValueError(
                    f"[{COHORT}] unknown sample phase {phase_raw!r}; refusing "
                    "to guess whether it is primary or recurrent.")
            role, treatment_phase = PHASE_MAP[key]
            t = _TITLE_RE.search(str(rec.get("Sample_title", "")))
            urls = sample_supplementary_urls(rec)
            rows.append({
                "gsm": rec["gsm"],
                "local_patient_id": local_patient,
                "site": SITE_MAP.get(local_patient[:2], "unknown"),
                "site_code": local_patient[:2],
                "patient_number": int(t.group("pnum")) if t else pd.NA,
                "timepoint_index": int(t.group("tp")) if t else pd.NA,
                "phase_raw": phase_raw,
                "sample_role": role,
                "treatment_phase": treatment_phase,
                "platform": str(rec.get("Sample_platform_id", "")),
                "instrument": str(rec.get("Sample_instrument_model", "")),
                "protocol": _first(rec.get("Sample_characteristics_ch1")),
                "supplementary_url": urls[0] if urls else pd.NA,
                "file_name": urls[0].rsplit("/", 1)[-1] if urls else pd.NA,
            })
        df = pd.DataFrame(rows)
        counts = df.groupby("local_patient_id")["gsm"].transform("size")
        df["is_paired"] = counts >= 2
        return df.sort_values(["local_patient_id", "timepoint_index"]).reset_index(drop=True)

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()
        samples = self.sample_records()

        patients = self._build_patients(samples)
        visits = self._build_visits(samples)
        molecular = self._build_molecular(samples)

        bundle = self.empty_bundle(
            patients=patients, visits=visits, molecular_samples=molecular)
        bundle.notes.update({
            "imaging": "GBM-CARE releases no imaging. Table empty by design.",
            "treatments": (
                "No treatment records in GEO. Samples are nonetheless labelled "
                "primary vs recurrent, i.e. post standard-of-care, which is the "
                "contrast Module C estimates."
            ),
            "symptoms": "No symptom data.",
            "outcomes": (
                "No survival or response outcomes in this GEO series; they are "
                "in the Synapse repository behind a DUA."
            ),
            "timing": (
                "tissue_collection_relative_day is NA for every specimen: the "
                "series releases an ordinal timepoint only."
            ),
            "cross_cohort": (
                "These patients must never be joined to MRI cohorts. ids are "
                "namespaced 'GBM-CARE::' so any such join yields zero rows."
            ),
        })
        return bundle

    def _build_patients(self, samples: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for local, grp in samples.groupby("local_patient_id"):
            roles = set(grp["sample_role"])
            rows.append({
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_patient_id": local,
                "sex": pd.NA,                      # not released in GEO
                "age_at_diagnosis_years": pd.NA,   # not released in GEO
                "idh_status": pd.NA,
                "mgmt_status": pd.NA,
                "histology": "glioblastoma",
                "t0_definition": "ordinal: timepoint 1 = primary resection",
                "site": grp["site"].iloc[0],
                "site_code": grp["site_code"].iloc[0],
                "n_specimens": int(len(grp)),
                "has_primary": "primary" in roles,
                "has_recurrence": bool(roles & {"first_recurrence",
                                                "second_recurrence"}),
                "has_matched_pair": ("primary" in roles
                                     and bool(roles & {"first_recurrence",
                                                       "second_recurrence"})),
            })
        return pd.DataFrame(rows)

    def _build_visits(self, samples: pd.DataFrame) -> pd.DataFrame:
        """One visit per specimen; time is ordinal, so relative days stay NA."""
        rows = []
        for _, r in samples.iterrows():
            local = r["local_patient_id"]
            tp = r["timepoint_index"]
            rows.append({
                "visit_id": make_key(COHORT, local, f"T{tp}"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "local_visit_id": f"{local}_T{tp}",
                "local_patient_id": local,
                "relative_time_days": pd.NA,
                "relative_time_days_precision": "ordinal_only",
                "timepoint_index": tp,
                "treatment_phase": r["treatment_phase"],
                "sample_role": r["sample_role"],
                "has_imaging": False,
                "has_molecular": True,
            })
        visits = pd.DataFrame(rows)
        visits = visits.sort_values(
            ["patient_id", "timepoint_index"]).reset_index(drop=True)
        visits["visit_index"] = visits.groupby("patient_id").cumcount()
        return visits

    def _build_molecular(self, samples: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in samples.iterrows():
            local = r["local_patient_id"]
            tp = r["timepoint_index"]
            rows.append({
                "specimen_id": make_key(COHORT, local, f"T{tp}"),
                "patient_id": make_patient_id(COHORT, local),
                "cohort": COHORT,
                "visit_id": make_key(COHORT, local, f"T{tp}"),
                "local_specimen_id": f"{local}_T{tp}",
                "assay": "snRNA_seq",
                "platform": "10x Chromium 3' v3 / Illumina NovaSeq 6000",
                "tissue_collection_relative_day": pd.NA,
                "timepoint_index": tp,
                "treatment_phase": r["treatment_phase"],
                "sample_role": r["sample_role"],
                "sample_role_raw": r["phase_raw"],
                "is_paired": bool(r["is_paired"]),
                "accession": r["gsm"],
                "file_name": r["file_name"],
                "supplementary_url": r["supplementary_url"],
                "site": r["site"],
            })
        return pd.DataFrame(rows)

    # -- convenience ------------------------------------------------------
    def paired_patients(self) -> pd.DataFrame:
        """Patients with a primary and at least one recurrent specimen."""
        samples = self.sample_records()
        out = []
        for local, grp in samples.groupby("local_patient_id"):
            roles = set(grp["sample_role"])
            if "primary" in roles and (roles & {"first_recurrence",
                                                "second_recurrence"}):
                out.append({
                    "local_patient_id": local,
                    "patient_id": make_patient_id(COHORT, local),
                    "site": grp["site"].iloc[0],
                    "n_specimens": len(grp),
                    "roles": ",".join(sorted(roles)),
                    "gsm_primary": grp.loc[grp.sample_role == "primary",
                                           "gsm"].iloc[0],
                    "gsm_recurrent": grp.loc[grp.sample_role == "first_recurrence",
                                             "gsm"].iloc[0]
                    if "first_recurrence" in roles else pd.NA,
                })
        return pd.DataFrame(out)

    def specimen_download_plan(self, local_patient_ids: list[str] | None = None
                               ) -> pd.DataFrame:
        """What would be downloaded, with sizes, before anything is fetched."""
        samples = self.sample_records()
        if local_patient_ids:
            samples = samples[samples.local_patient_id.isin(local_patient_ids)]
        return samples[["gsm", "local_patient_id", "sample_role",
                        "file_name", "supplementary_url"]].reset_index(drop=True)


def _first(value: object) -> object:
    if isinstance(value, list):
        return value[0] if value else pd.NA
    return value if value is not None else pd.NA
