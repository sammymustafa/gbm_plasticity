"""Institutional-cohort adapter (Phase 5): the switch-over point.

This is how the future PI cohort enters the pipeline **without changing any
model code**.  The adapter is entirely config-driven: a YAML block names your
files and maps your column spellings onto the canonical schema, and every
downstream module (longitudinal builder, modelling, metrics, leakage checks,
figures) runs unchanged.

Minimal config
--------------
.. code-block:: yaml

    cohorts:
      institutional:
        raw_subdir: institutional
        cohort_key: MYSITE           # id namespace, e.g. MYSITE::MRN-123
        t0_definition: "date of first resection"
        files:
          patients:   patients.csv
          visits:     visits.csv
          imaging:    imaging.csv        # optional
          treatments: treatments.csv     # optional
          symptoms:   symptoms_long.csv  # optional
          molecular_samples: specimens.csv   # optional
          outcomes:   outcomes.csv       # optional
        column_map:
          patients:
            local_patient_id: mrn
            sex: gender
            age_at_diagnosis_years: age_dx
            mgmt_status: mgmt
          visits:
            local_visit_id: scan_label
            local_patient_id: mrn
            relative_time_days: days_from_surgery
        value_map:
          patients:
            sex: {M: male, F: female}
            mgmt_status: {MET: methylated, UNMET: unmethylated}
        symptoms_format: long        # 'long' | 'wide'
        notes:
          enabled: false             # opt-in note extraction
          file: notes.csv
          text_column: note_text

Guarantees
----------
* No column is invented.  Anything you do not map is NA.
* Dates are never required: the pipeline runs on relative days.  If you
  supply calendar dates, set ``date_columns`` and ``anchor_date_column`` and
  the adapter converts them to relative days **and does not retain the
  calendar values**, which keeps derived tables de-identified.
* ``validate_bundle`` runs the same structural checks as for public cohorts.
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
from gbmplast.schema import (
    CANONICAL_TABLES,
    SPECS,
    CanonicalBundle,
    make_key,
    make_patient_id,
)

DEFAULT_FILES = {
    "patients": "patients.csv",
    "visits": "visits.csv",
    "imaging": "imaging.csv",
    "treatments": "treatments.csv",
    "symptoms": "symptoms.csv",
    "molecular_samples": "molecular_samples.csv",
    "outcomes": "outcomes.csv",
}

REQUIRED_TABLES = ("patients", "visits")


@register
class InstitutionalAdapter(DatasetAdapter):
    key = "institutional"

    # -- config helpers ---------------------------------------------------
    @property
    def cohort_key(self) -> str:
        return str(self.config.get("cohorts.institutional.cohort_key",
                                   "INSTITUTIONAL"))

    @property
    def _files(self) -> dict[str, str]:
        configured = self.config.get("cohorts.institutional.files", {}) or {}
        return {**DEFAULT_FILES, **configured}

    def _column_map(self, table: str) -> dict[str, str]:
        node = self.config.get("cohorts.institutional.column_map", {}) or {}
        return node.get(table, {}) or {}

    def _value_map(self, table: str) -> dict[str, dict]:
        node = self.config.get("cohorts.institutional.value_map", {}) or {}
        return node.get(table, {}) or {}

    def describe(self) -> CohortDescription:
        return CohortDescription(
            key=self.cohort_key,
            name=f"Institutional cohort ({self.cohort_key})",
            url="(local, not public)",
            access="Local institutional data under your own IRB and DUA.",
            license="Institutional; not redistributable.",
            citation="(internal cohort)",
            modalities=tuple(self.config.get(
                "cohorts.institutional.modalities",
                ["T1", "T1CE", "T2", "FLAIR", "DWI", "DSC", "SEG"])),
            t0_definition=str(self.config.get(
                "cohorts.institutional.t0_definition",
                "date of first resection")),
            known_limitations=(
                "Contents depend entirely on what the site supplies; run "
                "'gbmplast dictionary --dataset institutional' to see the "
                "actual availability and missingness before modelling.",
            ),
            requires_registration=True,
            notes={
                "deidentification": (
                    "Calendar dates, if supplied, are converted to relative "
                    "days and dropped from derived tables."
                ),
            },
        )

    def source_files(self) -> list[SourceFile]:
        files = self._files
        return [
            SourceFile(name, files[name],
                       f"Institutional {name} table.",
                       required=name in REQUIRED_TABLES,
                       url=None,
                       note=("Map your column names via "
                             f"cohorts.institutional.column_map.{name}"))
            for name in CANONICAL_TABLES
        ]

    # -- build ------------------------------------------------------------
    def build(self) -> CanonicalBundle:
        self.require_buildable()
        cohort = self.cohort_key
        tables: dict[str, pd.DataFrame] = {}

        for name in CANONICAL_TABLES:
            path = self.raw_dir / self._files[name]
            if not path.exists():
                tables[name] = SPECS[name].empty()
                continue
            raw = _read(path)
            renamed = _apply_column_map(raw, self._column_map(name))
            mapped = _apply_value_map(renamed, self._value_map(name))
            mapped = self._relativise_dates(mapped, name)
            tables[name] = self._finalise(mapped, name, cohort)

        if self._notes_enabled():
            tables["symptoms"] = self._merge_note_symptoms(tables["symptoms"])

        tables["visits"] = _index_visits(tables["visits"])
        bundle = self.empty_bundle(**tables)
        bundle.notes.update({
            "source": f"institutional cohort {cohort} from {self.raw_dir}",
            "note_extraction": ("enabled" if self._notes_enabled()
                                else "disabled (structured symptoms only)"),
        })
        return bundle

    # -- internals --------------------------------------------------------
    def _finalise(self, df: pd.DataFrame, table: str, cohort: str) -> pd.DataFrame:
        out = df.copy()
        out["cohort"] = cohort

        if "local_patient_id" in out.columns:
            out["patient_id"] = out["local_patient_id"].map(
                lambda p: make_patient_id(cohort, str(p).strip()))
        elif "patient_id" in out.columns:
            out["local_patient_id"] = out["patient_id"].astype(str)
            out["patient_id"] = out["local_patient_id"].map(
                lambda p: make_patient_id(cohort, str(p).strip())
                if "::" not in str(p) else str(p))

        # Namespace visit ids so they cannot collide with a public cohort.
        if table != "patients" and "visit_id" not in out.columns \
                and "local_visit_id" in out.columns:
            out["visit_id"] = [
                make_key(cohort, str(p), str(v))
                for p, v in zip(out.get("local_patient_id", ""),
                                out["local_visit_id"])]
        elif "visit_id" in out.columns:
            out["visit_id"] = out["visit_id"].map(
                lambda v: v if "::" in str(v) else make_key(cohort, str(v)))

        pk = SPECS[table].primary_key[0]
        if pk not in out.columns:
            base = {"patients": "local_patient_id",
                    "visits": "local_visit_id"}.get(table)
            if base and base in out.columns:
                out[pk] = out[base].map(lambda v: make_key(cohort, str(v)))
            else:
                out[pk] = [make_key(cohort, table, str(i)) for i in range(len(out))]

        if table == "patients" and "t0_definition" not in out.columns:
            out["t0_definition"] = self.describe().t0_definition

        if table == "symptoms" and len(out):
            from gbmplast.symptoms.structured import parse_presence

            # A site table of symptoms is structured by definition; its
            # presence cells may be 1/0, yes/no or True/False.
            if "source" not in out.columns:
                out["source"] = "structured"
            else:
                out["source"] = out["source"].fillna("structured")
            if "present" in out.columns:
                out["present"] = out["present"].map(parse_presence)
            if "temporality" not in out.columns:
                out["temporality"] = "current"
            else:
                out["temporality"] = out["temporality"].fillna("current")

        return SPECS[table].conform(out)

    def _relativise_dates(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Convert configured calendar dates to relative days, then drop them."""
        cfg = self.config.get("cohorts.institutional.date_columns", {}) or {}
        spec = cfg.get(table)
        if not spec:
            return df
        anchor_col = spec.get("anchor")
        targets = spec.get("targets", {}) or {}
        if not anchor_col or anchor_col not in df.columns:
            return df
        out = df.copy()
        anchor = pd.to_datetime(out[anchor_col], errors="coerce")
        for src, dest in targets.items():
            if src not in out.columns:
                continue
            value = pd.to_datetime(out[src], errors="coerce")
            out[dest] = (value - anchor).dt.days.astype("Int64")
            out = out.drop(columns=[src])
        return out.drop(columns=[anchor_col])

    def _notes_enabled(self) -> bool:
        return bool(self.config.get("cohorts.institutional.notes.enabled", False))

    def _merge_note_symptoms(self, structured: pd.DataFrame) -> pd.DataFrame:
        from gbmplast.symptoms.notes import ExtractionConfig, extract_from_documents

        node = self.config.get("cohorts.institutional.notes", {}) or {}
        path = self.raw_dir / node.get("file", "notes.csv")
        if not path.exists():
            return structured
        docs = _read(path)
        text_col = node.get("text_column", "text")
        if text_col not in docs.columns:
            raise ValueError(
                f"[{self.cohort_key}] notes file {path} has no column "
                f"{text_col!r}; set cohorts.institutional.notes.text_column.")
        extracted, mentions = extract_from_documents(
            docs, cohort=self.cohort_key, text_col=text_col,
            document_id_col=node.get("document_id_column", "document_id"),
            patient_col=node.get("patient_column", "patient_id"),
            visit_col=node.get("visit_column", "visit_id"),
            day_col=node.get("day_column", "document_relative_day"),
            config=ExtractionConfig(
                window=int(node.get("window", 60)),
                extra_synonyms={k: tuple(v) for k, v in
                                (node.get("extra_synonyms", {}) or {}).items()},
            ))
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        mentions.to_csv(self.interim_dir / "note_symptom_mentions.csv", index=False)
        combined = pd.concat([structured, extracted], ignore_index=True)
        combined.attrs["note_extraction_warning"] = (
            "Note-derived symptom rows (source='note_extraction') are "
            "unvalidated. Validate a sample and report agreement before use."
        )
        return combined

    # -- scaffolding helper ----------------------------------------------
    def write_templates(self, outdir: Path | None = None) -> Path:
        """Write empty CSV templates with the canonical columns for the site."""
        out = Path(outdir) if outdir else self.raw_dir
        out.mkdir(parents=True, exist_ok=True)
        for name, spec in SPECS.items():
            spec.empty().to_csv(out / self._files[name], index=False)
        (out / "README_TEMPLATES.md").write_text(
            "# Institutional intake templates\n\n"
            "One CSV per canonical table. Fill in what you have and leave the\n"
            "rest empty -- blank means 'not available', which the pipeline\n"
            "preserves rather than imputing.\n\n"
            "Required: patients.csv, visits.csv.\n\n"
            "If your column names differ, do NOT rename your exports: add a\n"
            "`column_map` block to configs/institutional.yaml instead.\n\n"
            "Times are integer days relative to each patient's t0 anchor\n"
            "(see cohorts.institutional.t0_definition). If you have calendar\n"
            "dates, configure `date_columns` and the adapter converts them\n"
            "and drops the calendar values.\n",
            encoding="utf-8")
        return out


# -- helpers ---------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    sep = "\t" if suffix in {".tsv"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _apply_column_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """mapping is canonical_name -> site_column_name."""
    if not mapping:
        return df
    rename = {site: canonical for canonical, site in mapping.items()
              if site in df.columns}
    missing = {c: s for c, s in mapping.items() if s not in df.columns}
    out = df.rename(columns=rename)
    if missing:
        out.attrs["unmapped_columns"] = missing
    return out


def _apply_value_map(df: pd.DataFrame,
                     mapping: dict[str, dict]) -> pd.DataFrame:
    if not mapping:
        return df
    out = df.copy()
    for column, lookup in mapping.items():
        if column in out.columns and lookup:
            out[column] = out[column].map(lambda v: lookup.get(v, v))
    return out


def _index_visits(visits: pd.DataFrame) -> pd.DataFrame:
    if visits.empty or "patient_id" not in visits.columns:
        return visits
    out = visits.copy()
    if "relative_time_days" in out.columns:
        out["_t"] = pd.to_numeric(out["relative_time_days"], errors="coerce")
    else:
        out["_t"] = np.nan
    out = out.sort_values(["patient_id", "_t"], na_position="last")
    out["visit_index"] = out.groupby("patient_id").cumcount()
    return out.drop(columns=["_t"]).reset_index(drop=True)
