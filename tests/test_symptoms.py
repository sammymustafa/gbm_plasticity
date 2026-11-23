"""Phase 5: structured symptoms and the optional note extractor."""

from __future__ import annotations

import pandas as pd
import pytest

from gbmplast.symptoms.notes import (
    ExtractionConfig,
    apply_validation,
    extract_from_documents,
    extract_mentions,
    sample_for_validation,
    validation_agreement,
)
from gbmplast.symptoms.schema import SYMPTOM_KEYS, TERMS_BY_KEY
from gbmplast.symptoms.structured import coverage_matrix, from_long, parse_presence


@pytest.mark.parametrize("value,expected", [
    (1, True), (0, False), (1.0, True), (0.0, False),
    ("1.0", True), ("0.0", False), ("yes", True), ("NO", False),
    (True, True), (False, False),
    (None, None), ("maybe", None), (2, None), ("", None),
])
def test_parse_presence(value, expected):
    out = parse_presence(value)
    if expected is None:
        assert pd.isna(out)
    else:
        assert out is expected


def test_vocabulary_is_well_formed():
    assert len(SYMPTOM_KEYS) == len(set(SYMPTOM_KEYS))
    for key in SYMPTOM_KEYS:
        term = TERMS_BY_KEY[key]
        assert term.synonyms
        assert all(s == s.lower() for s in term.synonyms), key


def test_unknown_symptom_terms_are_dropped_and_recorded():
    df = pd.DataFrame({"patient_id": ["p"], "symptom": ["not_a_symptom"],
                       "present": [1]})
    out = from_long(df, cohort="T")
    assert len(out) == 0
    assert out.attrs["unknown_symptom_terms"] == ["not_a_symptom"]


def test_coverage_separates_absent_from_unassessed():
    df = pd.DataFrame({
        "patient_id": ["a", "b", "c"],
        "symptom": ["headache"] * 3,
        "present": [1, 0, None],
    })
    coverage = coverage_matrix(from_long(df, cohort="T")).set_index("symptom")
    row = coverage.loc["headache"]
    assert (row.n_present, row.n_absent, row.n_unknown) == (1, 1, 1)
    assert row.pct_assessed == pytest.approx(66.7, abs=0.1)


# -- negation / temporality / experiencer ---------------------------------

@pytest.mark.parametrize("text,symptom,negated", [
    ("Patient denies seizures.", "seizure", True),
    ("No headache reported.", "headache", True),
    ("Reports intermittent headache.", "headache", False),
    ("Negative for aphasia.", "aphasia", True),
    ("New expressive aphasia today.", "aphasia", False),
])
def test_negation_detection(text, symptom, negated):
    mentions = [m for m in extract_mentions(text, document_id="d",
                                            patient_id="p")
                if m.symptom == symptom]
    assert mentions, f"no {symptom} mention found in {text!r}"
    assert mentions[0].negated is negated


@pytest.mark.parametrize("text,expected", [
    ("History of seizure in 2019.", "historical"),
    ("Will monitor for seizure activity.", "hypothetical"),
    ("Witnessed seizure this morning.", "current"),
    ("Status post seizure workup.", "historical"),
])
def test_temporality_detection(text, expected):
    mentions = [m for m in extract_mentions(text, document_id="d",
                                            patient_id="p")
                if m.symptom == "seizure"]
    assert mentions
    assert mentions[0].temporality == expected


def test_family_history_is_attributed_to_someone_else():
    mentions = extract_mentions("Family history of epilepsy.",
                                document_id="d", patient_id="p")
    seizure = [m for m in mentions if m.symptom == "seizure"]
    assert seizure
    assert seizure[0].experiencer == "other"


def test_negation_does_not_cross_a_clause_break():
    """'No headache. Seizure witnessed.' must not negate the seizure."""
    mentions = extract_mentions("No headache. Seizure witnessed overnight.",
                                document_id="d", patient_id="p")
    by_symptom = {m.symptom: m for m in mentions}
    assert by_symptom["headache"].negated is True
    assert by_symptom["seizure"].negated is False


def test_extraction_carries_provenance_and_is_unvalidated():
    docs = pd.DataFrame({
        "document_id": ["n1"], "patient_id": ["p1"],
        "document_relative_day": [10],
        "text": ["Denies seizures. New aphasia noted."]})
    symptoms, mentions = extract_from_documents(docs, cohort="T")
    assert len(symptoms) == 2
    assert (symptoms["source"] == "note_extraction").all()
    assert symptoms["validated_by_human"].isna().all()
    assert symptoms["provenance"].str.contains("n1:").all()
    assert "unvalidated" in symptoms.attrs["warning"]
    for _, row in mentions.iterrows():
        start, end = int(row["start"]), int(row["end"])
        assert docs.text.iloc[0][start:end].lower() in \
            row["matched_text"].lower() + row["matched_text"].lower()


def test_ambiguous_mentions_stay_na():
    """Historical-only mentions assert neither presence nor absence."""
    docs = pd.DataFrame({
        "document_id": ["n1"], "patient_id": ["p1"],
        "text": ["History of seizure in 2019."]})
    symptoms, _ = extract_from_documents(docs, cohort="T")
    row = symptoms[symptoms.symptom == "seizure"].iloc[0]
    assert pd.isna(row["present"])
    assert row["temporality"] == "historical"


def test_validation_round_trip():
    docs = pd.DataFrame({
        "document_id": ["n1", "n2"], "patient_id": ["p1", "p2"],
        "text": ["New seizure overnight.", "Denies seizures."]})
    symptoms, mentions = extract_from_documents(docs, cohort="T")
    sample = sample_for_validation(symptoms, mentions, n_per_symptom=10)
    assert "reviewer_present" in sample.columns
    assert sample["reviewer_present"].isna().all()

    reviewed = sample.copy()
    reviewed["reviewer_present"] = ["TRUE"] * len(reviewed)
    reviewed["reviewer_id"] = "rev1"

    agreement = validation_agreement(symptoms, reviewed)
    assert "agreement" in agreement.columns

    applied = apply_validation(symptoms, reviewed)
    updated = applied[applied.symptom_id.isin(reviewed.symptom_id)]
    assert (updated["validated_by_human"] == True).all()   # noqa: E712
    assert (updated["source"] == "manual_review").all()


def test_extra_synonyms_are_honoured():
    config = ExtractionConfig(extra_synonyms={"seizure": ("spell",)})
    mentions = extract_mentions("Witnessed spell this morning.",
                                document_id="d", patient_id="p", config=config)
    assert any(m.symptom == "seizure" for m in mentions)


@pytest.mark.requires_data
def test_public_cohorts_carry_no_symptoms(config, available_datasets):
    """No public dataset here releases symptoms; none may appear to."""
    from gbmplast.adapters import get_adapter

    for key in ("lumiere", "upenn_gbm", "ucsd_ptgbm", "gbm_care"):
        if not available_datasets.get(key):
            continue
        bundle = get_adapter(key, config).build()
        assert len(bundle.symptoms) == 0, (
            f"{key} produced symptom rows; no public cohort releases them")
