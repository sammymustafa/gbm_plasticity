"""OPTIONAL clinical-note symptom extraction (Phase 5).

Scope and honesty
-----------------
This module is **optional** and is never used on public data: none of the
public cohorts in this project release clinical notes.  It exists so that the
institutional cohort can be ingested without changing any model code.

It is a *rule-based* extractor with explicit negation and temporality
detection, in the lineage of NegEx/ConText.  It is deliberately simple and
auditable rather than a black box, and every extracted value carries:

* ``provenance``        -- document id and character offsets of the mention
* ``negated``           -- a negation cue governs the mention
* ``temporality``       -- current / historical / hypothetical
* ``validated_by_human`` -- NA until a reviewer fills it in

**Extraction output is not evidence until validated.**  Use
:func:`sample_for_validation` to draw a stratified review set and
:func:`apply_validation` to fold reviewer decisions back in, then report
agreement with :func:`validation_agreement`.  Any downstream analysis on
note-derived symptoms should state the measured agreement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.schema import SYMPTOMS, make_key, make_patient_id
from gbmplast.symptoms.schema import SYMPTOM_TERMS, TERMS_BY_KEY

# -- cue lexicons ----------------------------------------------------------

#: Cues that negate what follows them, within the same clause.
NEGATION_PRE = (
    "no", "not", "without", "denies", "denied", "denying", "negative for",
    "absent", "free of", "ruled out", "r/o", "rules out", "no evidence of",
    "no sign of", "no signs of", "no history of", "never had", "nor",
    "resolved", "unremarkable for",
)

#: Cues that negate what precedes them.
NEGATION_POST = (
    "is absent", "are absent", "was absent", "were absent",
    "is negative", "are negative", "not present", "ruled out",
)

#: Cues that make a mention historical rather than current.
HISTORICAL_CUES = (
    "history of", "h/o", "hx of", "hx:", "past medical history",
    "pmh", "previously", "prior", "in the past", "years ago",
    "months ago", "status post", "s/p", "remote",
)

#: Cues that make a mention hypothetical / conditional.
HYPOTHETICAL_CUES = (
    "if", "should", "in case of", "risk of", "concern for", "to rule out",
    "monitor for", "watch for", "return if", "call if", "may develop",
    "possible", "versus", "vs", "rule out", "r/o", "query", "?",
)

#: Mentions attributed to someone other than the patient.
EXPERIENCER_CUES = ("family history", "fh", "mother", "father", "sibling",
                    "brother", "sister", "son", "daughter")

#: Tokens that end a clause, limiting how far a cue can reach.
CLAUSE_BREAKS = (";", ".", ":", " but ", " however ", " although ",
                 " except ", "\n")

#: How many characters before/after a mention a cue may act over.
DEFAULT_WINDOW = 60


@dataclass
class Mention:
    document_id: str
    patient_id: str
    symptom: str
    matched_text: str
    start: int
    end: int
    negated: bool
    temporality: str
    experiencer: str
    cue: str | None
    snippet: str

    def provenance(self) -> str:
        return f"{self.document_id}:{self.start}-{self.end}"


@dataclass
class ExtractionConfig:
    window: int = DEFAULT_WINDOW
    enabled_symptoms: tuple[str, ...] = tuple(t.key for t in SYMPTOM_TERMS)
    extra_synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    require_word_boundary: bool = True


def _patterns(cfg: ExtractionConfig) -> dict[str, re.Pattern]:
    out: dict[str, re.Pattern] = {}
    for term in SYMPTOM_TERMS:
        if term.key not in cfg.enabled_symptoms:
            continue
        syns = list(term.synonyms) + list(cfg.extra_synonyms.get(term.key, ()))
        syns = sorted({s.strip().lower() for s in syns if s.strip()},
                      key=len, reverse=True)
        body = "|".join(re.escape(s) for s in syns)
        pattern = rf"\b(?:{body})\b" if cfg.require_word_boundary else f"(?:{body})"
        out[term.key] = re.compile(pattern, re.IGNORECASE)
    return out


def _clause_left(text: str, index: int, window: int) -> str:
    start = max(0, index - window)
    chunk = text[start:index]
    cut = 0
    lowered = chunk.lower()
    for brk in CLAUSE_BREAKS:
        pos = lowered.rfind(brk)
        if pos != -1:
            cut = max(cut, pos + len(brk))
    return chunk[cut:]


def _clause_right(text: str, index: int, window: int) -> str:
    chunk = text[index:index + window]
    lowered = chunk.lower()
    cut = len(chunk)
    for brk in CLAUSE_BREAKS:
        pos = lowered.find(brk)
        if pos != -1:
            cut = min(cut, pos)
    return chunk[:cut]


def _find_cue(haystack: str, cues: tuple[str, ...]) -> str | None:
    low = haystack.lower()
    for cue in cues:
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", low):
            return cue
    return None


def extract_mentions(
    text: str,
    *,
    document_id: str,
    patient_id: str,
    config: ExtractionConfig | None = None,
) -> list[Mention]:
    """Find symptom mentions with negation / temporality / experiencer flags."""
    cfg = config or ExtractionConfig()
    patterns = _patterns(cfg)
    mentions: list[Mention] = []
    for key, pattern in patterns.items():
        for m in pattern.finditer(text):
            left = _clause_left(text, m.start(), cfg.window)
            right = _clause_right(text, m.end(), cfg.window)

            neg_cue = _find_cue(left, NEGATION_PRE) or _find_cue(right, NEGATION_POST)
            hist_cue = _find_cue(left, HISTORICAL_CUES)
            hypo_cue = _find_cue(left, HYPOTHETICAL_CUES) or _find_cue(right, ("?",))
            exp_cue = _find_cue(left, EXPERIENCER_CUES)

            if hist_cue:
                temporality = "historical"
            elif hypo_cue:
                temporality = "hypothetical"
            else:
                temporality = "current"

            snippet_start = max(0, m.start() - cfg.window)
            mentions.append(Mention(
                document_id=document_id,
                patient_id=patient_id,
                symptom=key,
                matched_text=m.group(0),
                start=m.start(),
                end=m.end(),
                negated=neg_cue is not None,
                temporality=temporality,
                experiencer="other" if exp_cue else "patient",
                cue=neg_cue or hist_cue or hypo_cue or exp_cue,
                snippet=text[snippet_start:m.end() + cfg.window].replace("\n", " "),
            ))
    return sorted(mentions, key=lambda x: (x.symptom, x.start))


def extract_from_documents(
    documents: pd.DataFrame,
    *,
    cohort: str,
    text_col: str = "text",
    document_id_col: str = "document_id",
    patient_col: str = "patient_id",
    visit_col: str | None = "visit_id",
    day_col: str | None = "document_relative_day",
    config: ExtractionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract from a note table.

    Returns ``(symptoms, mentions)``:

    * ``symptoms`` -- canonical rows, one per (document, symptom). A symptom
      is recorded ``present=True`` only when at least one mention is
      non-negated, current and attributed to the patient; ``present=False``
      when every mention is negated; otherwise NA (asserted but ambiguous).
    * ``mentions`` -- the full audit trail, one row per mention.
    """
    cfg = config or ExtractionConfig()
    mention_rows: list[dict] = []
    symptom_rows: list[dict] = []

    for _, doc in documents.iterrows():
        text = doc[text_col]
        if not isinstance(text, str) or not text.strip():
            continue
        doc_id = str(doc[document_id_col])
        local_pt = str(doc[patient_col])
        pid = local_pt if "::" in local_pt else make_patient_id(cohort, local_pt)
        mentions = extract_mentions(text, document_id=doc_id, patient_id=pid,
                                    config=cfg)
        for mention in mentions:
            mention_rows.append({
                "document_id": doc_id, "patient_id": pid,
                "symptom": mention.symptom, "matched_text": mention.matched_text,
                "start": mention.start, "end": mention.end,
                "negated": mention.negated, "temporality": mention.temporality,
                "experiencer": mention.experiencer, "cue": mention.cue,
                "snippet": mention.snippet, "provenance": mention.provenance(),
            })

        by_symptom: dict[str, list[Mention]] = {}
        for mention in mentions:
            by_symptom.setdefault(mention.symptom, []).append(mention)

        for key, group in by_symptom.items():
            affirmed = [m for m in group
                        if not m.negated and m.temporality == "current"
                        and m.experiencer == "patient"]
            negated_all = all(m.negated for m in group)
            if affirmed:
                present: object = True
                chosen = affirmed[0]
            elif negated_all:
                present = False
                chosen = group[0]
            else:
                present = pd.NA           # historical/hypothetical only
                chosen = group[0]
            symptom_rows.append({
                "symptom_id": make_key(cohort, local_pt, key, doc_id),
                "patient_id": pid,
                "cohort": cohort,
                "visit_id": (str(doc[visit_col])
                             if visit_col and visit_col in documents.columns
                             and not pd.isna(doc.get(visit_col)) else pd.NA),
                "symptom": key,
                "present": present,
                "assessment_relative_day": _opt_int(doc, day_col),
                "source": "note_extraction",
                "provenance": "; ".join(m.provenance() for m in group),
                "negated": bool(negated_all),
                "temporality": chosen.temporality,
                "n_mentions": len(group),
                "n_affirmed": len(affirmed),
                "extractor_version": "rule-based v1 (NegEx/ConText-style)",
                "validated_by_human": pd.NA,
            })

    symptoms = SYMPTOMS.conform(pd.DataFrame(symptom_rows))
    mentions_df = pd.DataFrame(mention_rows)
    symptoms.attrs["warning"] = (
        "Note-derived symptoms are unvalidated extractions, not clinical "
        "facts. Validate a sample with sample_for_validation() and report "
        "agreement before using them in any analysis."
    )
    return symptoms, mentions_df


# -- manual-validation hooks ----------------------------------------------

def sample_for_validation(
    symptoms: pd.DataFrame,
    mentions: pd.DataFrame,
    *,
    n_per_symptom: int = 20,
    seed: int = 20251123,
    stratify_by_present: bool = True,
) -> pd.DataFrame:
    """Draw a reproducible stratified review set with blank reviewer columns."""
    rng = np.random.default_rng(seed)
    picks: list[pd.DataFrame] = []
    extracted = symptoms[symptoms["source"] == "note_extraction"]
    for key, grp in extracted.groupby("symptom"):
        if stratify_by_present:
            strata = [grp[grp["present"] == True],      # noqa: E712
                      grp[grp["present"] == False],     # noqa: E712
                      grp[grp["present"].isna()]]
        else:
            strata = [grp]
        per = max(1, n_per_symptom // max(1, sum(1 for s in strata if len(s))))
        for stratum in strata:
            if not len(stratum):
                continue
            take = min(per, len(stratum))
            idx = rng.choice(len(stratum), size=take, replace=False)
            picks.append(stratum.iloc[sorted(idx)])
    if not picks:
        return pd.DataFrame(columns=["symptom_id", "patient_id", "symptom",
                                     "extracted_present", "snippet",
                                     "reviewer_present", "reviewer_id",
                                     "reviewer_note"])
    sample = pd.concat(picks, ignore_index=True)
    snippets = (mentions.groupby(["patient_id", "symptom"])["snippet"]
                .apply(lambda s: " ||| ".join(s.head(3))).rename("snippet")
                if len(mentions) else None)
    out = sample[["symptom_id", "patient_id", "cohort", "symptom", "present",
                  "provenance", "temporality", "negated"]].copy()
    out = out.rename(columns={"present": "extracted_present"})
    if snippets is not None:
        out = out.merge(snippets, on=["patient_id", "symptom"], how="left")
    else:
        out["snippet"] = pd.NA
    out["reviewer_present"] = pd.NA       # reviewer fills: TRUE/FALSE/UNKNOWN
    out["reviewer_id"] = pd.NA
    out["reviewer_note"] = pd.NA
    return out.reset_index(drop=True)


def apply_validation(symptoms: pd.DataFrame,
                     reviewed: pd.DataFrame) -> pd.DataFrame:
    """Overwrite extracted values with reviewer decisions where provided."""
    from gbmplast.symptoms.structured import parse_presence

    out = symptoms.copy()
    decisions = {}
    for _, r in reviewed.iterrows():
        decision = parse_presence(r.get("reviewer_present"))
        if decision is pd.NA and str(r.get("reviewer_present", "")).strip().lower() \
                not in {"unknown", "na", ""}:
            continue
        decisions[r["symptom_id"]] = (decision, r.get("reviewer_id"))
    mask = out["symptom_id"].isin(decisions)
    for idx in out.index[mask]:
        decision, reviewer = decisions[out.at[idx, "symptom_id"]]
        out.at[idx, "present"] = decision
        out.at[idx, "validated_by_human"] = True
        out.at[idx, "source"] = "manual_review"
        if "reviewer_id" in out.columns:
            out.at[idx, "reviewer_id"] = reviewer
    return out


def validation_agreement(symptoms: pd.DataFrame,
                         reviewed: pd.DataFrame) -> pd.DataFrame:
    """Per-symptom agreement between extractor and reviewer."""
    from gbmplast.symptoms.structured import parse_presence

    merged = reviewed.copy()
    merged["reviewer_value"] = merged["reviewer_present"].map(parse_presence)
    merged["extracted_value"] = merged["extracted_present"].map(parse_presence)
    rows = []
    for key, grp in merged.groupby("symptom"):
        both = grp.dropna(subset=["reviewer_value", "extracted_value"])
        if not len(both):
            rows.append({"symptom": key, "n_reviewed": len(grp),
                         "n_comparable": 0, "agreement": pd.NA,
                         "tp": 0, "fp": 0, "fn": 0, "tn": 0})
            continue
        ext = both["extracted_value"].astype(bool)
        rev = both["reviewer_value"].astype(bool)
        tp = int(((ext) & (rev)).sum()); fp = int(((ext) & (~rev)).sum())
        fn = int(((~ext) & (rev)).sum()); tn = int(((~ext) & (~rev)).sum())
        rows.append({
            "symptom": key, "n_reviewed": len(grp), "n_comparable": len(both),
            "agreement": round((tp + tn) / len(both), 3),
            "ppv": round(tp / (tp + fp), 3) if (tp + fp) else pd.NA,
            "sensitivity": round(tp / (tp + fn), 3) if (tp + fn) else pd.NA,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def _opt_int(row: pd.Series, col: str | None) -> object:
    if not col or col not in row.index or pd.isna(row[col]):
        return pd.NA
    try:
        return int(float(row[col]))
    except (TypeError, ValueError):
        return pd.NA
