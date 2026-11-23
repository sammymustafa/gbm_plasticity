"""Controlled vocabulary for the clinical-symptom module (Phase 5).

None of the public cohorts used in this project release symptom data.  This
vocabulary exists for the future institutional cohort and is deliberately
kept separate from the public adapters so that no public dataset can appear
to carry symptoms it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymptomTerm:
    key: str
    label: str
    synonyms: tuple[str, ...]
    negation_sensitive: bool = True
    note: str = ""


#: Core neuro-oncology symptoms requested for the institutional cohort.
SYMPTOM_TERMS: tuple[SymptomTerm, ...] = (
    SymptomTerm("seizure", "Seizure",
                ("seizure", "seizures", "convulsion", "convulsive",
                 "epilepsy", "epileptic", "status epilepticus", "ictal",
                 "focal seizure", "generalized seizure", "gtc", "gtcs")),
    SymptomTerm("headache", "Headache",
                ("headache", "headaches", "cephalgia", "cephalalgia",
                 "head pain")),
    SymptomTerm("focal_weakness", "Focal weakness",
                ("hemiparesis", "hemiplegia", "weakness", "paresis",
                 "monoparesis", "foot drop", "drift", "pronator drift",
                 "motor deficit")),
    SymptomTerm("aphasia", "Aphasia / language disturbance",
                ("aphasia", "dysphasia", "expressive aphasia",
                 "receptive aphasia", "word finding difficulty",
                 "anomia", "dysnomia", "speech difficulty", "dysarthria")),
    SymptomTerm("cognitive_change", "Cognitive change",
                ("cognitive decline", "cognitive change", "confusion",
                 "memory loss", "memory impairment", "disorientation",
                 "encephalopathy", "inattention", "executive dysfunction")),
    SymptomTerm("sensory_deficit", "Sensory deficit",
                ("numbness", "paresthesia", "paraesthesia", "hypoesthesia",
                 "sensory loss", "sensory deficit", "tingling")),
    SymptomTerm("visual_change", "Visual change",
                ("visual field cut", "hemianopia", "hemianopsia", "diplopia",
                 "blurred vision", "visual loss", "papilledema")),
    SymptomTerm("nausea_vomiting", "Nausea / vomiting",
                ("nausea", "vomiting", "emesis")),
    SymptomTerm("gait_ataxia", "Gait disturbance / ataxia",
                ("ataxia", "gait instability", "unsteady gait",
                 "gait disturbance", "imbalance", "dysmetria")),
    SymptomTerm("fatigue", "Fatigue",
                ("fatigue", "tiredness", "lethargy", "somnolence")),
)

SYMPTOM_KEYS: tuple[str, ...] = tuple(t.key for t in SYMPTOM_TERMS)
TERMS_BY_KEY: dict[str, SymptomTerm] = {t.key: t for t in SYMPTOM_TERMS}

#: Structured performance-status and exposure fields (not free text).
PERFORMANCE_FIELDS: tuple[str, ...] = (
    "kps",                      # Karnofsky Performance Status, 0-100
    "ecog",                     # ECOG 0-5
    "mmse",
    "moca",
)

STEROID_FIELDS: tuple[str, ...] = (
    "steroid_agent",            # e.g. dexamethasone
    "steroid_dose_mg_per_day",
    "steroid_start_relative_day",
    "steroid_stop_relative_day",
    "steroid_on_at_visit",      # boolean
)

#: Antiseizure medication, commonly a confounder for seizure reporting.
ASM_FIELDS: tuple[str, ...] = (
    "asm_agent",
    "asm_start_relative_day",
    "asm_stop_relative_day",
    "asm_on_at_visit",
)

#: How a symptom value was established.
SOURCE_VALUES = ("structured", "note_extraction", "manual_review")

#: Temporality assertion for an extracted mention.
TEMPORALITY_VALUES = ("current", "historical", "hypothetical", "unknown")

#: Severity scale left deliberately open: institutions differ (CTCAE grade,
#: mild/moderate/severe, numeric). The adapter records the raw value plus the
#: scale name rather than forcing a mapping.
SEVERITY_SCALES = ("ctcae_v5", "mild_moderate_severe", "numeric", "other")
