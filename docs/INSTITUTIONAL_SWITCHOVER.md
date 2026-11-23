# Switching to institutional data

The point of the canonical schema is that **your cohort enters through
configuration, not code**. Nothing in `gbmplast.modeling`,
`gbmplast.longitudinal` or `gbmplast.report` knows which cohort it is
looking at.

## 1. Export the tables

```bash
gbmplast templates --outdir data/raw/institutional
```

Writes one empty CSV per canonical table with the full column set, plus a
README. Fill in what you have; **leave the rest blank**. Blank means "not
available", which the pipeline preserves rather than imputing.

Only `patients.csv` and `visits.csv` are required.

## 2. Map your column names

Do **not** rename your exports. Add a config block instead, so the mapping is
reviewable and your extract stays byte-identical to what your data warehouse
produced.

```yaml
# configs/institutional.yaml
cohorts:
  institutional:
    raw_subdir: /secure/mysite/gbm_export
    cohort_key: MYSITE
    t0_definition: "date of first resection"
    files:
      patients:          patients.csv
      visits:            visits.csv
      imaging:           imaging.csv
      treatments:        treatments.csv
      symptoms:          symptoms_long.csv
      molecular_samples: specimens.csv
      outcomes:          outcomes.csv
    column_map:
      patients:
        local_patient_id:       mrn
        sex:                    gender
        age_at_diagnosis_years: age_at_dx
        mgmt_status:            mgmt_result
        idh_status:             idh_result
      visits:
        local_patient_id:   mrn
        local_visit_id:     accession
        relative_time_days: days_from_surgery
    value_map:
      patients:
        sex:         {M: male, F: female}
        mgmt_status: {MET: methylated, UNMET: unmethylated, IND: unknown}
```

## 3. Calendar dates

The pipeline runs on relative days. If your export has real dates, let the
adapter convert them - it **drops the calendar values afterwards**, so
derived tables under `data/processed/` carry no dates:

```yaml
    date_columns:
      visits:
        anchor: surgery_date
        targets:
          scan_date: relative_time_days
      treatments:
        anchor: surgery_date
        targets:
          tx_start: start_relative_day
          tx_end:   end_relative_day
```

## 4. Symptoms

Public cohorts release none, so this is yours alone.

**Structured** (preferred). A long table of
`patient_id, symptom, present, assessment_relative_day`, with `symptom` drawn
from `gbmplast.symptoms.schema.SYMPTOM_KEYS`:

```
seizure, headache, focal_weakness, aphasia, cognitive_change,
sensory_deficit, visual_change, nausea_vomiting, gait_ataxia, fatigue
```

A missing row means **not assessed**; `present = 0` means **assessed and
absent**. The pipeline keeps those apart everywhere.

**Notes** (optional, opt-in):

```yaml
    notes:
      enabled: true
      file: deidentified_notes.csv
      text_column: note_text
      document_id_column: note_id
      patient_column: mrn
      day_column: note_relative_day
      extra_synonyms:
        seizure: ["spell", "event with LOC"]
```

Extraction records negation, temporality, experiencer and character-offset
provenance, and every extracted row is `validated_by_human = NA` until a
reviewer fills it in.

**Validate before you use them:**

```python
from gbmplast.symptoms.notes import sample_for_validation, apply_validation, validation_agreement

sample = sample_for_validation(symptoms, mentions, n_per_symptom=20)
sample.to_csv("review_me.csv", index=False)       # reviewer fills reviewer_present
# ... after review ...
reviewed = pd.read_csv("reviewed.csv")
print(validation_agreement(symptoms, reviewed))   # report this in any paper
symptoms = apply_validation(symptoms, reviewed)
```

## 5. Build and check

```bash
gbmplast canonicalize --dataset institutional -c institutional
```

```bash
gbmplast validate --dataset institutional -c institutional --verbose
```

```bash
gbmplast dictionary --dataset institutional -c institutional
```

Read `results/data_dictionary/institutional/` **before modelling**. It tells
you how many patients have 2+ dated visits, which modalities are present,
which outcomes carry a censoring indicator, and where the missingness is.

## 6. Run the modules

```bash
gbmplast longitudinal --dataset institutional -c institutional
```

```bash
gbmplast model next-visit --dataset institutional -c institutional --outcome rano
```

```bash
gbmplast model survival --dataset institutional -c institutional
```

Survival runs only if you supply genuine event times **and** a censoring
indicator; otherwise it refuses, exactly as it does for LUMIERE.

## 7. Imaging

Point the imaging table's `relative_path` at your files and reuse the QC and
extraction utilities directly:

```python
from gbmplast.imaging.qc import qc_image_mask_pair
from gbmplast.imaging.features import build_extractor, extract_study

qc = qc_image_mask_pair(image, mask, cohort="MYSITE", patient="p", study="s",
                        sequence="T1CE", expected_labels={1: "enhancing"})
features = extract_study({"T1CE": image}, mask,
                         labels={1: "enhancing"},
                         identifiers={"patient_id": "MYSITE::p"},
                         extractor=build_extractor(preset="default"))
```

Run `qc_image_mask_pair` on **every** study before extracting. Grid mismatch
between an image and its mask is the most common cause of silently wrong
radiomics, and it is a hard failure here.

## What to decide before you model

The pipeline makes you state these; they are not defaults to accept quietly:

1. **t0** - what does relative day 0 mean for your patients?
2. **Prediction time** - at which visit is the prediction made?
3. **Lookback** - how far back may features reach?
4. **Horizon** - over what window is the outcome observed?
5. **Eligibility and exclusions** - who enters, and who is removed, and why?
6. **Censoring** - for any time-to-event outcome, where does the indicator
   come from?

Write them into the `analysis:` block of your config. They are copied into
`estimand.csv` beside every result.

## Immunotherapy extension

For the eventual paired MRI + tissue + immunotherapy cohort:

- Record immunotherapy as `treatments.treatment_type = "immunotherapy"` with
  real start/stop relative days. Treatment timing is the main confounder in
  every analysis here, and recording it is what lets it be adjusted for.
- Put tissue specimens in `molecular_samples` with
  `tissue_collection_relative_day` populated - the public molecular cohort has
  only ordinal timing, which is what blocks a time-resolved analysis there.
- With both, `gbmplast.molecular.paired` works on your specimens, and -
  because the same patients have imaging - the MRI and tissue arms can be
  joined **within your cohort**, which is precisely what no public dataset
  here permits.
