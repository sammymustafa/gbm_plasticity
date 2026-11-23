# gbmplast - glioblastoma plasticity research pipeline

A reproducible Python pipeline for studying **glioblastoma tumour-cell
plasticity, tumour evolution and treatment resistance** across public
multimodal cohorts, built so that an institutional cohort can be swapped in
later **without changing any model code**.

Three independently valid, interoperable modules:

| Module | Question | Cohort | Status |
|---|---|---|---|
| **A** | Does next-visit progression follow from the current imaging trajectory? | LUMIERE | runs end to end |
| **B** | Can molecular status, and tumour vs treatment-related change, be read off MRI? | UPENN-GBM, UCSD-PTGBM | runs end to end |
| **C** | How do malignant and immune programmes change between primary and recurrent tumour? | GBM-CARE (+ GLASS, optional) | runs end to end |

They share a **canonical schema**, not patients. Patient ids are namespaced
(`LUMIERE::Patient-001`), so a cross-cohort join returns zero rows by
construction, and the pipeline never presents a molecular observation and an
imaging observation as though they came from the same person.

---

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"
```

PyRadiomics needs two extra steps on Python 3.11 - see
[`docs/INSTALL.md`](docs/INSTALL.md).

```bash
gbmplast datasets
```

```bash
gbmplast acquire --dataset lumiere,upenn_gbm,ucsd_ptgbm,gbm_care
```

```bash
gbmplast canonicalize --dataset all && gbmplast validate --dataset all
```

```bash
gbmplast dictionary --dataset all
```

---

## Running each module

### Module A - longitudinal MRI progression (LUMIERE)

A development subset of images is pulled **out of the 32.5 GB archive with
HTTP range requests** - about 800 MB for five patients, not 32 GB:

```bash
gbmplast acquire --dataset lumiere --images --n-patients 5 --dry-run
```

```bash
gbmplast acquire --dataset lumiere --images --n-patients 5
```

```bash
gbmplast imaging-qc --dataset lumiere --limit 12
```

```bash
gbmplast features --dataset lumiere --limit 8
```

```bash
gbmplast longitudinal --dataset lumiere -c lumiere
```

```bash
gbmplast model next-visit --dataset lumiere --outcome rano -c lumiere
```

```bash
gbmplast model next-visit --dataset lumiere --outcome volume -c lumiere
```

### Module B - radiogenomics and pseudoprogression

```bash
gbmplast radiogenomics --dataset upenn_gbm --target mgmt
```

```bash
gbmplast radiogenomics --dataset ucsd_ptgbm
```

```bash
gbmplast model survival --dataset upenn_gbm
```

### Module C - molecular plasticity (GBM-CARE)

```bash
gbmplast acquire --dataset gbm_care --specimens --n-patients 56 --dry-run
```

```bash
gbmplast acquire --dataset gbm_care --specimens --n-patients 56 --max-bytes 6
```

```bash
gbmplast molecular pseudobulk --dataset gbm_care
```

```bash
gbmplast molecular plasticity --dataset gbm_care
```

### The report

```bash
gbmplast report
```

Writes `results/RESULTS_REPORT.md`, assembled from the analysis CSVs so it
cannot drift from the numbers.

---

## What each cohort actually contains

Verified against the downloaded tables, not assumed. Full detail in
`results/data_dictionary/` after `gbmplast dictionary`.

| | LUMIERE | UPENN-GBM | UCSD-PTGBM | GBM-CARE |
|---|---|---|---|---|
| Patients | 91 | 630 | 178 | 59 |
| Timepoints | 655 visits | 671 scans | 243 scans | 121 specimens |
| Longitudinal | **all 91 have 2+ imaged studies** | 41 of 630 | 49 of 178 | 56 matched pairs |
| Imaging | T1/T1CE/T2/FLAIR + 2 automated segmentations | + DTI, DSC | + RSI, DSC, ASL (Aspera-only) | none |
| Molecular | IDH, MGMT (patient level) | IDH1, MGMT | MGMT, IDH, 1p19q, ATRX | snRNA-seq |
| Treatments | none | extent of resection | **surgery/RT/chemo/bevacizumab with dates** | none |
| Symptoms | none | KPS only | none | none |
| Survival | time, **no censoring flag** | time + censoring | time + censoring | none |

Three consequences the pipeline enforces rather than papers over:

- **LUMIERE survival is not modellable.** No vital-status column means a Cox
  model would treat every patient as having died, so it is refused.
- **UCSD positives cannot be restricted to pathology.** The negative class
  carries an adjudication tier (6 biopsy-proven, 9 clinically confirmed, 7
  radiation necrosis, 29 non-specific); the positives do not.
- **GBM-CARE has no collection dates.** Timing is ordinal
  (primary / 1st recurrent / 2nd recurrent), so relative days stay NA.

---

## Design

```
src/gbmplast/
  schema.py          canonical tables + controlled vocabularies
  config.py          YAML config, paths, seeding
  validate.py        structural + integrity checks
  adapters/          one per cohort; the ONLY place file layouts are known
  acquire/           downloads, incl. a range-based remote-ZIP reader
  imaging/           IO + geometry, QC, PyRadiomics, visual QC
  longitudinal/      transition tables, mixed models, next-visit prediction
  modeling/          estimands, grouped CV, leakage, metrics, Cox, SHAP
  molecular/         snRNA-seq, CNV validation, signatures, paired stats
  symptoms/          Phase 5: structured + optional note extraction
  report/            data dictionaries, figures, report assembly
```

### Canonical tables

`patients`, `visits`, `imaging`, `treatments`, `symptoms`,
`molecular_samples`, `outcomes`. Every adapter emits all seven; a domain a
cohort does not release is an **empty table with a note explaining why**,
never invented rows.

All times are relative integer days from a cohort-defined `t0`.
Missingness is preserved: *unknown* stays distinct from *absent* throughout
(`imaging.available` is `NA` for UCSD because the cohort documents a protocol
but publishes no per-scan flags).

### Methodological guarantees

- **Patient-grouped CV everywhere.** Every timepoint, scan and specimen of one
  patient stays in one fold. A hard assertion fails the run otherwise.
- **Nothing is fitted before splitting.** Imputation, scaling, feature
  selection and hyperparameters are all fitted inside the training fold.
- **No future information.** Predictors come only from visits at or before the
  prediction time; next-visit quantities carry a `next_` prefix and a guard
  refuses any predictor so named. A test perturbs each patient's *final*
  visit and asserts no predictor moves.
- **Simple models first.** A naive carry-forward baseline and a prevalence
  baseline are fitted alongside every learned model, and compared with a
  paired patient-level bootstrap.
- **Patient-level bootstrap CIs**, because a patient contributing several
  transitions is one observation, not several.
- **Estimands declared in code** before fitting: prediction time, lookback,
  horizon, eligibility, exclusions, acknowledged confounders, and an explicit
  statement that nothing here is causal.

---

## Headline results

Produced by the commands above; see `results/RESULTS_REPORT.md`.

- **Module A.** Next-visit RANO progression: the naive carry-forward baseline
  reaches AUROC 0.698 (95% CI 0.645-0.747) and **no learned model beats it**.
  Imaging-only is significantly worse (-0.091, p = 0.011); combined overlaps
  it (p = 0.30). For volume change, imaging models reach R² 0.24-0.28 against
  ~0 for the baselines.
- **Module B.** MGMT from pre-operative radiomics sits at chance under proper
  grouped splitting (AUROC 0.459, 95% CI 0.389-0.529, n = 262). Tumour vs
  treatment-related change from clinical covariates alone reaches AUROC 0.658
  (0.563-0.749); the imaging arm is implemented but Aspera-gated.
- **Survival.** Penalised Cox on UPENN-GBM recovers the expected prognostic
  structure - age +0.31, MGMT methylation -0.157 - with C-index 0.648.
- **Pipeline correctness.** Locally extracted PyRadiomics features reproduce
  the published LUMIERE values **exactly** (0.000 relative difference across
  shape, first-order, GLCM and GLRLM) once the cohort's own extraction
  settings are used.

---

## Switching to your own cohort

No model code changes. Write a config block that maps your column names onto
the canonical schema:

```bash
gbmplast templates --outdir data/raw/institutional
```

```yaml
cohorts:
  institutional:
    raw_subdir: /secure/mysite/export      # absolute paths are fine
    cohort_key: MYSITE
    t0_definition: "date of first resection"
    files:
      patients: patients.csv
      visits:   visits.csv
    column_map:
      patients: {local_patient_id: mrn, sex: gender, mgmt_status: mgmt}
      visits:   {local_patient_id: mrn, local_visit_id: scan_label}
    value_map:
      patients: {sex: {M: male, F: female}}
    date_columns:                 # calendar dates -> relative days, then dropped
      visits: {anchor: surgery_date, targets: {scan_date: relative_time_days}}
    notes:
      enabled: false              # optional clinical-note extraction
```

```bash
gbmplast canonicalize --dataset institutional -c institutional
```

```bash
gbmplast validate --dataset institutional -c institutional
```

Then every module runs unchanged. See
[`docs/INSTITUTIONAL_SWITCHOVER.md`](docs/INSTITUTIONAL_SWITCHOVER.md).

**Symptoms** (Phase 5) are supported only for your cohort - no public dataset
here releases them. Structured intake keeps *not assessed* distinct from
*assessed and absent*; the optional note extractor records negation,
temporality, experiencer and character-offset provenance, and ships
validation hooks because **extracted symptoms are not evidence until
reviewed**.

---

## Tests

```bash
pytest
```

```bash
pytest -m "not requires_data"
```

Covers patient-ID linkage and namespacing, longitudinal ordering, leakage
(grouped splits, future information, backward information flow,
baseline state carry-over), missing modalities and missingness preservation,
masking geometry, and survival identifiability. Tests marked
`requires_data` skip cleanly when the raw data is absent.

---

## Data use

Each cohort has its own terms, recorded in `adapter.describe()`:

- **LUMIERE** - Figshare, non-commercial use.
- **UPENN-GBM**, **UCSD-PTGBM** - TCIA, CC BY 4.0, citation required.
- **GBM-CARE** - GEO processed matrices are public; raw sequencing is withheld
  for patient privacy. Consortium cell-state annotations are on Synapse behind
  a data-use agreement.
- **GLASS** - Synapse account and consortium terms required.

Raw data is excluded from version control. This repository stores no
credentials.
