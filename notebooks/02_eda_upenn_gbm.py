# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # UPENN-GBM - exploratory data analysis
#
# [TCIA collection UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/):
# 630 patients, 671 scans, advanced MRI with CaPTk radiomics, and - unlike
# LUMIERE - **genuine survival censoring**.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))

from gbmplast.adapters import get_adapter
from gbmplast.config import load_config
from gbmplast.report import dictionary as D
from gbmplast.report.theme import apply_theme, series_color
from gbmplast.validate import validate_bundle

apply_theme()
config = load_config()
adapter = get_adapter("upenn_gbm", config)
bundle = adapter.build()
bundle.summary()

# %% [markdown]
# ## The scan-id scheme
#
# `UPENN-GBM-00036_11` and `UPENN-GBM-00036_21` are the **same patient**:
# `_11` is the baseline pre-operative scan, `_21` a later recurrence scan.
# Treating them as two patients would break grouped cross-validation.

# %%
visits = bundle.visits
print(visits.scan_role.value_counts().to_string())
print(f"\npatients: {visits.patient_id.nunique()}; scans: {len(visits)}")
print(f"patients with both scans: "
      f"{int((visits.groupby('patient_id').size() == 2).sum())}")

# %%
visits[visits.patient_id.isin(
    visits.patient_id[visits.duplicated('patient_id', keep=False)].unique()[:4]
)][["local_patient_id", "local_visit_id", "scan_role",
    "relative_time_days", "psp_tp_score"]].sort_values("local_patient_id")

# %% [markdown]
# ## Survival has real censoring
#
# `Survival_Status` distinguishes *Deceased* from *Alive*, *Lost to Follow-up*
# and *Deceased - uncertain date of death*. Censored rows take their time from
# `Survival_Censor`. This is what makes a Cox model admissible here and not on
# LUMIERE.

# %%
outcomes = bundle.outcomes
survival = outcomes[outcomes.outcome_name == "overall_survival"]
print(survival.outcome_label.value_counts().to_string())
print(f"\nevents: {int(survival.event_indicator.sum())}, "
      f"censored: {int((survival.event_indicator == False).sum())}")

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, (label, group) in enumerate(survival.groupby(survival.event_indicator)):
    ax.hist(pd.to_numeric(group.outcome_value, errors="coerce").dropna(),
            bins=40, alpha=0.7, color=series_color(i),
            label="event (deceased)" if label else "censored")
ax.set_xlabel("time from anchor scan (days)")
ax.set_ylabel("patients")
ax.set_title("Overall survival, by event status")
ax.legend()
plt.show()

# %% [markdown]
# ## Molecular markers and their missingness

# %%
patients = bundle.patients
for col in ("mgmt_status", "idh_status", "sex"):
    print(f"{col}:")
    print(patients[col].value_counts(dropna=False).to_string())
    print()
print("KPS available for", int(patients.kps.notna().sum()), "patients")

# %% [markdown]
# ## Imaging availability
#
# DTI and DSC are not available for every scan; the canonical table records
# that per series rather than assuming a complete protocol.

# %%
D.modality_completeness(bundle)

# %% [markdown]
# ## CaPTk radiomic features
#
# 11 image types / parametric maps x 3 subregions (ET, ED, NC) x two
# segmentation flavours. These are **CaPTk** features, not PyRadiomics, and are
# not comparable with the LUMIERE feature set.

# %%
features = adapter.load_captk_features(segmentation="automatic")
print("shape:", features.shape)
feature_cols = adapter.feature_columns(features)
print("feature columns:", len(feature_cols))
print("\nexamples:", feature_cols[:4], "...", feature_cols[-2:])

# %%
prefixes = pd.Series([c.split("_")[0] for c in feature_cols]).value_counts()
prefixes.head(12)

# %% [markdown]
# ## The PsP/TP score
#
# Present for the 60 recurrence scans on a 1-6 scale. The collection page does
# not publish the **direction** of the scale, so this pipeline carries it as a
# covariate and never uses it as an outcome label.

# %%
visits[visits.psp_tp_score.notna()].psp_tp_score.value_counts().sort_index()

# %% [markdown]
# ## Analysis set for the MGMT question

# %%
mgmt = outcomes[outcomes.outcome_name == "mgmt_methylation"]
baseline = visits[visits.scan_role == "baseline_preoperative"]
eligible = mgmt[mgmt.visit_id.isin(set(baseline.visit_id))]
merged = eligible.merge(features[["visit_id"]], on="visit_id", how="inner")
print(f"definite MGMT calls: {len(mgmt)}")
print(f"... on a baseline pre-operative scan: {len(eligible)}")
print(f"... with CaPTk radiomics: {len(merged)}")
eligible.outcome_label.value_counts()

# %%
report = validate_bundle(bundle)
report.to_frame()[["check", "table", "severity", "message"]]
