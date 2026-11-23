# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # UCSD-PTGBM - exploratory data analysis
#
# [TCIA collection UCSD-PTGBM](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/):
# 178 patients, 243 **post-treatment** timepoints, with expert labels
# separating residual/recurrent tumour from treatment-related change - and an
# explicit record of how each negative judgement was made.

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
adapter = get_adapter("ucsd_ptgbm", config)
bundle = adapter.build()
bundle.summary()

# %% [markdown]
# ## The cohort's own data dictionary
#
# Read it rather than inferring column meanings. It states, for example, that
# `Overall survival` is "days between initial surgery and death".

# %%
adapter.data_dictionary_official()

# %% [markdown]
# ## Time anchoring
#
# Every released offset is **days from that scan's acquisition** and may be
# negative. The adapter re-anchors to initial surgery so that this cohort
# shares one convention with the others.

# %%
visits = bundle.visits
print("timepoints per patient:",
      visits.groupby("patient_id").size().value_counts().to_dict())
print(f"\nvisits with a known offset from surgery: "
      f"{int(visits.relative_time_days.notna().sum())} of {len(visits)}")
visits[["local_visit_id", "timepoint_index", "relative_time_days",
        "days_acq_to_last_surgery", "days_acq_to_last_radiation"]].head(8)

# %% [markdown]
# ## The outcome, and how it was adjudicated
#
# This is the cohort's distinguishing feature. 192 timepoints are
# tumour-positive and 51 are treatment-related change - and the negatives carry
# the **basis** of the judgement.

# %%
print(visits.tumor_status.value_counts().to_string())
print()
pd.crosstab(visits.tumor_status_detail, visits.tumor_status_ascertainment)

# %% [markdown]
# A limitation worth stating plainly: the collection abstract says 45 of the
# 192 positives are pathology-proven, but **no released column flags which
# ones**. So a histopathology-restricted comparison is possible on the negative
# class only, and this pipeline refuses to construct one for the positives.

# %%
outcomes = bundle.outcomes
tumour = outcomes[outcomes.outcome_name == "tumor_vs_treatment_change"]
tumour.groupby(["outcome_label", "ascertainment"]).size()

# %% [markdown]
# ## Treatments
#
# Unusually for a public imaging cohort, treatment episodes are released with
# relative dates.

# %%
treatments = bundle.treatments
print(treatments.treatment_type.value_counts().to_string())
print(f"\nepisodes with a known start day: "
      f"{int(treatments.start_relative_day.notna().sum())} of {len(treatments)}")
treatments.head(8)[["local_patient_id" if "local_patient_id" in treatments
                    else "patient_id", "treatment_type", "treatment_name",
                    "start_relative_day", "end_relative_day",
                    "extent_of_resection"]]

# %% [markdown]
# ## Molecular markers

# %%
patients = bundle.patients
for col in ("mgmt_status", "idh_status", "codeletion_1p19q", "atrx_status"):
    print(f"{col}: {patients[col].value_counts(dropna=False).to_dict()}")

# %% [markdown]
# ## Survival
#
# OS is populated for a subset; PFS has a time but **no progression-event
# indicator**, so it cannot be used for time-to-event modelling.

# %%
survival = outcomes[outcomes.outcome_type == "time_to_event"]
survival.groupby("outcome_name").agg(
    n=("outcome_value", "size"),
    n_with_time=("outcome_value", lambda s: int(s.notna().sum())),
    n_events=("event_indicator", lambda s: int((s == True).sum())),
    n_censoring_unknown=("event_indicator", lambda s: int(s.isna().sum())))

# %% [markdown]
# One released PFS value is negative (-8 days), which is an error in the source
# table. It is reported rather than silently dropped or clipped.

# %%
negative = survival[pd.to_numeric(survival.outcome_value,
                                  errors="coerce") < 0]
negative[["patient_id", "outcome_name", "outcome_value", "event_indicator"]]

# %% [markdown]
# ## Imaging
#
# The protocol is documented, but the clinical tables carry no per-scan
# availability flags, so `available` is **NA (unknown)** rather than True.
# Imaging itself needs IBM Aspera and is not retrievable headlessly.

# %%
D.modality_completeness(bundle)

# %%
report = validate_bundle(bundle)
report.to_frame()[["check", "table", "severity", "message"]]
