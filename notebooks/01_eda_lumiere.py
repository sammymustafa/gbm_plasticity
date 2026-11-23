# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LUMIERE - exploratory data analysis
#
# Longitudinal glioblastoma MRI with expert RANO evaluation
# ([Figshare collection 5904905](https://figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905)).
#
# This notebook describes what the cohort actually contains, including what it
# does **not** contain. Run `gbmplast acquire --dataset lumiere` first.

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
from gbmplast.report.theme import ROLE, apply_theme, series_color
from gbmplast.validate import validate_bundle

apply_theme()
config = load_config("lumiere")
adapter = get_adapter("lumiere", config)
print(adapter.describe().name)
print("raw dir:", adapter.raw_dir)

# %% [markdown]
# ## What the cohort releases, and what it does not

# %%
description = adapter.describe()
print("t0:", description.t0_definition)
print("\nKnown limitations:")
for item in description.known_limitations:
    print(" -", item)

# %%
adapter.availability()[["source_key", "relative_path", "required", "present",
                        "size_bytes"]]

# %% [markdown]
# ## Canonical tables

# %%
bundle = adapter.build()
bundle.summary()

# %%
for name, note in sorted(bundle.notes.items()):
    print(f"[{name}] {note}\n")

# %% [markdown]
# Validation: structural checks on ids, ordering and survival identifiability.

# %%
report = validate_bundle(bundle)
report.to_frame()[["check", "table", "severity", "message", "n_affected"]]

# %% [markdown]
# ## Patients

# %%
patients = bundle.patients
print(f"{len(patients)} patients")
display_cols = ["sex", "idh_status", "mgmt_status"]
for col in display_cols:
    print(f"\n{col}:")
    print(patients[col].value_counts(dropna=False).to_string())

# %% [markdown]
# `IDH1 neg, Sequencing required` is an **inconclusive immunostain**, mapped to
# `indeterminate` rather than to `wildtype`. Collapsing it into wild-type would
# invent 10 definite calls.

# %%
patients.groupby(["idh_status_raw", "idh_status"], dropna=False).size()

# %% [markdown]
# ## Visits and the RANO course

# %%
visits = bundle.visits
print(f"{len(visits)} visits "
      f"({int(visits.has_imaging.sum())} imaged, "
      f"{int(visits.has_expert_rating.sum())} rated)")
print(f"rated but NOT imaged: "
      f"{int(((~visits.has_imaging) & visits.has_expert_rating).sum())}")
visits.rano_status.value_counts(dropna=False)

# %% [markdown]
# Three studies carry two expert ratings each. Two are genuine disagreements and
# are flagged rather than silently resolved.

# %%
visits[visits.rano_conflict][["local_patient_id", "local_visit_id",
                              "rano_status", "rano_status_raw",
                              "rano_n_assessments"]]

# %%
per_patient = visits.groupby("patient_id").size()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(per_patient, bins=range(1, int(per_patient.max()) + 2),
        color=series_color(0), edgecolor="#fcfcfb")
ax.set_xlabel("visits per patient")
ax.set_ylabel("patients")
ax.set_title(f"Follow-up depth (median {per_patient.median():.0f}, "
             f"max {per_patient.max()})")
plt.show()

# %% [markdown]
# ## Modality completeness

# %%
D.modality_completeness(bundle)

# %% [markdown]
# ## Radiomics: the absent-label question
#
# A row exists for every (study, sequence, label), but when the segmentation
# found no voxels of a label **every feature is NaN**. Those are volume zero,
# not missing data - and they are concentrated in post-operative and
# complete-response studies.

# %%
radiomics = adapter.load_radiomics("HD-GLIO-AUTO")
ct1 = radiomics[(radiomics.sequence == "CT1")
                & (radiomics.subregion == "enhancing")]
feature_cols = [c for c in ct1.columns if c.startswith("original_")]
absent = ct1[feature_cols].isna().all(axis=1)
print(f"{int(absent.sum())} of {len(ct1)} studies have NO enhancing label")

status = visits.set_index("visit_id").loc[
    ct1.loc[absent.to_numpy(), "visit_id"], "rano_status"]
status.value_counts(dropna=False)

# %% [markdown]
# ## Longitudinal transitions

# %%
from gbmplast.longitudinal.build import build_longitudinal

result = build_longitudinal(adapter, config)
for line in result.provenance:
    print(" -", line)

# %%
table = result.table
table[["patient_id", "index_local_visit_id", "next_local_visit_id",
       "interscan_interval_days", "cur_enhancing__volume_mm3",
       "prev_enhancing__volume_mm3", "next_rano_status"]].head(10)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].hist(table.interscan_interval_days, bins=30, color=series_color(0),
             edgecolor="#fcfcfb")
axes[0].set_xlabel("days between consecutive studies")
axes[0].set_ylabel("transitions")
axes[0].set_title("Inter-scan interval")

for i, subregion in enumerate(("enhancing", "non_enhancing")):
    values = pd.to_numeric(table[f"cur_{subregion}__volume_mm3"],
                           errors="coerce").dropna() / 1000.0
    axes[1].hist(values, bins=40, alpha=0.65,
                 color=ROLE.get(subregion, series_color(i)),
                 label=subregion.replace("_", "-"))
axes[1].set_xlabel("volume at the index visit (mL)")
axes[1].set_ylabel("transitions")
axes[1].set_title("Subregion volume")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The outcome
#
# 385 transitions have an expert RANO response assessment at the next visit.
# The class balance (about 64% PD) reflects that RANO ratings were made on a
# **selected** set of timepoints, not on every follow-up study - so this is not
# the progression rate of routine surveillance imaging.

# %%
eligible = table[table.next_rano_is_response_assessment == True]  # noqa: E712
print(f"{len(eligible)} eligible transitions, "
      f"{eligible.patient_id.nunique()} patients")
eligible.next_rano_status.value_counts()

# %% [markdown]
# ## Survival is not modellable here
#
# The cohort releases a survival time in weeks with **no vital-status column**,
# so every patient would be treated as having died. The pipeline refuses to fit
# a Cox model on this cohort.

# %%
from gbmplast.modeling.survival import can_fit_survival

print(can_fit_survival(bundle, "overall_survival"))
