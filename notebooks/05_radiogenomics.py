# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # Module B - radiogenomics and pseudoprogression
#
# Two independent analyses that share code but never patients:
#
# 1. **UPENN-GBM** - MGMT promoter methylation from baseline pre-operative MRI.
# 2. **UCSD-PTGBM** - tumour vs treatment-related change at a post-treatment
#    timepoint.
#
# Run the CLI first:
#
# ```
# gbmplast radiogenomics --dataset upenn_gbm --target mgmt
# gbmplast radiogenomics --dataset ucsd_ptgbm
# ```

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))

from gbmplast.config import load_config
from gbmplast.report.theme import apply_theme, series_color

apply_theme()
config = load_config()
results = config.paths.results


def load(dataset: str, analysis: str, name: str) -> pd.DataFrame | None:
    path = results / dataset / analysis / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


# %% [markdown]
# ## UPENN-GBM: MGMT from pre-operative MRI
#
# The estimand is declared before any model is fitted, and travels with the
# results.

# %%
estimand = load("upenn_gbm", "radiogenomics_mgmt", "estimand")
if estimand is not None:
    for _, row in estimand.iterrows():
        print(f"{row['field']:28s} {row['value']}")

# %%
load("upenn_gbm", "radiogenomics_mgmt", "cohort_flow")

# %% [markdown]
# ### Performance
#
# All three feature blocks sit at chance. Under patient-grouped splitting with
# feature selection inside the fold, pre-operative radiomics do not predict
# MGMT status in this cohort. Published claims of high AUC for this task
# frequently come from row-level splits or selection performed before
# splitting; neither is done here.

# %%
metrics = load("upenn_gbm", "radiogenomics_mgmt", "model_metrics")
if metrics is not None:
    keep = [c for c in ("model", "n_rows", "n_patients", "prevalence",
                        "auroc", "auprc", "brier", "calibration_slope")
            if c in metrics.columns]
    display = metrics[keep].round(4)
    print(display.to_string(index=False))

# %%
ci = load("upenn_gbm", "radiogenomics_mgmt", "model_metric_ci")
if ci is not None:
    auroc = ci[ci.metric == "auroc"]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(auroc) + 1.4))
    y = np.arange(len(auroc))
    ax.axvline(0.5, color="#b9b8b1", linestyle=(0, (4, 3)), linewidth=1.2)
    for i, (_, row) in enumerate(auroc.iterrows()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=series_color(0),
                linewidth=2)
        ax.scatter([row.estimate], [i], s=42, color=series_color(0),
                   edgecolor="white", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("__", " + ") for m in auroc.model],
                       fontsize=8)
    ax.set_xlabel("AUROC (patient bootstrap 95% CI)")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("MGMT from baseline MRI - every block includes 0.5")
    ax.grid(axis="y", visible=False)
    plt.show()

# %% [markdown]
# ## UCSD-PTGBM: tumour vs treatment-related change
#
# The cohort's distinguishing feature is that it records **how** each negative
# judgement was made.

# %%
load("ucsd_ptgbm", "radiogenomics_tumor_vs_treatment", "label_adjudication")

# %% [markdown]
# The released tables do not flag which tumour-positive timepoints are
# biopsy-proven, so a histopathology-restricted comparison is possible on the
# negative class only. The sensitivity analysis says so explicitly rather than
# producing a number that would imply otherwise.

# %%
load("ucsd_ptgbm", "radiogenomics_tumor_vs_treatment",
     "adjudication_sensitivity")

# %%
metrics = load("ucsd_ptgbm", "radiogenomics_tumor_vs_treatment",
               "model_metrics")
if metrics is not None:
    keep = [c for c in ("model", "n_rows", "n_patients", "prevalence",
                        "auroc", "auprc", "brier", "calibration_slope")
            if c in metrics.columns]
    print(metrics[keep].round(4).to_string(index=False))

# %% [markdown]
# Note the prevalence beside AUPRC: the positive class is 79%, so an AUPRC of
# 0.86 is only modestly above the floor. The cohort was deliberately enriched
# for pathology-proven cases, so this is **not** the prevalence of tumour in
# routine post-treatment imaging.

# %%
importance = None
directory = results / "ucsd_ptgbm" / "radiogenomics_tumor_vs_treatment"
for path in directory.glob("permutation_importance__*.csv"):
    importance = pd.read_csv(path)
if importance is not None:
    print(importance.round(4).to_string(index=False))
    print("\nOnly features with sign_stable = True held a consistent "
          "direction across folds.")

# %% [markdown]
# ## Why the imaging arm is missing here
#
# UCSD-PTGBM distributes its 33 GB NIfTI bundle through IBM Aspera, which is
# not usable from a headless environment. The join is implemented: dropping an
# `imaging_features.csv` keyed by `local_visit_id` into the cohort's raw
# directory activates an imaging block with no change to model code.

# %%
from gbmplast.adapters import get_adapter

adapter = get_adapter("ucsd_ptgbm", config)
print("expected imaging feature table:",
      adapter.raw_dir / "imaging_features.csv")
print("present:", (adapter.raw_dir / "imaging_features.csv").exists())
