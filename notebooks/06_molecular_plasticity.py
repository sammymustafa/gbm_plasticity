# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # Module C - molecular tumour-cell plasticity
#
# Paired primary and recurrent snRNA-seq from GBM-CARE: **directly observed
# molecular change**, in contrast with the MRI surrogates of Modules A and B.
#
# ```
# gbmplast molecular pseudobulk --dataset gbm_care
# gbmplast molecular plasticity --dataset gbm_care
# ```

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))

from gbmplast.adapters import get_adapter
from gbmplast.config import load_config
from gbmplast.molecular import signatures as S
from gbmplast.report.theme import DIVERGING, apply_theme, series_color

apply_theme()
config = load_config()
adapter = get_adapter("gbm_care", config)
plasticity = config.paths.results / "gbm_care" / "plasticity"
processed = adapter.processed_dir


def load(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


# %% [markdown]
# ## Inference is at the patient level
#
# Each sample's thousands of nuclei collapse to one pseudobulk profile before
# any test. 56 patients remain 56 observations; they do not become hundreds of
# thousands of correlated pseudo-replicates.

# %%
composition = load(processed, "sample_composition")
if composition is not None:
    print(f"samples processed: {len(composition)}")
    print(f"with a usable malignant pseudobulk: "
          f"{int(composition.malignant_pseudobulk_available.sum())}")
    print(composition[["n_cells", "n_malignant_compartment",
                       "malignant_fraction"]].describe().round(3).to_string())

# %% [markdown]
# ## Validating the malignant compartment
#
# The compartment is defined by canonical tumour markers, then checked against
# the GBM-defining chromosome-7-gain / chromosome-10-loss expression signature
# relative to immune and endothelial cells.
#
# Per-**cell** copy-number calling was attempted and proved unreliable on this
# sparse data, so it is used as a per-**sample** validation instead. That
# choice is recorded rather than hidden.

# %%
validation = load(processed, "cnv_validation")
if validation is not None:
    print(validation.verdict.value_counts().to_string())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    values = pd.to_numeric(validation.cnv_separation, errors="coerce").dropna()
    ax.hist(values, bins=30, color=series_color(0), edgecolor="#fcfcfb")
    ax.axvline(0.02, color="#b9b8b1", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xlabel("chr7 gain minus chr10 loss: compartment vs reference")
    ax.set_ylabel("samples")
    ax.set_title("Copy-number validation of the malignant compartment")
    plt.show()

# %% [markdown]
# ## Signature coverage
#
# A score over a third of a signature is a different quantity from a score over
# all of it, so coverage is reported rather than assumed.

# %%
coverage = load(plasticity, "signature_coverage")
if coverage is not None:
    print(coverage[["signature", "n_genes_in_signature", "n_genes_found",
                    "coverage", "below_min_coverage"]].to_string(index=False))

# %% [markdown]
# ## Paired within-patient change
#
# Primary to first recurrence, in the malignant pseudobulk.

# %%
effects = load(plasticity, "paired_effects")
if effects is not None:
    tested = effects[effects.status == "tested"].sort_values("wilcoxon_fdr")
    keep = ["programme", "n_patients", "mean_delta", "delta_ci_low",
            "delta_ci_high", "cohens_dz", "n_increased", "n_decreased",
            "wilcoxon_p", "wilcoxon_fdr"]
    print(tested[keep].head(20).round(4).to_string(index=False))
    print(f"\nFDR < 0.05: {int((tested.wilcoxon_fdr < 0.05).sum())} of "
          f"{len(tested)} programmes")

# %% [markdown]
# ## Dominant-state switching
#
# **This is not lineage conversion.** A change in which signature scores
# highest is a change in the composition of the malignant compartment;
# selection, a shift in the microenvironment, or sampling a different part of
# the tumour can each produce it.

# %%
transitions = load(plasticity, "state_transitions")
if transitions is not None:
    print(f"switched dominant state: {int(transitions.switched.sum())} of "
          f"{len(transitions)} patients")
    print()
    print(pd.crosstab(transitions.primary_state,
                      transitions.recurrent_state).to_string())

# %% [markdown]
# ## Paired trajectories for individual programmes

# %%
paired = load(plasticity, "paired_table")
if paired is not None:
    programmes = [p for p in ("state_MES", "state_AC", "state_OPC",
                              "state_NPC", "HLA_I", "IFN_GAMMA")
                  if f"{p}__primary" in paired.columns]
    fig, axes = plt.subplots(1, len(programmes),
                             figsize=(2.3 * len(programmes), 4.2),
                             squeeze=False, sharey=False)
    for k, programme in enumerate(programmes):
        ax = axes[0][k]
        a = pd.to_numeric(paired[f"{programme}__primary"], errors="coerce")
        b = pd.to_numeric(paired[f"{programme}__recurrent"], errors="coerce")
        ok = a.notna() & b.notna()
        for x0, x1 in zip(a[ok], b[ok]):
            ax.plot([0, 1], [x0, x1],
                    color=DIVERGING[2] if x1 > x0 else DIVERGING[0],
                    alpha=0.5, linewidth=1.1)
        ax.plot([0, 1], [a[ok].mean(), b[ok].mean()], color="#0b0b0b",
                linewidth=2.4, marker="o", markersize=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prim", "rec"], fontsize=8)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(programme.replace("state_", ""), fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Paired programme scores, primary to recurrent", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()

# %% [markdown]
# ## What these numbers do not support

# %%
interpretation = plasticity / "INTERPRETATION.txt"
if interpretation.exists():
    print(interpretation.read_text())

# %% [markdown]
# ## And what cannot be done with them
#
# These patients have **no imaging**. The GBM-CARE ids are namespaced
# `GBM-CARE::`, so any attempt to join them to a LUMIERE, UPENN-GBM or
# UCSD-PTGBM patient returns nothing. That is deliberate: a molecular
# observation here and an imaging observation there are about different
# people, and the pipeline will not let them be combined.

# %%
from gbmplast.adapters import get_adapter as _get

care = _get("gbm_care", config).build().patients.patient_id
try:
    lumiere = _get("lumiere", config).build().patients.patient_id
    overlap = set(care) & set(lumiere)
    print(f"GBM-CARE patients: {len(care)}")
    print(f"LUMIERE patients: {len(lumiere)}")
    print(f"overlap: {len(overlap)}")
except FileNotFoundError:
    print("LUMIERE not downloaded")
