# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# %% [markdown]
# # GBM-CARE - exploratory data analysis
#
# [GEO GSE274546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274546):
# single-nucleus RNA-seq from the GBM CARE consortium. **56 patients with a
# matched primary and first-recurrent tumour** - the basis of Module C, and
# the only cohort here with directly observed molecular change.

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
from gbmplast.report import dictionary as D
from gbmplast.report.theme import apply_theme, series_color
from gbmplast.validate import validate_bundle

apply_theme()
config = load_config()
adapter = get_adapter("gbm_care", config)
bundle = adapter.build()
bundle.summary()

# %% [markdown]
# ## Specimen structure
#
# Parsed from the GEO SOFT records: `!Sample_source_name_ch1` carries the
# patient code and the phase, `!Sample_title` the ordinal timepoint.

# %%
samples = adapter.sample_records()
print(samples.sample_role.value_counts().to_string())
print(f"\npatients: {samples.local_patient_id.nunique()}")
print(f"specimens: {len(samples)}")
samples[["gsm", "local_patient_id", "site_code", "sample_role",
         "timepoint_index", "file_name"]].head(8)

# %% [markdown]
# Seven contributing sites and two dissociation protocols. Site is recorded on
# every sample and should be treated as a batch variable in any extension.

# %%
samples.site.value_counts()

# %%
paired = adapter.paired_patients()
print(f"patients with a matched primary + recurrent pair: {len(paired)}")
samples.groupby("local_patient_id").size().value_counts()

# %% [markdown]
# ## What this cohort does NOT release
#
# No collection dates, no clinical covariates, no treatment records, no
# survival, and no consortium cell-state annotations (those are on Synapse
# behind a data-use agreement). The canonical tables reflect that rather than
# filling the gaps.

# %%
for name, note in sorted(bundle.notes.items()):
    print(f"[{name}] {note}\n")

# %%
molecular = bundle.molecular_samples
print("collection day known for:",
      int(molecular.tissue_collection_relative_day.notna().sum()), "specimens")
print("ordinal timepoint known for:",
      int(molecular.timepoint_index.notna().sum()), "specimens")
D.molecular_pairing(bundle).T

# %% [markdown]
# ## Signature definitions
#
# Published gene programmes, reproduced so that the scoring is auditable.
# These are **not** the consortium's per-cell labels.

# %%
pd.DataFrame([{"key": s.key, "kind": s.kind, "n_genes": len(s.genes),
               "source": s.source} for s in S.ALL_SIGNATURES])

# %% [markdown]
# ## One specimen, end to end
#
# Needs `gbmplast acquire --dataset gbm_care --specimens --n-patients 4`.

# %%
specimen_dir = adapter.raw_dir / "specimens"
files = sorted(specimen_dir.glob("*_umi_counts.RDS.gz")) if specimen_dir.exists() else []
print(f"{len(files)} specimen matrices available locally")

sample = cells = cnv = None
gene_info = config.paths.raw / "reference" / "Homo_sapiens.gene_info.gz"

# %%
if files:
    from gbmplast.molecular.snrna import assign_cell_types, read_umi_rds

    sample = read_umi_rds(files[0])
    print(f"{files[0].name}: {sample.n_genes} genes x {sample.n_cells} nuclei")
    cells = assign_cell_types(sample)
    print(cells.cell_type.value_counts(normalize=True).round(3).to_string())
    print("\nassignment method:", cells.attrs["method"])

# %% [markdown]
# Marker-based assignment alone cannot separate AC-like malignant cells from
# reactive astrocytes - both express AQP4, SLC1A2 and GJA1 - so scoring
# "malignant states" on a marker-defined compartment would be close to
# circular. The compartment is therefore **validated** against chromosome 7
# gain / chromosome 10 loss, measured relative to immune and endothelial cells
# that cannot be tumour.

# %%
if sample is not None and gene_info.exists():
    from gbmplast.molecular.cnv import (infer_chromosome_scores,
                                        validate_malignant_compartment)

    cnv = infer_chromosome_scores(sample, cells, gene_info_path=gene_info)
    for key, value in validate_malignant_compartment(cnv, cells).items():
        print(f"  {key}: {value}")
elif sample is not None:
    print("gene_info not downloaded; run gbmplast acquire --dataset gbm_care")

# %%
if cnv is not None and cnv.status == "ok":
    scores = cnv.cell_scores
    in_compartment = (cells.cell_type == "malignant_glial").to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(scores[scores.is_reference_cell].gbm_cnv_score, bins=50,
            alpha=0.75, color=series_color(0),
            label="immune / endothelial reference")
    ax.hist(scores[in_compartment].gbm_cnv_score, bins=50, alpha=0.75,
            color=series_color(1), label="malignant compartment")
    ax.set_xlabel("chr7 gain minus chr10 loss (expression score)")
    ax.set_ylabel("nuclei")
    ax.set_title("Copy-number validation of the malignant compartment")
    ax.legend()
    plt.show()

# %% [markdown]
# ## Composition across the cohort
#
# After `gbmplast molecular pseudobulk --dataset gbm_care`.

# %%
composition_path = adapter.processed_dir / "sample_composition.csv"
if composition_path.exists():
    composition = pd.read_csv(composition_path)
    frac_cols = [c for c in composition.columns if c.startswith("frac_")]
    medians = composition[frac_cols].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(range(len(medians)), medians.to_numpy(), color=series_color(0))
    ax.set_yticks(range(len(medians)))
    ax.set_yticklabels([c.replace("frac_", "") for c in medians.index])
    ax.set_xlabel("median fraction of nuclei")
    ax.set_title("Compartment composition across samples")
    ax.grid(axis="y", visible=False)
    plt.show()
    print(composition[["n_cells", "n_malignant_compartment",
                       "malignant_fraction"]].describe().round(3).to_string())
else:
    print("run: gbmplast molecular pseudobulk --dataset gbm_care")

# %% [markdown]
# Nuclei-based protocols under-recover immune cells relative to whole-cell
# dissociation, so immune fractions here are not a measure of immune
# infiltration in the tumour.

# %%
report = validate_bundle(bundle)
report.to_frame()[["check", "table", "severity", "message"]]
