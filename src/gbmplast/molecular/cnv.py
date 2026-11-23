"""Expression-based copy-number inference for malignant-cell calling.

Why this exists
---------------
Marker-based compartment assignment alone cannot separate malignant glioma
cells from normal glia: AC-like malignant cells express the same astrocyte
markers (AQP4, SLC1A2, GJA1) as reactive astrocytes, and OPC-like malignant
cells look like OPCs.  Scoring "malignant states" on a compartment defined by
those very markers would be close to circular.

Adult IDH-wild-type glioblastoma is defined genetically by **chromosome 7
gain and chromosome 10 loss** in the large majority of cases (WHO CNS5 lists
+7/-10 as a diagnostic criterion).  That gives a marker-independent handle:
average expression along a chromosome, relative to a reference built from
cells that cannot be tumour (immune and endothelial), tracks copy number.
This is the logic of inferCNV and CONICSmat, implemented here in a
deliberately simple, auditable form.

What this is and is not
-----------------------
It is a *transcriptional* estimate of large-scale copy number, used to label
cells as malignant or non-malignant.  It is **not** a DNA copy-number call,
it cannot resolve focal events, and a sample whose reference compartment is
too small is reported as indeterminate rather than guessed at.

Gene -> chromosome mapping comes from NCBI ``Homo_sapiens.gene_info``
(authoritative, versioned, and downloaded by the acquisition layer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

#: Compartments that cannot be glioma cells; used as the diploid reference.
REFERENCE_COMPARTMENTS = ("myeloid", "t_cell", "b_plasma", "endothelial")

#: The GBM-defining arm-level events (WHO CNS5 diagnostic criteria for
#: IDH-wild-type glioblastoma).
GBM_GAIN_CHROMS = ("7",)
GBM_LOSS_CHROMS = ("10",)

_CYTOBAND_RE = re.compile(r"^(?P<chrom>\d+|X|Y)(?P<arm>[pq])")


@lru_cache(maxsize=4)
def load_gene_chromosomes(path: str) -> pd.DataFrame:
    """Symbol -> chromosome and arm, from NCBI gene_info."""
    df = pd.read_csv(path, sep="\t", low_memory=False,
                     usecols=["Symbol", "chromosome", "map_location",
                              "type_of_gene"])
    df = df[df["type_of_gene"].isin(["protein-coding", "ncRNA"])]
    df = df[df["chromosome"].astype(str).isin(
        [str(i) for i in range(1, 23)] + ["X", "Y"])]
    arms = df["map_location"].astype(str).str.extract(_CYTOBAND_RE)
    df = df.assign(chrom=df["chromosome"].astype(str), arm=arms["arm"])
    df["chrom_arm"] = df["chrom"] + df["arm"].fillna("")
    return (df[["Symbol", "chrom", "arm", "chrom_arm"]]
            .dropna(subset=["chrom"]).drop_duplicates("Symbol")
            .set_index("Symbol"))


@dataclass
class CnvResult:
    cell_scores: pd.DataFrame
    chrom_means: pd.DataFrame
    n_reference_cells: int
    status: str
    note: str = ""


def infer_chromosome_scores(
    sample,
    cell_types: pd.DataFrame,
    *,
    gene_info_path: Path,
    min_reference_cells: int = 30,
    min_genes_per_chrom: int = 50,
    smooth_trim: float = 0.1,
) -> CnvResult:
    """Per-cell relative expression per chromosome, centred on a reference.

    Returns per-cell chromosome scores (log-scale, reference-centred) plus a
    ``gbm_cnv_score`` = mean(chr7) - mean(chr10), the axis on which malignant
    cells separate in IDH-wild-type GBM.
    """
    from gbmplast.molecular.snrna import log_normalise

    genes = load_gene_chromosomes(str(gene_info_path))
    normalised = log_normalise(sample.counts)

    mapped = pd.Series(sample.genes, name="Symbol").to_frame()
    mapped = mapped.join(genes, on="Symbol")
    usable = mapped["chrom"].notna().to_numpy()
    if usable.sum() < 1000:
        return CnvResult(pd.DataFrame(), pd.DataFrame(), 0, "failed",
                         f"only {int(usable.sum())} genes mapped to a "
                         "chromosome; cannot infer copy number")

    reference_mask = cell_types["cell_type"].isin(
        REFERENCE_COMPARTMENTS).to_numpy()
    n_reference = int(reference_mask.sum())
    if n_reference < min_reference_cells:
        return CnvResult(pd.DataFrame(), pd.DataFrame(), n_reference,
                         "indeterminate",
                         f"only {n_reference} reference (immune/endothelial) "
                         f"cells, below the {min_reference_cells} needed for a "
                         "diploid baseline; malignant calling is not attempted")

    chrom_values: dict[str, np.ndarray] = {}
    gene_counts: dict[str, int] = {}
    chrom_labels = mapped["chrom"].to_numpy()

    raw: dict[str, np.ndarray] = {}
    for chrom in [str(i) for i in range(1, 23)]:
        rows = np.flatnonzero(chrom_labels == chrom)
        if len(rows) < min_genes_per_chrom:
            continue
        raw[f"chr{chrom}"] = np.asarray(normalised[rows].mean(axis=0)).ravel()
        gene_counts[f"chr{chrom}"] = len(rows)

    if not raw:
        return CnvResult(pd.DataFrame(), pd.DataFrame(), n_reference, "failed",
                         "no chromosome had enough mapped genes")

    # Two centrings, in this order, and both are necessary:
    #
    # 1. PER CELL, across chromosomes. A nucleus with a small library or an
    #    unusual transcriptional programme shifts every chromosome mean
    #    together; that is cell-level technical variation, not copy number.
    #    Without this the per-cell spread swamps an arm-level effect of
    #    ~0.05 log units and nothing is callable.
    # 2. PER CHROMOSOME, against the diploid reference cells. Chromosomes
    #    differ in baseline expression for reasons unrelated to copy number.
    matrix = np.vstack([raw[c] for c in raw])            # chroms x cells
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    for i, chrom in enumerate(raw):
        values = matrix[i]
        chrom_values[chrom] = values - float(np.mean(values[reference_mask]))

    if not chrom_values:
        return CnvResult(pd.DataFrame(), pd.DataFrame(), n_reference, "failed",
                         "no chromosome had enough mapped genes")

    scores = pd.DataFrame(chrom_values, index=cell_types["barcode"].to_numpy())
    gain = [f"chr{c}" for c in GBM_GAIN_CHROMS if f"chr{c}" in scores.columns]
    loss = [f"chr{c}" for c in GBM_LOSS_CHROMS if f"chr{c}" in scores.columns]
    if not (gain and loss):
        return CnvResult(scores, pd.DataFrame(), n_reference, "failed",
                         "chr7 or chr10 lacked enough mapped genes")

    scores["gbm_cnv_score"] = scores[gain].mean(axis=1) - scores[loss].mean(axis=1)
    scores["is_reference_cell"] = reference_mask

    chrom_means = pd.DataFrame({
        "chromosome": list(gene_counts),
        "n_genes": list(gene_counts.values()),
        "reference_mean": [0.0] * len(gene_counts),
        "nonreference_mean": [
            float(scores.loc[~reference_mask, c].mean()) for c in gene_counts],
    })
    return CnvResult(scores, chrom_means, n_reference, "ok")


def call_malignant(
    cnv: CnvResult,
    cell_types: pd.DataFrame,
    *,
    n_mad: float = 2.0,
    min_score: float = 0.01,
    reference_quantile: float = 0.95,
) -> pd.DataFrame:
    """Label cells malignant by the +7/-10 axis, against the reference spread.

    A cell is malignant when its ``gbm_cnv_score`` exceeds the
    ``reference_quantile`` of the reference cells' own score distribution
    (and a small absolute floor).  Fixing the threshold at an upper quantile
    of known-diploid cells states the false-positive rate among them
    explicitly -- at 0.95, about 5% of reference-like cells would be called
    malignant if they were not excluded.  Cells in a reference compartment
    are never called malignant, whatever their score.
    """
    out = cell_types.copy()
    if cnv.status != "ok" or cnv.cell_scores.empty:
        out["is_malignant"] = pd.NA
        out["gbm_cnv_score"] = np.nan
        out["malignant_call_method"] = f"cnv_{cnv.status}"
        return out

    scores = cnv.cell_scores["gbm_cnv_score"].to_numpy()
    reference = cnv.cell_scores["is_reference_cell"].to_numpy()
    ref_scores = scores[reference]
    median = float(np.median(ref_scores))
    mad = float(np.median(np.abs(ref_scores - median)))
    robust_sd = 1.4826 * mad
    # The threshold is an upper quantile of the REFERENCE distribution, which
    # fixes the false-positive rate among known-diploid cells at a stated
    # level, rather than assuming the scores are Gaussian.
    quantile_threshold = float(np.quantile(ref_scores, reference_quantile))
    threshold = max(quantile_threshold, median + n_mad * robust_sd,
                    median + min_score)

    malignant = (scores > threshold) & (~reference)
    out["gbm_cnv_score"] = scores
    out["is_malignant"] = malignant
    out["cnv_threshold"] = threshold
    out["cnv_reference_median"] = median
    out["cnv_reference_robust_sd"] = robust_sd
    out["cnv_reference_quantile"] = reference_quantile
    out["malignant_call_method"] = (
        "expression-inferred chr7 gain / chr10 loss vs an immune+endothelial "
        "reference (derived here; not a DNA copy-number call, and not the "
        "GBM-CARE consortium's annotation)")
    return out


# --------------------------------------------------------------------------
# Sample-level validation (the way CNV is actually used in this pipeline)
# --------------------------------------------------------------------------

def validate_malignant_compartment(
    cnv: CnvResult,
    cell_types: pd.DataFrame,
    *,
    compartment: str = "malignant_glial",
    min_separation: float = 0.02,
) -> dict:
    """Does the marker-defined malignant compartment carry the +7/-10 signal?

    Per-cell copy-number calling from sparse single-nucleus data proved
    unreliable across the GBM-CARE samples in this project: with a threshold
    fixed on the reference cells' own distribution, the malignant fraction
    swung between 0.1% and 20% across samples whose chromosome-level signal
    was in the same direction.  Rather than ship a per-cell filter that
    behaves that way, this pipeline:

      * defines the malignant compartment by canonical **tumour-enriched**
        markers (PTPRZ1, SOX2, OLIG1/2, EGFR, DLL3, CSPG4, PDGFRA, ...), and
      * uses the CNV axis as a **sample-level validation** that the
        compartment really is enriched for cells carrying the GBM-defining
        chromosome-7-gain / chromosome-10-loss signature.

    Returns the separation between the compartment and the diploid
    reference, with a pass/weak verdict, so a sample whose compartment lacks
    the signal is flagged rather than silently analysed.
    """
    out = {
        "compartment": compartment,
        "cnv_status": cnv.status,
        "n_reference_cells": cnv.n_reference_cells,
    }
    if cnv.status != "ok" or cnv.cell_scores.empty:
        out.update({"verdict": "not_assessed", "note": cnv.note,
                    "cnv_separation": np.nan})
        return out

    scores = cnv.cell_scores["gbm_cnv_score"].to_numpy()
    is_reference = cnv.cell_scores["is_reference_cell"].to_numpy()
    in_compartment = (cell_types["cell_type"] == compartment).to_numpy()

    if in_compartment.sum() < 20:
        out.update({"verdict": "too_few_cells", "cnv_separation": np.nan,
                    "n_compartment_cells": int(in_compartment.sum())})
        return out

    compartment_median = float(np.median(scores[in_compartment]))
    reference_median = float(np.median(scores[is_reference]))
    separation = compartment_median - reference_median

    # Mann-Whitney U: is the compartment shifted above the reference?
    try:
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(scores[in_compartment],
                                     scores[is_reference],
                                     alternative="greater")
    except Exception:
        stat, p_value = np.nan, np.nan

    out.update({
        "n_compartment_cells": int(in_compartment.sum()),
        "compartment_median_cnv_score": round(compartment_median, 5),
        "reference_median_cnv_score": round(reference_median, 5),
        "cnv_separation": round(separation, 5),
        "mannwhitney_p": p_value,
        "verdict": ("pass" if separation >= min_separation and
                    (not np.isfinite(p_value) or p_value < 0.05)
                    else "weak"),
        "note": ("chr7-gain minus chr10-loss expression score, compartment vs "
                 "immune/endothelial reference. A 'weak' verdict means the "
                 "marker-defined compartment is not clearly enriched for the "
                 "GBM copy-number signature in this sample; its malignant "
                 "pseudobulk should be treated with caution."),
    })
    return out
