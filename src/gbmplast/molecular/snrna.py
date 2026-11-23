"""Single-nucleus RNA-seq handling for GBM-CARE.

GEO releases one dense gene x nucleus UMI matrix per sample as a gzipped R
``.RDS``.  Each parses to roughly 33,500 genes by 2,000-8,000 nuclei, which
is about 1.4 GB in memory as float64, so this module:

* parses one sample at a time with the pure-Python ``rdata`` reader (no R
  dependency),
* converts to a sparse integer matrix immediately,
* assigns each nucleus to a compartment with canonical markers,
* aggregates to pseudobulk -- both all-cell and malignant-restricted -- and
  discards the dense array before moving on.

Pseudobulk, not per-cell averaging, is what the statistics downstream use:
a sample's nuclei are not independent observations, and treating them as
such would turn 56 patients into hundreds of thousands of fake replicates.
"""

from __future__ import annotations

import gzip
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.molecular.signatures import CELL_TYPE_MARKERS


@dataclass
class SampleMatrix:
    """A parsed sample: sparse counts plus gene and barcode labels."""

    accession: str
    genes: np.ndarray
    barcodes: np.ndarray
    counts: object                      # scipy.sparse.csr_matrix (genes x cells)
    source_file: str = ""

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.counts.shape[1])


def read_umi_rds(path: Path, *, min_counts_per_cell: int = 200,
                 min_cells_per_gene: int = 3) -> SampleMatrix:
    """Parse a GEO ``*_umi_counts.RDS.gz`` into a sparse matrix.

    Light QC only -- the cohort has already produced these matrices with
    CellRanger, and aggressive filtering here would silently change what the
    pseudobulk represents.
    """
    import rdata
    from scipy import sparse

    path = Path(path)
    with warnings.catch_warnings():
        # The reader warns about the .gz extension on an RDS; harmless.
        warnings.simplefilter("ignore", UserWarning)
        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)

    array, genes, barcodes = _unpack(converted)
    dense = np.asarray(array)
    if dense.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D matrix, got {dense.shape}")

    # Orient as genes x cells using the label lengths.
    if dense.shape[0] != len(genes) and dense.shape[1] == len(genes):
        dense = dense.T
        genes, barcodes = genes, barcodes

    counts = sparse.csr_matrix(np.rint(dense).astype(np.int32))
    del dense

    genes = np.asarray([str(g) for g in genes])
    barcodes = np.asarray([str(b) for b in barcodes])

    # Light QC
    per_cell = np.asarray(counts.sum(axis=0)).ravel()
    keep_cells = per_cell >= min_counts_per_cell
    per_gene = np.asarray((counts > 0).sum(axis=1)).ravel()
    keep_genes = per_gene >= min_cells_per_gene
    counts = counts[keep_genes][:, keep_cells]

    return SampleMatrix(
        accession=path.name.split("_")[0],
        genes=genes[keep_genes],
        barcodes=barcodes[keep_cells],
        counts=counts.tocsr(),
        source_file=str(path),
    )


def _unpack(converted) -> tuple[np.ndarray, list, list]:
    """Pull (values, row labels, column labels) out of whatever rdata returns."""
    # xarray DataArray: coordinates carry the dimnames.
    if hasattr(converted, "dims") and hasattr(converted, "coords"):
        coords = [converted.coords[d].values for d in converted.dims]
        return converted.values, list(coords[0]), list(coords[1])
    if isinstance(converted, dict):
        # A one-entry dict keyed by the R object's name.
        if len(converted) == 1:
            return _unpack(next(iter(converted.values())))
    if isinstance(converted, pd.DataFrame):
        return converted.to_numpy(), list(converted.index), list(converted.columns)
    raise TypeError(
        f"unsupported RDS payload {type(converted)}; expected a labelled "
        "matrix. If the cohort changed its release format, extend _unpack "
        "rather than guessing an orientation."
    )


# --------------------------------------------------------------------------
# normalisation and cell typing
# --------------------------------------------------------------------------

def log_normalise(counts, *, scale: float = 1e4):
    """CP10K + log1p, the standard single-cell normalisation."""
    from scipy import sparse

    totals = np.asarray(counts.sum(axis=0)).ravel()
    totals[totals == 0] = 1.0
    scaled = counts.multiply(sparse.csr_matrix(scale / totals))
    return scaled.log1p().tocsr()


def assign_cell_types(
    sample: SampleMatrix,
    *,
    markers: dict[str, tuple[str, ...]] | None = None,
    min_margin: float = 0.05,
) -> pd.DataFrame:
    """Score each nucleus against canonical compartment markers.

    This is a **marker-based assignment derived here**, not the GBM-CARE
    consortium's published per-cell annotation (which is Synapse-gated).
    Every downstream column built from it carries that label.

    A nucleus whose best and second-best compartment scores differ by less
    than ``min_margin`` is left ``ambiguous`` rather than forced into a
    class.
    """
    markers = markers or CELL_TYPE_MARKERS
    normalised = log_normalise(sample.counts)
    gene_index = {g: i for i, g in enumerate(sample.genes)}

    scores = {}
    coverage = {}
    for cell_type, genes in markers.items():
        rows = [gene_index[g] for g in genes if g in gene_index]
        coverage[cell_type] = len(rows) / len(genes) if genes else 0.0
        if not rows:
            scores[cell_type] = np.zeros(sample.n_cells)
            continue
        block = normalised[rows]
        scores[cell_type] = np.asarray(block.mean(axis=0)).ravel()

    table = pd.DataFrame(scores, index=sample.barcodes)
    # z-score each compartment across nuclei so that compartments with
    # intrinsically higher expression do not win by construction.
    z = (table - table.mean(axis=0)) / table.std(axis=0).replace(0, np.nan)
    z = z.fillna(0.0)

    ordered = np.sort(z.to_numpy(), axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    assignment = z.idxmax(axis=1)
    assignment[margin < min_margin] = "ambiguous"

    out = pd.DataFrame({
        "barcode": sample.barcodes,
        "cell_type": assignment.to_numpy(),
        "assignment_margin": margin,
        "n_counts": np.asarray(sample.counts.sum(axis=0)).ravel(),
        "n_genes": np.asarray((sample.counts > 0).sum(axis=0)).ravel(),
    })
    out["is_malignant_compartment"] = out["cell_type"] == "malignant_glial"
    out.attrs["marker_coverage"] = coverage
    out.attrs["method"] = ("marker-based z-scored assignment derived in this "
                           "pipeline; NOT the GBM-CARE consortium annotation")
    return out


# --------------------------------------------------------------------------
# pseudobulk
# --------------------------------------------------------------------------

def pseudobulk(
    sample: SampleMatrix,
    *,
    cell_mask: np.ndarray | None = None,
    scale: float = 1e6,
) -> pd.Series:
    """Sum counts over the selected nuclei, then CPM + log1p.

    Summing raw counts before normalising (rather than averaging per-cell
    normalised values) is what makes this a pseudobulk profile.
    """
    counts = sample.counts if cell_mask is None else sample.counts[:, cell_mask]
    if counts.shape[1] == 0:
        return pd.Series(np.nan, index=sample.genes)
    summed = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    total = summed.sum()
    if total <= 0:
        return pd.Series(np.nan, index=sample.genes)
    return pd.Series(np.log1p(summed / total * scale), index=sample.genes)


@dataclass
class SampleSummary:
    accession: str
    n_cells_total: int
    n_cells_kept: dict = field(default_factory=dict)
    composition: dict = field(default_factory=dict)
    marker_coverage: dict = field(default_factory=dict)


def process_sample(
    path: Path,
    *,
    min_malignant_cells: int = 50,
) -> tuple[pd.Series, pd.Series, SampleSummary, pd.DataFrame]:
    """Parse one sample and return its pseudobulk profiles and composition.

    Returns ``(all_cell_pseudobulk, malignant_pseudobulk, summary,
    cell_table)``.  The malignant profile is NaN when the sample yields
    fewer than ``min_malignant_cells`` nuclei in the malignant compartment,
    rather than being computed from a handful of cells.
    """
    sample = read_umi_rds(path)
    cells = assign_cell_types(sample)

    composition = (cells["cell_type"].value_counts(normalize=True)
                   .round(4).to_dict())
    counts_by_type = cells["cell_type"].value_counts().to_dict()

    all_cell = pseudobulk(sample)
    malignant_mask = cells["is_malignant_compartment"].to_numpy()
    n_malignant = int(malignant_mask.sum())
    malignant = (pseudobulk(sample, cell_mask=malignant_mask)
                 if n_malignant >= min_malignant_cells
                 else pd.Series(np.nan, index=sample.genes))

    summary = SampleSummary(
        accession=sample.accession,
        n_cells_total=sample.n_cells,
        n_cells_kept={"malignant": n_malignant, **counts_by_type},
        composition=composition,
        marker_coverage=cells.attrs.get("marker_coverage", {}),
    )
    return all_cell, malignant, summary, cells
