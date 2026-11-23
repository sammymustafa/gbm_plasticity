"""One specimen, one process.

Why a subprocess per sample
---------------------------
The pure-Python ``rdata`` reader materialises the matrix as Python objects
before handing back an array: for a 33,500 x 6,400 matrix that is ~213
million Python floats, about **7 GB of peak RSS for a single sample**.  The
allocator does not return that to the operating system, so a long sequential
loop creeps upward and is eventually killed -- which is exactly what happened
on the first full run (OOM at 14 GB after 90 of 115 samples).

Running each sample in its own short-lived process makes the reclamation
unconditional: when the process exits, the memory is gone.  Each worker also
writes its own cache file, so an interrupted run resumes instead of starting
over.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def cache_paths(cache_dir: Path, accession: str) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return (cache_dir / f"{accession}.parquet",
            cache_dir / f"{accession}.json")


def is_cached(cache_dir: Path, accession: str) -> bool:
    profile_path, meta_path = cache_paths(cache_dir, accession)
    return profile_path.exists() and meta_path.exists()


def process_specimen(
    path: str | Path,
    cache_dir: str | Path,
    gene_info_path: str | Path | None = None,
    min_malignant_cells: int = 50,
) -> dict:
    """Parse one specimen, write its cache, return a small summary.

    Returns only metadata; the profiles go to disk, so nothing large is
    pickled back to the parent.
    """
    from gbmplast.molecular.snrna import (
        assign_cell_types,
        pseudobulk,
        read_umi_rds,
    )

    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    accession = path.name.split("_")[0]
    profile_path, meta_path = cache_paths(cache_dir, accession)

    try:
        sample = read_umi_rds(path)
        cells = assign_cell_types(sample)
    except Exception as exc:
        return {"accession": accession, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}

    malignant_mask = cells["is_malignant_compartment"].to_numpy()
    n_malignant = int(malignant_mask.sum())

    profiles = pd.DataFrame({
        "all_cells": pseudobulk(sample),
        "malignant": (pseudobulk(sample, cell_mask=malignant_mask)
                      if n_malignant >= min_malignant_cells
                      else pd.Series(np.nan, index=sample.genes)),
    })
    profiles.index.name = "gene"
    profiles.to_parquet(profile_path)

    meta = {
        "accession": accession,
        "status": "ok",
        "source_file": path.name,
        "n_cells": int(sample.n_cells),
        "n_genes": int(sample.n_genes),
        "n_malignant_compartment": n_malignant,
        "malignant_fraction": round(n_malignant / max(1, sample.n_cells), 4),
        "malignant_pseudobulk_available": bool(n_malignant >= min_malignant_cells),
        "composition": {str(k): float(v) for k, v in
                        cells["cell_type"].value_counts(normalize=True)
                        .round(4).items()},
        "counts_by_type": {str(k): int(v) for k, v in
                           cells["cell_type"].value_counts().items()},
        "marker_coverage": {str(k): round(float(v), 3) for k, v in
                            cells.attrs.get("marker_coverage", {}).items()},
        "assignment_method": cells.attrs.get("method", ""),
    }

    if gene_info_path and Path(gene_info_path).exists():
        try:
            from gbmplast.molecular.cnv import (
                infer_chromosome_scores,
                validate_malignant_compartment,
            )

            cnv = infer_chromosome_scores(sample, cells,
                                          gene_info_path=Path(gene_info_path))
            validation = validate_malignant_compartment(cnv, cells)
            meta["cnv_validation"] = {
                k: (float(v) if isinstance(v, (int, float, np.floating))
                    and not isinstance(v, bool) else v)
                for k, v in validation.items()}
        except Exception as exc:
            meta["cnv_validation"] = {"verdict": "not_assessed",
                                      "note": f"{type(exc).__name__}: {exc}"}

    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> int:
    """Entry point when this module is run as a one-shot worker."""
    import argparse

    parser = argparse.ArgumentParser(description="process one GBM-CARE specimen")
    parser.add_argument("path")
    parser.add_argument("cache_dir")
    parser.add_argument("--gene-info", default=None)
    parser.add_argument("--min-malignant-cells", type=int, default=50)
    args = parser.parse_args()
    result = process_specimen(args.path, args.cache_dir, args.gene_info,
                              args.min_malignant_cells)
    print(json.dumps({"accession": result.get("accession"),
                      "status": result.get("status"),
                      "error": result.get("error", "")}))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
