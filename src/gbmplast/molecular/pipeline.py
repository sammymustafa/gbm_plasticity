"""Module C driver: pseudobulk construction and paired plasticity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters import get_adapter
from gbmplast.config import Config
from gbmplast.molecular import signatures as S
from gbmplast.molecular.cnv import (
    infer_chromosome_scores,
    validate_malignant_compartment,
)
from gbmplast.molecular.paired import (
    baseline_predictors_of_recurrence,
    paired_effects,
    paired_table,
    state_switch_table,
)
from gbmplast.molecular.snrna import (
    assign_cell_types,
    pseudobulk,
    read_umi_rds,
)

GENE_INFO = "reference/Homo_sapiens.gene_info.gz"


# ==========================================================================
# pseudobulk
# ==========================================================================

def run_pseudobulk(config: Config, *, dataset: str = "gbm_care",
                   limit: int | None = None) -> int:
    adapter = get_adapter(dataset, config)
    if not adapter.is_buildable()[0]:
        print(f"  {dataset}: metadata absent; "
              f"run: gbmplast acquire --dataset {dataset}")
        return 0

    specimen_dir = adapter.raw_dir / "specimens"
    files = sorted(specimen_dir.glob("*_umi_counts.RDS.gz")) if specimen_dir.exists() else []
    if not files:
        print(f"  no specimen matrices in {specimen_dir}\n"
              f"  run: gbmplast acquire --dataset {dataset} --specimens "
              "--n-patients 56")
        return 0
    if limit:
        files = files[:limit]

    gene_info = config.paths.raw / GENE_INFO
    use_cnv = gene_info.exists()
    if not use_cnv:
        print(f"  (gene_info not found at {gene_info}; CNV validation of the "
              "malignant compartment will be skipped)")

    samples = adapter.sample_records().set_index("gsm")
    outdir = adapter.processed_dir
    outdir.mkdir(parents=True, exist_ok=True)

    all_profiles: dict[str, pd.Series] = {}
    malignant_profiles: dict[str, pd.Series] = {}
    composition_rows: list[dict] = []
    validation_rows: list[dict] = []

    print(f"  processing {len(files)} specimens "
          f"(one at a time; each dense matrix is discarded after pseudobulk)")
    for i, path in enumerate(files, 1):
        accession = path.name.split("_")[0]
        try:
            sample = read_umi_rds(path)
            cells = assign_cell_types(sample)
        except Exception as exc:
            print(f"   ! {accession}: {type(exc).__name__}: {exc}")
            continue

        meta = samples.loc[accession] if accession in samples.index else None
        malignant_mask = cells["is_malignant_compartment"].to_numpy()
        n_malignant = int(malignant_mask.sum())

        all_profiles[accession] = pseudobulk(sample)
        malignant_profiles[accession] = (
            pseudobulk(sample, cell_mask=malignant_mask)
            if n_malignant >= 50 else pd.Series(np.nan, index=sample.genes))

        row = {
            "accession": accession,
            "local_patient_id": meta["local_patient_id"] if meta is not None else pd.NA,
            "sample_role": meta["sample_role"] if meta is not None else pd.NA,
            "site": meta["site"] if meta is not None else pd.NA,
            "n_cells": sample.n_cells,
            "n_genes": sample.n_genes,
            "n_malignant_compartment": n_malignant,
            "malignant_fraction": round(n_malignant / max(1, sample.n_cells), 4),
            "malignant_pseudobulk_available": n_malignant >= 50,
        }
        for cell_type, fraction in (cells["cell_type"].value_counts(normalize=True)
                                    .round(4).items()):
            row[f"frac_{cell_type}"] = fraction
        composition_rows.append(row)

        if use_cnv:
            cnv = infer_chromosome_scores(sample, cells, gene_info_path=gene_info)
            validation = validate_malignant_compartment(cnv, cells)
            validation["accession"] = accession
            validation["local_patient_id"] = row["local_patient_id"]
            validation["sample_role"] = row["sample_role"]
            validation_rows.append(validation)

        del sample
        if i % 10 == 0 or i == len(files):
            print(f"   {i}/{len(files)} processed")

    if not all_profiles:
        print("  no specimens could be processed")
        return 1

    all_matrix = pd.DataFrame(all_profiles).sort_index()
    malignant_matrix = pd.DataFrame(malignant_profiles).sort_index()
    all_matrix.to_parquet(outdir / "pseudobulk_all_cells.parquet")
    malignant_matrix.to_parquet(outdir / "pseudobulk_malignant.parquet")

    composition = pd.DataFrame(composition_rows)
    composition.to_csv(outdir / "sample_composition.csv", index=False)
    print(f"\n  pseudobulk: {all_matrix.shape[0]} genes x "
          f"{all_matrix.shape[1]} samples -> {outdir}")
    print(f"  malignant pseudobulk available for "
          f"{int(composition['malignant_pseudobulk_available'].sum())} samples")

    if validation_rows:
        validation = pd.DataFrame(validation_rows)
        validation.to_csv(outdir / "cnv_validation.csv", index=False)
        verdicts = validation["verdict"].value_counts().to_dict()
        print(f"  CNV validation of the malignant compartment: {verdicts}")
        passing = validation[validation["verdict"] == "pass"]
        if len(passing):
            print(f"    median chr7-gain/chr10-loss separation "
                  f"(compartment vs reference): "
                  f"{passing['cnv_separation'].median():.4f}")

    print("\n  composition (median fraction across samples):")
    frac_cols = [c for c in composition.columns if c.startswith("frac_")]
    medians = composition[frac_cols].median().sort_values(ascending=False)
    for name, value in medians.items():
        print(f"    {name.replace('frac_', ''):20s} {value:.3f}")
    return 0


# ==========================================================================
# plasticity
# ==========================================================================

def run_plasticity(config: Config, *, dataset: str = "gbm_care") -> int:
    adapter = get_adapter(dataset, config)
    processed = adapter.processed_dir
    malignant_path = processed / "pseudobulk_malignant.parquet"
    all_path = processed / "pseudobulk_all_cells.parquet"
    if not malignant_path.exists():
        print(f"  {malignant_path} not found; run: "
              f"gbmplast molecular pseudobulk --dataset {dataset}")
        return 0

    outdir = config.paths.results / dataset / "plasticity"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = config.paths.figures / dataset / "plasticity"

    malignant = pd.read_parquet(malignant_path)
    all_cells = pd.read_parquet(all_path) if all_path.exists() else None
    composition = pd.read_csv(processed / "sample_composition.csv")
    samples = adapter.sample_records()

    n_bootstrap = int(config.get("reproducibility.n_bootstrap", 2000))
    seed = int(config.get("reproducibility.bootstrap_seed", 777))

    print("== Module C: malignant-state and immune-programme plasticity ==")
    print(f"  malignant pseudobulk: {malignant.shape[0]} genes x "
          f"{malignant.shape[1]} samples")

    # ---- score programmes -------------------------------------------------
    mal_scores, mal_coverage = S.score_all(malignant)
    mal_scores = S.combine_state_groups(mal_scores)
    weights = S.state_proportions(mal_scores)
    mal_scores = pd.concat([mal_scores, weights], axis=1)
    mal_coverage.to_csv(outdir / "signature_coverage.csv", index=False)

    low = mal_coverage[mal_coverage["below_min_coverage"]]
    if len(low):
        print(f"  ! {len(low)} signatures below 40% gene coverage: "
              f"{', '.join(low['signature'])}")

    scores = mal_scores.reset_index().rename(columns={"index": "accession"})
    scores = scores.merge(
        samples[["gsm", "local_patient_id", "sample_role", "site"]],
        left_on="accession", right_on="gsm", how="left").drop(columns=["gsm"])
    scores = scores.merge(
        composition[["accession", "n_cells", "n_malignant_compartment",
                     "malignant_fraction", "malignant_pseudobulk_available"]],
        on="accession", how="left")

    # Immune composition is a property of the WHOLE sample, not of the
    # malignant compartment, so it is scored on the all-cell pseudobulk.
    if all_cells is not None:
        immune_sigs = tuple(s for s in S.IMMUNE)
        env_scores, env_coverage = S.score_all(all_cells, immune_sigs)
        env_scores = env_scores.add_prefix("env_")
        scores = scores.merge(env_scores.reset_index()
                              .rename(columns={"index": "accession"}),
                              on="accession", how="left")
        env_coverage.to_csv(outdir / "signature_coverage_environment.csv",
                            index=False)
        for cell_type in ("myeloid", "t_cell", "oligodendrocyte", "neuron",
                          "endothelial", "astrocyte_normal"):
            col = f"frac_{cell_type}"
            if col in composition.columns:
                scores = scores.merge(composition[["accession", col]],
                                      on="accession", how="left")

    usable = scores[scores["malignant_pseudobulk_available"] == True]  # noqa: E712
    print(f"  samples with a usable malignant pseudobulk: {len(usable)} "
          f"of {len(scores)}")
    scores.to_csv(outdir / "sample_scores.csv", index=False)

    # ---- paired analysis ---------------------------------------------------
    programme_cols = (
        [f"state_{g}" for g in S.STATE_GROUPS]
        + [f"state_weight_{g}" for g in S.STATE_GROUPS]
        + [s.key for s in S.IMMUNE if s.key in usable.columns]
        + [s.key for s in S.OTHER if s.key in usable.columns]
        + [c for c in usable.columns if c.startswith("env_")]
        + [c for c in usable.columns if c.startswith("frac_")]
    )
    programme_cols = [c for c in dict.fromkeys(programme_cols)
                      if c in usable.columns]

    paired = paired_table(usable, value_cols=programme_cols)
    print(f"\n  matched primary/recurrent pairs with usable profiles: "
          f"{len(paired)}")
    if len(paired) < 5:
        print("  too few complete pairs for paired inference; stopping here")
        paired.to_csv(outdir / "paired_table.csv", index=False)
        return 0
    paired.to_csv(outdir / "paired_table.csv", index=False)

    effects = paired_effects(paired, programme_cols, n_bootstrap=n_bootstrap,
                             seed=seed)
    effects.to_csv(outdir / "paired_effects.csv", index=False)

    tested = effects[effects["status"] == "tested"].copy()
    tested = tested.sort_values("wilcoxon_fdr")
    show = ["programme", "n_patients", "mean_primary", "mean_recurrent",
            "mean_delta", "delta_ci_low", "delta_ci_high", "cohens_dz",
            "n_increased", "n_decreased", "wilcoxon_p", "wilcoxon_fdr"]
    print("\n== paired primary -> recurrent change "
          "(malignant pseudobulk; patient-level) ==")
    print(tested[show].head(18).round(4).to_string(index=False))

    significant = tested[tested["wilcoxon_fdr"] < 0.05]
    print(f"\n  programmes with FDR < 0.05: {len(significant)} of {len(tested)}")

    # ---- dominant-state switching -----------------------------------------
    pairs, matrix = state_switch_table(usable)
    pairs.to_csv(outdir / "state_transitions.csv", index=False)
    matrix.to_csv(outdir / "state_transition_matrix.csv")
    print("\n== dominant malignant state: primary -> recurrent ==")
    print(matrix.to_string())
    print(f"  switched dominant state: {int(pairs['switched'].sum())} of "
          f"{len(pairs)} patients")
    print("  NOTE: this describes the dominant signature of the malignant "
          "pseudobulk, NOT lineage conversion of individual cells.")

    # ---- baseline predictors (exploratory) ---------------------------------
    predictors = [c for c in programme_cols if c.startswith("state_")]
    exploratory = baseline_predictors_of_recurrence(
        paired, outcome="state_MES", predictors=predictors)
    exploratory.to_csv(outdir / "baseline_predictors_of_recurrent_MES.csv",
                       index=False)
    print("\n== exploratory: primary-sample programmes vs recurrent MES ==")
    print(exploratory.round(4).to_string(index=False))

    _write_interpretation(outdir, tested, pairs)

    try:
        from gbmplast.report.figures_molecular import plasticity_figures
        written = plasticity_figures(paired, tested, pairs, matrix, usable,
                                     figdir)
        if written:
            print(f"\nfigures -> {figdir} ({len(written)} files)")
    except Exception as exc:
        print(f"  (figures skipped: {exc})")
    return 0


def _write_interpretation(outdir: Path, tested: pd.DataFrame,
                          pairs: pd.DataFrame) -> None:
    """Record, beside the numbers, what they do and do not support."""
    lines = [
        "Interpretation limits for Module C (GBM-CARE)",
        "=" * 60,
        "",
        "What these numbers are:",
        "  * Paired within-patient differences in SIGNATURE SCORES computed on "
        "    sample-level pseudobulk of a marker-defined malignant compartment.",
        "  * Inference is at the patient level (one pair per patient).",
        "",
        "What they are NOT:",
        "  * NOT evidence of cell lineage conversion. A shift in which "
        "    signature scores highest is a change in the composition of the "
        "    malignant compartment, which selection, microenvironmental shift "
        "    or differential sampling of the tumour can all produce.",
        "  * NOT evidence of immune function. HLA-I, B2M, TAP and interferon "
        "    scores are mRNA abundances; they do not demonstrate surface "
        "    presentation, antigen processing or T-cell recognition.",
        "  * NOT a stem-cell assay. The stemness score is a marker average; "
        "    self-renewal requires orthogonal functional experiments.",
        "  * NOT causal. Patients received standard of care between the two "
        "    resections, but treatment is unrecorded in this GEO series, so "
        "    primary-vs-recurrent differences mix therapy effects, natural "
        "    evolution and sampling.",
        "",
        "Known confounders and caveats:",
        "  * The malignant compartment is defined by canonical markers here; "
        "    the consortium's own per-cell annotations are Synapse-gated and "
        "    were not used. CNV validation is reported per sample.",
        "  * Seven contributing sites with two dissociation protocols; site "
        "    is recorded on every sample and should be treated as a batch "
        "    variable in any extension.",
        "  * Immune composition is scored on the all-cell pseudobulk, not the "
        "    malignant compartment, and nuclei-based protocols under-recover "
        "    immune cells relative to whole-cell dissociation.",
        "",
    ]
    if len(tested):
        top = tested.head(5)
        lines.append("Largest paired changes by FDR:")
        for _, r in top.iterrows():
            lines.append(
                f"  {r['programme']}: delta={r['mean_delta']:+.4f} "
                f"[{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}], "
                f"dz={r['cohens_dz']:+.2f}, n={int(r['n_patients'])}, "
                f"FDR={r['wilcoxon_fdr']:.4g}")
    if len(pairs):
        lines += ["", f"Dominant-state switching: "
                      f"{int(pairs['switched'].sum())}/{len(pairs)} patients."]
    (outdir / "INTERPRETATION.txt").write_text("\n".join(lines),
                                               encoding="utf-8")
