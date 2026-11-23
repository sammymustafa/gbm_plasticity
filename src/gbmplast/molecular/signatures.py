"""Published gene programmes used to score malignant and immune states.

Provenance matters here.  These are *published* gene sets, reproduced so
that the scoring is transparent and auditable; they are **not** the GBM-CARE
consortium's own per-cell annotations, which live on Synapse behind a DUA.
Any column derived from them is labelled ``derived_here`` so a reader never
mistakes a marker-based score for a consortium label.

Primary sources
---------------
Malignant cell states (MES/AC/OPC/NPC)
    Neftel C, Laffy J, Filbin MG, et al. *An Integrative Model of Cellular
    States, Plasticity, and Genetics for Glioblastoma.* Cell 178:835-849
    (2019).  The four meta-modules, with MES and NPC each split into the two
    published subprogrammes.

Antigen presentation / interferon / immune composition
    Canonical HLA class I, B2M and peptide-loading genes; interferon-alpha
    and interferon-gamma response genes as in the MSigDB Hallmark
    collection (Liberzon A, et al. Cell Syst 1:417-425, 2015).

Stemness
    Classical neural-stem/progenitor and glioma-stem markers as used in the
    glioma literature (e.g. SOX2, OLIG2, NES, PROM1).  Reported as a
    *signature score*, never as evidence of stem-cell function.

Cell-type markers
    Canonical lineage markers for the major compartments of the GBM
    microenvironment.

Scores are means of log-normalised expression over the genes present,
reported alongside the number of genes actually found, because a score over
4 of 25 genes is not the same quantity as a score over 25 of 25.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Signature:
    key: str
    label: str
    genes: tuple[str, ...]
    source: str
    kind: str          # 'malignant_state' | 'immune' | 'stemness' | 'other'
    note: str = ""


# --------------------------------------------------------------------------
# Neftel malignant states
# --------------------------------------------------------------------------

NEFTEL = (
    Signature(
        "MES1", "Mesenchymal-like 1 (hypoxia-independent)",
        ("CHI3L1", "ANXA1", "ANXA2", "CD44", "VIM", "MT2A", "C1S", "NAMPT",
         "EFEMP1", "C1R", "SOD2", "IFITM3", "TIMP1", "SPP1", "A2M", "S100A11",
         "MT1X", "S100A10", "FN1", "LGALS1", "S100A16", "CLIC1", "MGST1",
         "RCAN1", "TAGLN2", "NPC2", "SERPING1", "EMP1", "APOE", "CTSB"),
        "Neftel et al. Cell 2019, meta-module MES1", "malignant_state"),
    Signature(
        "MES2", "Mesenchymal-like 2 (hypoxia-associated)",
        ("HILPDA", "ADM", "DDIT3", "NDRG1", "HERPUD1", "DNAJB9", "TRIB3",
         "ENO2", "AKAP12", "SQSTM1", "MT1X", "ATF3", "NAMPT", "NRN1",
         "SLC2A1", "BNIP3", "LGALS3", "INSIG2", "IGFBP3", "VEGFA", "PGK1",
         "SERPINE1", "PPP1R15A", "EGLN3", "ANGPTL4", "LDHA", "P4HA1"),
        "Neftel et al. Cell 2019, meta-module MES2", "malignant_state",
        note="Hypoxia-driven; overlaps the hypoxia response by design."),
    Signature(
        "AC", "Astrocyte-like",
        ("CST3", "S100B", "SLC1A3", "HEPN1", "HOPX", "MT3", "SPARCL1",
         "MLC1", "GFAP", "FABP7", "BCAN", "PON2", "METTL7B", "SPARC",
         "GATM", "GJA1", "CLU", "ALDOC", "PMP2", "EDNRB", "PTPRZ1", "ATP1A2",
         "AQP4", "SLC1A2", "TTYH1"),
        "Neftel et al. Cell 2019, meta-module AC", "malignant_state"),
    Signature(
        "OPC", "Oligodendrocyte-progenitor-like",
        ("BCAN", "PLP1", "GPR17", "FIBIN", "LHFPL3", "OLIG1", "PSAT1",
         "SCRG1", "OMG", "APOD", "SIRT2", "TNR", "THY1", "PHYHIPL", "SOX2-OT",
         "NKAIN4", "PTPRZ1", "VCAN", "DBI", "CSPG4", "PDGFRA", "SOX10",
         "LIMA1", "CA10", "RNF13"),
        "Neftel et al. Cell 2019, meta-module OPC", "malignant_state"),
    Signature(
        "NPC1", "Neural-progenitor-like 1",
        ("DLL3", "DLL1", "SOX4", "TUBB3", "HES6", "TAGLN3", "NEU4", "MARCKSL1",
         "CD24", "STMN1", "TCF12", "BEX1", "OLIG1", "MAP2", "FXYD6", "PTPRS",
         "MLLT11", "NPPA", "BCAN", "MEST", "ASCL1", "BTG2", "CCND2", "SOX11"),
        "Neftel et al. Cell 2019, meta-module NPC1", "malignant_state"),
    Signature(
        "NPC2", "Neural-progenitor-like 2",
        ("STMN2", "CD24", "RND3", "TUBB3", "MIAT", "DCX", "NSG1", "ELAVL4",
         "MLLT11", "DLX5-AS1", "SOX11", "NREP", "FNBP1L", "TAGLN3", "STMN1",
         "SNAP25", "TMSB15A", "NFIB", "DPYSL3", "PAK3", "NFIX", "IGFBPL1",
         "STMN4", "TUBA1A", "CRABP1"),
        "Neftel et al. Cell 2019, meta-module NPC2", "malignant_state"),
)

#: The four states the brief names, formed by combining the published
#: subprogrammes exactly as Neftel et al. do.
STATE_GROUPS = {
    "MES": ("MES1", "MES2"),
    "NPC": ("NPC1", "NPC2"),
    "AC": ("AC",),
    "OPC": ("OPC",),
}


# --------------------------------------------------------------------------
# Immune / antigen presentation / interferon
# --------------------------------------------------------------------------

IMMUNE = (
    Signature(
        "HLA_I", "HLA class I heavy chains",
        ("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F"),
        "Canonical MHC class I heavy-chain genes", "immune",
        note="Expression only. Says nothing about surface protein or "
             "peptide presentation."),
    Signature(
        "B2M", "Beta-2 microglobulin",
        ("B2M",), "Canonical", "immune",
        note="Single gene; reported for transparency, not as a signature."),
    Signature(
        "TAP_PLC", "Peptide-loading complex (TAP1/TAP2/TAPBP and chaperones)",
        ("TAP1", "TAP2", "TAPBP", "CALR", "CANX", "PDIA3", "ERAP1", "ERAP2"),
        "Canonical antigen-processing machinery", "immune"),
    Signature(
        "HLA_II", "HLA class II",
        ("HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
         "HLA-DQB1", "CD74", "HLA-DMA", "HLA-DMB"),
        "Canonical MHC class II genes", "immune",
        note="In GBM, largely myeloid-derived; interpret with the cell-type "
             "composition beside it."),
    Signature(
        "IFN_ALPHA", "Interferon-alpha response",
        ("ISG15", "IFI6", "IFI27", "IFIT1", "IFIT3", "IFITM1", "IFITM3",
         "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "STAT1",
         "STAT2", "IRF7", "IRF9", "BST2", "XAF1", "SAMD9", "SAMD9L",
         "EIF2AK2", "ADAR", "USP18"),
        "MSigDB Hallmark IFN-alpha response (Liberzon 2015), core members",
        "immune"),
    Signature(
        "IFN_GAMMA", "Interferon-gamma response",
        ("GBP1", "GBP2", "GBP4", "GBP5", "CXCL9", "CXCL10", "CXCL11", "IDO1",
         "STAT1", "IRF1", "SOCS1", "PSMB8", "PSMB9", "TAP1", "CIITA",
         "NLRC5", "ICAM1", "VCAM1", "FGL2", "CD274"),
        "MSigDB Hallmark IFN-gamma response (Liberzon 2015), core members",
        "immune"),
    Signature(
        "T_CELL", "T lymphocytes",
        ("CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC2", "IL7R", "CD8A",
         "CD8B", "CD4", "THEMIS", "SKAP1", "LCK", "ITK"),
        "Canonical T-cell lineage markers", "immune"),
    Signature(
        "T_EXHAUSTION", "T-cell inhibitory receptors",
        ("PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1",
         "CD244", "BTLA"),
        "Canonical inhibitory-receptor set", "immune",
        note="Inhibitory-receptor expression. NOT evidence of exhaustion as "
             "a functional state."),
    Signature(
        "MYELOID", "Myeloid cells (macrophage/microglia)",
        ("CD14", "AIF1", "ITGAM", "CSF1R", "TYROBP", "FCER1G", "C1QA",
         "C1QB", "C1QC", "CD68", "MRC1", "MSR1", "CD163", "LYZ", "SPI1"),
        "Canonical myeloid markers", "immune"),
    Signature(
        "MICROGLIA", "Microglia-specific",
        ("P2RY12", "TMEM119", "CX3CR1", "SALL1", "OLFML3", "SIGLEC8",
         "GPR34", "SELPLG"),
        "Canonical microglial identity genes", "immune"),
    Signature(
        "IMMUNOSUPPRESSIVE", "Immunosuppressive myeloid programme",
        ("IL10", "TGFB1", "TGFB2", "VEGFA", "ARG1", "ARG2", "IDO1", "CD274",
         "PDCD1LG2", "LGALS9", "LGALS3", "CCL2", "CCL22", "SPP1", "TREM2",
         "MARCO", "CSF1"),
        "Commonly used immunosuppressive/M2-associated genes", "immune",
        note="A transcriptional programme, not a demonstration of "
             "suppressive function."),
)


# --------------------------------------------------------------------------
# Stemness and proliferation
# --------------------------------------------------------------------------

OTHER = (
    Signature(
        "STEMNESS", "Neural/glioma stem-progenitor programme",
        ("SOX2", "OLIG2", "NES", "PROM1", "FUT4", "L1CAM", "MSI1", "NOTCH1",
         "HES1", "HES5", "ID1", "ID3", "ASCL1", "POU3F2", "SALL2", "SOX9",
         "CD44", "FABP7", "PTPRZ1", "BMI1"),
        "Canonical neural-stem/glioma-stem markers used in the glioma "
        "literature", "stemness",
        note="A signature score. Stem-cell function requires orthogonal "
             "assays (serial transplantation, clonogenicity) that no "
             "expression dataset can substitute for."),
    Signature(
        "CYCLE_G1S", "Cell cycle G1/S",
        ("MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "PCNA", "RRM2", "TYMS",
         "GINS2", "CDC6", "CDC45", "UNG", "SLBP", "UHRF1", "CLSPN", "E2F1"),
        "Standard G1/S genes (Tirosh et al. Science 2016)", "other"),
    Signature(
        "CYCLE_G2M", "Cell cycle G2/M",
        ("CCNB1", "CCNB2", "CDK1", "TOP2A", "MKI67", "AURKA", "AURKB",
         "BUB1", "PLK1", "TPX2", "UBE2C", "BIRC5", "CENPF", "NUSAP1",
         "TACC3", "HMGB2"),
        "Standard G2/M genes (Tirosh et al. Science 2016)", "other"),
    Signature(
        "HYPOXIA", "Hypoxia response",
        ("VEGFA", "SLC2A1", "LDHA", "PGK1", "CA9", "ADM", "NDRG1", "BNIP3",
         "P4HA1", "EGLN3", "ANKRD37", "PDK1", "ALDOA", "ENO1"),
        "Canonical HIF-target genes", "other"),
)

ALL_SIGNATURES: tuple[Signature, ...] = NEFTEL + IMMUNE + OTHER
BY_KEY: dict[str, Signature] = {s.key: s for s in ALL_SIGNATURES}


# --------------------------------------------------------------------------
# Cell-type markers (for compartment assignment, not scoring)
# --------------------------------------------------------------------------

CELL_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "malignant_glial": ("PTPRZ1", "SOX2", "OLIG1", "OLIG2", "EGFR", "GFAP",
                        "BCAN", "FABP7", "SOX9", "NES", "DLL3", "CSPG4",
                        "PDGFRA", "VIM", "CHI3L1", "ATP1A2", "TNC"),
    "oligodendrocyte": ("MBP", "PLP1", "MOG", "MAG", "CNP", "MOBP", "CLDN11",
                        "ST18", "CTNNA3", "MYRF"),
    "myeloid": ("CD14", "AIF1", "CSF1R", "TYROBP", "FCER1G", "C1QA", "C1QB",
                "C1QC", "CD68", "P2RY12", "CX3CR1", "ITGAM", "SPI1", "DOCK8"),
    "t_cell": ("CD3D", "CD3E", "CD2", "TRAC", "SKAP1", "THEMIS", "IL7R",
               "CD8A", "ITK", "CD247"),
    "b_plasma": ("CD79A", "MS4A1", "IGHM", "IGKC", "MZB1", "JCHAIN"),
    "endothelial": ("CLDN5", "FLT1", "PECAM1", "VWF", "ABCB1", "EGFL7",
                    "ERG", "CDH5"),
    "pericyte_mural": ("PDGFRB", "RGS5", "ACTA2", "NOTCH3", "DCN", "COL1A2",
                       "MYL9", "TAGLN"),
    "neuron": ("SYT1", "SNAP25", "RBFOX3", "NEFL", "NRGN", "GAD1", "GAD2",
               "SLC17A7", "SYN1", "MEG3"),
    "astrocyte_normal": ("AQP4", "SLC1A2", "GJA1", "ALDH1L1", "MLC1",
                         "SLC1A3", "ATP1A2"),
}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_signature(
    expression: pd.DataFrame,
    signature: Signature | str,
    *,
    gene_axis: str = "index",
) -> tuple[pd.Series, dict]:
    """Mean log-normalised expression over the signature's present genes.

    ``expression`` is genes x samples (``gene_axis='index'``) or
    samples x genes.  Returns ``(score per sample, coverage info)``.  The
    coverage dict is not optional decoration: a score computed over a third
    of a signature is a different quantity and must be reported as such.
    """
    sig = signature if isinstance(signature, Signature) else BY_KEY[signature]
    frame = expression if gene_axis == "index" else expression.T
    present = [g for g in sig.genes if g in frame.index]
    coverage = {
        "signature": sig.key,
        "n_genes_in_signature": len(sig.genes),
        "n_genes_found": len(present),
        "coverage": round(len(present) / len(sig.genes), 3) if sig.genes else 0.0,
        "missing_genes": ";".join(g for g in sig.genes if g not in frame.index),
    }
    if not present:
        return pd.Series(np.nan, index=frame.columns, name=sig.key), coverage
    return frame.loc[present].mean(axis=0).rename(sig.key), coverage


def score_all(
    expression: pd.DataFrame,
    signatures: tuple[Signature, ...] = ALL_SIGNATURES,
    *,
    gene_axis: str = "index",
    min_coverage: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every signature; returns ``(scores, coverage_table)``.

    Signatures below ``min_coverage`` are scored but flagged, so a reader
    can drop them rather than silently trusting a quarter-measured score.
    """
    scores, coverages = {}, []
    for sig in signatures:
        series, coverage = score_signature(expression, sig, gene_axis=gene_axis)
        coverage["below_min_coverage"] = coverage["coverage"] < min_coverage
        coverage["label"] = sig.label
        coverage["source"] = sig.source
        coverage["kind"] = sig.kind
        coverage["note"] = sig.note
        scores[sig.key] = series
        coverages.append(coverage)
    return pd.DataFrame(scores), pd.DataFrame(coverages)


def combine_state_groups(scores: pd.DataFrame) -> pd.DataFrame:
    """Combine published subprogrammes into the four named states."""
    out = scores.copy()
    for group, members in STATE_GROUPS.items():
        present = [m for m in members if m in scores.columns]
        if present:
            out[f"state_{group}"] = scores[present].mean(axis=1)
    return out


def state_proportions(state_scores: pd.DataFrame) -> pd.DataFrame:
    """Softmax-free relative weighting of the four malignant states.

    Each sample's four state scores are shifted to be non-negative and
    normalised to sum to 1.  This is a *relative weighting of signature
    scores*, not a cell-type deconvolution, and the column names say so.
    """
    cols = [f"state_{g}" for g in STATE_GROUPS if f"state_{g}" in state_scores]
    if not cols:
        return pd.DataFrame(index=state_scores.index)
    block = state_scores[cols].copy()
    shifted = block.sub(block.min(axis=1), axis=0)
    total = shifted.sum(axis=1)
    out = shifted.div(total.where(total > 0, np.nan), axis=0)
    out.columns = [f"state_weight_{c.replace('state_', '')}" for c in cols]
    out["dominant_state"] = block.idxmax(axis=1).str.replace("state_", "",
                                                             regex=False)
    return out
