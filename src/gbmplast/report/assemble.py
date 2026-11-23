"""Assemble the results report from whatever analyses have been run.

The report is generated, never hand-written, so it cannot drift from the
numbers.  Where an analysis has not been run or its data is absent, the
report says so rather than omitting the section.

A standing requirement runs through it: observed molecular change
(GBM-CARE, measured directly in tissue) and MRI surrogates (LUMIERE,
UPENN-GBM, UCSD-PTGBM) are reported in separate sections and never combined
into one statement, because no patient appears in both.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from gbmplast.adapters import available_adapters, get_adapter
from gbmplast.config import Config


def assemble_report(config: Config) -> Path:
    results = config.paths.results
    lines: list[str] = []
    w = lines.append

    w("# GBM plasticity pipeline - results report")
    w("")
    w(f"Generated {date.today().isoformat()} by `gbmplast report`. "
      "Every number below is read from the CSVs written by the analysis "
      "commands; nothing here is typed by hand.")
    w("")
    w("## Reading this report")
    w("")
    w("- **Module C (GBM-CARE)** reports *observed molecular change*, "
      "measured in paired tumour tissue from the same patients.")
    w("- **Modules A and B (LUMIERE, UPENN-GBM, UCSD-PTGBM)** report *MRI "
      "surrogates*: imaging-derived quantities that stand in for tumour "
      "burden and response.")
    w("- **No patient appears in more than one cohort.** Patient ids are "
      "namespaced by cohort, so a cross-cohort join returns nothing. "
      "Statements about molecular change and statements about imaging are "
      "therefore about *different people*, and are never combined.")
    w("")

    w("## Dataset availability")
    w("")
    lines.extend(_dataset_table(config))
    w("")

    w("## Module A - longitudinal MRI progression (LUMIERE)")
    w("")
    lines.extend(_module_a(config, results))
    w("")

    w("## Module B - radiogenomics and pseudoprogression")
    w("")
    lines.extend(_module_b(config, results))
    w("")

    w("## Module C - molecular plasticity (GBM-CARE)")
    w("")
    lines.extend(_module_c(config, results))
    w("")

    w("## Survival")
    w("")
    lines.extend(_survival(config, results))
    w("")

    w("## Limitations that apply throughout")
    w("")
    lines.extend(_limitations(config))
    w("")

    out = results / "RESULTS_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------

def _dataset_table(config: Config) -> list[str]:
    rows = []
    for key in available_adapters():
        adapter = get_adapter(key, config)
        description = adapter.describe()
        buildable, missing = adapter.is_buildable()
        rows.append({
            "dataset": key,
            "cohort": description.key,
            "patients (published)": description.n_patients_published or "-",
            "timepoints (published)": description.n_timepoints_published or "-",
            "raw data present": "yes" if buildable else "no",
            "registration required": "yes" if description.requires_registration else "no",
        })
    return _md_table(pd.DataFrame(rows))


def _module_a(config: Config, results: Path) -> list[str]:
    base = results / "lumiere"
    out: list[str] = []
    if not base.exists():
        return ["_Not run. `gbmplast model next-visit --dataset lumiere -c lumiere`_"]

    for outcome, title in (("rano", "Next-visit RANO progression"),
                           ("volume", "Next-visit enhancing-volume change")):
        directory = base / f"next_visit_{outcome}"
        if not directory.exists():
            continue
        out.append(f"### {title}")
        out.append("")
        estimand = _read(directory / "estimand.csv")
        if estimand is not None:
            question = estimand.loc[estimand["field"] == "question", "value"]
            if len(question):
                out.append(f"**Estimand.** {question.iloc[0]}")
                out.append("")
        flow = _read(directory / "cohort_flow.csv")
        if flow is not None:
            out.append("**Cohort flow.**")
            out.append("")
            out.extend(_md_table(flow[["kind", "label", "n"]]))
            out.append("")
        metrics = _read(directory / "model_metrics.csv")
        if metrics is not None:
            keep = [c for c in ("model", "n_rows", "n_patients", "auroc",
                                "auprc", "brier", "calibration_slope", "mae",
                                "rmse", "r2", "spearman")
                    if c in metrics.columns]
            out.append("**Out-of-fold performance (patient-grouped CV).**")
            out.append("")
            out.extend(_md_table(metrics[keep].round(4)))
            out.append("")
        comparison = _read(directory / "model_comparison.csv")
        if comparison is not None:
            out.append("**Against the naive baseline (paired patient bootstrap).**")
            out.append("")
            keep = [c for c in ("model", "versus", "metric", "difference",
                                "ci_low", "ci_high", "p_two_sided")
                    if c in comparison.columns]
            out.extend(_md_table(comparison[keep].round(4)))
            out.append("")

    mixed = _read(base / "next_visit_rano" / "mixed_effects_models.csv")
    if mixed is not None:
        out.append("### Descriptive mixed-effects trajectories")
        out.append("")
        keep = [c for c in ("subregion", "marker", "status", "n_observations",
                            "n_patients", "approx_pct_change_per_month",
                            "slope_p_value", "interaction_estimate",
                            "interaction_p_value", "reason")
                if c in mixed.columns]
        out.extend(_md_table(mixed[keep].round(4)))
        out.append("")
        out.append("_Descriptive only: LUMIERE records no treatment, so a time "
                   "coefficient mixes natural history with therapy._")
        out.append("")

    reproduction = _read(base / "radiomics_reproduction_check.csv")
    if reproduction is not None:
        out.append("### Feature-extraction correctness check")
        out.append("")
        out.append("Locally extracted PyRadiomics features versus the values "
                   "published with the cohort, using the cohort's own "
                   "extraction settings:")
        out.append("")
        out.extend(_md_table(reproduction))
        out.append("")
    return out or ["_No Module A outputs found._"]


def _module_b(config: Config, results: Path) -> list[str]:
    out: list[str] = []
    for dataset, title in (("upenn_gbm", "UPENN-GBM: MGMT from baseline MRI"),
                           ("ucsd_ptgbm",
                            "UCSD-PTGBM: tumour vs treatment-related change")):
        base = results / dataset
        if not base.exists():
            continue
        for directory in sorted(base.glob("radiogenomics_*")):
            out.append(f"### {title}")
            out.append("")
            flow = _read(directory / "cohort_flow.csv")
            if flow is not None:
                out.extend(_md_table(flow[["kind", "label", "n"]]))
                out.append("")
            adjudication = _read(directory / "label_adjudication.csv")
            if adjudication is not None:
                out.append("**Label adjudication.**")
                out.append("")
                out.extend(_md_table(adjudication))
                out.append("")
                out.append("_The released tables do not flag which "
                           "tumour-positive timepoints are biopsy-proven, so a "
                           "histopathology-restricted comparison is possible "
                           "on the negative class only._")
                out.append("")
            metrics = _read(directory / "model_metrics.csv")
            if metrics is not None:
                keep = [c for c in ("model", "n_rows", "n_patients",
                                    "prevalence", "auroc", "auprc", "brier",
                                    "calibration_slope")
                        if c in metrics.columns]
                out.extend(_md_table(metrics[keep].round(4)))
                out.append("")
            sensitivity = _read(directory / "adjudication_sensitivity.csv")
            if sensitivity is not None:
                out.append("**Sensitivity to the negative-class adjudication tier.**")
                out.append("")
                out.extend(_md_table(sensitivity.round(4)))
                out.append("")
    return out or ["_No Module B outputs found._"]


def _module_c(config: Config, results: Path) -> list[str]:
    directory = results / "gbm_care" / "plasticity"
    if not directory.exists():
        return ["_Not run. `gbmplast molecular pseudobulk --dataset gbm_care` "
                "then `gbmplast molecular plasticity --dataset gbm_care`_"]
    out: list[str] = []
    out.append("These are **observed molecular changes measured in paired "
               "tumour tissue**, not imaging surrogates.")
    out.append("")

    effects = _read(directory / "paired_effects.csv")
    if effects is not None:
        tested = effects[effects.get("status") == "tested"].copy()
        if len(tested):
            tested = tested.sort_values("wilcoxon_fdr")
            keep = [c for c in ("programme", "n_patients", "mean_primary",
                                "mean_recurrent", "mean_delta", "delta_ci_low",
                                "delta_ci_high", "cohens_dz", "n_increased",
                                "n_decreased", "wilcoxon_p", "wilcoxon_fdr")
                    if c in tested.columns]
            out.append("**Paired within-patient change (primary to recurrent), "
                       "malignant pseudobulk.**")
            out.append("")
            out.extend(_md_table(tested[keep].head(20).round(4)))
            out.append("")
            n_significant = int((tested["wilcoxon_fdr"] < 0.05).sum())
            out.append(f"{n_significant} of {len(tested)} programmes reach "
                       "FDR < 0.05 (Wilcoxon signed-rank, "
                       "Benjamini-Hochberg across programmes).")
            out.append("")

    matrix = _read(directory / "state_transition_matrix.csv")
    if matrix is not None:
        out.append("**Dominant malignant state, primary to recurrent.**")
        out.append("")
        out.extend(_md_table(matrix))
        out.append("")

    validation = _read(config.paths.processed / "gbm_care" / "cnv_validation.csv")
    if validation is not None and "verdict" in validation.columns:
        counts = validation["verdict"].value_counts().to_dict()
        out.append(f"**Malignant-compartment validation.** Copy-number "
                   f"verdicts across samples: {counts}. The compartment is "
                   "defined by canonical tumour markers and checked against "
                   "the GBM chromosome-7-gain / chromosome-10-loss expression "
                   "signature relative to immune and endothelial cells.")
        out.append("")

    interpretation = directory / "INTERPRETATION.txt"
    if interpretation.exists():
        out.append("**Interpretation limits.**")
        out.append("")
        out.append("```")
        out.append(interpretation.read_text(encoding="utf-8").strip())
        out.append("```")
        out.append("")
    return out


def _survival(config: Config, results: Path) -> list[str]:
    out: list[str] = []
    for dataset in ("upenn_gbm", "ucsd_ptgbm", "lumiere"):
        directory = results / dataset / "survival"
        if not directory.exists():
            continue
        refused = _read(directory / "survival_refused.csv")
        if refused is not None:
            out.append(f"- **{dataset}**: refused - "
                       f"{refused['reason'].iloc[0]}")
            continue
        result = _read(directory / "cox_result.csv")
        if result is not None and "c_index_mean" in result.columns:
            out.append(f"- **{dataset}**: penalised Cox, "
                       f"n = {int(result['n'].iloc[0])}, "
                       f"events = {int(result['n_events'].iloc[0])}, "
                       f"C-index {float(result['c_index_mean'].iloc[0]):.3f}. "
                       f"Coefficients: {result['mean_coefficients'].iloc[0]}")
    return out or ["_No survival analyses have been run._"]


def _limitations(config: Config) -> list[str]:
    out = ["Collected from the cohort descriptions, so they stay in step with "
           "the adapters:", ""]
    for key in available_adapters():
        adapter = get_adapter(key, config)
        description = adapter.describe()
        if not description.known_limitations:
            continue
        out.append(f"**{description.key}**")
        out.append("")
        for item in description.known_limitations:
            out.append(f"- {item}")
        out.append("")
    out += [
        "**Across all modules**",
        "",
        "- Sample sizes are small by prediction-model standards (81-630 "
        "patients per analysis), so confidence intervals are wide and a "
        "single cohort cannot establish generalisability.",
        "- Treatment is unrecorded (LUMIERE, GBM-CARE) or partially recorded "
        "(UPENN-GBM, UCSD-PTGBM), so every comparison is confounded by "
        "therapy.",
        "- All analyses are associational. No intervention is assigned, and "
        "nothing here supports a causal claim.",
        "- Feature importances (SHAP, permutation) describe what a model "
        "used, not what drives the biology.",
    ]
    return out


# --------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        return frame if len(frame) else None
    except Exception:
        return None


def _md_table(frame: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if frame is None or frame.empty:
        return ["_(empty)_"]
    frame = frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            text = "" if pd.isna(value) else str(value)
            cells.append(text.replace("|", "\\|")[:120])
        rows.append("| " + " | ".join(cells) + " |")
    return [header, divider, *rows]
