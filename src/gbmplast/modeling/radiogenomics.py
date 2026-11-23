"""Module B: radiogenomics and pseudoprogression.

Two independent analyses, each loaded through its own adapter.  They share
only code -- never patients, and never a row.

UPENN-GBM  -- predict MGMT promoter methylation from baseline pre-operative
              MRI radiomics and clinical covariates. Clinical-only vs
              MRI-only vs combined.
UCSD-PTGBM -- separate tumour-positive from treatment-related change at a
              post-treatment timepoint, using the cohort's explicit expert
              labels and its adjudication tiers.

Rules enforced here
-------------------
* Patient-level grouped CV: every timepoint of one patient stays in one fold.
* Feature selection and hyperparameter search happen inside the training
  fold (:mod:`gbmplast.modeling.evaluate`).
* Only covariates knowable at the prediction time are used.  For UCSD that
  means a treatment variable enters only when the cohort states it happened
  *before* the scan.
* Class imbalance is handled by balanced class weights and reported with
  AUPRC and prevalence, not accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters import get_adapter
from gbmplast.config import Config
from gbmplast.modeling import metrics as M
from gbmplast.modeling.estimand import CohortFlow, Estimand
from gbmplast.modeling.evaluate import evaluate_classifier, summarise_selection
from gbmplast.modeling.leakage import audit_analysis
from gbmplast.modeling.splits import split_indices
from gbmplast.modeling.models import (
    CLASSIFIER_GRIDS,
    PrevalenceClassifier,
    elastic_net_classifier,
    gradient_boosting_classifier,
)

DEFAULT_TARGETS = {
    "upenn_gbm": "mgmt",
    "ucsd_ptgbm": "tumor_vs_treatment",
}


def run_radiogenomics(config: Config, *, dataset: str,
                      target: str | None = None) -> int:
    adapter = get_adapter(dataset, config)
    if not adapter.is_buildable()[0]:
        print(f"  {dataset}: raw data absent; "
              f"run: gbmplast acquire --dataset {dataset}")
        return 0
    target = target or DEFAULT_TARGETS[dataset]
    if dataset == "upenn_gbm":
        return _run_upenn(config, adapter, target)
    if dataset == "ucsd_ptgbm":
        return _run_ucsd(config, adapter)
    print(f"  {dataset}: no radiogenomics analysis defined")
    return 0


# ==========================================================================
# UPENN-GBM: MGMT (and IDH) from baseline MRI
# ==========================================================================

def _upenn_estimand(target: str) -> Estimand:
    label = {"mgmt": "MGMT promoter methylation",
             "idh": "IDH1 mutation"}[target]
    return Estimand(
        name=f"upenn_{target}_from_baseline_mri",
        question=(
            f"Can {label} status be predicted from pre-operative MRI "
            "radiomics and routinely available clinical covariates?"),
        cohort="UPENN-GBM",
        unit_of_analysis="baseline pre-operative scan (one per patient)",
        outcome=f"{target}_label",
        outcome_type="binary",
        positive_label=("methylated" if target == "mgmt" else "mutant"),
        prediction_time="the baseline pre-operative scan, before surgery",
        lookback_window_days=0,
        outcome_horizon_days=None,
        eligibility=(
            "scan is the baseline pre-operative study (suffix _11)",
            f"{label} is reported as a definite call",
            "CaPTk radiomic features are available for the scan",
        ),
        exclusions=(
            "recurrence scans (suffix _21): a post-treatment scan is not a "
            "baseline, and mixing the two would put two scans of one patient "
            "in the analysis",
            "Indeterminate / NOS-NEC / Not Available marker calls",
        ),
        confounders_acknowledged=(
            "Scanner and protocol vary across the cohort; radiomic features "
            "are sensitive to acquisition, and no scanner harmonisation is "
            "applied here.",
            "MGMT assay method is not released per patient.",
        ),
        notes=(
            "Radiomics are CaPTk-derived as released by the cohort, not "
            "PyRadiomics, and are not comparable with the LUMIERE features.",
        ),
    )


def _run_upenn(config: Config, adapter, target: str) -> int:
    bundle = adapter.build()
    outcome_name = {"mgmt": "mgmt_methylation", "idh": "idh_mutation"}[target]
    estimand = _upenn_estimand(target)
    outdir = config.paths.results / adapter.key / f"radiogenomics_{target}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(estimand.describe())

    outcomes = bundle.outcomes[bundle.outcomes["outcome_name"] == outcome_name]
    flow = CohortFlow()
    flow.step("scans in cohort", len(bundle.visits))
    flow.step(f"scans with a definite {target.upper()} call", len(outcomes))

    baseline_visits = bundle.visits[bundle.visits["scan_role"]
                                    == "baseline_preoperative"]
    outcomes = outcomes[outcomes["visit_id"].isin(set(baseline_visits["visit_id"]))]
    flow.step("restricted to baseline pre-operative scans", len(outcomes))

    try:
        features = adapter.load_captk_features(segmentation="automatic")
    except FileNotFoundError as exc:
        print(f"  CaPTk features unavailable: {exc}")
        return 0
    flow.step("scans with CaPTk radiomics", len(features))

    data = outcomes[["patient_id", "visit_id", "outcome_value",
                     "outcome_label"]].merge(
        features, on=["patient_id", "visit_id"], how="inner")
    flow.exclude("no radiomics for the scan",
                 len(outcomes) - len(data))

    clinical = bundle.patients[[
        "patient_id", "age_at_diagnosis_years", "sex", "kps"]].copy()
    clinical["sex_female"] = clinical["sex"].map(
        {"female": 1.0, "male": 0.0}).astype(float)
    clinical["kps"] = pd.to_numeric(clinical["kps"], errors="coerce")
    eor = bundle.treatments[bundle.treatments["treatment_type"] == "surgery"][
        ["patient_id", "extent_of_resection"]].drop_duplicates("patient_id")
    clinical = clinical.merge(eor, on="patient_id", how="left")
    clinical["gtr_over_90pct"] = clinical["extent_of_resection"].map(
        {"GTR_over_90pct": 1.0, "less_than_90pct": 0.0}).astype(float)
    data = data.merge(
        clinical[["patient_id", "age_at_diagnosis_years", "sex_female",
                  "kps", "gtr_over_90pct"]],
        on="patient_id", how="left")

    data["y"] = pd.to_numeric(data["outcome_value"], errors="coerce")
    data = data[data["y"].notna()].reset_index(drop=True)

    # One row per patient here by construction; assert it rather than assume.
    dup = data["patient_id"].duplicated().sum()
    if dup:
        raise AssertionError(
            f"{dup} patients contribute more than one baseline row; grouped "
            "CV would still be correct but the estimand says one scan per "
            "patient -- investigate before modelling.")

    flow.step("analysis set", len(data))
    flow.step("patients", data["patient_id"].nunique())
    flow.step(f"positive ({estimand.positive_label})", int(data["y"].sum()))
    print("\nCohort flow:")
    print(flow.describe())

    clinical_cols = [c for c in ("age_at_diagnosis_years", "sex_female", "kps",
                                 "gtr_over_90pct")
                     if _usable(data, c)]
    radiomic_cols = [c for c in adapter.feature_columns(features)
                     if c in data.columns and _usable(data, c)]
    combined_cols = clinical_cols + radiomic_cols
    print(f"\nFeature blocks: clinical={len(clinical_cols)}, "
          f"radiomics={len(radiomic_cols)}, combined={len(combined_cols)}")

    estimand.to_frame().to_csv(outdir / "estimand.csv", index=False)
    flow.to_frame().to_csv(outdir / "cohort_flow.csv", index=False)

    if len(data) < 50:
        print("\n  fewer than 50 eligible scans; refusing to model")
        return 0

    blocks = {"clinical_only": clinical_cols, "mri_only": radiomic_cols,
              "combined": combined_cols}
    _run_leakage_audit(data, analysis=f"upenn_gbm/{target}", blocks=blocks,
                       outcome_col="y", config=config, outdir=outdir)
    results = _fit_ladder(data, outcome_col="y", blocks=blocks, config=config)
    _write_and_report(results, outdir,
                      config.paths.figures / adapter.key / f"radiogenomics_{target}",
                      title=f"UPENN-GBM: {estimand.positive_label} from baseline MRI",
                      data=data, flow=flow)
    return 0


# ==========================================================================
# UCSD-PTGBM: tumour vs treatment-related change
# ==========================================================================

def _ucsd_estimand() -> Estimand:
    return Estimand(
        name="ucsd_tumor_vs_treatment_change",
        question=(
            "At a post-treatment MRI timepoint, does the lesion represent "
            "residual/recurrent tumour rather than treatment-related change "
            "(pseudoprogression, radiation necrosis or non-specific "
            "post-treatment change)?"),
        cohort="UCSD-PTGBM",
        unit_of_analysis="post-treatment MRI timepoint",
        outcome="tumor_vs_treatment_change",
        outcome_type="binary",
        positive_label="tumor_positive",
        prediction_time="the post-treatment MRI acquisition",
        lookback_window_days=None,
        outcome_horizon_days=None,
        eligibility=(
            "timepoint is in the released cohort (243 scans, 178 patients)",
            "an expert label is available for the timepoint",
        ),
        exclusions=(
            "none by label: all 243 timepoints carry a label",
            "covariates are used only where the cohort states the event "
            "preceded the scan",
        ),
        confounders_acknowledged=(
            "Bevacizumab reduces enhancement independently of tumour burden; "
            "27 patients received it.",
            "Time since radiotherapy strongly shapes the prior probability "
            "of pseudoprogression and is itself clinically chosen.",
            "The cohort was deliberately enriched for pathology-proven "
            "cases, so the observed 79% tumour-positive rate is NOT the "
            "prevalence in routine post-treatment imaging.",
            "Multiple timepoints per patient (49 patients contribute 2-4), "
            "handled by patient-level grouping.",
        ),
        notes=(
            "Adjudication differs by class: the negative class carries an "
            "explicit tier (6 biopsy-proven, 9 clinically confirmed, 7 "
            "radiation necrosis, 29 non-specific) while the released tables "
            "do not flag which positives are biopsy-proven. A "
            "histopathology-only sensitivity analysis is therefore possible "
            "on the negative side only.",
            "Image-derived features are unavailable in this environment: the "
            "33 GB NIfTI bundle is distributed through IBM Aspera. The "
            "analysis therefore runs on clinical/treatment covariates, and "
            "the imaging interface is implemented and left ready.",
        ),
    )


def _run_ucsd(config: Config, adapter) -> int:
    bundle = adapter.build()
    estimand = _ucsd_estimand()
    outdir = config.paths.results / adapter.key / "radiogenomics_tumor_vs_treatment"
    outdir.mkdir(parents=True, exist_ok=True)
    print(estimand.describe())

    outcomes = bundle.outcomes[
        bundle.outcomes["outcome_name"] == "tumor_vs_treatment_change"].copy()
    visits = bundle.visits.copy()
    data = outcomes[["patient_id", "visit_id", "outcome_value", "outcome_label",
                     "ascertainment", "outcome_detail"]].merge(
        visits, on=["patient_id", "visit_id"], how="left", suffixes=("", "_v"))

    flow = CohortFlow()
    flow.step("timepoints in cohort", len(visits))
    flow.step("timepoints with an expert label", len(data))
    flow.step("patients", data["patient_id"].nunique())

    data["y"] = pd.to_numeric(data["outcome_value"], errors="coerce")

    # ---- covariates knowable at scan time --------------------------------
    patients = bundle.patients[[
        "patient_id", "age_at_first_scan_years", "sex", "mgmt_status",
        "idh_status", "who_grade"]].copy()
    patients["sex_female"] = patients["sex"].map(
        {"female": 1.0, "male": 0.0}).astype(float)
    patients["mgmt_methylated"] = patients["mgmt_status"].map(
        {"methylated": 1.0, "unmethylated": 0.0}).astype(float)
    patients["idh_mutant"] = patients["idh_status"].map(
        {"mutant": 1.0, "wildtype": 0.0}).astype(float)
    data = data.merge(patients[["patient_id", "age_at_first_scan_years",
                                "sex_female", "mgmt_methylated", "idh_mutant"]],
                      on="patient_id", how="left")

    # These three are released as "days from acquisition to the LAST <event>
    # PRIOR TO the scan", so they are knowable at prediction time by
    # construction. Convert to a positive "days since" and keep the sign
    # convention explicit.
    for src, dest in (("days_acq_to_last_surgery", "days_since_last_surgery"),
                      ("days_acq_to_last_radiation", "days_since_last_radiation"),
                      ("days_acq_to_last_avastin", "days_since_last_avastin")):
        value = pd.to_numeric(data.get(src), errors="coerce")
        # Released values are negative when the event preceded the scan.
        data[dest] = np.where(value <= 0, -value, np.nan)
        data[f"{dest}_known"] = value.notna().astype(float)

    treatments = bundle.treatments.copy()
    for ttype, name, col in (
        ("radiotherapy", None, "had_radiotherapy"),
        ("chemotherapy", "bevacizumab", "had_bevacizumab"),
    ):
        sel = treatments[treatments["treatment_type"] == ttype]
        if name:
            sel = sel[sel["treatment_name"] == name]
        flagged = set(sel["patient_id"])
        data[col] = data["patient_id"].isin(flagged).astype(float)
    eor = treatments[treatments["treatment_type"] == "surgery"][
        ["patient_id", "extent_of_resection"]].drop_duplicates("patient_id")
    data = data.merge(eor, on="patient_id", how="left")
    data["gtr"] = (data["extent_of_resection"] == "GTR").astype(float)

    data["timepoint_index_num"] = pd.to_numeric(data["timepoint_index"],
                                                errors="coerce")
    data["days_since_surgery_at_scan"] = pd.to_numeric(
        data["relative_time_days"], errors="coerce")

    flow.step("analysis set", len(data))
    flow.step("positive (tumour)", int(data["y"].sum()))
    flow.step("negative (treatment-related change)", int((data["y"] == 0).sum()))
    print("\nCohort flow:")
    print(flow.describe())

    print("\nAdjudication of the labels:")
    adj = (data.groupby(["outcome_label", "outcome_detail", "ascertainment"])
           .size().reset_index(name="n"))
    print(adj.to_string(index=False))
    adj.to_csv(outdir / "label_adjudication.csv", index=False)

    clinical_cols = [c for c in (
        "age_at_first_scan_years", "sex_female", "mgmt_methylated",
        "idh_mutant", "gtr", "had_radiotherapy", "had_bevacizumab",
        "timepoint_index_num") if _usable(data, c)]
    timing_cols = [c for c in (
        "days_since_last_surgery", "days_since_last_radiation",
        "days_since_last_avastin", "days_since_surgery_at_scan",
        "days_since_last_surgery_known", "days_since_last_radiation_known",
        "days_since_last_avastin_known") if _usable(data, c)]
    combined_cols = clinical_cols + timing_cols
    print(f"\nFeature blocks: clinical/molecular={len(clinical_cols)}, "
          f"treatment-timing={len(timing_cols)}, combined={len(combined_cols)}")

    # Imaging features would join here; the cohort's NIfTI bundle needs
    # IBM Aspera and is not retrievable in this environment.
    imaging_cols = _load_ucsd_imaging_features(adapter, data)
    if imaging_cols:
        combined_cols = combined_cols + imaging_cols
        print(f"  plus imaging features: {len(imaging_cols)}")
    else:
        print("  imaging features: unavailable (Aspera-only distribution); "
              "the interface is implemented and the analysis proceeds on "
              "clinical and treatment-timing covariates only")

    estimand.to_frame().to_csv(outdir / "estimand.csv", index=False)
    flow.to_frame().to_csv(outdir / "cohort_flow.csv", index=False)

    blocks = {"clinical_only": clinical_cols,
              "treatment_timing_only": timing_cols,
              "combined": combined_cols}
    if imaging_cols:
        blocks["imaging_only"] = imaging_cols
    _run_leakage_audit(data, analysis="ucsd_ptgbm/tumor_vs_treatment",
                       blocks=blocks, outcome_col="y", config=config,
                       outdir=outdir, treatments=bundle.treatments,
                       prediction_time_col="relative_time_days")
    results = _fit_ladder(data, outcome_col="y", blocks=blocks, config=config)
    _write_and_report(results, outdir,
                      config.paths.figures / adapter.key / "radiogenomics_tumor_vs_treatment",
                      title="UCSD-PTGBM: tumour vs treatment-related change",
                      data=data, flow=flow)

    # Sensitivity analysis: restrict the negative class to the tiers where
    # the cohort states how the judgement was made.
    _ucsd_adjudication_sensitivity(data, blocks, config, outdir)
    return 0


def _load_ucsd_imaging_features(adapter, data: pd.DataFrame) -> list[str]:
    """Join locally-present UCSD imaging features, if any.

    The interface is implemented so that dropping a feature table into
    ``<raw>/imaging_features.csv`` (keyed by ``local_visit_id``) activates
    the imaging arm without touching model code.
    """
    path = adapter.raw_dir / "imaging_features.csv"
    if not path.exists():
        return []
    features = pd.read_csv(path, low_memory=False)
    if "local_visit_id" not in features.columns:
        print(f"  ! {path} lacks a 'local_visit_id' column; ignored")
        return []
    merged = data.merge(features, on="local_visit_id", how="left",
                        suffixes=("", "_img"))
    new_cols = [c for c in features.columns if c != "local_visit_id"]
    for col in new_cols:
        data[col] = merged[col].to_numpy()
    return [c for c in new_cols if _usable(data, c)]


def _ucsd_adjudication_sensitivity(data: pd.DataFrame, blocks: dict,
                                   config: Config, outdir: Path) -> None:
    """Re-fit with the negative class restricted by adjudication tier."""
    tiers = {
        "histopathology_only_negatives": ("histopathology",),
        "pathology_or_clinical_negatives": ("histopathology",
                                            "clinical_consensus"),
    }
    rows = []
    for label, keep in tiers.items():
        subset = data[(data["y"] == 1) |
                      (data["ascertainment"].isin(keep))].copy()
        n_neg = int((subset["y"] == 0).sum())
        if n_neg < 10:
            rows.append({"sensitivity": label, "n": len(subset),
                         "n_negative": n_neg, "status": "skipped",
                         "reason": f"only {n_neg} negatives after restriction; "
                                   "too few to model"})
            continue
        results = _fit_ladder(subset, outcome_col="y",
                              blocks={"combined": blocks["combined"]},
                              config=config, quiet=True)
        for r in results:
            rows.append({
                "sensitivity": label, "model": r.model, "status": "fitted",
                "n": r.n_rows, "n_patients": r.n_patients,
                "n_negative": n_neg,
                "auroc": r.metrics.get("auroc"),
                "auprc": r.metrics.get("auprc"),
                "prevalence": r.metrics.get("prevalence"),
            })
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "adjudication_sensitivity.csv", index=False)
    print("\n== sensitivity to negative-class adjudication tier ==")
    print(out.to_string(index=False))
    print("  (the positive class cannot be restricted this way: the released "
          "tables do not flag which tumour-positive timepoints are "
          "biopsy-proven)")


# ==========================================================================
# shared
# ==========================================================================

def _run_leakage_audit(data: pd.DataFrame, *, analysis: str, blocks: dict,
                       outcome_col: str, config: Config, outdir: Path,
                       treatments: pd.DataFrame | None = None,
                       specimen_col: str | None = None,
                       prediction_time_col: str | None = None) -> None:
    combined = blocks.get("combined") or next(iter(blocks.values()), [])
    if not combined:
        return
    y = pd.to_numeric(data[outcome_col], errors="coerce")
    audit = audit_analysis(
        data, analysis=analysis, feature_cols=combined,
        outcome_col=outcome_col,
        folds=split_indices(data[combined], y, data["patient_id"],
                            n_splits=int(config.get("modeling.cv.n_splits", 5)),
                            seed=config.seed, stratify=True),
        treatments=treatments, specimen_col=specimen_col,
        prediction_time_col=prediction_time_col)
    print()
    print(audit.describe())
    outdir.mkdir(parents=True, exist_ok=True)
    audit.to_frame().to_csv(outdir / "leakage_audit.csv", index=False)
    audit.raise_if_failed()


def _fit_ladder(data: pd.DataFrame, *, outcome_col: str, blocks: dict,
                config: Config, quiet: bool = False) -> list:
    seed = config.seed
    common = dict(outcome_col=outcome_col,
                  n_splits=int(config.get("modeling.cv.n_splits", 5)),
                  inner_splits=int(config.get("modeling.inner_cv.n_splits", 3)),
                  seed=seed,
                  n_bootstrap=int(config.get("reproducibility.n_bootstrap", 2000)),
                  bootstrap_seed=int(config.get("reproducibility.bootstrap_seed",
                                                777)))
    results = [evaluate_classifier(
        data, feature_cols=[], model_name="prevalence_baseline",
        baseline=PrevalenceClassifier(), **common)]
    for label, cols in blocks.items():
        if len(cols) < 2:
            continue
        results.append(evaluate_classifier(
            data, feature_cols=cols, model_name=f"{label}__elastic_net",
            estimator=elastic_net_classifier(seed),
            param_grid=CLASSIFIER_GRIDS["elastic_net"], **common))
        results.append(evaluate_classifier(
            data, feature_cols=cols, model_name=f"{label}__gradient_boosting",
            estimator=gradient_boosting_classifier(seed),
            param_grid=CLASSIFIER_GRIDS["gradient_boosting"], **common))
    return results


def _write_and_report(results: list, outdir: Path, figdir: Path, *,
                      title: str, data: pd.DataFrame,
                      flow: CohortFlow | None = None) -> None:
    rows = []
    for r in results:
        row = {"model": r.model, "n": r.n_rows, "patients": r.n_patients,
               "prevalence": _fmt(r.metrics.get("prevalence")),
               "AUROC": ("undefined"
                         if r.metrics.get("discrimination_defined") is False
                         else _fmt(r.metrics.get("auroc"))),
               "AUPRC": _fmt(r.metrics.get("auprc")),
               "Brier": _fmt(r.metrics.get("brier")),
               "CalSlope": _fmt(r.metrics.get("calibration_slope"))}
        if r.metric_ci is not None:
            sel = r.metric_ci[r.metric_ci.metric == "auroc"]
            if len(sel) and r.metrics.get("discrimination_defined") is not False:
                row["AUROC 95% CI"] = (f"{sel.ci_low.iloc[0]:.3f}-"
                                       f"{sel.ci_high.iloc[0]:.3f}")
        rows.append(row)
    table = pd.DataFrame(rows)
    print("\n== out-of-fold performance (patient-grouped CV) ==")
    print(table.to_string(index=False))

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.metric_row() for r in results]).to_csv(
        outdir / "model_metrics.csv", index=False)
    frames = []
    for r in results:
        if r.metric_ci is not None:
            ci = r.metric_ci.copy(); ci.insert(0, "model", r.model)
            frames.append(ci)
        r.predictions.to_csv(outdir / f"predictions__{r.model}.csv", index=False)
        stability = summarise_selection(
            r.selected_features, len(r.folds) if r.folds is not None else 5)
        if not stability.empty:
            stability.head(40).to_csv(
                outdir / f"feature_selection__{r.model}.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            outdir / "model_metric_ci.csv", index=False)

    best = max((r for r in results
                if r.metrics.get("discrimination_defined") is not False),
               key=lambda r: r.metrics.get("auroc", 0), default=None)
    _attribute(results, data, outdir)
    if best is not None and best.selected_features is not None:
        stability = summarise_selection(best.selected_features,
                                        len(best.folds) if best.folds is not None else 5)
        if not stability.empty:
            print(f"\n== most stably selected features ({_pretty(best.model)}) ==")
            print(stability.head(10).to_string(index=False))

    try:
        from gbmplast.report.figures import (
            calibration_plot, cohort_flow_diagram, metric_comparison,
            pr_curves, roc_curves)
        figdir = Path(figdir)
        if flow is not None:
            cohort_flow_diagram(flow.to_frame(), figdir / "cohort_flow.png",
                                title=f"Analysis cohort flow - {title}")
        roc_curves(results, figdir / "roc_curves.png", title=title)
        pr_curves(results, figdir / "precision_recall.png", title=title)
        calibration_plot(results, figdir / "calibration.png", title=title)
        metric_comparison(results, figdir / "auroc_comparison.png",
                          metric="auroc", title=title)
        print(f"\nfigures -> {figdir}")
    except Exception as exc:
        print(f"  (figures skipped: {exc})")


def _attribute(results: list, data: pd.DataFrame, outdir: Path) -> None:
    """Permutation importance, where the sample size makes it interpretable."""
    from gbmplast.modeling.explain import CAUTION, explain_if_valid
    from gbmplast.modeling.models import gradient_boosting_classifier

    candidates = [r for r in results
                  if r.model.endswith("gradient_boosting")
                  and r.selected_features is not None]
    if not candidates:
        return
    best = max(candidates, key=lambda r: r.metrics.get("auroc", 0))
    block = best.model.split("__")[0]
    columns = sorted({str(f).replace("missingindicator_", "")
                      for f in best.selected_features["feature"]})
    columns = [c for c in columns if c in data.columns]
    if not columns:
        return
    importance, reason = explain_if_valid(
        data, feature_cols=columns, outcome_col="y",
        estimator=gradient_boosting_classifier())
    print(f"\n== feature attribution ({block}) ==")
    print(f"  {reason}")
    if importance is None or importance.empty:
        pd.DataFrame([{"status": "refused", "reason": reason}]).to_csv(
            outdir / "attribution_refused.csv", index=False)
        return
    importance.to_csv(outdir / f"permutation_importance__{best.model}.csv",
                      index=False)
    print(importance.head(10).round(4).to_string(index=False))
    print(f"  {CAUTION}")


def _usable(df: pd.DataFrame, column: str, *, min_present: float = 0.3,
            min_unique: int = 2) -> bool:
    if column not in df.columns:
        return False
    s = pd.to_numeric(df[column], errors="coerce")
    if s.notna().mean() < min_present:
        return False
    return s.dropna().nunique() >= min_unique


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{float(value):.3f}"


def _pretty(name: str) -> str:
    return name.replace("__", " + ").replace("_", " ")
