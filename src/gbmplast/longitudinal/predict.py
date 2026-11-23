"""Module A: next-visit progression prediction on LUMIERE.

Two estimands, both declared before any model is fitted:

``rano``    next-visit expert RANO progression (PD vs CR/PR/SD), binary
``volume``  next-visit relative change in enhancing-tumour volume, continuous

Model ladder (brief: compare simple models first)
  prevalence / carry-forward  -> last_observation -> clinical -> imaging
  -> combined, each as elastic net and gradient boosting where applicable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters import get_adapter
from gbmplast.config import Config
from gbmplast.longitudinal.build import build_longitudinal
from gbmplast.modeling import metrics as M
from gbmplast.modeling.estimand import CohortFlow, Estimand
from gbmplast.modeling.evaluate import (
    evaluate_classifier,
    evaluate_regressor,
    summarise_selection,
)
from gbmplast.modeling.models import (
    CLASSIFIER_GRIDS,
    REGRESSOR_GRIDS,
    CarryForwardRegressor,
    LastObservationClassifier,
    MeanRegressor,
    PrevalenceClassifier,
    elastic_net_classifier,
    elastic_net_regressor,
    gradient_boosting_classifier,
    gradient_boosting_regressor,
)


def rano_estimand(cohort: str) -> Estimand:
    return Estimand(
        name="next_visit_rano_progression",
        question=(
            "Among patient-visits with at least one prior imaged study, what "
            "is the probability that the NEXT expert RANO assessment is "
            "progressive disease, using only information available up to and "
            "including the prediction visit?"),
        cohort=cohort,
        unit_of_analysis="patient-visit transition (index visit -> next visit)",
        outcome="next_rano_progression",
        outcome_type="binary",
        positive_label="PD",
        prediction_time="the index visit of a consecutive imaged pair",
        lookback_window_days=365,
        outcome_horizon_days=365,
        eligibility=(
            "patient has >= 2 dated studies",
            "index visit has radiomics for the primary segmentation",
            "next visit carries an expert RANO rating in {CR, PR, SD, PD}",
            "inter-scan interval between 7 and 365 days",
        ),
        exclusions=(
            "Pre-Op and Post-Op ratings are surgical anchors, not response "
            "assessments, and cannot serve as a progression outcome",
            "transitions whose next visit was never rated",
            "same-week repeat studies (interval < 7 days)",
        ),
        confounders_acknowledged=(
            "Treatment is not recorded in LUMIERE, so the probability of "
            "progression is confounded by unobserved therapy (including "
            "second-line agents and bevacizumab, which alters enhancement).",
            "RANO ratings were made with knowledge of the imaging the model "
            "also sees, so the label is not independent of the features.",
            "Scan intervals are clinically driven: a shorter interval often "
            "reflects clinical suspicion of progression.",
            "Timepoints are week-resolution, so intervals carry +/- ~3 days "
            "of error.",
        ),
        notes=(
            "Expert ratings exist for a RANO-guided selection of timepoints, "
            "not all studies, so the eligible set is not a random sample of "
            "follow-up visits.",
        ),
    )


#: Minimum index enhancing volume for a volume-change outcome, in mm^3.
#:
#: The released HD-GLIO-AUTO masks contain lesions as small as 10 mm^3 -- ten
#: 1 mm voxels -- and a ratio against such a denominator is noise: observed
#: relative changes reach +306 (24 mm^3 -> 7376 mm^3), giving the raw
#: relative-change outcome a mean of 8.0 with an SD of 35.4. That is not a
#: modellable quantity, so this module:
#:   * uses the symmetric LOG VOLUME RATIO as the primary outcome, which is
#:     well behaved and stays defined when a volume is exactly 0, and
#:   * applies this floor to the secondary relative-change outcome.
MIN_INDEX_VOLUME_MM3 = 100.0


def volume_estimand(cohort: str, *, outcome_col: str,
                    min_index_volume_mm3: float = MIN_INDEX_VOLUME_MM3
                    ) -> Estimand:
    is_log = outcome_col.startswith("next_log_ratio")
    return Estimand(
        name=("next_visit_enhancing_log_volume_ratio" if is_log
              else "next_visit_enhancing_relative_change"),
        question=(
            "What is the log ratio of enhancing-tumour volume between the "
            "next imaged study and the prediction visit, using only "
            "information available up to and including the prediction visit?"
            if is_log else
            "What is the relative change in enhancing-tumour volume between "
            "the prediction visit and the next imaged study, using only "
            "information available up to and including the prediction visit?"),
        cohort=cohort,
        unit_of_analysis="patient-visit transition",
        outcome=outcome_col,
        outcome_type="continuous",
        prediction_time="the index visit of a consecutive imaged pair",
        lookback_window_days=365,
        outcome_horizon_days=365,
        eligibility=(
            "both visits have a segmented enhancing subregion",
            "inter-scan interval between 7 and 365 days",
        ) + ((
            "log1p ratio is used, so a zero volume at either visit is "
            "retained (complete response is a real observation, not a gap)",
        ) if is_log else (
            f"index enhancing volume >= {min_index_volume_mm3:.0f} mm^3, a "
            "segmentation-reliability floor: a ratio against a ten-voxel "
            "lesion is noise",
        )),
        exclusions=((
            "transitions where either volume is missing because the study "
            "was never segmented",
        ) if is_log else (
            f"index enhancing volume below {min_index_volume_mm3:.0f} mm^3",
            "transitions where the index enhancing volume is 0 -- common "
            "after gross total resection, and excluded here rather than "
            "silently producing an unbounded relative change",
        )),
        confounders_acknowledged=(
            "Unobserved treatment, including anti-angiogenic therapy which "
            "reduces enhancement without reducing tumour burden.",
            "Automated segmentation error propagates directly into both the "
            "predictor and the outcome.",
            "Week-resolution timing adds error to any rate-based feature.",
        ),
        notes=(
            "The log ratio is symmetric in growth and shrinkage; the raw "
            "relative change is not, and is reported only as a secondary "
            "outcome under a volume floor.",
        ),
    )


# --------------------------------------------------------------------------

def run_next_visit_analysis(config: Config, *, dataset: str = "lumiere",
                            outcome: str = "rano") -> int:
    adapter = get_adapter(dataset, config)
    if not adapter.is_buildable()[0]:
        print(f"  {dataset}: raw data absent; "
              f"run: gbmplast acquire --dataset {dataset}")
        return 0

    result = build_longitudinal(adapter, config)
    if result is None or result.table.empty:
        print(f"  {dataset}: no longitudinal transitions available")
        return 0

    table = result.table
    cohort = adapter.describe().key
    outdir = config.paths.results / dataset / f"next_visit_{outcome}"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = config.paths.figures / dataset / f"next_visit_{outcome}"

    seed = config.seed
    n_bootstrap = int(config.get("reproducibility.n_bootstrap", 2000))
    bootstrap_seed = int(config.get("reproducibility.bootstrap_seed", 777))
    n_splits = int(config.get("modeling.cv.n_splits", 5))
    inner_splits = int(config.get("modeling.inner_cv.n_splits", 3))

    if outcome == "rano":
        estimand = rano_estimand(cohort)
        eligible, flow = _rano_eligible(table)
        outcome_col = "next_rano_progression"
    else:
        outcome_col = str(config.get("analysis.next_visit_volume.outcome",
                                     "next_log_ratio_enhancing"))
        floor = float(config.get("analysis.next_visit_volume.min_index_volume_mm3",
                                 MIN_INDEX_VOLUME_MM3))
        estimand = volume_estimand(cohort, outcome_col=outcome_col,
                                   min_index_volume_mm3=floor)
        eligible, flow = _volume_eligible(table, outcome_col=outcome_col,
                                          min_index_volume_mm3=floor)

    print(estimand.describe())
    print("\nCohort flow:")
    print(flow.describe())
    estimand.to_frame().to_csv(outdir / "estimand.csv", index=False)
    flow.to_frame().to_csv(outdir / "cohort_flow.csv", index=False)
    pd.DataFrame({"provenance": result.provenance}).to_csv(
        outdir / "longitudinal_provenance.csv", index=False)
    if result.excluded is not None and not result.excluded.empty:
        result.excluded.to_csv(outdir / "longitudinal_exclusions.csv",
                               index=False)

    if len(eligible) < 30:
        print(f"\n  only {len(eligible)} eligible rows; refusing to fit "
              "models on a sample this small")
        return 0

    clinical = result.features("clinical")
    imaging = [c for c in result.features("imaging", "dynamics")
               if _usable(eligible, c)]
    history = [c for c in result.features("history") if _usable(eligible, c)]
    combined = clinical + imaging + history

    print(f"\nFeature blocks: clinical={len(clinical)}, "
          f"imaging+dynamics={len(imaging)}, history={len(history)}, "
          f"combined={len(combined)}")

    results = (_run_rano_models if outcome == "rano" else _run_volume_models)(
        eligible, outcome_col=outcome_col, clinical=clinical,
        imaging=imaging, history=history, combined=combined,
        seed=seed, n_splits=n_splits, inner_splits=inner_splits,
        n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)

    _write_results(results, outdir, outcome=outcome)
    _print_results(results, outcome=outcome)

    # Model comparison against the naive baseline, paired on patients.
    comparison = _compare_to_baseline(results, outcome=outcome,
                                      n_bootstrap=n_bootstrap,
                                      seed=bootstrap_seed)
    if comparison is not None and not comparison.empty:
        comparison.to_csv(outdir / "model_comparison.csv", index=False)
        print("\n== vs naive baseline (paired patient bootstrap) ==")
        print(comparison.to_string(index=False))

    try:
        from gbmplast.report.figures import next_visit_figures
        written = next_visit_figures(results, eligible, figdir, outcome=outcome)
        if written:
            print(f"\nfigures -> {figdir} ({len(written)} files)")
    except Exception as exc:
        print(f"  (figures skipped: {exc})")

    try:
        from gbmplast.longitudinal.mixed import run_descriptive_models
        descriptive = run_descriptive_models(table, outdir)
        if descriptive is not None:
            print("\n== descriptive mixed-effects models ==")
            print(descriptive.to_string(index=False))
    except Exception as exc:
        print(f"  (mixed-effects models skipped: {exc})")

    return 0


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------

def _rano_eligible(table: pd.DataFrame) -> tuple[pd.DataFrame, CohortFlow]:
    flow = CohortFlow()
    flow.step("transitions built", len(table))
    flow.step("patients", table["patient_id"].nunique())

    rated = table[table["next_rano_is_response_assessment"] == True]  # noqa: E712
    flow.exclude("next visit has no RANO response assessment "
                 "(unrated, or Pre-Op/Post-Op only)", len(table) - len(rated))
    flow.step("next visit carries a RANO response assessment", len(rated))

    eligible = rated[rated["next_rano_progression"].notna()].copy()
    flow.exclude("outcome not derivable", len(rated) - len(eligible))
    flow.step("analysis set", len(eligible))
    flow.step("patients in analysis set", eligible["patient_id"].nunique())
    flow.step("events (next visit PD)",
              int(eligible["next_rano_progression"].sum()))
    return eligible.reset_index(drop=True), flow


def _volume_eligible(table: pd.DataFrame, *, outcome_col: str,
                     min_index_volume_mm3: float
                     ) -> tuple[pd.DataFrame, CohortFlow]:
    flow = CohortFlow()
    flow.step("transitions built", len(table))

    has_outcome = table[table[outcome_col].notna()].copy()
    flow.exclude(f"{outcome_col} undefined (a volume was never segmented)",
                 len(table) - len(has_outcome))
    finite = has_outcome[np.isfinite(
        pd.to_numeric(has_outcome[outcome_col], errors="coerce"))].copy()
    flow.exclude("non-finite outcome", len(has_outcome) - len(finite))

    if not outcome_col.startswith("next_log_ratio"):
        # The relative-change outcome needs a denominator floor; the log
        # ratio does not, because log1p keeps it finite at zero volume.
        volume = pd.to_numeric(finite["cur_enhancing__volume_mm3"],
                               errors="coerce")
        kept = finite[volume >= min_index_volume_mm3].copy()
        flow.exclude(
            f"index enhancing volume < {min_index_volume_mm3:.0f} mm^3 "
            "(segmentation-reliability floor)", len(finite) - len(kept))
        finite = kept

    flow.step("analysis set", len(finite))
    flow.step("patients in analysis set", finite["patient_id"].nunique())
    return finite.reset_index(drop=True), flow


def _usable(df: pd.DataFrame, column: str, *,
            min_present: float = 0.3, min_unique: int = 3) -> bool:
    """Drop features that are mostly missing or near-constant in this sample."""
    if column not in df.columns:
        return False
    s = pd.to_numeric(df[column], errors="coerce")
    if s.notna().mean() < min_present:
        return False
    return s.dropna().nunique() >= min_unique


# --------------------------------------------------------------------------
# model ladders
# --------------------------------------------------------------------------

def _run_rano_models(eligible, *, outcome_col, clinical, imaging, history,
                     combined, seed, n_splits, inner_splits, n_bootstrap,
                     bootstrap_seed) -> list:
    common = dict(outcome_col=outcome_col, n_splits=n_splits,
                  inner_splits=inner_splits, seed=seed,
                  n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
    out = []
    out.append(evaluate_classifier(
        eligible, feature_cols=[], model_name="prevalence_baseline",
        baseline=PrevalenceClassifier(), **common))
    out.append(evaluate_classifier(
        eligible, feature_cols=[], model_name="last_observation_baseline",
        baseline=LastObservationClassifier("index_rano_status"), **common))
    for label, cols in (("clinical_only", clinical),
                        ("imaging_only", imaging),
                        ("combined", combined)):
        if len(cols) < 2:
            continue
        out.append(evaluate_classifier(
            eligible, feature_cols=cols,
            model_name=f"{label}__elastic_net",
            estimator=elastic_net_classifier(seed),
            param_grid=CLASSIFIER_GRIDS["elastic_net"], **common))
        out.append(evaluate_classifier(
            eligible, feature_cols=cols,
            model_name=f"{label}__gradient_boosting",
            estimator=gradient_boosting_classifier(seed),
            param_grid=CLASSIFIER_GRIDS["gradient_boosting"], **common))
    return out


def _run_volume_models(eligible, *, outcome_col, clinical, imaging, history,
                       combined, seed, n_splits, inner_splits, n_bootstrap,
                       bootstrap_seed) -> list:
    common = dict(outcome_col=outcome_col, n_splits=n_splits,
                  inner_splits=inner_splits, seed=seed,
                  n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
    out = []
    out.append(evaluate_regressor(
        eligible, feature_cols=[], model_name="mean_baseline",
        baseline=MeanRegressor(), **common))
    carry_col = ("log_ratio_enhancing"
                 if outcome_col.startswith("next_log_ratio")
                 else "rel_change_enhancing")
    out.append(evaluate_regressor(
        eligible, feature_cols=[], model_name="carry_forward_baseline",
        baseline=CarryForwardRegressor(carry_col), **common))
    for label, cols in (("clinical_only", clinical),
                        ("imaging_only", imaging),
                        ("combined", combined)):
        if len(cols) < 2:
            continue
        out.append(evaluate_regressor(
            eligible, feature_cols=cols,
            model_name=f"{label}__elastic_net",
            estimator=elastic_net_regressor(seed),
            param_grid=REGRESSOR_GRIDS["elastic_net"], **common))
        out.append(evaluate_regressor(
            eligible, feature_cols=cols,
            model_name=f"{label}__gradient_boosting",
            estimator=gradient_boosting_regressor(seed),
            param_grid=REGRESSOR_GRIDS["gradient_boosting"], **common))
    return out


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _write_results(results: list, outdir: Path, *, outcome: str) -> None:
    pd.DataFrame([r.metric_row() for r in results]).to_csv(
        outdir / "model_metrics.csv", index=False)
    frames = []
    for r in results:
        if r.metric_ci is not None:
            ci = r.metric_ci.copy()
            ci.insert(0, "model", r.model)
            frames.append(ci)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            outdir / "model_metric_ci.csv", index=False)
    for r in results:
        r.predictions.to_csv(
            outdir / f"predictions__{r.model}.csv", index=False)
        if r.calibration is not None and not r.calibration.empty:
            r.calibration.to_csv(
                outdir / f"calibration__{r.model}.csv", index=False)
        if r.folds is not None:
            r.folds.to_csv(outdir / f"folds__{r.model}.csv", index=False)
        stability = summarise_selection(
            r.selected_features, len(r.folds) if r.folds is not None else 5)
        if not stability.empty:
            stability.to_csv(
                outdir / f"feature_selection__{r.model}.csv", index=False)
        if r.best_params:
            pd.DataFrame(r.best_params).to_csv(
                outdir / f"best_params__{r.model}.csv", index=False)
    notes = [{"model": r.model, "note": n} for r in results for n in r.notes]
    if notes:
        pd.DataFrame(notes).to_csv(outdir / "model_notes.csv", index=False)


def _print_results(results: list, *, outcome: str) -> None:
    rows = []
    for r in results:
        row = {"model": r.model, "n": r.n_rows, "patients": r.n_patients}
        if outcome == "rano":
            row.update({
                "AUROC": _fmt(r.metrics.get("auroc")),
                "AUPRC": _fmt(r.metrics.get("auprc")),
                "Brier": _fmt(r.metrics.get("brier")),
                "BrierSkill": _fmt(r.metrics.get("brier_skill_score")),
                "CalSlope": _fmt(r.metrics.get("calibration_slope")),
            })
            if r.metric_ci is not None:
                auroc = r.metric_ci[r.metric_ci.metric == "auroc"]
                if len(auroc):
                    row["AUROC 95% CI"] = (
                        f"{auroc.ci_low.iloc[0]:.3f}-{auroc.ci_high.iloc[0]:.3f}")
            if r.metrics.get("discrimination_defined") is False:
                row["AUROC"] = "undefined"
                row["AUROC 95% CI"] = "(constant within fold)"
        else:
            row.update({
                "MAE": _fmt(r.metrics.get("mae")),
                "MAE(mean)": _fmt(r.metrics.get("mae_baseline_mean")),
                "RMSE": _fmt(r.metrics.get("rmse")),
                "R2": _fmt(r.metrics.get("r2")),
                "Spearman": _fmt(r.metrics.get("spearman")),
            })
        rows.append(row)
    print("\n== out-of-fold performance (patient-grouped CV) ==")
    print(pd.DataFrame(rows).to_string(index=False))


def _compare_to_baseline(results: list, *, outcome: str, n_bootstrap: int,
                         seed: int) -> pd.DataFrame | None:
    baseline_name = ("last_observation_baseline" if outcome == "rano"
                     else "carry_forward_baseline")
    baseline = next((r for r in results if r.model == baseline_name), None)
    if baseline is None:
        return None
    metric_fn = (M.binary_metrics if outcome == "rano" else M.regression_metrics)
    metric_key = "auroc" if outcome == "rano" else "mae"
    pred_col = "y_prob" if outcome == "rano" else "y_pred"

    base = baseline.predictions.dropna(subset=[pred_col]).set_index(
        baseline.predictions.columns[0] if False else None) \
        if False else baseline.predictions
    rows = []
    for r in results:
        if r.model == baseline_name:
            continue
        merged = _align(r.predictions, base, pred_col)
        if merged is None or len(merged) < 20:
            continue
        diff = M.paired_bootstrap_difference(
            merged["y_true"], merged[f"{pred_col}_model"],
            merged[f"{pred_col}_base"], merged["patient_id"],
            metric_fn=metric_fn, metric_key=metric_key,
            n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"model": r.model, "versus": baseline_name, **diff})
    return pd.DataFrame(rows)


def _align(model_preds: pd.DataFrame, base_preds: pd.DataFrame,
           pred_col: str) -> pd.DataFrame | None:
    key = "index_visit_id" if "index_visit_id" in model_preds.columns else None
    if key is None:
        return None
    a = model_preds[[key, "patient_id", "y_true", pred_col]].rename(
        columns={pred_col: f"{pred_col}_model"})
    b = base_preds[[key, pred_col]].rename(columns={pred_col: f"{pred_col}_base"})
    merged = a.merge(b, on=key, how="inner").dropna(
        subset=[f"{pred_col}_model", f"{pred_col}_base", "y_true"])
    return merged


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{float(value):.3f}"
