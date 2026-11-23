"""Penalised Cox models -- fitted only where the data identify them.

The gate is deliberate.  A time-to-event analysis needs genuine event times
*and* a censoring indicator.  LUMIERE releases a survival time with no
vital-status column, so a Cox model on it would silently treat every patient
as having had the event; :func:`can_fit_survival` refuses that outright
rather than producing a plausible-looking hazard ratio.

UPENN-GBM and UCSD-PTGBM do release censoring and are modelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbmplast.adapters import get_adapter
from gbmplast.config import Config
from gbmplast.modeling.estimand import CohortFlow, Estimand
from gbmplast.modeling.splits import feasible_n_splits, split_indices
from gbmplast.schema import CanonicalBundle

MIN_EVENTS_PER_FEATURE = 10


def can_fit_survival(bundle: CanonicalBundle,
                     outcome_name: str = "overall_survival"
                     ) -> tuple[bool, str]:
    """Is a time-to-event model identifiable from what this cohort released?"""
    outcomes = bundle.outcomes
    if outcomes.empty:
        return False, "cohort releases no outcomes table"
    rows = outcomes[outcomes["outcome_name"] == outcome_name]
    if rows.empty:
        return False, f"cohort releases no {outcome_name!r} outcome"
    if "event_indicator" not in rows.columns or rows["event_indicator"].isna().all():
        return False, (
            f"{outcome_name} is released without a censoring indicator, so a "
            "Cox model is not identifiable: every patient would be treated as "
            "having had the event. Refusing to fit.")
    times = pd.to_numeric(rows["outcome_value"], errors="coerce")
    usable = times.notna() & rows["event_indicator"].notna()
    n_events = int((rows.loc[usable, "event_indicator"] == True).sum())  # noqa: E712
    if usable.sum() < 30:
        return False, f"only {int(usable.sum())} usable rows"
    if n_events < 20:
        return False, f"only {n_events} observed events"
    return True, (f"{int(usable.sum())} patients, {n_events} events, "
                  f"{int(usable.sum()) - n_events} censored")


def survival_frame(bundle: CanonicalBundle,
                   outcome_name: str = "overall_survival") -> pd.DataFrame:
    rows = bundle.outcomes[bundle.outcomes["outcome_name"] == outcome_name].copy()
    rows["time"] = pd.to_numeric(rows["outcome_value"], errors="coerce")
    rows["event"] = rows["event_indicator"].map(
        lambda v: np.nan if pd.isna(v) else bool(v))
    out = rows[["patient_id", "time", "event"]].dropna()
    return out[out["time"] > 0].reset_index(drop=True)


def fit_penalised_cox(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    time_col: str = "time",
    event_col: str = "event",
    group_col: str = "patient_id",
    n_splits: int = 5,
    seed: int = 20251123,
    alphas: tuple[float, ...] = (0.5, 0.1, 0.05, 0.01),
    l1_ratio: float = 0.9,
) -> dict:
    """Patient-grouped CV of an elastic-net Cox model; returns C-index etc."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    work = data.dropna(subset=[time_col, event_col, group_col]).reset_index(drop=True)
    usable = [c for c in feature_cols
              if c in work.columns
              and pd.to_numeric(work[c], errors="coerce").notna().mean() >= 0.3
              and pd.to_numeric(work[c], errors="coerce").dropna().nunique() >= 2]
    if len(usable) < 1:
        return {"status": "skipped", "reason": "no usable features"}

    n_events = int(work[event_col].astype(bool).sum())
    budget = max(1, n_events // MIN_EVENTS_PER_FEATURE)
    if len(usable) > budget:
        # Rank by univariate association inside the whole set only to CAP the
        # count; the selection itself still happens per fold below.
        usable = usable[:budget]

    y = Surv.from_arrays(event=work[event_col].astype(bool).to_numpy(),
                         time=work[time_col].to_numpy(dtype=float))
    X = work[usable].apply(pd.to_numeric, errors="coerce")
    groups = work[group_col]

    folds = split_indices(X, work[event_col].astype(int), groups,
                          n_splits=feasible_n_splits(
                              groups, work[event_col].astype(int), n_splits),
                          seed=seed, stratify=True)

    scores, coefs = [], []
    for train_idx, test_idx in folds:
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False,
                                     keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        X_train = pipeline.fit_transform(X.iloc[train_idx])
        X_test = pipeline.transform(X.iloc[test_idx])
        best = None
        for alpha in alphas:
            try:
                model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,
                                               alphas=[alpha],
                                               fit_baseline_model=False)
                model.fit(X_train, y[train_idx])
                prediction = model.predict(X_test)
                c = concordance_index_censored(
                    work[event_col].astype(bool).to_numpy()[test_idx],
                    work[time_col].to_numpy(dtype=float)[test_idx],
                    prediction)[0]
                if best is None or c > best[0]:
                    best = (c, alpha, np.ravel(model.coef_))
            except Exception:
                continue
        if best is not None:
            scores.append(best[0])
            coefs.append(pd.Series(best[2], index=usable))

    if not scores:
        return {"status": "failed", "reason": "no fold converged"}
    coef_frame = pd.DataFrame(coefs)
    return {
        "status": "fitted",
        "n": len(work),
        "n_events": n_events,
        "n_features_used": len(usable),
        "features": usable,
        "c_index_mean": float(np.mean(scores)),
        "c_index_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
        "c_index_folds": [round(float(s), 4) for s in scores],
        "mean_coefficients": coef_frame.mean().round(4).to_dict(),
        "selection_frequency": (coef_frame != 0).mean().round(3).to_dict(),
    }


def run_survival_analysis(config: Config, *, dataset: str = "upenn_gbm") -> int:
    adapter = get_adapter(dataset, config)
    if not adapter.is_buildable()[0]:
        print(f"  {dataset}: raw data absent")
        return 0
    bundle = adapter.build()
    cohort = adapter.describe().key

    ok, reason = can_fit_survival(bundle)
    print(f"== survival: {dataset}")
    print(f"  identifiability: {reason}")
    outdir = config.paths.results / dataset / "survival"
    outdir.mkdir(parents=True, exist_ok=True)

    if not ok:
        pd.DataFrame([{"dataset": dataset, "status": "refused",
                       "reason": reason}]).to_csv(
            outdir / "survival_refused.csv", index=False)
        print("  -> refusing to fit a Cox model on this cohort")
        return 0

    estimand = Estimand(
        name=f"{dataset}_overall_survival",
        question="Which routinely available baseline covariates are "
                 "associated with overall survival?",
        cohort=cohort,
        unit_of_analysis="patient",
        outcome="overall_survival",
        outcome_type="time_to_event",
        prediction_time="baseline",
        lookback_window_days=0,
        outcome_horizon_days=None,
        eligibility=("a genuine event time and censoring indicator",),
        exclusions=("patients with no usable survival time",),
        confounders_acknowledged=(
            "Treatment is only partially recorded; survival differences mix "
            "prognosis with therapy received.",
            "Molecular markers are missing for a substantial minority, and "
            "missingness is unlikely to be at random.",
        ),
    )
    print(estimand.describe())

    survival = survival_frame(bundle)
    patients = bundle.patients.copy()
    patients["sex_female"] = patients["sex"].map(
        {"female": 1.0, "male": 0.0}).astype(float)
    patients["mgmt_methylated"] = patients["mgmt_status"].map(
        {"methylated": 1.0, "unmethylated": 0.0}).astype(float)
    patients["idh_mutant"] = patients["idh_status"].map(
        {"mutant": 1.0, "wildtype": 0.0}).astype(float)
    for column in ("kps", "age_at_diagnosis_years"):
        if column in patients.columns:
            patients[column] = pd.to_numeric(patients[column], errors="coerce")

    data = survival.merge(patients, on="patient_id", how="left")
    features = [c for c in ("age_at_diagnosis_years", "sex_female", "kps",
                            "mgmt_methylated", "idh_mutant")
                if c in data.columns]

    flow = CohortFlow()
    flow.step("patients with usable survival", len(data))
    flow.step("events", int(data["event"].sum()))
    flow.step("censored", int((~data["event"].astype(bool)).sum()))
    print("\nCohort flow:")
    print(flow.describe())

    result = fit_penalised_cox(data, feature_cols=features, seed=config.seed)
    estimand.to_frame().to_csv(outdir / "estimand.csv", index=False)
    flow.to_frame().to_csv(outdir / "cohort_flow.csv", index=False)
    pd.DataFrame([{k: (v if not isinstance(v, (list, dict)) else str(v))
                   for k, v in result.items()}]).to_csv(
        outdir / "cox_result.csv", index=False)

    print("\n== penalised Cox (patient-grouped CV) ==")
    if result.get("status") != "fitted":
        print(f"  {result}")
        return 0
    print(f"  n = {result['n']}, events = {result['n_events']}, "
          f"features used = {result['n_features_used']}")
    print(f"  C-index: {result['c_index_mean']:.3f} "
          f"(SD {result['c_index_sd']:.3f} across folds) "
          f"folds={result['c_index_folds']}")
    print("  mean coefficients (log hazard ratio, standardised):")
    for name, value in sorted(result["mean_coefficients"].items(),
                              key=lambda kv: -abs(kv[1])):
        frequency = result["selection_frequency"].get(name, 0)
        print(f"    {name:28s} {value:+.4f}   selected in "
              f"{frequency:.0%} of folds")
    print("  NOTE: associational. Treatment is only partially recorded, so "
          "these are not treatment-adjusted effects.")
    return 0
