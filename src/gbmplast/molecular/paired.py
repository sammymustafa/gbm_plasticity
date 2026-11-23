"""Paired within-patient statistics for primary vs recurrent specimens.

The unit of inference is the **patient**, not the nucleus.  GBM-CARE
contributes 56 patients with a matched primary and first-recurrent sample;
each sample's thousands of nuclei are collapsed to one pseudobulk profile
first, so the paired tests have 56 degrees of freedom, not hundreds of
thousands.

Reported for every programme
----------------------------
* the paired difference (recurrent - primary) per patient
* a paired effect size with a confidence interval
* both a parametric (paired t) and a rank-based (Wilcoxon) p-value
* Benjamini-Hochberg FDR across the programmes tested
* the number of patients actually contributing, which varies by programme
  when a sample's malignant compartment was too small to profile
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_table(
    scores: pd.DataFrame,
    *,
    patient_col: str = "local_patient_id",
    role_col: str = "sample_role",
    primary: str = "primary",
    recurrent: str = "first_recurrence",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Wide table: one row per patient, ``<col>_primary`` / ``<col>_recurrent``."""
    value_cols = value_cols or [
        c for c in scores.columns
        if c not in {patient_col, role_col, "accession", "specimen_id",
                     "visit_id", "patient_id", "cohort", "site"}
        and pd.api.types.is_numeric_dtype(scores[c])]

    a = scores[scores[role_col] == primary].set_index(patient_col)[value_cols]
    b = scores[scores[role_col] == recurrent].set_index(patient_col)[value_cols]
    shared = sorted(set(a.index) & set(b.index))
    out = pd.DataFrame(index=shared)
    for col in value_cols:
        out[f"{col}__primary"] = a.loc[shared, col]
        out[f"{col}__recurrent"] = b.loc[shared, col]
        out[f"{col}__delta"] = out[f"{col}__recurrent"] - out[f"{col}__primary"]
    out.index.name = patient_col
    return out.reset_index()


def paired_effects(
    paired: pd.DataFrame,
    value_cols: list[str],
    *,
    patient_col: str = "local_patient_id",
    n_bootstrap: int = 2000,
    seed: int = 777,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired effect sizes with CIs, two tests, and BH-FDR across programmes."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    rows = []
    for col in value_cols:
        delta_col = f"{col}__delta"
        if delta_col not in paired.columns:
            continue
        delta = pd.to_numeric(paired[delta_col], errors="coerce")
        delta = delta[np.isfinite(delta)]
        n = len(delta)
        if n < 5:
            rows.append({"programme": col, "n_patients": n,
                         "status": "skipped",
                         "reason": f"only {n} complete pairs"})
            continue

        values = delta.to_numpy(dtype=float)
        mean_delta = float(values.mean())
        sd = float(values.std(ddof=1))
        # Cohen's dz for a paired design.
        dz = mean_delta / sd if sd > 0 else np.nan

        try:
            t_stat, t_p = stats.ttest_rel(
                pd.to_numeric(paired[f"{col}__recurrent"], errors="coerce"),
                pd.to_numeric(paired[f"{col}__primary"], errors="coerce"),
                nan_policy="omit")
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = stats.wilcoxon(values, alternative="two-sided",
                                         zero_method="wilcox")
        except Exception:
            w_stat, w_p = np.nan, np.nan

        # Patient-level bootstrap CI for the mean paired difference.
        draws = np.array([
            rng.choice(values, size=n, replace=True).mean()
            for _ in range(n_bootstrap)])
        rows.append({
            "programme": col,
            "n_patients": n,
            "status": "tested",
            "mean_primary": float(pd.to_numeric(
                paired[f"{col}__primary"], errors="coerce").mean()),
            "mean_recurrent": float(pd.to_numeric(
                paired[f"{col}__recurrent"], errors="coerce").mean()),
            "mean_delta": mean_delta,
            "sd_delta": sd,
            "cohens_dz": dz,
            "delta_ci_low": float(np.quantile(draws, alpha / 2)),
            "delta_ci_high": float(np.quantile(draws, 1 - alpha / 2)),
            "n_increased": int((values > 0).sum()),
            "n_decreased": int((values < 0).sum()),
            "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        })

    out = pd.DataFrame(rows)
    tested = out["status"] == "tested" if "status" in out.columns else pd.Series(dtype=bool)
    if tested.any():
        for source, target in (("wilcoxon_p", "wilcoxon_fdr"),
                               ("paired_t_p", "paired_t_fdr")):
            p = out.loc[tested, source]
            out.loc[tested, target] = _benjamini_hochberg(p.to_numpy())
    return out


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up FDR. NaN p-values pass through as NaN."""
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    values = p[ok]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(m)
    result[order] = adjusted
    out[ok] = result
    return out


def state_switch_table(
    scores: pd.DataFrame,
    *,
    patient_col: str = "local_patient_id",
    role_col: str = "sample_role",
    state_col: str = "dominant_state",
    primary: str = "primary",
    recurrent: str = "first_recurrence",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary -> recurrent dominant-state transitions.

    A caution the brief asks for explicitly: a change in which signature
    scores highest is **not** evidence that individual cells converted from
    one lineage to another.  It is a change in the bulk composition of the
    malignant compartment, which can arise from selection, from a shift in
    the microenvironment, or from sampling a different part of the tumour.
    """
    a = scores[scores[role_col] == primary].set_index(patient_col)[state_col]
    b = scores[scores[role_col] == recurrent].set_index(patient_col)[state_col]
    shared = sorted(set(a.index) & set(b.index))
    pairs = pd.DataFrame({
        patient_col: shared,
        "primary_state": a.loc[shared].to_numpy(),
        "recurrent_state": b.loc[shared].to_numpy(),
    })
    pairs["switched"] = pairs["primary_state"] != pairs["recurrent_state"]
    matrix = pd.crosstab(pairs["primary_state"], pairs["recurrent_state"])
    matrix.attrs["caution"] = (
        "Transitions describe the dominant signature of the malignant "
        "pseudobulk, not lineage conversion of individual cells."
    )
    return pairs, matrix


def baseline_predictors_of_recurrence(
    paired: pd.DataFrame,
    *,
    outcome: str,
    predictors: list[str],
    min_patients_per_predictor: int = 10,
) -> pd.DataFrame:
    """Do primary-sample programmes predict the recurrent state?

    Deliberately conservative: with ~56 patients this is explored one
    predictor at a time with a rank correlation, never as a multivariable
    model, and the module refuses outright when the sample would not support
    even that.
    """
    from scipy import stats

    outcome_col = f"{outcome}__recurrent"
    if outcome_col not in paired.columns:
        return pd.DataFrame([{"status": "skipped",
                              "reason": f"{outcome_col} absent"}])
    y = pd.to_numeric(paired[outcome_col], errors="coerce")
    rows = []
    budget = len(y.dropna()) // min_patients_per_predictor
    if budget < 1:
        return pd.DataFrame([{
            "status": "refused",
            "reason": (f"{int(y.notna().sum())} patients support no "
                       f"predictor at {min_patients_per_predictor} per "
                       "predictor")}])
    for name in predictors:
        col = f"{name}__primary"
        if col not in paired.columns:
            continue
        x = pd.to_numeric(paired[col], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < min_patients_per_predictor:
            rows.append({"predictor": name, "n": int(ok.sum()),
                         "status": "skipped",
                         "reason": "too few complete pairs"})
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"predictor": name, "outcome": outcome_col,
                     "n": int(ok.sum()), "status": "tested",
                     "spearman_rho": float(rho), "p_value": float(p)})
    out = pd.DataFrame(rows)
    if "status" in out.columns:
        tested = out["status"] == "tested"
        if tested.any():
            out.loc[tested, "fdr"] = _benjamini_hochberg(
                out.loc[tested, "p_value"].to_numpy())
    out.attrs["caution"] = (
        "Univariable exploration only. These are associations within one "
        "cohort of ~56 patients, not validated predictors."
    )
    return out
