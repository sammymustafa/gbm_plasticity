"""Descriptive hierarchical / mixed-effects models of longitudinal change.

These are *descriptive*, not predictive: they estimate how log tumour volume
changes with time since surgery while allowing each patient their own
intercept (and, where estimable, their own slope).  A patient-level random
effect is what keeps 477 correlated observations from being treated as 477
independent ones.

No causal reading is available: treatment is unrecorded in LUMIERE, so a
time coefficient mixes natural history with the effect of whatever therapy
the patient received.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_volume_trajectory(
    table: pd.DataFrame,
    *,
    subregion: str = "enhancing",
    random_slope: bool = True,
) -> dict:
    """Mixed model: log1p(volume) ~ time + (1 [+ time] | patient).

    Fitted on the index-visit volumes of the transition table, which is one
    row per (patient, visit) pair by construction.
    """
    import statsmodels.formula.api as smf

    volume_col = f"cur_{subregion}__volume_mm3"
    needed = {volume_col, "time_since_t0_days", "patient_id"}
    if not needed.issubset(table.columns):
        return {"subregion": subregion, "status": "skipped",
                "reason": f"missing columns {sorted(needed - set(table.columns))}"}

    df = table[[volume_col, "time_since_t0_days", "patient_id"]].copy()
    df.columns = ["volume", "time_days", "patient"]
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df = df.dropna()
    if len(df) < 30 or df["patient"].nunique() < 10:
        return {"subregion": subregion, "status": "skipped",
                "reason": f"insufficient data (n={len(df)}, "
                          f"patients={df['patient'].nunique()})"}

    df["log_volume"] = np.log1p(df["volume"])
    df["time_months"] = df["time_days"] / 30.44

    try:
        if random_slope:
            model = smf.mixedlm("log_volume ~ time_months", df,
                                groups=df["patient"],
                                re_formula="~time_months")
            fit = model.fit(method="lbfgs", maxiter=500, reml=True)
        else:
            raise RuntimeError("random intercept requested")
    except Exception:
        try:
            model = smf.mixedlm("log_volume ~ time_months", df,
                                groups=df["patient"])
            fit = model.fit(method="lbfgs", maxiter=500, reml=True)
            random_slope = False
        except Exception as exc:
            return {"subregion": subregion, "status": "failed",
                    "reason": str(exc)[:200]}

    params = fit.params
    conf = fit.conf_int()
    slope_name = "time_months"
    out = {
        "subregion": subregion,
        "status": "fitted",
        "model": ("log1p(volume) ~ time_months + (1 + time_months | patient)"
                  if random_slope else
                  "log1p(volume) ~ time_months + (1 | patient)"),
        "n_observations": len(df),
        "n_patients": int(df["patient"].nunique()),
        "intercept": float(params.get("Intercept", np.nan)),
        "slope_per_month_log1p": float(params.get(slope_name, np.nan)),
        "slope_ci_low": (float(conf.loc[slope_name, 0])
                         if slope_name in conf.index else np.nan),
        "slope_ci_high": (float(conf.loc[slope_name, 1])
                          if slope_name in conf.index else np.nan),
        "slope_p_value": (float(fit.pvalues.get(slope_name, np.nan))),
        "group_var": float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else np.nan,
        "residual_var": float(fit.scale),
        "converged": bool(getattr(fit, "converged", True)),
    }
    # Fold change per month on the volume scale, for readability.
    if np.isfinite(out["slope_per_month_log1p"]):
        out["approx_pct_change_per_month"] = round(
            100.0 * (np.exp(out["slope_per_month_log1p"]) - 1.0), 2)
    out["interpretation"] = (
        "Descriptive only. Treatment is unrecorded in LUMIERE, so the time "
        "coefficient mixes natural history with therapy effects and must not "
        "be read causally."
    )
    return out


def fit_change_by_marker(
    table: pd.DataFrame,
    *,
    subregion: str = "enhancing",
    marker: str = "mgmt_methylated",
) -> dict:
    """Does the trajectory slope differ by a baseline molecular marker?

    Fits an interaction with a patient random intercept.  Reported as an
    association, with the marker's missingness stated.
    """
    import statsmodels.formula.api as smf

    volume_col = f"cur_{subregion}__volume_mm3"
    needed = {volume_col, "time_since_t0_days", "patient_id", marker}
    if not needed.issubset(table.columns):
        return {"subregion": subregion, "marker": marker, "status": "skipped",
                "reason": f"missing columns {sorted(needed - set(table.columns))}"}

    df = table[[volume_col, "time_since_t0_days", "patient_id", marker]].copy()
    df.columns = ["volume", "time_days", "patient", "marker"]
    for col in ("volume", "time_days", "marker"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna()
    if len(df) < 40 or df["patient"].nunique() < 15 or df["marker"].nunique() < 2:
        return {"subregion": subregion, "marker": marker, "status": "skipped",
                "reason": f"insufficient data after dropping missing marker "
                          f"(n={len(df)} of {n_before}, "
                          f"patients={df['patient'].nunique()})"}
    df["log_volume"] = np.log1p(df["volume"])
    df["time_months"] = df["time_days"] / 30.44
    try:
        fit = smf.mixedlm("log_volume ~ time_months * marker", df,
                          groups=df["patient"]).fit(method="lbfgs", maxiter=500)
    except Exception as exc:
        return {"subregion": subregion, "marker": marker, "status": "failed",
                "reason": str(exc)[:200]}
    term = "time_months:marker"
    conf = fit.conf_int()
    return {
        "subregion": subregion,
        "marker": marker,
        "status": "fitted",
        "model": "log1p(volume) ~ time_months * marker + (1 | patient)",
        "n_observations": len(df),
        "n_observations_dropped_missing_marker": int(n_before - len(df)),
        "n_patients": int(df["patient"].nunique()),
        "interaction_estimate": float(fit.params.get(term, np.nan)),
        "interaction_ci_low": (float(conf.loc[term, 0])
                               if term in conf.index else np.nan),
        "interaction_ci_high": (float(conf.loc[term, 1])
                                if term in conf.index else np.nan),
        "interaction_p_value": float(fit.pvalues.get(term, np.nan)),
        "interpretation": (
            "Association only, on the subset with a non-missing marker; not "
            "adjusted for treatment, which is unrecorded."),
    }


def run_descriptive_models(table: pd.DataFrame,
                           outdir: Path | None = None) -> pd.DataFrame | None:
    rows = []
    for subregion in ("enhancing", "non_enhancing"):
        rows.append(fit_volume_trajectory(table, subregion=subregion))
    for marker in ("mgmt_methylated", "idh_mutant"):
        rows.append(fit_change_by_marker(table, marker=marker))
    out = pd.DataFrame(rows)
    if outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out.to_csv(Path(outdir) / "mixed_effects_models.csv", index=False)
    cols = [c for c in ("subregion", "marker", "status", "n_observations",
                        "n_patients", "slope_per_month_log1p",
                        "approx_pct_change_per_month", "slope_ci_low",
                        "slope_ci_high", "slope_p_value",
                        "interaction_estimate", "interaction_p_value",
                        "reason")
            if c in out.columns]
    return out[cols]
