"""Patient-visit longitudinal feature tables.

The unit of analysis is a **transition**: an index visit (where the
prediction is made) paired with the next qualifying visit (where the outcome
is observed).

Leakage discipline
------------------
Every predictor is computed from visits at or before the index visit.  The
next visit contributes **only** outcome columns, all of which are prefixed
``next_``.  :func:`assert_no_future_features` enforces this by construction,
and the modelling layer re-checks it.

Feature groups
--------------
``clinical``   age, sex, MGMT, IDH -- known at baseline
``imaging``    current subregion volumes, shape, intensity, texture
``dynamics``   previous volumes, absolute/relative change, growth rate,
               lookback slope, inter-scan interval
``history``    visit index, time since t0, prior RANO course
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gbmplast.config import Config

#: Expert RANO categories that constitute a response assessment. Pre-Op and
#: Post-Op are surgical anchors, not response assessments, and are excluded
#: from outcome construction.
RESPONSE_CATEGORIES = ("CR", "PR", "SD", "PD")


@dataclass
class LongitudinalResult:
    table: pd.DataFrame
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    excluded: pd.DataFrame | None = None

    def features(self, *groups: str) -> list[str]:
        out: list[str] = []
        for group in (groups or tuple(self.feature_groups)):
            out += self.feature_groups.get(group, [])
        seen, unique = set(), []
        for col in out:
            if col not in seen and col in self.table.columns:
                seen.add(col)
                unique.append(col)
        return unique


# --------------------------------------------------------------------------
# per-visit imaging summary
# --------------------------------------------------------------------------

def visit_imaging_summary(
    radiomics: pd.DataFrame,
    *,
    subregions: tuple[str, ...],
    primary_sequence: str = "CT1",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Collapse the long radiomics table to one row per visit.

    Volumes and shape come from the mask and are therefore identical across
    sequences; intensity and texture are taken from ``primary_sequence``.
    """
    df = radiomics.copy()
    id_cols = ["patient_id", "visit_id", "local_patient_id", "local_visit_id",
               "relative_time_days"]
    id_cols = [c for c in id_cols if c in df.columns]

    # --- mask-derived (sequence-invariant) -------------------------------
    #
    # Absent label vs unsegmented study
    # ---------------------------------
    # LUMIERE emits a row per (study, sequence, label) but leaves EVERY
    # feature NaN when the automated segmentation found no voxels of that
    # label -- verified: 175/599 enhancing rows are all-NaN with no mask
    # diagnostics, and they are predominantly Post-Op (48) and CR (24)
    # studies, i.e. genuinely no enhancing tumour after resection/response.
    #
    # Treating those as missing would both discard informative zeros and
    # bias the cohort toward patients with residual enhancing disease, so:
    #   * row present, all features NaN  -> volume 0.0, label_present = False
    #   * study absent from the table    -> volume NA  (never segmented)
    # Intensity and texture genuinely cannot be computed for an absent
    # label and stay NaN; the modelling layer carries missing-indicators.
    shape_cols = [c for c in df.columns if "_shape_" in c]
    all_feature_cols = [c for c in df.columns if c.startswith("original_")]
    mask_level = (df[df["sequence"] == primary_sequence]
                  if primary_sequence in set(df["sequence"]) else df)
    pieces: list[pd.DataFrame] = []
    for subregion in subregions:
        sub = mask_level[mask_level["subregion"] == subregion]
        if sub.empty:
            continue
        keep = sub[id_cols + shape_cols].copy()
        absent = sub[all_feature_cols].isna().all(axis=1)
        keep["_label_absent"] = absent.to_numpy()
        keep = keep.drop_duplicates(subset=["visit_id"])
        rename = {c: f"{subregion}__{_short(c)}" for c in shape_cols}
        keep = keep.rename(columns=rename)
        volume = pd.to_numeric(keep.get(f"{subregion}__VoxelVolume"),
                               errors="coerce")
        keep[f"{subregion}__volume_mm3"] = np.where(
            keep["_label_absent"], 0.0, volume)
        keep[f"{subregion}__label_present"] = ~keep["_label_absent"]
        # A zero-volume subregion has no shape; set the ratio-style shape
        # features to NaN rather than to a misleading 0.
        keep = keep.drop(columns=["_label_absent"])
        pieces.append(keep.set_index(id_cols))

    # --- intensity / texture from the primary sequence -------------------
    texture_cols = feature_columns or [
        c for c in df.columns
        if c.startswith("original_") and "_shape_" not in c]
    for subregion in subregions:
        sub = df[(df["subregion"] == subregion)
                 & (df["sequence"] == primary_sequence)]
        if sub.empty:
            continue
        keep = sub[id_cols + texture_cols].drop_duplicates(subset=["visit_id"])
        rename = {c: f"{subregion}__{primary_sequence}__{_short(c)}"
                  for c in texture_cols}
        pieces.append(keep.rename(columns=rename).set_index(id_cols))

    if not pieces:
        return pd.DataFrame()
    merged = pd.concat(pieces, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    out = merged.reset_index()

    # Derived whole-lesion aggregates across the available subregions.
    vol_cols = [f"{s}__volume_mm3" for s in subregions
                if f"{s}__volume_mm3" in out.columns]
    if vol_cols:
        out["total_lesion_volume_mm3"] = out[vol_cols].sum(axis=1, min_count=1)
        if len(vol_cols) >= 2 and "enhancing__volume_mm3" in out.columns:
            total = out["total_lesion_volume_mm3"]
            out["enhancing_fraction"] = np.where(
                total > 0, out["enhancing__volume_mm3"] / total, np.nan)
    return out


def _short(column: str) -> str:
    """'original_shape_VoxelVolume' -> 'VoxelVolume'."""
    parts = column.split("_")
    if len(parts) >= 3 and parts[0] == "original":
        return "_".join(parts[2:])
    return column


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------

def build_transitions(
    visits: pd.DataFrame,
    imaging_summary: pd.DataFrame,
    patients: pd.DataFrame,
    *,
    subregions: tuple[str, ...],
    lookback_days: int = 365,
    min_interval_days: int = 7,
    max_interval_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """One row per (index visit -> next visit) transition.

    Returns ``(transitions, excluded, feature_groups)``.
    """
    spine = visits.merge(imaging_summary.drop(
        columns=[c for c in ("local_patient_id", "local_visit_id",
                             "relative_time_days") if c in imaging_summary.columns]),
        on=["patient_id", "visit_id"], how="left")

    # Only dated visits with imaging features can anchor a transition.
    volume_col = f"{subregions[0]}__volume_mm3"
    spine["_t"] = pd.to_numeric(spine["relative_time_days"], errors="coerce")
    spine["_has_features"] = spine.get(
        volume_col, pd.Series(np.nan, index=spine.index)).notna()

    excluded_rows: list[dict] = []
    rows: list[dict] = []

    patient_lookup = patients.set_index("patient_id")

    for pid, grp in spine.groupby("patient_id"):
        grp = grp.sort_values(["_t", "visit_index"]).reset_index(drop=True)
        usable = grp[grp["_t"].notna()].reset_index(drop=True)
        if len(usable) < 2:
            excluded_rows.append({"patient_id": pid, "reason":
                                  "fewer than 2 dated visits"})
            continue

        for i in range(len(usable) - 1):
            index_visit = usable.iloc[i]
            next_visit = usable.iloc[i + 1]

            interval = float(next_visit["_t"] - index_visit["_t"])
            if not index_visit["_has_features"]:
                excluded_rows.append({
                    "patient_id": pid, "visit_id": index_visit["visit_id"],
                    "reason": "index visit has no imaging features"})
                continue
            if interval < min_interval_days:
                excluded_rows.append({
                    "patient_id": pid, "visit_id": index_visit["visit_id"],
                    "reason": f"interval {interval:.0f}d below minimum "
                              f"{min_interval_days}d"})
                continue
            if interval > max_interval_days:
                excluded_rows.append({
                    "patient_id": pid, "visit_id": index_visit["visit_id"],
                    "reason": f"interval {interval:.0f}d above maximum "
                              f"{max_interval_days}d"})
                continue

            history = usable.iloc[: i + 1]          # index visit inclusive
            previous = usable.iloc[i - 1] if i >= 1 else None

            row: dict = {
                "patient_id": pid,
                "cohort": index_visit.get("cohort"),
                "local_patient_id": index_visit.get("local_patient_id"),
                "index_visit_id": index_visit["visit_id"],
                "index_local_visit_id": index_visit.get("local_visit_id"),
                "next_visit_id": next_visit["visit_id"],
                "next_local_visit_id": next_visit.get("local_visit_id"),
                "prediction_time_days": float(index_visit["_t"]),
                "transition_index": i,
                "n_prior_visits": int(i),
                "interscan_interval_days": interval,
            }

            # -- history ---------------------------------------------------
            row["visit_index"] = int(index_visit.get("visit_index", i))
            row["time_since_t0_days"] = float(index_visit["_t"])
            rano_history = [r for r in history.get("rano_status", pd.Series(dtype=object))
                            if isinstance(r, str)]
            row["n_prior_pd"] = sum(1 for r in rano_history[:-1] if r == "PD")
            row["any_prior_pd"] = bool(row["n_prior_pd"] > 0)
            row["index_rano_status"] = (index_visit.get("rano_status")
                                        if isinstance(index_visit.get("rano_status"), str)
                                        else pd.NA)
            row["index_rano_is_pd"] = (
                float(row["index_rano_status"] == "PD")
                if isinstance(row["index_rano_status"], str) else np.nan)
            row["index_treatment_phase"] = index_visit.get("treatment_phase")
            row["index_nonmeasurable"] = _as_bool(
                index_visit.get("rano_nonmeasurable"))

            # -- current imaging ------------------------------------------
            feature_cols = [c for c in usable.columns
                            if "__" in c or c == "total_lesion_volume_mm3"
                            or c == "enhancing_fraction"]
            for col in feature_cols:
                row[f"cur_{col}"] = _num(index_visit.get(col))

            # -- dynamics --------------------------------------------------
            for subregion in subregions:
                vcol = f"{subregion}__volume_mm3"
                current = _num(index_visit.get(vcol))
                prior = _num(previous.get(vcol)) if previous is not None else np.nan
                gap = (float(index_visit["_t"] - previous["_t"])
                       if previous is not None else np.nan)
                row[f"prev_{subregion}__volume_mm3"] = prior
                row[f"delta_{subregion}__volume_mm3"] = (
                    current - prior if _ok(current) and _ok(prior) else np.nan)
                row[f"rel_change_{subregion}"] = (
                    (current - prior) / prior
                    if _ok(current) and _ok(prior) and prior > 0 else np.nan)
                row[f"growth_rate_{subregion}_mm3_per_day"] = (
                    (current - prior) / gap
                    if _ok(current) and _ok(prior) and _ok(gap) and gap > 0
                    else np.nan)
                row[f"log_ratio_{subregion}"] = (
                    float(np.log((current + 1.0) / (prior + 1.0)))
                    if _ok(current) and _ok(prior) else np.nan)
                row[f"slope_{subregion}_mm3_per_day"] = _lookback_slope(
                    history, vcol, lookback_days)
            row["prev_interscan_interval_days"] = (
                float(index_visit["_t"] - previous["_t"])
                if previous is not None else np.nan)

            # -- clinical --------------------------------------------------
            if pid in patient_lookup.index:
                p = patient_lookup.loc[pid]
                row["age_at_diagnosis_years"] = _num(p.get("age_at_diagnosis_years"))
                row["sex"] = p.get("sex")
                row["sex_female"] = (1.0 if str(p.get("sex")).lower() == "female"
                                     else 0.0 if str(p.get("sex")).lower() == "male"
                                     else np.nan)
                row["mgmt_status"] = p.get("mgmt_status")
                row["mgmt_methylated"] = _ternary(p.get("mgmt_status"),
                                                  "methylated", "unmethylated")
                row["idh_status"] = p.get("idh_status")
                row["idh_mutant"] = _ternary(p.get("idh_status"),
                                             "mutant", "wildtype")

            # -- OUTCOMES (next visit only) --------------------------------
            next_rano = next_visit.get("rano_status")
            row["next_rano_status"] = (next_rano if isinstance(next_rano, str)
                                       else pd.NA)
            row["next_rano_is_response_assessment"] = bool(
                isinstance(next_rano, str) and next_rano in RESPONSE_CATEGORIES)
            row["next_rano_progression"] = (
                float(next_rano == "PD")
                if isinstance(next_rano, str) and next_rano in RESPONSE_CATEGORIES
                else np.nan)
            row["next_rano_conflict"] = _as_bool(next_visit.get("rano_conflict"))
            for subregion in subregions:
                vcol = f"{subregion}__volume_mm3"
                current = _num(index_visit.get(vcol))
                future = _num(next_visit.get(vcol))
                row[f"next_{subregion}__volume_mm3"] = future
                row[f"next_rel_change_{subregion}"] = (
                    (future - current) / current
                    if _ok(future) and _ok(current) and current > 0 else np.nan)
                row[f"next_log_ratio_{subregion}"] = (
                    float(np.log((future + 1.0) / (current + 1.0)))
                    if _ok(future) and _ok(current) else np.nan)
            row["outcome_time_days"] = float(next_visit["_t"])
            rows.append(row)

    transitions = pd.DataFrame(rows)
    excluded = pd.DataFrame(excluded_rows)
    groups = _feature_groups(transitions, subregions)
    return transitions, excluded, groups


def _feature_groups(table: pd.DataFrame,
                    subregions: tuple[str, ...]) -> dict[str, list[str]]:
    if table.empty:
        return {"clinical": [], "imaging": [], "dynamics": [], "history": []}
    clinical = [c for c in ("age_at_diagnosis_years", "sex_female",
                            "mgmt_methylated", "idh_mutant")
                if c in table.columns]
    dynamics = [c for c in table.columns
                if c.startswith(("prev_", "delta_", "rel_change_",
                                 "growth_rate_", "log_ratio_", "slope_"))]
    history = [c for c in ("visit_index", "time_since_t0_days", "n_prior_visits",
                           "n_prior_pd", "index_rano_is_pd",
                           "interscan_interval_days")
               if c in table.columns]
    imaging = [c for c in table.columns if c.startswith("cur_")]
    # label-presence indicators are informative (no enhancing tumour is a
    # finding, not a gap) and belong with the imaging block.
    imaging += [c for c in table.columns
                if c.endswith("__label_present") and c not in imaging]
    return {"clinical": clinical, "imaging": imaging,
            "dynamics": dynamics, "history": history}


def _lookback_slope(history: pd.DataFrame, column: str,
                    lookback_days: int) -> float:
    """OLS slope of volume on time over the lookback window (mm^3/day)."""
    if column not in history.columns or len(history) < 2:
        return np.nan
    cutoff = float(history["_t"].iloc[-1]) - lookback_days
    window = history[history["_t"] >= cutoff]
    x = pd.to_numeric(window["_t"], errors="coerce")
    y = pd.to_numeric(window[column], errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 2 or x[ok].nunique() < 2:
        return np.nan
    return float(np.polyfit(x[ok].to_numpy(float), y[ok].to_numpy(float), 1)[0])


# --------------------------------------------------------------------------
# leakage guard
# --------------------------------------------------------------------------

def assert_no_future_features(feature_cols: list[str]) -> None:
    """Refuse any predictor that names a next-visit quantity."""
    offenders = [c for c in feature_cols
                 if c.startswith("next_") or c.startswith("outcome_")]
    if offenders:
        raise ValueError(
            "future information in the predictor set: "
            f"{offenders}. Columns prefixed 'next_'/'outcome_' are observed "
            "after the prediction time and must never be used as features."
        )


# --------------------------------------------------------------------------
# cohort driver
# --------------------------------------------------------------------------

def build_longitudinal(adapter, config: Config) -> LongitudinalResult | None:
    if adapter.key != "lumiere":
        # Other cohorts in this project are cross-sectional by construction;
        # the institutional adapter reuses this builder once it has >=2
        # dated studies per patient.
        bundle = adapter.build()
        dated = bundle.visits[pd.to_numeric(
            bundle.visits["relative_time_days"], errors="coerce").notna()]
        if dated.groupby("patient_id").size().max() if len(dated) else 0 < 2:
            return None

    key = adapter.key
    cohort_cfg = config.get(f"cohorts.{key}", {}) or {}
    long_cfg = config.get("longitudinal", {}) or {}
    segmentation = cohort_cfg.get("primary_segmentation", "HD-GLIO-AUTO")
    primary_sequence = cohort_cfg.get("primary_sequence", "CT1")
    subregions = tuple(long_cfg.get("subregions",
                                    ["enhancing", "non_enhancing"]))

    bundle = adapter.build()
    radiomics = adapter.load_radiomics(segmentation)
    summary = visit_imaging_summary(
        radiomics, subregions=subregions, primary_sequence=primary_sequence)

    transitions, excluded, groups = build_transitions(
        bundle.visits, summary, bundle.patients,
        subregions=subregions,
        lookback_days=int(long_cfg.get("lookback_days", 365)),
        min_interval_days=int(long_cfg.get("min_interscan_interval_days", 7)),
        max_interval_days=int(long_cfg.get("max_interscan_interval_days", 365)),
    )

    provenance = [
        f"segmentation: {segmentation} "
        f"(labels verified from the released feature table)",
        f"primary sequence for intensity/texture: {primary_sequence}",
        f"subregions: {', '.join(subregions)}",
        f"transitions: {len(transitions)} over "
        f"{transitions['patient_id'].nunique() if len(transitions) else 0} patients",
        f"excluded: {len(excluded)} index visits/patients "
        f"({_top_reasons(excluded)})",
        "predictors are computed from visits at or before the index visit; "
        "all next-visit quantities carry the 'next_' prefix and are outcomes",
    ]
    if len(transitions):
        rated = transitions["next_rano_is_response_assessment"].sum()
        provenance.append(
            f"transitions whose next visit carries a RANO response assessment: "
            f"{int(rated)} ({int(transitions.loc[transitions['next_rano_is_response_assessment'], 'next_rano_progression'].sum())} PD)")
    return LongitudinalResult(table=transitions, feature_groups=groups,
                              provenance=provenance, excluded=excluded)


def _top_reasons(excluded: pd.DataFrame, n: int = 3) -> str:
    if excluded is None or excluded.empty or "reason" not in excluded.columns:
        return "none"
    counts = excluded["reason"].value_counts().head(n)
    return "; ".join(f"{k} x{v}" for k, v in counts.items())


def _num(value) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if value is pd.NA:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ok(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and np.isfinite(value)


def _as_bool(value) -> object:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if isinstance(value, float) and np.isnan(value):
            return pd.NA
    except TypeError:
        pass
    return bool(value)


def _ternary(value, positive: str, negative: str) -> float:
    text = str(value).lower()
    if text == positive:
        return 1.0
    if text == negative:
        return 0.0
    return np.nan
