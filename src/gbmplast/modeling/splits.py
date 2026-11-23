"""Cross-validation that groups by patient, always.

Every split in this project is patient-level.  Repeat scans, repeat tissue
samples and multiple timepoints from one patient must never straddle a fold:
with 477 transitions from 81 LUMIERE patients, a row-level split would
inflate performance badly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)


def make_splitter(
    *,
    scheme: str = "stratified_group_kfold",
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 20251123,
):
    if scheme == "stratified_group_kfold":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle,
                                    random_state=seed if shuffle else None)
    if scheme == "group_kfold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"unknown CV scheme {scheme!r}")


def feasible_n_splits(groups: pd.Series, y: pd.Series | None = None,
                      requested: int = 5) -> int:
    """Shrink the fold count when the data cannot support it.

    A stratified grouped split needs at least one group of each class per
    fold; silently producing empty folds would make the metrics meaningless.
    """
    n_groups = pd.Series(groups).nunique()
    n = min(requested, n_groups)
    if y is not None:
        y = pd.Series(y)
        per_class_groups = (pd.DataFrame({"g": pd.Series(groups).values,
                                          "y": y.values})
                            .groupby("y")["g"].nunique())
        if len(per_class_groups):
            n = min(n, int(per_class_groups.min()))
    return max(2, int(n))


def split_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    scheme: str = "stratified_group_kfold",
    n_splits: int = 5,
    seed: int = 20251123,
    stratify: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    effective = feasible_n_splits(groups, y if stratify else None, n_splits)
    if stratify and scheme == "stratified_group_kfold":
        splitter = make_splitter(scheme=scheme, n_splits=effective, seed=seed)
        folds = list(splitter.split(X, y, groups=groups))
    else:
        splitter = make_splitter(scheme="group_kfold", n_splits=effective)
        folds = list(splitter.split(X, y, groups=groups))
    return folds


def assert_group_disjoint(groups: pd.Series,
                          folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Hard check that no group appears in both sides of any split."""
    groups = pd.Series(groups).reset_index(drop=True)
    for i, (train_idx, test_idx) in enumerate(folds):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(
                f"fold {i}: {len(overlap)} group(s) appear in both train and "
                f"test, e.g. {sorted(overlap)[:3]}. Grouped CV is broken."
            )


def fold_summary(y: pd.Series, groups: pd.Series,
                 folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    rows = []
    for i, (train_idx, test_idx) in enumerate(folds):
        y_test = y.iloc[test_idx]
        rows.append({
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_patients": groups.iloc[train_idx].nunique(),
            "n_test_patients": groups.iloc[test_idx].nunique(),
            "test_positive": (int(y_test.sum())
                              if set(pd.unique(y.dropna())) <= {0, 1, 0.0, 1.0}
                              else np.nan),
            "test_positive_rate": (round(float(y_test.mean()), 4)
                                   if set(pd.unique(y.dropna())) <= {0, 1, 0.0, 1.0}
                                   else np.nan),
        })
    return pd.DataFrame(rows)


def patient_bootstrap_indices(
    groups: pd.Series,
    *,
    n_bootstrap: int = 2000,
    seed: int = 777,
) -> list[np.ndarray]:
    """Row indices for patient-level (cluster) bootstrap replicates.

    Resampling patients, not rows, is what makes the confidence intervals
    honest when a patient contributes several transitions.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    unique = groups.unique()
    by_group = {g: np.flatnonzero((groups == g).to_numpy()) for g in unique}
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        picked = rng.choice(unique, size=len(unique), replace=True)
        out.append(np.concatenate([by_group[g] for g in picked]))
    return out
