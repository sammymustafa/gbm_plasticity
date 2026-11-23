"""Model zoo: baselines first, then regularised linear, then trees.

The ordering is deliberate.  A clinical prediction claim is only interesting
relative to a naive rule, so every analysis fits:

1. ``last_observation``   -- carry the current state forward. For next-visit
   RANO this means "predict PD iff the current visit is PD"; for volume
   change it means "predict the same relative change as last interval".
2. ``prevalence``         -- predict the training prevalence for everyone.
3. ``clinical_only``      -- age, sex, MGMT, IDH.
4. ``imaging_only``       -- radiomics at the index visit (+ dynamics).
5. ``combined``           -- both blocks.

Neural networks are deliberately absent: with 81 patients and 477
transitions they cannot be honestly validated here, and the brief asks for
simple models to be compared first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSpec:
    name: str
    kind: str                       # 'classifier' | 'regressor' | 'baseline'
    feature_groups: tuple[str, ...]
    estimator_factory: object = None
    param_grid: dict = field(default_factory=dict)
    description: str = ""
    baseline_column: str | None = None


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def make_preprocessor(numeric_cols: list[str], *,
                      add_indicator: bool = True,
                      strategy: str = "median") -> ColumnTransformer:
    """Impute + scale, fitted **inside** each training fold only.

    ``add_indicator`` keeps missingness as information rather than erasing
    it: whether a subregion was segmented at all is itself predictive.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=strategy,
                                 add_indicator=add_indicator,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric, numeric_cols)],
                             remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def elastic_net_classifier(seed: int = 20251123) -> Pipeline:
    """Logistic regression with an elastic-net penalty (saga)."""
    return LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
        max_iter=8000, class_weight="balanced", random_state=seed)


def elastic_net_regressor(seed: int = 20251123) -> ElasticNet:
    return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000,
                      random_state=seed)


def gradient_boosting_classifier(seed: int = 20251123) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=2,
        subsample=0.8, random_state=seed)


def gradient_boosting_regressor(seed: int = 20251123) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=2,
        subsample=0.8, random_state=seed)


CLASSIFIER_GRIDS = {
    "elastic_net": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    },
    "gradient_boosting": {
        "model__n_estimators": [150, 300],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    },
}

REGRESSOR_GRIDS = {
    "elastic_net": {
        "model__alpha": [0.01, 0.1, 1.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    },
    "gradient_boosting": {
        "model__n_estimators": [150, 300],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    },
}


def build_pipeline(numeric_cols: list[str], estimator, *,
                   add_indicator: bool = True,
                   strategy: str = "median") -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(numeric_cols, add_indicator=add_indicator,
                                   strategy=strategy)),
        ("model", estimator),
    ])


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

class LastObservationClassifier:
    """Predict the outcome from the current observed state.

    For next-visit RANO progression the naive rule is "this visit is PD, so
    the next one will be too".  Predictions are the smoothed empirical rate
    of the outcome within each level of the carry-forward column, estimated
    on the training fold only.
    """

    def __init__(self, column: str, prior_weight: float = 2.0) -> None:
        self.column = column
        self.prior_weight = prior_weight
        self.rates_: dict = {}
        self.prevalence_: float = 0.5

    def fit(self, X: pd.DataFrame, y) -> "LastObservationClassifier":
        y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        values = pd.Series(X[self.column].to_numpy()).reset_index(drop=True)
        self.prevalence_ = float(y.mean())
        self.rates_ = {}
        frame = pd.DataFrame({"v": values, "y": y}).dropna()
        for level, grp in frame.groupby("v"):
            n = len(grp)
            # Laplace-style shrinkage toward the training prevalence keeps a
            # level seen twice from asserting 0.0 or 1.0.
            self.rates_[level] = float(
                (grp["y"].sum() + self.prior_weight * self.prevalence_)
                / (n + self.prior_weight))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.Series(X[self.column].to_numpy())
        p = values.map(lambda v: self.rates_.get(v, self.prevalence_))
        p = p.fillna(self.prevalence_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PrevalenceClassifier:
    """Predict the training prevalence for everyone. The floor for AUPRC."""

    def __init__(self) -> None:
        self.prevalence_ = 0.5

    def fit(self, X, y) -> "PrevalenceClassifier":
        self.prevalence_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), self.prevalence_, dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CarryForwardRegressor:
    """Predict the next relative change as the last observed relative change.

    Falls back to the training mean where no previous interval exists, which
    is the honest naive rule for a patient's first transition.
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.fallback_ = 0.0

    def fit(self, X: pd.DataFrame, y) -> "CarryForwardRegressor":
        self.fallback_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(pd.Series(X[self.column].to_numpy()),
                               errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, self.fallback_)


class MeanRegressor:
    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X, y) -> "MeanRegressor":
        self.mean_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)
