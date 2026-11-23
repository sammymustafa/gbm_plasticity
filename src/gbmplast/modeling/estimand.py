"""Explicit estimand declaration.

Nothing in this project is fitted without first stating, in code, what is
being estimated: the target, the moment at which the prediction is made,
how far back features may look, how far forward the outcome is observed, who
is eligible, and what is excluded.  The declaration travels with the results
so a reader never has to infer it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class Estimand:
    name: str
    question: str
    cohort: str
    unit_of_analysis: str
    outcome: str
    outcome_type: str                  # 'binary' | 'continuous' | 'time_to_event'
    prediction_time: str
    lookback_window_days: int | None
    outcome_horizon_days: int | None
    eligibility: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    positive_label: str | None = None
    grouping_variable: str = "patient_id"
    confounders_acknowledged: tuple[str, ...] = ()
    causal_claim: str = (
        "None. This is a predictive/associational analysis; no intervention "
        "is assigned and treatment is not randomised."
    )
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {k: ("; ".join(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"field": k, "value": v}
                             for k, v in self.as_dict().items()])

    def describe(self) -> str:
        lines = [f"Estimand: {self.name}", f"  question: {self.question}",
                 f"  cohort: {self.cohort}",
                 f"  unit: {self.unit_of_analysis}",
                 f"  outcome: {self.outcome} ({self.outcome_type})"]
        if self.positive_label:
            lines.append(f"  positive class: {self.positive_label}")
        lines += [f"  prediction time: {self.prediction_time}",
                  f"  lookback: {self.lookback_window_days} days",
                  f"  outcome horizon: {self.outcome_horizon_days} days",
                  f"  grouping: {self.grouping_variable}"]
        for label, items in (("eligibility", self.eligibility),
                             ("exclusions", self.exclusions),
                             ("confounders acknowledged",
                              self.confounders_acknowledged),
                             ("notes", self.notes)):
            for item in items:
                lines.append(f"  {label}: {item}")
        lines.append(f"  causal claim: {self.causal_claim}")
        return "\n".join(lines)


@dataclass
class CohortFlow:
    """A CONSORT-style accounting of who entered the analysis and why not."""

    steps: list[tuple[str, int]] = field(default_factory=list)
    exclusions: list[tuple[str, int]] = field(default_factory=list)

    def step(self, label: str, n: int) -> "CohortFlow":
        self.steps.append((label, int(n)))
        return self

    def exclude(self, reason: str, n: int) -> "CohortFlow":
        if n:
            self.exclusions.append((reason, int(n)))
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [{"kind": "step", "label": k, "n": v} for k, v in self.steps]
        rows += [{"kind": "exclusion", "label": k, "n": v}
                 for k, v in self.exclusions]
        return pd.DataFrame(rows)

    def describe(self) -> str:
        lines = []
        for label, n in self.steps:
            lines.append(f"  {label}: n = {n}")
        for reason, n in self.exclusions:
            lines.append(f"    - excluded ({reason}): {n}")
        return "\n".join(lines)
