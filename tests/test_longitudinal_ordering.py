"""Longitudinal ordering, timepoint parsing and transition construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbmplast.adapters.lumiere import parse_timepoint
from tests.conftest import require_dataset


@pytest.mark.parametrize("label,expected", [
    ("week-000", (0, 0)),
    ("week-044", (44, 0)),
    ("week-000-1", (0, 1)),
    ("week-000-2", (0, 2)),
    ("week-106", (106, 0)),
])
def test_parse_timepoint(label, expected):
    assert parse_timepoint(label) == expected


@pytest.mark.parametrize("bad", ["week-", "wk-003", "week-abc", "", "003"])
def test_parse_timepoint_refuses_garbage(bad):
    """An unparseable label must raise, never silently become 0."""
    with pytest.raises(ValueError):
        parse_timepoint(bad)


@pytest.mark.requires_data
def test_visit_index_follows_time(config):
    adapter = require_dataset(config, "lumiere")
    visits = adapter.build().visits
    for pid, grp in visits.groupby("patient_id"):
        ordered = grp.sort_values("visit_index")
        times = pd.to_numeric(ordered["relative_time_days"], errors="coerce")
        assert times.dropna().is_monotonic_increasing, pid
        assert list(ordered["visit_index"]) == list(range(len(ordered))), pid


@pytest.mark.requires_data
def test_transitions_are_forward_in_time(config):
    from gbmplast.longitudinal.build import build_longitudinal

    adapter = require_dataset(config, "lumiere")
    result = build_longitudinal(adapter, config)
    assert result is not None and not result.table.empty
    t = result.table
    assert (t["outcome_time_days"] > t["prediction_time_days"]).all()
    assert (t["interscan_interval_days"] > 0).all()


@pytest.mark.requires_data
def test_transition_pairs_are_consecutive(config):
    """No transition may skip over an intervening dated visit."""
    from gbmplast.longitudinal.build import build_longitudinal

    adapter = require_dataset(config, "lumiere")
    bundle = adapter.build()
    result = build_longitudinal(adapter, config)
    visits = bundle.visits.copy()
    visits["_t"] = pd.to_numeric(visits["relative_time_days"], errors="coerce")

    by_patient = {pid: grp.sort_values("_t")["visit_id"].tolist()
                  for pid, grp in visits[visits["_t"].notna()].groupby("patient_id")}
    for _, row in result.table.iterrows():
        order = by_patient[row["patient_id"]]
        i = order.index(row["index_visit_id"])
        j = order.index(row["next_visit_id"])
        assert j == i + 1, (
            f"{row['patient_id']}: transition skips a visit ({i} -> {j})")


def test_lookback_slope_uses_only_the_window():
    from gbmplast.longitudinal.build import _lookback_slope

    history = pd.DataFrame({
        "_t": [0.0, 100.0, 200.0, 300.0],
        "v": [100.0, 200.0, 300.0, 400.0],
    })
    full = _lookback_slope(history, "v", 1000)
    assert full == pytest.approx(1.0, rel=1e-6)
    # A 150-day window keeps only the last two points.
    windowed = _lookback_slope(history, "v", 150)
    assert windowed == pytest.approx(1.0, rel=1e-6)
    short = _lookback_slope(history.iloc[:1], "v", 1000)
    assert np.isnan(short)
