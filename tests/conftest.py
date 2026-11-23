"""Shared fixtures.

Tests split into two kinds:

* pure tests that build synthetic data and exercise the logic -- these run
  anywhere, in CI, with no downloads;
* ``@pytest.mark.requires_data`` tests that run against real cohort tables
  when they are present under ``data/raw/`` and skip cleanly when not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbmplast.config import load_config, project_root


@pytest.fixture(scope="session")
def root() -> Path:
    return project_root()


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def available_datasets(config) -> dict:
    from gbmplast.adapters import available_adapters, get_adapter

    out = {}
    for key in available_adapters():
        adapter = get_adapter(key, config)
        out[key] = adapter.is_buildable()[0]
    return out


def require_dataset(config, key: str):
    """Return a built bundle, or skip when the raw data is absent."""
    from gbmplast.adapters import get_adapter

    adapter = get_adapter(key, config)
    if not adapter.is_buildable()[0]:
        pytest.skip(f"{key}: raw data not present under data/raw/")
    return adapter


@pytest.fixture
def synthetic_longitudinal() -> pd.DataFrame:
    """A small, fully-controlled transition table with known properties."""
    rng = np.random.default_rng(11)
    rows = []
    for p in range(12):
        n_visits = rng.integers(3, 6)
        volume = float(rng.uniform(2000, 12000))
        for v in range(n_visits - 1):
            previous = volume
            volume = max(10.0, volume * float(rng.uniform(0.7, 1.5)))
            rows.append({
                "patient_id": f"TEST::P{p:03d}",
                "local_patient_id": f"P{p:03d}",
                "index_visit_id": f"TEST::P{p:03d}::v{v}",
                "next_visit_id": f"TEST::P{p:03d}::v{v + 1}",
                "prediction_time_days": 70.0 * v,
                "time_since_t0_days": 70.0 * v,
                "visit_index": v,
                "transition_index": v,
                "interscan_interval_days": 70.0,
                "cur_enhancing__volume_mm3": volume,
                "prev_enhancing__volume_mm3": previous,
                "rel_change_enhancing": (volume - previous) / previous,
                "age_at_diagnosis_years": 55.0 + p,
                "sex_female": float(p % 2),
                "mgmt_methylated": float(p % 3 == 0),
                "index_rano_status": "PD" if v % 2 else "SD",
                "index_rano_is_pd": float(v % 2),
                "next_rano_status": "PD" if (v + p) % 2 else "SD",
                "next_rano_is_response_assessment": True,
                "next_rano_progression": float((v + p) % 2),
                "next_enhancing__volume_mm3": volume * 1.1,
                "next_rel_change_enhancing": 0.1,
                "next_log_ratio_enhancing": float(np.log(1.1)),
                "outcome_time_days": 70.0 * (v + 1),
            })
    return pd.DataFrame(rows)
