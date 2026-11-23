"""TCIA acquisition.

Only the *metadata* API is used here.  Image series are large and several
collections (UCSD-PTGBM) distribute NIfTI bundles through IBM Aspera, which
is not usable headlessly; the adapters document that and degrade gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from gbmplast.acquire.http import DEFAULT_TIMEOUT, USER_AGENT, download

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
UPLOADS = "https://www.cancerimagingarchive.net/wp-content/uploads/"


def _get(endpoint: str, **params) -> list[dict]:
    r = requests.get(f"{NBIA}/{endpoint}", params=params, timeout=DEFAULT_TIMEOUT,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


def collections() -> list[str]:
    return [c["Collection"] for c in _get("getCollectionValues")]


def patients(collection: str) -> pd.DataFrame:
    return pd.DataFrame(_get("getPatient", Collection=collection))


def series(collection: str) -> pd.DataFrame:
    """Every series in a collection, with FileSize so volume is explicit."""
    return pd.DataFrame(_get("getSeries", Collection=collection))


def collection_size_gb(collection: str) -> float:
    df = series(collection)
    if df.empty or "FileSize" not in df:
        return float("nan")
    return float(pd.to_numeric(df["FileSize"], errors="coerce").sum()) / 1e9


def download_supporting_file(filename: str, dest: Path, **kwargs) -> Path:
    """Download a supporting table published under the TCIA uploads path."""
    return download(UPLOADS + filename, dest, **kwargs).path
