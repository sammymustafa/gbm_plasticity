"""NCBI GEO acquisition: series/sample metadata and per-sample supplementaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from gbmplast.acquire.http import DEFAULT_TIMEOUT, USER_AGENT, download

GEO_ACC = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
GEO_HOST = "https://ftp.ncbi.nlm.nih.gov"
GEO_FTP = GEO_HOST + "/geo"


def fetch_soft(accession: str, dest: Path, *, targ: str = "gsm",
               view: str = "brief", overwrite: bool = False) -> Path:
    """Download a GEO SOFT text record (``targ='self'`` or ``'gsm'``)."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest
    params = {"acc": accession, "targ": targ, "form": "text", "view": view}
    r = requests.get(GEO_ACC, params=params, timeout=DEFAULT_TIMEOUT,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.text, encoding="utf-8")
    return dest


def parse_soft(path: Path) -> list[dict[str, object]]:
    """Parse a SOFT text file into one dict per ``^SAMPLE`` record.

    Repeated keys are collected into lists so that multi-line fields such as
    ``!Sample_characteristics_ch1`` are never silently truncated.
    """
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("^SAMPLE"):
            current = {"gsm": line.split("=", 1)[1].strip()}
            records.append(current)
            continue
        if current is None or not line.startswith("!") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.lstrip("!").strip()
        value = value.strip()
        if key in current:
            existing = current[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                current[key] = [existing, value]
        else:
            current[key] = value
    return records


def sample_supplementary_urls(record: dict[str, object]) -> list[str]:
    """Every ``!Sample_supplementary_file*`` URL, normalised to HTTPS."""
    urls: list[str] = []
    for key, value in record.items():
        if not key.startswith("Sample_supplementary_file"):
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            v = str(v).strip()
            if v.lower() in {"none", "null", ""}:
                continue
            urls.append(v.replace("ftp://ftp.ncbi.nlm.nih.gov", GEO_HOST))
    return urls


def download_sample_supplementary(url: str, dest_dir: Path, *,
                                  overwrite: bool = False,
                                  allow_large: bool = False) -> Path:
    name = url.rsplit("/", 1)[-1]
    return download(url, Path(dest_dir) / name, overwrite=overwrite,
                    allow_large=allow_large).path
