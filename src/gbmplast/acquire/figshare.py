"""Figshare acquisition (LUMIERE lives in Figshare collection 5904905)."""

from __future__ import annotations

from pathlib import Path

import requests

from gbmplast.acquire.http import DEFAULT_TIMEOUT, USER_AGENT, download

API = "https://api.figshare.com/v2"
DOWNLOAD = "https://ndownloader.figshare.com/files/{file_id}"


def collection_articles(collection_id: int | str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(f"{API}/collections/{collection_id}/articles",
                         params={"page": page, "page_size": 100},
                         timeout=DEFAULT_TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


def article_files(article_id: int | str) -> list[dict]:
    r = requests.get(f"{API}/articles/{article_id}", timeout=DEFAULT_TIMEOUT,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json().get("files", [])


def file_url(file_id: int | str) -> str:
    return DOWNLOAD.format(file_id=file_id)


def download_file(file_id: int | str, dest: Path, **kwargs) -> Path:
    return download(file_url(file_id), dest, **kwargs).path
