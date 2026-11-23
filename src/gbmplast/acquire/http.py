"""Shared HTTP download helpers: retries, resume, size reporting.

Every acquisition path in this project is explicit about how many bytes it
is about to pull.  Nothing downloads hundreds of gigabytes implicitly:
callers that target a large archive must pass ``allow_large=True`` or use the
range-based reader in :mod:`gbmplast.acquire.remote_zip`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_TIMEOUT = (30, 300)          # (connect, read) seconds
DEFAULT_CHUNK = 1 << 20              # 1 MiB
LARGE_FILE_BYTES = 2 * 1024 ** 3     # 2 GiB -- above this, require opt-in

USER_AGENT = "gbmplast/0.1 (public-dataset research pipeline)"


@dataclass
class DownloadResult:
    url: str
    path: Path
    bytes_written: int
    skipped: bool
    status: int | None = None


class DownloadError(RuntimeError):
    pass


def head(url: str, *, timeout=DEFAULT_TIMEOUT) -> requests.Response:
    return requests.head(url, allow_redirects=True, timeout=timeout,
                         headers={"User-Agent": USER_AGENT})


def remote_size(url: str) -> int | None:
    """Total size in bytes, or None if the server will not say.

    HEAD is tried first, then a 1-byte Range GET: Figshare redirects to a
    signed S3 URL whose signature covers GET only, so HEAD comes back 403
    with no Content-Length while a Range GET returns 206 plus a
    Content-Range carrying the true total.
    """
    try:
        r = head(url)
        value = r.headers.get("Content-Length")
        if value is not None and r.status_code < 400:
            return int(value)
    except Exception:
        pass
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0",
                                       "User-Agent": USER_AGENT},
                         stream=True, timeout=DEFAULT_TIMEOUT)
        content_range = r.headers.get("Content-Range")
        r.close()
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total)
    except Exception:
        pass
    return None


def supports_ranges(url: str) -> bool:
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0",
                                       "User-Agent": USER_AGENT},
                         stream=True, timeout=DEFAULT_TIMEOUT)
        ok = r.status_code == 206
        r.close()
        return ok
    except Exception:
        return False


def download(
    url: str,
    dest: Path,
    *,
    overwrite: bool = False,
    allow_large: bool = False,
    max_retries: int = 4,
    expected_size: int | None = None,
    quiet: bool = False,
) -> DownloadResult:
    """Download ``url`` to ``dest``, creating parents.

    Refuses files above :data:`LARGE_FILE_BYTES` unless ``allow_large`` is
    set, so an accidental multi-hundred-GB pull cannot happen silently.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return DownloadResult(url, dest, dest.stat().st_size, skipped=True)

    size = expected_size if expected_size is not None else remote_size(url)
    if size is not None and size > LARGE_FILE_BYTES and not allow_large:
        raise DownloadError(
            f"{url} is {size / 1e9:.1f} GB, above the {LARGE_FILE_BYTES / 1e9:.1f} GB "
            "guard. Pass allow_large=True (CLI: --allow-large) if you really "
            "want the whole archive, or use the range-based extractor to pull "
            "only the members you need."
        )
    if size is None and not allow_large:
        # Fail closed. A server that will not declare a size must not be able
        # to stream an unbounded file: the transfer is capped below and
        # aborted if it crosses the guard.
        cap = LARGE_FILE_BYTES
    else:
        cap = None

    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT,
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                written = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(DEFAULT_CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if cap is not None and written > cap:
                            fh.close()
                            tmp.unlink(missing_ok=True)
                            raise DownloadError(
                                f"{url} did not declare a size and exceeded "
                                f"the {cap / 1e9:.1f} GB guard after "
                                f"{written / 1e9:.2f} GB; aborted. Pass "
                                "allow_large=True to permit it, or extract "
                                "only the members you need."
                            )
                tmp.replace(dest)
                if not quiet:
                    print(f"  downloaded {dest.name} ({written / 1e6:.2f} MB)")
                return DownloadResult(url, dest, written, skipped=False,
                                      status=r.status_code)
        except DownloadError:           # guard tripped: a retry would repeat it
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:        # network flakiness -> retry
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2
    raise DownloadError(f"failed to download {url} after {max_retries} "
                        f"attempts: {last_error}")
