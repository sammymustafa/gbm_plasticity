"""Read individual members out of a remote ZIP using HTTP range requests.

LUMIERE publishes its imaging as one 32.5 GB archive and its CoLlAGe features
as a 54 GB archive.  Pulling either in full to develop a pipeline would be
wasteful and is explicitly out of scope, but the central directory of a ZIP
lives at its end, so the whole index can be fetched in a few kilobytes and
individual members extracted with further range requests.

Supports ZIP64 (required above 4 GB) and both stored and deflated members.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import requests

from gbmplast.acquire.http import DEFAULT_TIMEOUT, USER_AGENT, supports_ranges

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"


@dataclass
class ZipMember:
    name: str
    compress_type: int
    compressed_size: int
    file_size: int
    header_offset: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


class RemoteZipError(RuntimeError):
    pass


class RemoteZip:
    """Minimal random-access reader for a ZIP served over HTTP(S).

    >>> rz = RemoteZip(url)                     # doctest: +SKIP
    >>> [m.name for m in rz.members[:3]]        # doctest: +SKIP
    >>> rz.extract("Imaging/Patient-001/week-000-1/CT1.nii.gz", out_dir)
    """

    def __init__(self, url: str, *, session: requests.Session | None = None,
                 verify_ranges: bool = True) -> None:
        self.url = url
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if verify_ranges and not supports_ranges(url):
            raise RemoteZipError(
                f"{url} does not advertise HTTP range support; the whole "
                "archive would have to be downloaded."
            )
        self.size = self._content_length()
        self._members: dict[str, ZipMember] | None = None

    # -- low level --------------------------------------------------------
    def _content_length(self) -> int:
        r = self.session.get(self.url, stream=True, timeout=DEFAULT_TIMEOUT,
                             headers={"Range": "bytes=0-0"})
        r.close()
        rng = r.headers.get("Content-Range")
        if rng and "/" in rng:
            total = rng.rsplit("/", 1)[1]
            if total.isdigit():
                return int(total)
        length = r.headers.get("Content-Length")
        if length and length.isdigit() and r.status_code == 200:
            return int(length)
        raise RemoteZipError(f"could not determine remote size of {self.url}")

    def _get_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = min(start + length - 1, self.size - 1)
        r = self.session.get(self.url, timeout=DEFAULT_TIMEOUT,
                             headers={"Range": f"bytes={start}-{end}"})
        if r.status_code not in (206, 200):
            raise RemoteZipError(
                f"range request {start}-{end} returned HTTP {r.status_code}")
        return r.content

    # -- central directory ------------------------------------------------
    def _locate_central_directory(self) -> tuple[int, int, int]:
        """Return (cd_offset, cd_size, n_entries), handling ZIP64."""
        tail_len = min(66_000, self.size)
        tail = self._get_range(self.size - tail_len, tail_len)
        idx = tail.rfind(EOCD_SIG)
        if idx < 0:
            raise RemoteZipError("end-of-central-directory record not found")
        n_entries, cd_size, cd_offset = struct.unpack_from("<HII", tail, idx + 10)

        # ZIP64 is signalled by 0xFFFF / 0xFFFFFFFF sentinels.
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_entries == 0xFFFF:
            loc = tail.rfind(EOCD64_LOCATOR_SIG, 0, idx)
            if loc < 0:
                raise RemoteZipError("ZIP64 locator not found")
            eocd64_offset = struct.unpack_from("<Q", tail, loc + 8)[0]
            head = self._get_range(eocd64_offset, 56)
            if head[:4] != EOCD64_SIG:
                raise RemoteZipError("ZIP64 end-of-central-directory not found")
            n_entries = struct.unpack_from("<Q", head, 32)[0]
            cd_size = struct.unpack_from("<Q", head, 40)[0]
            cd_offset = struct.unpack_from("<Q", head, 48)[0]
        return cd_offset, cd_size, n_entries

    def _read_central_directory(self) -> dict[str, ZipMember]:
        cd_offset, cd_size, _ = self._locate_central_directory()
        blob = self._get_range(cd_offset, cd_size)
        members: dict[str, ZipMember] = {}
        pos = 0
        while pos + 46 <= len(blob):
            if blob[pos:pos + 4] != CENTRAL_SIG:
                break
            (compress_type,) = struct.unpack_from("<H", blob, pos + 10)
            csize, usize = struct.unpack_from("<II", blob, pos + 20)
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", blob, pos + 28)
            (header_offset,) = struct.unpack_from("<I", blob, pos + 42)
            name = blob[pos + 46: pos + 46 + name_len].decode("utf-8", "replace")
            extra = blob[pos + 46 + name_len: pos + 46 + name_len + extra_len]

            if 0xFFFFFFFF in (csize, usize, header_offset):
                usize, csize, header_offset = _zip64_extra(
                    extra, usize, csize, header_offset)

            members[name] = ZipMember(name, compress_type, csize, usize,
                                      header_offset)
            pos += 46 + name_len + extra_len + comment_len
        if not members:
            raise RemoteZipError("central directory contained no entries")
        return members

    # -- public -----------------------------------------------------------
    @property
    def members(self) -> list[ZipMember]:
        if self._members is None:
            self._members = self._read_central_directory()
        return list(self._members.values())

    def namelist(self) -> list[str]:
        return [m.name for m in self.members]

    def member(self, name: str) -> ZipMember:
        if self._members is None:
            self._members = self._read_central_directory()
        if name not in self._members:
            raise KeyError(f"{name!r} not in remote zip")
        return self._members[name]

    def read(self, name: str) -> bytes:
        """Fetch and decompress a single member."""
        m = self.member(name)
        # Local file header: 30 bytes + name + extra (lengths differ from
        # the central directory's, so they must be read from the local header).
        head = self._get_range(m.header_offset, 30)
        if head[:4] != b"PK\x03\x04":
            raise RemoteZipError(f"bad local header for {name!r}")
        name_len, extra_len = struct.unpack_from("<HH", head, 26)
        data_start = m.header_offset + 30 + name_len + extra_len
        raw = self._get_range(data_start, m.compressed_size)
        if m.compress_type == 0:
            return raw
        if m.compress_type == 8:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
        raise RemoteZipError(
            f"unsupported compression method {m.compress_type} for {name!r}")

    def extract(self, name: str, out_dir: Path, *,
                flatten: bool = False) -> Path:
        data = self.read(name)
        out_dir = Path(out_dir)
        target = out_dir / (Path(name).name if flatten else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def total_bytes(self, names: list[str]) -> int:
        return sum(self.member(n).compressed_size for n in names)


def _zip64_extra(extra: bytes, usize: int, csize: int,
                 offset: int) -> tuple[int, int, int]:
    """Replace 0xFFFFFFFF sentinels from the ZIP64 extra field (id 0x0001)."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4: pos + 4 + data_size]
        if header_id == 0x0001:
            cursor = 0
            if usize == 0xFFFFFFFF and cursor + 8 <= len(body):
                usize = struct.unpack_from("<Q", body, cursor)[0]; cursor += 8
            if csize == 0xFFFFFFFF and cursor + 8 <= len(body):
                csize = struct.unpack_from("<Q", body, cursor)[0]; cursor += 8
            if offset == 0xFFFFFFFF and cursor + 8 <= len(body):
                offset = struct.unpack_from("<Q", body, cursor)[0]; cursor += 8
            break
        pos += 4 + data_size
    return usize, csize, offset
