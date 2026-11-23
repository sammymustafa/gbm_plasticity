"""Per-dataset acquisition. Every path states its byte cost up front.

Nothing here downloads a multi-gigabyte archive implicitly.  Image-level
acquisition for LUMIERE uses HTTP range requests against the published ZIP so
that a development subset costs megabytes rather than 32.5 GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gbmplast.acquire import figshare, geo, tcia
from gbmplast.acquire.http import download, remote_size
from gbmplast.acquire.remote_zip import RemoteZip
from gbmplast.config import Config

LUMIERE_IMAGING_FILE_ID = 38249697        # Imaging-v202211.zip (32.5 GB)
LUMIERE_COLLAGE_FILE_ID = 38246988        # Radiomics-CoLlAGe-v202211.zip (54 GB)
GSE_CARE = "GSE274546"


@dataclass
class AcquisitionReport:
    dataset: str
    downloaded: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]
    bytes_written: int
    messages: list[str]

    def to_frame(self) -> pd.DataFrame:
        rows = [{"dataset": self.dataset, "status": "downloaded", "item": i}
                for i in self.downloaded]
        rows += [{"dataset": self.dataset, "status": "skipped", "item": i}
                 for i in self.skipped]
        rows += [{"dataset": self.dataset, "status": "failed",
                  "item": i, "error": e} for i, e in self.failed]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        parts = [f"{self.dataset}: {len(self.downloaded)} downloaded, "
                 f"{len(self.skipped)} already present, "
                 f"{len(self.failed)} failed, "
                 f"{self.bytes_written / 1e6:.1f} MB written"]
        parts += [f"  - {m}" for m in self.messages]
        for item, err in self.failed:
            parts.append(f"  ! {item}: {err}")
        return "\n".join(parts)


# --------------------------------------------------------------------------

def acquire_tables(adapter, *, overwrite: bool = False) -> AcquisitionReport:
    """Download every source file an adapter declares a URL for."""
    downloaded, skipped, failed, messages = [], [], [], []
    total = 0
    for sf in adapter.source_files():
        dest = sf.resolve(adapter.raw_dir)
        if sf.url is None:
            if not dest.exists():
                messages.append(
                    f"{sf.relative_path}: no public URL "
                    f"({sf.note or 'manual acquisition required'})")
            continue
        if dest.exists() and not overwrite:
            skipped.append(sf.relative_path)
            continue
        # Bulk archives are declared by the adapter, not discovered by a
        # network probe: a server that refuses HEAD must not be able to make
        # a 32 GB file look like an ordinary table.
        if getattr(sf, "large", False):
            approx = getattr(sf, "approx_bytes", None)
            size_text = f"{approx / 1e9:.1f} GB" if approx else "bulk archive"
            messages.append(
                f"{sf.relative_path}: {size_text} -- skipped by default. "
                f"{sf.note or 'Use the dedicated subset command.'}")
            continue
        size = remote_size(sf.url)
        if size is not None and size > 2 * 1024 ** 3:
            messages.append(
                f"{sf.relative_path}: {size / 1e9:.1f} GB -- skipped. Use the "
                "dedicated image command to pull a subset.")
            continue
        try:
            result = download(sf.url, dest, overwrite=overwrite, quiet=True)
            downloaded.append(sf.relative_path)
            total += result.bytes_written
        except Exception as exc:
            failed.append((sf.relative_path, str(exc)))
    return AcquisitionReport(adapter.key, downloaded, skipped, failed,
                             total, messages)


# --------------------------------------------------------------------------

def acquire_lumiere_images(
    config: Config,
    adapter,
    *,
    n_patients: int = 5,
    patients: list[str] | None = None,
    sequences: tuple[str, ...] = ("CT1", "FLAIR"),
    include_segmentation: bool = True,
    registered: bool = True,
    dry_run: bool = False,
    max_bytes: int = 2 * 1024 ** 3,
) -> AcquisitionReport:
    """Pull a development subset of LUMIERE images out of the 32.5 GB archive.

    Uses HTTP range requests: only the selected members are transferred.
    Reports the exact byte cost and refuses to exceed ``max_bytes``.
    """
    messages: list[str] = []
    url = figshare.file_url(LUMIERE_IMAGING_FILE_ID)
    out_root = adapter.raw_dir / "images"

    rz = RemoteZip(url)
    messages.append(f"remote archive {rz.size / 1e9:.2f} GB, "
                    f"{len(rz.namelist())} members indexed via range requests")

    bundle = adapter.build()
    visits = bundle.visits
    imaged = visits[visits["has_imaging"] == True]          # noqa: E712

    if patients:
        chosen = [p for p in patients]
    else:
        # Prefer patients with the most imaged studies and an expert rating,
        # deterministically ordered so the subset is reproducible.
        counts = (imaged.groupby("local_patient_id")
                  .agg(n_studies=("visit_id", "size"),
                       n_rated=("has_expert_rating", "sum"))
                  .sort_values(["n_studies", "n_rated"], ascending=False))
        chosen = list(counts.index[:n_patients])
    messages.append(f"selected patients: {', '.join(chosen)}")

    wanted: list[str] = []
    for local_patient in chosen:
        sub = imaged[imaged["local_patient_id"] == local_patient]
        for _, v in sub.iterrows():
            tp = v["local_visit_id"]
            for seq in sequences:
                wanted.append(f"Imaging/{local_patient}/{tp}/{seq}.nii.gz")
            if include_segmentation:
                if registered:
                    wanted.append(
                        f"Imaging/{local_patient}/{tp}/HD-GLIO-AUTO-segmentation/"
                        "registered/segmentation.nii.gz")
                    for seq in sequences:
                        wanted.append(
                            f"Imaging/{local_patient}/{tp}/"
                            f"HD-GLIO-AUTO-segmentation/registered/"
                            f"{seq}_r2s_bet_reg.nii.gz")
                else:
                    for seq in sequences:
                        wanted.append(
                            f"Imaging/{local_patient}/{tp}/"
                            f"HD-GLIO-AUTO-segmentation/native/"
                            f"segmentation_{seq}_origspace.nii.gz")

    available = set(rz.namelist())
    present = [n for n in wanted if n in available]
    absent = [n for n in wanted if n not in available]
    if absent:
        messages.append(f"{len(absent)} requested members are not in the "
                        "archive (expected: the cohort's completeness table "
                        "marks some sequences missing)")

    cost = rz.total_bytes(present)
    messages.append(f"{len(present)} members, {cost / 1e6:.1f} MB compressed "
                    f"(of {rz.size / 1e9:.2f} GB archive)")
    if cost > max_bytes:
        raise RuntimeError(
            f"subset would transfer {cost / 1e9:.2f} GB, above the "
            f"{max_bytes / 1e9:.2f} GB guard; reduce --n-patients or pass a "
            "larger --max-bytes")
    if dry_run:
        return AcquisitionReport("lumiere-images", [], present, [], 0,
                                 messages + ["dry run: nothing transferred"])

    downloaded, failed, total = [], [], 0
    for name in present:
        target = out_root / name
        if target.exists():
            continue
        try:
            path = rz.extract(name, out_root)
            downloaded.append(name)
            total += path.stat().st_size
        except Exception as exc:
            failed.append((name, str(exc)))
    return AcquisitionReport("lumiere-images", downloaded,
                             [n for n in present if n not in downloaded],
                             failed, total, messages)


# --------------------------------------------------------------------------

def acquire_gbm_care_metadata(adapter, *, overwrite: bool = False
                              ) -> AcquisitionReport:
    """Fetch the GEO SOFT records (small) for GSE274546."""
    downloaded, skipped, failed = [], [], []
    total = 0
    targets = [
        (f"{GSE_CARE}_samples.soft.txt", "gsm", "brief"),
        (f"{GSE_CARE}_series.soft.txt", "self", "full"),
    ]
    for filename, targ, view in targets:
        dest = adapter.raw_dir / filename
        if dest.exists() and not overwrite:
            skipped.append(filename)
            continue
        try:
            geo.fetch_soft(GSE_CARE, dest, targ=targ, view=view,
                           overwrite=overwrite)
            downloaded.append(filename)
            total += dest.stat().st_size
        except Exception as exc:
            failed.append((filename, str(exc)))
    return AcquisitionReport("gbm_care", downloaded, skipped, failed, total,
                             ["GEO SOFT metadata only; expression matrices are "
                              "fetched per specimen by "
                              "'gbmplast acquire --dataset gbm_care --specimens'"])


def acquire_gbm_care_specimens(
    adapter,
    *,
    patients: list[str] | None = None,
    n_patients: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    max_bytes: int = 8 * 1024 ** 3,
) -> AcquisitionReport:
    """Download per-specimen UMI count matrices (~20-50 MB each).

    Defaults to *nothing*: a patient list or ``n_patients`` must be given, so
    all 121 samples (~4 GB) are never pulled by accident.
    """
    samples = adapter.sample_records()
    paired = adapter.paired_patients()

    if patients:
        chosen = list(patients)
    elif n_patients:
        # Deterministic: paired patients first, in published order.
        chosen = list(paired["local_patient_id"].head(n_patients))
    else:
        raise ValueError(
            "Specify --patients or --n-patients. Downloading all 121 "
            "specimens (~4 GB) is never implicit.")

    subset = samples[samples["local_patient_id"].isin(chosen)]
    messages = [f"{len(chosen)} patients -> {len(subset)} specimens"]

    sizes = []
    for _, r in subset.iterrows():
        size = remote_size(r["supplementary_url"])
        sizes.append(size or 0)
    total_expected = sum(sizes)
    messages.append(f"expected transfer {total_expected / 1e6:.1f} MB")
    if total_expected > max_bytes:
        raise RuntimeError(
            f"would transfer {total_expected / 1e9:.2f} GB, above the "
            f"{max_bytes / 1e9:.2f} GB guard")
    if dry_run:
        return AcquisitionReport("gbm_care-specimens", [],
                                 list(subset["file_name"]), [], 0,
                                 messages + ["dry run: nothing transferred"])

    out = adapter.raw_dir / "specimens"
    downloaded, skipped, failed, total = [], [], [], 0
    for _, r in subset.iterrows():
        dest = out / r["file_name"]
        if dest.exists() and not overwrite:
            skipped.append(r["file_name"])
            continue
        try:
            result = download(r["supplementary_url"], dest, overwrite=overwrite,
                              quiet=True)
            downloaded.append(r["file_name"])
            total += result.bytes_written
        except Exception as exc:
            failed.append((r["file_name"], str(exc)))
    return AcquisitionReport("gbm_care-specimens", downloaded, skipped, failed,
                             total, messages)


# --------------------------------------------------------------------------

def tcia_inventory(collection: str, outdir: Path) -> pd.DataFrame:
    """Record what a TCIA collection contains, including its size in GB."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    series = tcia.series(collection)
    if not series.empty:
        series.to_csv(outdir / f"{collection}_series_inventory.csv", index=False)
    patients = tcia.patients(collection)
    if not patients.empty:
        patients.to_csv(outdir / f"{collection}_patients.csv", index=False)
    return series
