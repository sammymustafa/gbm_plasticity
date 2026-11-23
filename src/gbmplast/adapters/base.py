"""Dataset-adapter architecture.

An adapter is the *only* place that knows a cohort's file layout, column
spellings and label encodings.  Everything downstream consumes the canonical
tables, which is what makes the institutional switch-over in Phase 5 a
configuration change rather than a code change.

Contract
--------
``describe()``   -> static metadata: access terms, what the cohort releases.
``availability()`` -> what is actually present on disk right now.
``build()``      -> the seven canonical tables (raises if raw data missing).
``data_dictionary()`` -> per-source-file column documentation.

An adapter must never invent a value.  If a field is not released, the
column is present and NA.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from gbmplast.config import Config
from gbmplast.schema import SPECS, CanonicalBundle


@dataclass
class SourceFile:
    """One expected raw input."""

    key: str
    relative_path: str
    description: str
    required: bool = True
    url: str | None = None
    note: str | None = None

    def resolve(self, root: Path) -> Path:
        return root / self.relative_path


@dataclass
class CohortDescription:
    key: str
    name: str
    url: str
    access: str
    license: str
    citation: str
    modalities: tuple[str, ...]
    n_patients_published: int | None = None
    n_timepoints_published: int | None = None
    t0_definition: str = ""
    known_limitations: tuple[str, ...] = ()
    requires_registration: bool = False
    notes: dict[str, str] = field(default_factory=dict)


class DatasetAdapter(abc.ABC):
    """Base class for every cohort adapter."""

    #: Short, stable key. Used for directory names and id namespacing.
    key: str = ""

    def __init__(self, config: Config, raw_dir: Path | None = None) -> None:
        self.config = config
        self._raw_dir = Path(raw_dir) if raw_dir is not None else None

    # -- locations --------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        if self._raw_dir is not None:
            return self._raw_dir
        configured = self.config.get(f"cohorts.{self.key}.raw_subdir")
        if configured:
            candidate = Path(configured)
            # An absolute path lets an institution point at its own data
            # directory outside the repository.
            if candidate.is_absolute():
                return candidate
            return self.config.paths.raw / candidate
        return self.config.paths.raw / self.key

    @property
    def interim_dir(self) -> Path:
        return self.config.paths.interim / self.key

    @property
    def processed_dir(self) -> Path:
        return self.config.paths.processed / self.key

    # -- required interface ----------------------------------------------
    @abc.abstractmethod
    def describe(self) -> CohortDescription:
        """Static facts about the cohort, independent of local files."""

    @abc.abstractmethod
    def source_files(self) -> list[SourceFile]:
        """Raw inputs this adapter reads."""

    @abc.abstractmethod
    def build(self) -> CanonicalBundle:
        """Produce the seven canonical tables."""

    def data_dictionary(self) -> pd.DataFrame:
        """Column-level documentation of the raw source files.

        Default implementation introspects whatever delimited/Excel files are
        present; adapters override to add curated descriptions.
        """
        rows: list[dict[str, Any]] = []
        for sf in self.source_files():
            path = sf.resolve(self.raw_dir)
            if not path.exists():
                rows.append({
                    "source_key": sf.key, "file": sf.relative_path,
                    "column": pd.NA, "dtype": pd.NA, "n_missing": pd.NA,
                    "pct_missing": pd.NA, "n_unique": pd.NA, "example": pd.NA,
                    "status": "absent",
                })
                continue
            frames = _read_any(path)
            for sheet, df in frames.items():
                for col in df.columns:
                    s = df[col]
                    nonnull = s.dropna()
                    rows.append({
                        "source_key": sf.key,
                        "file": sf.relative_path + (f"#{sheet}" if sheet else ""),
                        "column": str(col),
                        "dtype": str(s.dtype),
                        "n_rows": len(s),
                        "n_missing": int(s.isna().sum()),
                        "pct_missing": round(100.0 * s.isna().mean(), 2),
                        "n_unique": int(nonnull.nunique()),
                        "example": _example(nonnull),
                        "status": "present",
                    })
        return pd.DataFrame(rows)

    # -- availability -----------------------------------------------------
    def availability(self) -> pd.DataFrame:
        rows = []
        for sf in self.source_files():
            path = sf.resolve(self.raw_dir)
            exists = path.exists()
            rows.append({
                "cohort": self.key,
                "source_key": sf.key,
                "relative_path": sf.relative_path,
                "required": sf.required,
                "present": exists,
                "size_bytes": path.stat().st_size if exists else pd.NA,
                "description": sf.description,
                "url": sf.url,
                "note": sf.note,
            })
        return pd.DataFrame(rows)

    def is_buildable(self) -> tuple[bool, list[str]]:
        """Can build() succeed? Returns (ok, missing required paths)."""
        missing = [
            sf.relative_path for sf in self.source_files()
            if sf.required and not sf.resolve(self.raw_dir).exists()
        ]
        return (len(missing) == 0, missing)

    def require_buildable(self) -> None:
        ok, missing = self.is_buildable()
        if not ok:
            raise FileNotFoundError(
                f"[{self.key}] missing required raw files under {self.raw_dir}:\n  "
                + "\n  ".join(missing)
                + f"\n\nAcquire them with:  gbmplast acquire --dataset {self.key}\n"
                + f"See: {self.describe().url}"
            )

    # -- persistence ------------------------------------------------------
    def write(self, bundle: CanonicalBundle, outdir: Path | None = None) -> Path:
        out = Path(outdir) if outdir is not None else self.processed_dir
        out.mkdir(parents=True, exist_ok=True)
        for name, table in bundle.items():
            table.to_parquet(out / f"{name}.parquet", index=False)
            table.to_csv(out / f"{name}.csv", index=False)
        return out

    def load(self, outdir: Path | None = None) -> CanonicalBundle:
        out = Path(outdir) if outdir is not None else self.processed_dir
        tables = {}
        for name in SPECS:
            path = out / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run: gbmplast canonicalize --dataset {self.key}"
                )
            tables[name] = pd.read_parquet(path)
        return CanonicalBundle(cohort=self.key, **tables)

    # -- helpers for subclasses ------------------------------------------
    def empty_bundle(self, **tables: pd.DataFrame) -> CanonicalBundle:
        """Build a bundle, conforming each supplied table and defaulting the rest."""
        built = {}
        for name, spec in SPECS.items():
            df = tables.get(name)
            built[name] = spec.conform(df) if df is not None else spec.empty()
        return CanonicalBundle(cohort=self.key, **built)


def _example(series: pd.Series) -> Any:
    if len(series) == 0:
        return pd.NA
    value = series.iloc[0]
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _read_any(path: Path) -> dict[str, pd.DataFrame]:
    """Read csv/tsv/xlsx into {sheet_name_or_empty: frame}; {} if unreadable."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv"}:
            return {"": pd.read_csv(path, low_memory=False)}
        if suffix in {".tsv", ".txt"}:
            return {"": pd.read_csv(path, sep="\t", low_memory=False)}
        if suffix in {".xlsx", ".xls"}:
            book = pd.read_excel(path, sheet_name=None)
            return {str(k): v for k, v in book.items()}
    except Exception:
        return {}
    return {}


# -- registry -------------------------------------------------------------

_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty 'key'.")
    _REGISTRY[cls.key] = cls
    return cls


def get_adapter(key: str, config: Config, **kwargs: Any) -> DatasetAdapter:
    import gbmplast.adapters  # noqa: F401  (ensures modules are imported)

    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {key!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](config, **kwargs)


def available_adapters() -> list[str]:
    import gbmplast.adapters  # noqa: F401

    return sorted(_REGISTRY)
