"""Configuration, paths and reproducible seeding.

A run is fully described by one YAML file.  ``configs/default.yaml`` holds
project-wide settings; each cohort file layers on top of it.  Nothing in the
analysis code reads an environment variable or a hard-coded path.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    """Repository root, located from this file (src/gbmplast/config.py)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    figures: Path
    results: Path

    @classmethod
    def from_mapping(cls, root: Path, mapping: dict[str, str]) -> "Paths":
        def p(key: str, default: str) -> Path:
            value = Path(mapping.get(key, default))
            return value if value.is_absolute() else root / value

        return cls(
            root=root,
            raw=p("raw", "data/raw"),
            interim=p("interim", "data/interim"),
            processed=p("processed", "data/processed"),
            figures=p("figures", "figures"),
            results=p("results", "results"),
        )

    def ensure(self) -> "Paths":
        for d in (self.raw, self.interim, self.processed, self.figures, self.results):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def cohort_raw(self, cohort_key: str) -> Path:
        return self.raw / cohort_key

    def cohort_interim(self, cohort_key: str) -> Path:
        return self.interim / cohort_key

    def cohort_processed(self, cohort_key: str) -> Path:
        return self.processed / cohort_key


@dataclass
class Config:
    """Immutable-ish view over a merged YAML configuration."""

    data: dict[str, Any]
    paths: Paths
    source_files: list[Path] = field(default_factory=list)

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(
                f"Required config key {dotted!r} missing. Loaded from: "
                + ", ".join(str(p) for p in self.source_files)
            )
        return value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    # -- derived ----------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.get("reproducibility.seed", 20251123))

    @property
    def cohort_key(self) -> str | None:
        return self.get("cohort.key")

    def cohort_config(self, key: str) -> dict[str, Any]:
        cohorts = self.get("cohorts", {}) or {}
        if key not in cohorts:
            raise KeyError(f"No cohort block {key!r} in config.")
        return cohorts[key]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    *paths: str | Path,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Config:
    """Load and merge YAML configs left to right.

    ``configs/default.yaml`` is always loaded first when it exists, so a
    cohort file only needs to carry its own differences.
    """
    root = Path(root) if root is not None else project_root()
    loaded: list[Path] = []
    merged: dict[str, Any] = {}

    default = root / "configs" / "default.yaml"
    candidates: list[Path] = []
    if default.exists():
        candidates.append(default)
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            # allow either 'configs/lumiere.yaml' or just 'lumiere'
            if path.suffix == "":
                path = root / "configs" / f"{path}.yaml"
            else:
                path = root / path
        candidates.append(path)

    for path in candidates:
        if path.resolve() in {c.resolve() for c in loaded}:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as fh:
            content = yaml.safe_load(fh) or {}
        if not isinstance(content, dict):
            raise ValueError(f"Config {path} must contain a YAML mapping.")
        merged = _deep_merge(merged, content)
        loaded.append(path)

    if overrides:
        merged = _deep_merge(merged, _expand_dotted(overrides))

    paths_obj = Paths.from_mapping(root, merged.get("paths", {}) or {})
    return Config(data=merged, paths=paths_obj, source_files=loaded)


def _expand_dotted(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn {'model.alpha': 1} into {'model': {'alpha': 1}} for CLI overrides."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def set_seeds(seed: int) -> int:
    """Seed every RNG the pipeline can reach. Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


def rng(seed: int | None = None) -> np.random.Generator:
    """Preferred RNG for new code: an explicit Generator, not global state."""
    return np.random.default_rng(seed)
