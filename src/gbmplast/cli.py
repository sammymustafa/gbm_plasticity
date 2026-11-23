"""Command-line interface.

    gbmplast datasets                      list cohorts and what they contain
    gbmplast acquire   --dataset lumiere   fetch public tables (small)
    gbmplast canonicalize --dataset ...    build the seven canonical tables
    gbmplast validate  --dataset ...       structural + integrity checks
    gbmplast dictionary --dataset all      data dictionaries + audit matrices
    gbmplast imaging-qc --dataset lumiere  geometry/orientation/mask QC
    gbmplast features  --dataset lumiere   radiomics extraction from images
    gbmplast longitudinal --dataset lumiere  patient-visit trajectory table
    gbmplast model next-visit --dataset lumiere
    gbmplast radiogenomics --dataset upenn_gbm
    gbmplast molecular pseudobulk --dataset gbm_care
    gbmplast molecular plasticity --dataset gbm_care
    gbmplast report                        assemble the results report

Every command takes ``--config`` (repeatable) and ``--set key=value``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gbmplast import __version__
from gbmplast.adapters import available_adapters, get_adapter
from gbmplast.config import Config, load_config, set_seeds


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _coerce(value.strip())
    configs = list(getattr(args, "config", None) or [])
    cfg = load_config(*configs, overrides=overrides)
    cfg.paths.ensure()
    set_seeds(cfg.seed)
    return cfg


def _coerce(value: str) -> object:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_datasets(spec: str | None) -> list[str]:
    known = available_adapters()
    if spec in (None, "", "all"):
        return [k for k in known if k != "institutional"]
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [r for r in requested if r not in known]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {known}")
    return requested


def _emit(frame: pd.DataFrame, path: Path | None, *, label: str,
          max_rows: int = 40) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"  wrote {label}: {path}  ({len(frame)} rows)")
    else:
        with pd.option_context("display.width", 200,
                               "display.max_columns", 40):
            print(frame.head(max_rows).to_string(index=False))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_datasets(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.report.dictionary import cohort_overview

    frames = []
    for key in available_adapters():
        adapter = get_adapter(key, cfg)
        buildable, missing = adapter.is_buildable()
        bundle = None
        if buildable and not args.fast:
            try:
                bundle = adapter.build()
            except Exception as exc:        # degrade, never crash the listing
                print(f"  ! {key}: build failed ({exc})", file=sys.stderr)
        frames.append(cohort_overview(adapter, bundle))
    table = pd.concat(frames, ignore_index=True)
    cols = ["dataset", "cohort", "raw_present", "n_patients_published",
            "n_timepoints_published", "modalities", "requires_registration"]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].to_string(index=False))
    out = cfg.paths.results / "dataset_availability.csv"
    table.to_csv(out, index=False)
    print(f"\nfull table -> {out}")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.acquire import datasets as acq

    rc = 0
    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        adapter.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {key} -> {adapter.raw_dir}")

        if key == "gbm_care":
            report = acq.acquire_gbm_care_metadata(adapter,
                                                   overwrite=args.overwrite)
        else:
            report = acq.acquire_tables(adapter, overwrite=args.overwrite)
        print(report.summary())
        if report.failed:
            rc = 1

        if key == "lumiere" and args.images:
            try:
                sub = acq.acquire_lumiere_images(
                    cfg, adapter,
                    n_patients=args.n_patients,
                    patients=args.patients.split(",") if args.patients else None,
                    dry_run=args.dry_run,
                    max_bytes=int(args.max_bytes * 1024 ** 3))
                print(sub.summary())
            except Exception as exc:
                print(f"  ! image subset failed: {exc}", file=sys.stderr)
                rc = 1

        if key == "gbm_care" and args.specimens:
            try:
                sub = acq.acquire_gbm_care_specimens(
                    adapter,
                    patients=args.patients.split(",") if args.patients else None,
                    n_patients=args.n_patients if not args.patients else None,
                    overwrite=args.overwrite, dry_run=args.dry_run,
                    max_bytes=int(args.max_bytes * 1024 ** 3))
                print(sub.summary())
            except Exception as exc:
                print(f"  ! specimen download failed: {exc}", file=sys.stderr)
                rc = 1

        if key in {"ucsd_ptgbm", "upenn_gbm"} and args.inventory:
            collection = {"ucsd_ptgbm": "UCSD-PTGBM",
                          "upenn_gbm": "UPENN-GBM"}[key]
            try:
                series = acq.tcia_inventory(collection, adapter.interim_dir)
                print(f"  TCIA inventory: {len(series)} series")
            except Exception as exc:
                print(f"  ! TCIA inventory unavailable: {exc}", file=sys.stderr)

        if key == "glass":
            print(adapter.acquisition_instructions())
    return rc


def cmd_canonicalize(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    rc = 0
    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        buildable, missing = adapter.is_buildable()
        if not buildable:
            print(f"== {key}: SKIPPED (missing {', '.join(missing)})")
            print(f"   run: gbmplast acquire --dataset {key}")
            continue
        bundle = adapter.build()
        out = adapter.write(bundle)
        print(f"== {key} -> {out}")
        print(bundle.summary().to_string(index=False))
        for name, note in sorted(bundle.notes.items()):
            print(f"   note[{name}]: {note}")
    return rc


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.validate import validate_bundle

    frames, rc = [], 0
    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        if not adapter.is_buildable()[0]:
            print(f"== {key}: SKIPPED (raw data absent)")
            continue
        report = validate_bundle(adapter.build())
        frames.append(report.to_frame())
        print(f"== {key}: {len(report.errors)} errors, "
              f"{len(report.warnings)} warnings")
        for finding in report.findings:
            if finding.severity == "info" and not args.verbose:
                continue
            print(f"   [{finding.severity}] {finding.table}."
                  f"{finding.check}: {finding.message}")
        if report.errors:
            rc = 1
    if frames:
        out = cfg.paths.results / "validation_report.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"\n-> {out}")
    if rc and args.strict:
        return rc
    return 0 if not args.strict else rc


def cmd_dictionary(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.report.dictionary import canonical_dictionary, write_audit

    outroot = cfg.paths.results / "data_dictionary"
    outroot.mkdir(parents=True, exist_ok=True)
    canonical_dictionary().to_csv(outroot / "canonical_schema.csv", index=False)
    print(f"canonical schema -> {outroot / 'canonical_schema.csv'}")

    overviews, coverage, pairing, outcomes, modalities = [], [], [], [], []
    from gbmplast.report import dictionary as D

    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        if not adapter.is_buildable()[0]:
            print(f"== {key}: SKIPPED (raw data absent)")
            overviews.append(D.cohort_overview(adapter, None))
            continue
        bundle = adapter.build()
        written = write_audit(adapter, bundle, outroot / key)
        print(f"== {key}: {len(written)} audit tables -> {outroot / key}")
        overviews.append(D.cohort_overview(adapter, bundle))
        coverage.append(D.longitudinal_coverage(bundle))
        pairing.append(D.molecular_pairing(bundle))
        outcomes.append(D.outcome_summary(bundle))
        modalities.append(D.modality_completeness(bundle))

    for name, frames in (("dataset_availability", overviews),
                         ("longitudinal_coverage", coverage),
                         ("molecular_pairing", pairing),
                         ("outcome_summary", outcomes),
                         ("modality_completeness", modalities)):
        frames = [f for f in frames if f is not None and not f.empty]
        if frames:
            path = outroot / f"{name}.csv"
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            print(f"  combined {name} -> {path}")
    return 0


def cmd_imaging_qc(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.imaging.qc import run_cohort_qc

    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        print(f"== imaging QC: {key}")
        result = run_cohort_qc(adapter, limit=args.limit,
                               make_figures=not args.no_figures)
        if result is None or result.empty:
            print("   no local images found; "
                  f"run: gbmplast acquire --dataset {key} --images")
            continue
        out = cfg.paths.results / key / "imaging_qc.csv"
        _emit(result, out, label="imaging_qc")
        fails = result[result["status"] != "pass"]
        if len(fails):
            print(f"   {len(fails)} studies with QC findings:")
            print(fails[["study", "status", "findings"]].head(10)
                  .to_string(index=False))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.imaging.features import extract_cohort_features

    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        print(f"== radiomics: {key}")
        table = extract_cohort_features(adapter, limit=args.limit,
                                        sequences=tuple(args.sequences.split(","))
                                        if args.sequences else None)
        if table is None or table.empty:
            print("   no local images; nothing extracted")
            continue
        out = cfg.paths.processed / key / "radiomics_extracted.csv"
        _emit(table, out, label="radiomics")
    return 0


def cmd_longitudinal(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.longitudinal.build import build_longitudinal

    for key in _resolve_datasets(args.dataset):
        adapter = get_adapter(key, cfg)
        if not adapter.is_buildable()[0]:
            print(f"== {key}: SKIPPED (raw data absent)")
            continue
        print(f"== longitudinal: {key}")
        result = build_longitudinal(adapter, cfg)
        if result is None:
            print("   cohort does not support longitudinal trajectories "
                  "(fewer than 2 dated studies per patient)")
            continue
        out = cfg.paths.processed / key / "longitudinal.csv"
        _emit(result.table, out, label="longitudinal", max_rows=8)
        for line in result.provenance:
            print(f"   {line}")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    if args.task == "next-visit":
        from gbmplast.longitudinal.predict import run_next_visit_analysis
        return run_next_visit_analysis(cfg, dataset=args.dataset or "lumiere",
                                       outcome=args.outcome)
    if args.task == "survival":
        from gbmplast.modeling.survival import run_survival_analysis
        return run_survival_analysis(cfg, dataset=args.dataset or "upenn_gbm")
    raise SystemExit(f"unknown model task {args.task!r}")


def cmd_radiogenomics(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.modeling.radiogenomics import run_radiogenomics

    rc = 0
    for key in _resolve_datasets(args.dataset or "upenn_gbm,ucsd_ptgbm"):
        if key not in {"upenn_gbm", "ucsd_ptgbm"}:
            continue
        print(f"== radiogenomics: {key}")
        rc |= run_radiogenomics(cfg, dataset=key, target=args.target)
    return rc


def cmd_molecular(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    if args.task == "pseudobulk":
        from gbmplast.molecular.pipeline import run_pseudobulk
        return run_pseudobulk(cfg, dataset=args.dataset or "gbm_care",
                              limit=args.limit)
    if args.task == "plasticity":
        from gbmplast.molecular.pipeline import run_plasticity
        return run_plasticity(cfg, dataset=args.dataset or "gbm_care")
    raise SystemExit(f"unknown molecular task {args.task!r}")


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    from gbmplast.report.assemble import assemble_report
    path = assemble_report(cfg)
    print(f"report -> {path}")
    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    adapter = get_adapter("institutional", cfg)
    out = adapter.write_templates(Path(args.outdir) if args.outdir else None)
    print(f"institutional intake templates -> {out}")
    for p in sorted(out.glob("*")):
        print(f"   {p.name}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gbmplast",
        description="Glioblastoma plasticity research pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"gbmplast {__version__}")

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-c", "--config", action="append", default=[],
                         help="YAML config (repeatable; layered left to right)")
        sub.add_argument("--set", action="append", default=[],
                         metavar="KEY=VALUE", help="override a config key")

    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("datasets", help="list cohorts and availability")
    common(p)
    p.add_argument("--fast", action="store_true",
                   help="skip building bundles (listing only)")
    p.set_defaults(func=cmd_datasets)

    p = subs.add_parser("acquire", help="download public tables / subsets")
    common(p)
    p.add_argument("--dataset", default="all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--images", action="store_true",
                   help="LUMIERE: pull an image subset via HTTP range requests")
    p.add_argument("--specimens", action="store_true",
                   help="GBM-CARE: pull per-specimen expression matrices")
    p.add_argument("--inventory", action="store_true",
                   help="TCIA: record the collection's series inventory")
    p.add_argument("--n-patients", type=int, default=5)
    p.add_argument("--patients", default=None,
                   help="comma-separated local patient ids")
    p.add_argument("--dry-run", action="store_true",
                   help="report the byte cost without transferring")
    p.add_argument("--max-bytes", type=float, default=2.0,
                   metavar="GB", help="transfer guard in GB (default 2)")
    p.set_defaults(func=cmd_acquire)

    p = subs.add_parser("canonicalize", help="build the canonical tables")
    common(p)
    p.add_argument("--dataset", default="all")
    p.set_defaults(func=cmd_canonicalize)

    p = subs.add_parser("validate", help="structural and integrity checks")
    common(p)
    p.add_argument("--dataset", default="all")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero when any error is found")
    p.set_defaults(func=cmd_validate)

    p = subs.add_parser("dictionary", help="data dictionaries and audit tables")
    common(p)
    p.add_argument("--dataset", default="all")
    p.set_defaults(func=cmd_dictionary)

    p = subs.add_parser("imaging-qc", help="geometry/orientation/mask QC")
    common(p)
    p.add_argument("--dataset", default="lumiere")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_imaging_qc)

    p = subs.add_parser("features", help="extract radiomics from local images")
    common(p)
    p.add_argument("--dataset", default="lumiere")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--sequences", default=None)
    p.set_defaults(func=cmd_features)

    p = subs.add_parser("longitudinal", help="build patient-visit trajectories")
    common(p)
    p.add_argument("--dataset", default="lumiere")
    p.set_defaults(func=cmd_longitudinal)

    p = subs.add_parser("model", help="fit and evaluate prediction models")
    common(p)
    p.add_argument("task", choices=["next-visit", "survival"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--outcome", default="rano",
                   choices=["rano", "volume"])
    p.set_defaults(func=cmd_model)

    p = subs.add_parser("radiogenomics", help="Module B analyses")
    common(p)
    p.add_argument("--dataset", default=None)
    p.add_argument("--target", default=None,
                   help="mgmt | tumor_vs_treatment (default: per cohort)")
    p.set_defaults(func=cmd_radiogenomics)

    p = subs.add_parser("molecular", help="Module C analyses")
    common(p)
    p.add_argument("task", choices=["pseudobulk", "plasticity"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_molecular)

    p = subs.add_parser("report", help="assemble the results report")
    common(p)
    p.set_defaults(func=cmd_report)

    p = subs.add_parser("templates",
                        help="write institutional intake CSV templates")
    common(p)
    p.add_argument("--outdir", default=None)
    p.set_defaults(func=cmd_templates)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
