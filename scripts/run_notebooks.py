#!/usr/bin/env python
"""Convert every notebook source to .ipynb and execute it.

Notebooks are kept as jupytext ``py:percent`` sources so that diffs are
readable; the executed ``.ipynb`` is a build product.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
VENV_BIN = ROOT / ".venv" / "bin"
TIMEOUT_SECONDS = "1800"


def _tool(name: str) -> str:
    local = VENV_BIN / name
    return str(local) if local.exists() else name


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    sources = ([NOTEBOOKS / a for a in argv] if argv
               else sorted(NOTEBOOKS.glob("*.py")))
    if not sources:
        print("no notebook sources found")
        return 1

    failures: list[str] = []
    for source in sources:
        target = source.with_suffix(".ipynb")
        print(f"== {source.name}")
        convert = subprocess.run(
            [_tool("jupytext"), "--to", "notebook", str(source),
             "-o", str(target)],
            capture_output=True, text=True)
        if convert.returncode != 0:
            print(convert.stderr.strip()[-800:])
            failures.append(f"{source.name} (convert)")
            continue
        execute = subprocess.run(
            [_tool("jupyter"), "nbconvert", "--to", "notebook", "--execute",
             "--inplace", str(target),
             f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}"],
            capture_output=True, text=True)
        if execute.returncode != 0:
            print(execute.stderr.strip()[-1500:])
            failures.append(f"{source.name} (execute)")
        else:
            print(f"   executed -> {target.name}")

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print(f"\nall {len(sources)} notebooks executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
