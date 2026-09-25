#!/usr/bin/env python3
"""Link the frozen reference checkpoints into a sandbox run root.

    python scripts/tools/stage_reference_checkpoints.py results/runs/v2_canaleB --mode full

The runner already falls back to ``results/full`` when it resolves the frozen
receivers, so this helper is only needed when that tree lives outside the
repository (for example on a Drive mount in a Colab session): it makes
``<sandbox>/<mode>/ber_vs_snr`` point at it, which is where the experiments of
that run look first. Nothing is ever written inside the reference tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "results" / "full"
STAGED = ("ber_vs_snr", "jamming")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sandbox_root", type=Path, help="Run root, e.g. results/runs/<tag>")
    parser.add_argument("--mode", default="full", choices=["full", "fast"])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--copy", action="store_true", help="Copy instead of symlinking"
    )
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"reference tree not found: {source}")
    target_root = (Path(args.sandbox_root) / args.mode).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    staged = 0
    for name in STAGED:
        origin = source / name
        if not origin.is_dir():
            print(f"skipped {name}: not present in {source}")
            continue
        destination = target_root / name
        if destination.exists() or destination.is_symlink():
            print(f"skipped {name}: already staged at {destination}")
            continue
        if args.copy:
            shutil.copytree(origin, destination)
        else:
            os.symlink(origin, destination, target_is_directory=True)
        staged += 1
        print(f"staged {name} -> {destination}")

    print(f"done: {staged} reference tree(s) staged in {target_root}")


if __name__ == "__main__":
    main()
