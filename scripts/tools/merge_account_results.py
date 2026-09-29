#!/usr/bin/env python3
"""Merge the per-account Colab result trees into one complete ``results/full``.

Every Colab account writes its own root (``DHUC_RUN/<account>``) containing a
``full/<experiment>/<arch>/...`` tree, while the figure pipeline reads a single
tree.  This tool unions the accounts:

    python scripts/tools/merge_account_results.py \\
        ../paper/test_results/conv1d ../paper/test_results/lstm \\
        ../paper/test_results/mc_dlsk ../paper/test_results/qkv_ber \\
        ../paper/test_results/qkv_jam ../paper/test_results/_common \\
        --dest results --also ../paper/test_results/full

Rules:

* per-architecture directories are disjoint, so the union is additive and no
  result is overwritten;
* the per-account logs that would collide are written with the account prefix
  (``logs/conv1d_runner.log``), so no evidence is lost;
* intermediate plots (``*.pdf``, ``*.html``) and TensorBoard event files are
  skipped by default: the figures of the paper are rebuilt from the CSVs by
  ``scripts/make_all.sh``, so the merged tree stays small (``--keep-plots``
  copies them anyway);
* ``MERGE.json`` in the merged tree records sources, counts and collisions.

Note on ``--also``: it is a snapshot taken at merge time.  The aggregate tables
that ``scripts/make_all.sh`` writes afterwards (``operating_region_table.csv``,
``channel_generalization/generalization_table.*``) exist only in ``--dest``, so
refresh the second copy with a plain ``rsync -a --delete <dest>/full/ <also>/``
before committing the paper tree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

SKIP_SUFFIXES = (".pdf", ".html")
SKIP_PARTS = ("tensorboard",)


def _kept(path: Path, rel: Path, keep_plots: bool) -> bool:
    if not keep_plots and path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return not any(part in SKIP_PARTS for part in rel.parts)


def _dest_for(dest_full: Path, rel: Path, account: str) -> Tuple[Path, bool]:
    """Destination of ``rel``; on a real collision the account goes in the name.

    Files that exist once per account (``logs/runner.log``, and also the
    ``<experiment>/logs/checkpoint_provenance.json`` that each account writes for
    its own architecture) must not overwrite each other, so the account name is
    prefixed only when the plain destination is already taken.
    """
    out = dest_full / rel
    if not out.exists():
        return out, False
    out = dest_full / rel.parent / f"{account}_{rel.name}"
    index = 2
    while out.exists():
        out = dest_full / rel.parent / f"{account}{index}_{rel.name}"
        index += 1
    return out, True


def merge(sources: List[Path], dest: Path, keep_plots: bool, also: Path | None) -> Dict:
    dest_full = dest / "full"
    if dest_full.exists():
        shutil.rmtree(dest_full)
    dest_full.mkdir(parents=True)

    stats: Dict[str, Dict[str, int]] = {}
    collisions: List[str] = []
    for src in sources:
        full = src / "full"
        if not full.is_dir():
            raise SystemExit(f"{src}: no full/ tree (expected {full})")
        account = src.name
        kept = skipped = 0
        for path in sorted(p for p in full.rglob("*") if p.is_file()):
            rel = path.relative_to(full)
            if not _kept(path, rel, keep_plots):
                skipped += 1
                continue
            out, collided = _dest_for(dest_full, rel, account)
            if collided:
                collisions.append(f"{account}:{rel.as_posix()} -> {out.name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            kept += 1
        stats[account] = {"kept": kept, "skipped": skipped}
        print("merged %-10s kept=%4d skipped=%4d" % (account, kept, skipped))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dest": str(dest_full),
        "sources": [str(s) for s in sources],
        "keep_plots": bool(keep_plots),
        "accounts": stats,
        "collisions": collisions,
        "files": sorted(p.relative_to(dest_full).as_posix()
                        for p in dest_full.rglob("*") if p.is_file()),
    }
    (dest / "MERGE.json").write_text(json.dumps(manifest, indent=2))
    if collisions:
        print("collisions kept under account-prefixed names:", len(collisions))
    print("merged tree:", dest_full, "|", len(manifest["files"]), "files")

    if also is not None:
        if Path(also).exists():
            shutil.rmtree(also)
        shutil.copytree(dest_full, also)
        (Path(also).parent / "MERGE.json").write_text(json.dumps(manifest, indent=2))
        print("second copy:", also)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path,
                    help="account roots, each containing a full/ tree")
    ap.add_argument("--dest", type=Path, default=Path("results"),
                    help="results root the merged full/ is written to (default: results)")
    ap.add_argument("--also", type=Path, default=None,
                    help="optional second copy of the merged tree (e.g. the paper repo)")
    ap.add_argument("--keep-plots", action="store_true",
                    help="also copy the intermediate PDFs (skipped by default)")
    args = ap.parse_args()
    merge([Path(s) for s in args.sources], Path(args.dest),
          bool(args.keep_plots), args.also)


if __name__ == "__main__":
    main()
