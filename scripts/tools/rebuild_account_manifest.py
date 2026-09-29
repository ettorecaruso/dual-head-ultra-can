#!/usr/bin/env python3
"""Rebuild the per-account manifest of a Colab root from the artifacts it holds.

Five of the six campaign roots came back without ``manifest.json`` (only
``qkv_ber`` wrote one: the heavy zip in the last notebook cell failed on the
others).  The manifest is bookkeeping, not science, so it is rebuilt here from
what the runner actually wrote:

    python scripts/tools/rebuild_account_manifest.py ../paper/test_results/conv1d ...

For every account root (a directory containing ``full/``) the tool records

* the operating point and its source (``logs/config_used.yaml``);
* the jamming protocol actually used (cap, realizations, jammer types);
* the legs recovered from ``logs/runner.log`` with their wall-clock minutes;
* the checkpoint provenance of every evaluation leg;
* the artifact inventory and the files that are expected but absent.

The result is marked ``"rebuilt_locally": true`` so a reconstructed manifest is
never mistaken for the one the notebook wrote.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import yaml

STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
#: The runner brackets every leg with "EXPERIMENT <name>" and
#: "Experiment <name> completed successfully." (uppercase marker, lowercase end).
START = re.compile(r"EXPERIMENT ([a-z_]+)")
DONE = re.compile(r"Experiment ([a-z_]+) completed successfully")


def read_configs(full: Path) -> List[Path]:
    """Config snapshots that survived in this root, canonical name first."""
    snaps = [p for p in full.rglob("config_used.yaml") if p.is_file()]
    return sorted(snaps, key=lambda p: (p.name != "config_used.yaml", str(p)))


def read_protocol(full: Path) -> Dict:
    for path in read_configs(full):
        data = yaml.safe_load(path.read_text()) or {}
        out = {}
        for name in ("jamming", "jamming_interpretability"):
            section = (data.get("experiments") or {}).get(name) or {}
            if section:
                out[name] = {k: section.get(k) for k in
                             ("max_symbols", "n_realizations",
                              "n_realizations_by_arch", "jamming_types")}
        if out:
            out["_source"] = path.name
            return out
    return {}


def read_map_param(full: Path) -> Dict:
    for path in read_configs(full):
        data = yaml.safe_load(path.read_text()) or {}
        value = (data.get("data") or {}).get("map_param")
        if value is not None:
            return {"map_param": float(value), "map_param_source": path.name}
    return {}


def read_provenance(full: Path) -> Dict:
    merged = {}
    for path in sorted(full.rglob("*checkpoint_provenance.json")):
        merged[path.relative_to(full).as_posix()] = json.loads(path.read_text())
    return merged


def read_legs(full: Path) -> List[Dict]:
    """Legs and their wall-clock time, recovered from the runner log."""
    logs = sorted({p for p in list(full.glob("logs/runner.log"))
                   + list(full.glob("logs/*runner.log"))})
    legs: List[Dict] = []
    for log in logs:
        started: Dict[str, str] = {}
        finished: Dict[str, str] = {}
        last = None
        for line in log.read_text(errors="replace").splitlines():
            stamp = STAMP.match(line)
            if stamp:
                last = stamp.group(1)
            hit = DONE.search(line)
            if hit:
                finished.setdefault(hit.group(1), last)
                continue
            hit = START.search(line)
            if hit:
                started.setdefault(hit.group(1), last)
        for name, begin in started.items():
            end = finished.get(name)
            minutes = None
            if begin and end:
                fmt = "%Y-%m-%d %H:%M:%S"
                minutes = round((time.mktime(time.strptime(end, fmt))
                                 - time.mktime(time.strptime(begin, fmt))) / 60.0, 1)
            legs.append({"experiment": name, "start": begin, "end": end,
                         "minutes": minutes, "log": log.name})
    return legs


#: Expected artifacts per experiment, as a function of the architecture dir.
EXPECT = {
    "ber_vs_snr": lambda a: [f"{a}/metrics.csv", f"{a}/best_model.keras",
                             f"{a}/logs/history.csv"],
    "jamming": lambda a: [f"{a}/jamming/jamming_results_cw.csv",
                          f"{a}/jamming/jamming_results_barrage.csv",
                          f"{a}/jamming/jamming_results_partial_band.csv",
                          f"{a}/best_model.keras"],
    "jamming_interpretability": lambda a: [f"{a}/conditions.csv",
                                           f"{a}/conditions_realizations.csv",
                                           f"{a}/per_snr.csv"],
    "frequency_agility": lambda a: [f"{a}/frequency_agility_vs_dwell.csv",
                                    f"{a}/frequency_agility_vs_jsr.csv",
                                    f"{a}/frequency_agility_vs_reaction.csv"],
    "peer_estimation": lambda a: [f"{a}/peer_estimation.csv"],
    "classical_receivers": lambda a: ["ber_vs_doppler.csv", "ber_vs_k.csv"],
}


#: Directories that are never an architecture: the plot drops of the runner and
#: the per-experiment logs (both are regenerated by scripts/make_all.sh).
NON_ARCH_DIRS = {"logs", "plots"}


def _arch_dirs(root: Path) -> List[Path]:
    return [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name not in NON_ARCH_DIRS]


def inventory(full: Path) -> Dict:
    """Architectures per experiment and the artifacts that are missing."""
    archs: Dict[str, List[str]] = {}
    missing: List[str] = []
    for name, expect in EXPECT.items():
        root = full / name
        if not root.is_dir():
            continue
        if name == "classical_receivers":
            archs[name] = ["classical"]
            missing += [f"{name}/{rel}" for rel in expect("")
                        if not (root / rel).is_file()]
            continue
        found: List[str] = []
        if name == "ber_vs_snr":
            units = [(head, arch.name) for head in _arch_dirs(root)
                     for arch in _arch_dirs(head)]
            for head, arch in units:
                found.append(f"{head.name}/{arch}")
                missing += [f"ber_vs_snr/{head.name}/{rel}" for rel in expect(arch)
                            if not (head / rel).is_file()]
        elif name == "channel_generalization":
            for head in _arch_dirs(root):
                found.append(head.name)
                variants = [v.name for v in sorted(head.iterdir()) if v.is_dir()]
                wanted = [f"{head.name}/summary.csv"] + [
                    f"{head.name}/{v}/metrics.csv" for v in variants]
                missing += [f"{name}/{rel}" for rel in wanted
                            if not (root / rel).is_file()]
        else:
            for head in _arch_dirs(root):
                found.append(head.name)
                missing += [f"{name}/{rel}" for rel in expect(head.name)
                            if not (root / rel).is_file()]
        archs[name] = found
    return {"architectures": archs, "missing": missing}


def rebuild(account: Path, drive_root: str) -> Dict:
    full = account / "full"
    if not full.is_dir():
        raise SystemExit(f"{account}: no full/ tree")
    inv = inventory(full)
    return {
        "root": f"/content/drive/MyDrive/{drive_root}/{account.name}/full",
        "local_tree": str(full),
        "rebuilt_locally": True,
        "note": ("manifest ricostruito localmente dagli artefatti della root: i "
                 "notebook 10-15 non lo hanno scritto su Drive (la cella dello zip "
                 "e' fallita su 5 account su 6; solo qkv_ber ha il manifest "
                 "originale). Il commit non e' registrato dagli artefatti."),
        "files": sorted(p.relative_to(full).as_posix()
                        for p in full.rglob("*") if p.is_file()),
        **read_map_param(full),
        "protocol": read_protocol(full),
        "legs": read_legs(full),
        "checkpoint_provenance": read_provenance(full),
        **inv,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accounts", nargs="+", type=Path,
                    help="account roots (each containing a full/ tree)")
    ap.add_argument("--drive-root", default="DHUC_RUN",
                    help="Drive parent of the account roots (default: DHUC_RUN)")
    args = ap.parse_args()
    for account in args.accounts:
        manifest = rebuild(Path(account), args.drive_root)
        out = Path(account) / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2))
        print("%-10s mu=%s | leg=%2d | missing=%d | file=%3d -> %s"
              % (Path(account).name, manifest.get("map_param"), len(manifest["legs"]),
                 len(manifest["missing"]), len(manifest["files"]), out))
        if manifest["missing"]:
            print("   mancanti:", manifest["missing"][:8])


if __name__ == "__main__":
    main()

