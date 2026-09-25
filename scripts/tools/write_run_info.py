#!/usr/bin/env python3
"""Write the provenance manifest and the timings of one run directory.

    python scripts/tools/write_run_info.py <run_dir>

The manifest of a run is a separate file per run, ``RUN_INFO_<timestamp>.txt``,
so the historical ``results/full/logs/RUN_INFO.txt`` of the paper is never
touched. Alongside it the script writes ``timings.json``, built from the CSV
modification times of the run, which is what makes the per-cell calibration of a
Colab session comparable with the next one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
SKIP_DIRS = {"logs", "__pycache__"}
SKIP_SUFFIX = {".pyc", ".log"}


def _git(args: List[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except OSError as exc:
        return f"unavailable ({exc})"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["tf_gpus"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:  # tensorflow is optional for this helper
        info["tensorflow"] = f"unavailable ({type(exc).__name__})"
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    info["nvidia_smi"] = gpu.stdout.strip().splitlines() or []
    return info


def _artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(run_dir).parts)
        if parts & SKIP_DIRS or path.suffix in SKIP_SUFFIX:
            continue
        entries.append({
            "path": str(path.relative_to(REPO)),
            "size": int(path.stat().st_size),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256_16": _sha256(path)[:16],
        })
    return entries


def _timings(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    stamps = sorted(
        time.mktime(time.strptime(entry["mtime"], "%Y-%m-%dT%H:%M:%S"))
        for entry in artifacts
    )
    if not stamps:
        return {"n_artifacts": 0}
    return {
        "n_artifacts": len(artifacts),
        "first_artifact": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamps[0])),
        "last_artifact": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamps[-1])),
        "wall_seconds": round(stamps[-1] - stamps[0], 1),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory produced by the runner")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    artifacts = _artifacts(run_dir)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    timings = _timings(artifacts)
    report = {
        "run_dir": str(run_dir.relative_to(REPO)) if REPO in run_dir.parents else str(run_dir),
        "written": stamp,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_commit_short": _git(["rev-parse", "--short", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "environment": _environment(),
        "timings": timings,
        "artifacts": artifacts,
    }

    manifest = run_dir / f"RUN_INFO_{time.strftime('%Y-%m-%d')}.txt"
    lines = [
        f"run_dir      : {report['run_dir']}",
        f"written      : {stamp}",
        f"git_commit   : {report['git_commit']}",
        f"git_subject  : {_git(['log', '-1', '--pretty=%s'])}",
        f"python       : {report['environment']['python']}",
        f"tensorflow   : {report['environment'].get('tensorflow')}",
        f"tf_gpus      : {report['environment'].get('tf_gpus')}",
        f"nvidia_smi   : {report['environment'].get('nvidia_smi')}",
        f"timings      : {timings}",
        "",
        "--- artifact (sha256[:16]  size  mtime  path) ---",
    ]
    for entry in artifacts:
        lines.append(
            f"{entry['sha256_16']}  {entry['size']:>9}  {entry['mtime']}  {entry['path']}"
        )
    lines.append("")
    lines.append("--- git status ---")
    lines.append(report["git_status"] or "(clean)")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(run_dir / "timings.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved {manifest}")
    print(f"saved {run_dir / 'timings.json'}")


if __name__ == "__main__":
    main()
