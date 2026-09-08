#!/usr/bin/env python3
"""Jamming-aware training of the QKV receiver."""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

_HERE = Path(__file__).resolve().parent
_REPO = next(
    (str(p) for p in [_HERE, _HERE.parent, _HERE.parent.parent]
     if (p / "configs" / "base_config.yaml").exists()),
    str(_HERE),
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SNR_EVAL = [-1.0, 3.0, 7.0, 11.0, 15.0, 20.0]
JSR_FULL = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]
JAMMERS = ["cw", "barrage", "partial_band"]
JSR_AUG = [-6.0, -2.0, 2.0]
AUG_PROB = 0.5
BASE_SEED = 42


def load_cfg() -> Dict:
    from pathlib import Path as _P
    from src.experiments.runner import load_experiment_config
    from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH
    return load_experiment_config(
        experiment_name="jamming",
        mode="full",
        experiments_yaml_path=_P(_REPO) / "configs/experiments.yaml",
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
        cli_overrides=None,
    )


def ensure_canonical_data(cfg: Dict, canon: Path) -> None:
    
    from src.data.data_loader import build_snr_grid
    from src.data.dataset_generator import generate_dataset

    snr_grid = [float(v) for v in build_snr_grid(cfg["data"]["snr_range"], cfg["data"]["snr_step"])]
    echoes = [int(k) for k in cfg["data"]["echoes"]]
    n_exp = len(snr_grid) * len(echoes)

    def _missing(split: str, subset_snr: bool = False) -> bool:
        if not list(canon.glob(f"{split}_*.npz")):
            return True
        if subset_snr:
            have = {p.name for p in canon.glob(f"{split}_*.npz")}
            want = {f"{split}_snr{s:g}_echo{k}.npz" for s in SNR_EVAL for k in echoes}
            return not want.issubset(have)
        return len(list(canon.glob(f"{split}_*.npz"))) < n_exp

    for split, subset in (("train", False), ("val", False), ("test", True)):
        if _missing(split, subset):
            print(f"[data] genero split '{split}' in {canon.name} (puo' richiedere alcuni minuti)...")
            canon.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            generate_dataset(cfg, split, canon)
            print(f"[data] split '{split}' generato ({time.time()-t0:.0f}s)")


def canonical_dir(cfg: Dict) -> Path:
    from src.utils.dataset_utils import get_dataset_dir
    d = Path(get_dataset_dir(cfg))
    return d if d.is_absolute() else Path(_REPO) / d


def _augment_one(npz_path: Path, out_path: Path) -> None:
    from src.experiments.run_jamming import apply_jamming
    with np.load(npz_path) as z:
        x = z["x"].copy()
        data = {k: z[k] for k in ("bit", "tau", "f_d", "snr_db", "seed")}
        k_val = z["k"]
    n = x.shape[0]
    h = int(hashlib.sha1(npz_path.name.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    file_seed = BASE_SEED * 1000003 + h
    rng = np.random.default_rng(file_seed)
    jam = rng.random(n) < AUG_PROB
    if jam.any():
        types = rng.integers(0, len(JAMMERS), size=int(jam.sum()))
        jsrs = np.asarray(JSR_AUG)[rng.integers(0, len(JSR_AUG), size=int(jam.sum()))]
        idxs = np.nonzero(jam)[0]
        x_new = x.copy()
        for ti in range(len(JAMMERS)):
            for ji, jsr in enumerate(JSR_AUG):
                sel = idxs[(types == ti) & (jsrs == jsr)]
                if sel.size == 0:
                    continue
                sub_rng = np.random.default_rng(file_seed + ti * 101 + ji * 7)
                x_new[sel] = apply_jamming(x[sel], JAMMERS[ti], float(jsr), sub_rng)
        x = x_new
    np.savez(out_path, x=x, **data, k=k_val)


def build_augmented_train(aug_dir: Path, canon: Path) -> Path:
    
    files = sorted(canon.glob("train_*.npz"))
    if not files:
        raise FileNotFoundError(f"Nessun file train in {canon}")
    if len(list(aug_dir.glob("train_*.npz"))) >= len(files):
        print(f"[aug] train augmentato gia' presente in {aug_dir}")
        return aug_dir
    aug_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, f in enumerate(files):
        _augment_one(f, aug_dir / f.name)
        if (i + 1) % 7 == 0:
            print(f"[aug] {i+1}/{len(files)} file ({time.time()-t0:.0f}s)")
    print(f"[aug] train augmentato completato: {len(files)} file in {aug_dir} ({time.time()-t0:.0f}s)")
    return aug_dir


def _train(cfg: Dict, aug_dir: Path, canon: Path, out_dir: Path) -> Dict:
    import tensorflow as tf
    from src.data.data_loader import (build_reference_matrix, build_snr_grid,
                                      build_tf_dataset, load_npz_files,
                                      verify_snr_balance)
    from src.experiments.pipeline import build_model
    from src.training.trainer import Trainer

    snr_grid = [float(v) for v in build_snr_grid(cfg["data"]["snr_range"], cfg["data"]["snr_step"])]
    echoes = [int(k) for k in cfg["data"]["echoes"]]
    bs = int(cfg["training"]["batch_size"])
    seed = int(cfg["general"]["seed"])

    def _load(split: str, ddir: Path):
        d = load_npz_files(ddir, snr_grid, echoes, split, cfg)
        verify_snr_balance(d, snr_grid, echoes)
        d["x_ref"] = build_reference_matrix(d["bit"], d["seed"], cfg)
        return d

    train_data = _load("train", aug_dir)
    val_data = _load("val", canon)
    train_ds = build_tf_dataset(train_data, batch_size=bs, config=cfg, shuffle=True, seed=seed)
    val_ds = build_tf_dataset(val_data, batch_size=bs, config=cfg, shuffle=False, seed=seed)

    model = build_model(cfg, "qkv")
    cfg.setdefault("general", {})["run_output_dir"] = str(out_dir)
    trainer = Trainer(cfg, model, train_ds, val_ds)
    t0 = time.time()
    history = trainer.train()
    trainer.restore_best_weights()
    best = out_dir / "best_model.keras"
    model.save(best)
    print(f"[train] QKV jamming-aware: {len(history.history.get('loss', []))} epoche, best in {best} ({time.time()-t0:.0f}s)")
    return {"model": model, "history": history, "path": best}


def _probe(model, cfg, canon: Path, out_dir: Path) -> None:
    from src.data.data_loader import (build_reference_matrix, load_npz_files)
    print("[probe] caricamento test subset e riferimento ISAC (240k campioni)...")
    from src.experiments.jamming_interpretability import run_jamming_interpretability_probe
    echoes = [int(k) for k in cfg["data"]["echoes"]]
    test_data = load_npz_files(canon, SNR_EVAL, echoes, "test", cfg)
    t0 = time.time()
    test_data["x_ref"] = build_reference_matrix(test_data["bit"], test_data["seed"], cfg)
    print(f"[probe] test pronto ({test_data[chr(120)].shape[0]} campioni) in {time.time()-t0:.0f}s; avvio griglia (log per condizione attivi)...")
    run_jamming_interpretability_probe(model=model, arch="qkv", test_data=test_data, config=cfg,
                   out_dir=out_dir, jsr_values=JSR_FULL, jammer_types=JAMMERS,
                   ret_subset=3000, tag="jamming_aware_training")


def _compare(out_dir: Path) -> None:
    import pandas as pd
    base = Path(_REPO) / "results/full/jamming_interpretability/qkv/conditions.csv"
    new = out_dir / "conditions.csv"
    if not base.exists() or not new.exists():
        print("[compare] baseline o nuovo mancante; skip confronto")
        return
    a = pd.read_csv(base); b = pd.read_csv(new)
    m = a.merge(b, on=["jammer", "jsr_db"], suffixes=("_clean_trained", "_jam_aware"))
    cols = ["jammer", "jsr_db", "ber_clean_trained", "ber_jam_aware",
            "margin_mean_clean_trained", "margin_mean_jam_aware"]
    m[[c for c in cols if c in m.columns]].to_csv(out_dir / "comparison_clean_vs_jamaware.csv", index=False)
    print(f"[compare] salvato {out_dir/'comparison_clean_vs_jamaware.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-train", action="store_true", help="riusa best_model gia' addestrato")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    canon = canonical_dir(cfg)
    ensure_canonical_data(cfg, canon)
    aug_dir = canon.parent / (canon.name + "_jamaug")
    out_root = Path(_REPO) / "results/full/jamming_interpretability/jamming_aware_training"
    out_dir = out_root / "qkv"
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_dir = build_augmented_train(aug_dir, canon)

    best = out_dir / "best_model.keras"
    if args.skip_train and best.exists():
        print(f"[train] skip: riuso {best}")
        from src.utils.model_io import load_model
        model = load_model(best)
    else:
        print(f"[train] avvio jamming-aware QKV (canon={canon.name}, aug={aug_dir.name})")
        model = _train(cfg, aug_dir, canon, out_dir)["model"]

    print("[probe] valutazione griglia jamming_interpretability (clean + 3 jammer x 6 JSR)...")
    _probe(model, cfg, canon, out_dir)
    _compare(out_dir)
    print("FATTO ->", out_dir)


if __name__ == "__main__":
    main()
