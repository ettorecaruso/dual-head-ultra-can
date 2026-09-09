#!/usr/bin/env python3

"""Interpretability probe under jamming (zero training)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd

from src.data.data_loader import DataDict, _build_feature_matrix
from src.evaluation.metrics import bit_error_count, mse_delay_doppler
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH = 1024
_BASE_SEED = 42
_PARTIAL_BAND_FRACTION = 0.25

_CAPTURE_LAYERS: Dict[str, List[str]] = {
    "conv1d": ["conv1", "conv2", "pool_projection", "attn_weights", "attn_pool",
               "sensing_position", "sensing_delay_profile", "sensing_features"],
    "qkv": ["conv1", "conv2", "qkv_attention", "shared_gap", "pool_projection",
            "sensing_position", "sensing_delay_profile", "sensing_features"],
    "lstm": ["lstm_1", "dropout_lstm", "shared_gap", "lstm_proj",
             "sensing_position", "sensing_delay_profile", "sensing_features"],
    "mc_dlsk": ["mc_bilstm_1", "dropout_mc", "shared_gap", "mc_proj",
                "sensing_position", "sensing_delay_profile", "sensing_features"],
}
_QKV_LAYER_NAME = "qkv_attention"
_ATTN_POOL_LAYER = "attn_weights"


def _row_entropy(w: np.ndarray) -> np.ndarray:
    
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    eps = np.finfo(np.float64).eps
    return -np.sum(w * np.log(np.clip(w, eps, 1.0)), axis=-1)


def _pool_temporal(arr: np.ndarray) -> np.ndarray:
    """Mean temporal pooling for 3D layers ``(B,T,F)``; identity for 2D."""
    arr = np.asarray(arr)
    return arr.mean(axis=1) if arr.ndim == 3 else arr


def _build_probe(model: tf.keras.Model, arch: str) -> Tuple[tf.keras.Model, List[str]]:
    
    cap, outs = [], {}
    for name in _CAPTURE_LAYERS.get(arch, []):
        try:
            layer = model.get_layer(name)
            if layer.output is not None:
                outs[name] = layer.output
                cap.append(name)
        except Exception:
            logger.debug("jamming_interpretability %s: layer %s not present, skipping", arch, name)
    outs["comm_logits"] = model.output["comm"]
    outs["sensing"] = model.output["sensing"]
    probe = tf.keras.Model(model.inputs, outs)
    return probe, cap



def _run_condition(
    model: tf.keras.Model,
    probe: tf.keras.Model,
    cap: List[str],
    arch: str,
    xx: np.ndarray,
    bit: np.ndarray,
    tau: np.ndarray,
    f_d: np.ndarray,
    snr_db: np.ndarray,
    x_ref: np.ndarray,
    clean_store: Optional[Dict[str, np.ndarray]],
    ret_idx: np.ndarray,
    tau_max: float,
    fd_max: float,
    capture_clean: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    
    if not capture_clean and clean_store is None:
        raise ValueError("clean_store is required for non-clean conditions")
    n = xx.shape[0]
    pos = {int(i): p for p, i in enumerate(ret_idx)}
    acc_err, acc_sym = 0, 0
    margin_sum = 0.0
    tau_preds, fd_preds = [], []
    snr_groups: Dict[float, Dict[str, float]] = {}
    attn_ent_sum = attn_max_sum = attn_n = 0.0
    energy: Dict[str, float] = {k: 0.0 for k in cap}
    energy_n: Dict[str, float] = {k: 0.0 for k in cap}
    cos_acc: Dict[str, float] = {k: 0.0 for k in cap}
    cos_n: Dict[str, int] = {k: 0 for k in cap}
    collect: Optional[Dict[str, list]] = (
        {k: [] for k in cap if k != _ATTN_POOL_LAYER} if capture_clean else None
    )

    qkv_layer = model.get_layer(_QKV_LAYER_NAME) if arch == "qkv" else None
    if qkv_layer is not None:
        qkv_layer._store_attention_weights = True

    feat = _build_feature_matrix(xx, "iq", "none", reference=x_ref)

    for s in range(0, n, _BATCH):
        e = min(s + _BATCH, n)
        bx = tf.convert_to_tensor(feat[s:e])
        preds = probe(bx, training=False)
        comm_logits = preds["comm_logits"].numpy()
        sens = preds["sensing"].numpy()
        bits = bit[s:e]

        n_err, _ = bit_error_count(comm_logits, bits.astype(np.int64))
        acc_err += int(n_err)
        acc_sym += int(e - s)
        ar = np.arange(e - s)
        margin_sum += float(np.sum(comm_logits[ar, bits] - comm_logits[ar, 1 - bits]))

        tau_p = np.clip(sens[:, 0], 0.0, 1.0) * tau_max
        fd_p = np.clip(sens[:, 1], 0.0, 1.0) * fd_max
        tau_preds.append(tau_p)
        fd_preds.append(fd_p)
        for u in np.unique(snr_db[s:e]):
            m = snr_db[s:e] == u
            g = snr_groups.setdefault(float(u), {"err": 0.0, "sym": 0.0})
            g["err"] += float(np.count_nonzero(comm_logits[m].argmax(1) != bits[m]))
            g["sym"] += float(m.sum())

        if arch == "conv1d":
            alpha = preds[_ATTN_POOL_LAYER].numpy()
            attn_ent_sum += float(_row_entropy(alpha).sum())
            attn_max_sum += float(np.max(alpha[..., 0], axis=-1).sum())
            attn_n += float(e - s)
        elif qkv_layer is not None:
            a = qkv_layer.last_attention_weights
            if a is not None:
                an = a.numpy()
                attn_ent_sum += float(_row_entropy(an).sum())
                attn_max_sum += float(np.max(an, axis=-1).sum())
                attn_n += float(an.shape[0] * an.shape[1])

        for k in energy:
            act = preds[k].numpy()
            energy[k] += float(np.mean(act ** 2)) * float(e - s)
            energy_n[k] += float(e - s)
            if capture_clean:
                if k not in collect:
                    continue
                pooled = _pool_temporal(act)
                keep = [j for j in range(e - s) if int(s + j) in pos]
                if keep:
                    collect[k].append(pooled[keep].astype(np.float32))
            elif k in clean_store:
                pooled = _pool_temporal(act)
                keep = [j for j in range(e - s) if int(s + j) in pos]
                if keep:
                    pp = np.array([pos[int(s + j)] for j in keep], dtype=np.int64)
                    cl = clean_store[k][pp]
                    qq = pooled[keep]
                    denom = np.linalg.norm(cl, axis=1) * np.linalg.norm(qq, axis=1)
                    denom = np.where(denom < 1e-12, 1.0, denom)
                    cos_acc[k] += float(np.sum(np.sum(cl * qq, axis=1) / denom))
                    cos_n[k] += len(keep)

    clean_out: Optional[Dict[str, np.ndarray]] = None
    if capture_clean:
        clean_out = {k: np.concatenate(v, axis=0) for k, v in collect.items() if v}

    tau_pred = np.concatenate(tau_preds)
    fd_pred = np.concatenate(fd_preds)
    corr_tau = float(np.corrcoef(tau, tau_pred)[0, 1]) if np.std(tau) > 0 else float("nan")

    per_snr: List[Dict[str, Any]] = []
    for u in sorted(snr_groups):
        m = snr_db == u
        corr_t = (float(np.corrcoef(tau[m], tau_pred[m])[0, 1])
                  if (m.sum() > 1 and np.std(tau[m]) > 0) else float("nan"))
        per_snr.append({"snr_db": u, "n_err": int(snr_groups[u]["err"]),
                        "n_sym": int(snr_groups[u]["sym"]),
                        "ber": snr_groups[u]["err"] / max(1.0, snr_groups[u]["sym"]),
                        "corr_tau": corr_t})

    res: Dict[str, Any] = {"n_sym": acc_sym, "n_err": acc_err,
                           "ber": acc_err / max(1, acc_sym),
                           "mse_tau": float(np.mean((tau - tau_pred) ** 2)),
                           "mse_fd": float(np.mean((f_d - fd_pred) ** 2)),
                           "corr_tau": corr_tau,
                           "margin_mean": margin_sum / max(1, acc_sym)}
    res["attn_entropy"] = attn_ent_sum / attn_n if attn_n else float("nan")
    res["attn_maxw"] = attn_max_sum / attn_n if attn_n else float("nan")
    for k in energy:
        res[f"{k}_energy"] = energy[k] / max(1.0, energy_n[k])
        if capture_clean:
            res[f"{k}_cos"] = 1.0 if k in clean_out else float("nan")
        else:
            has_clean = clean_store is not None and k in clean_store
            res[f"{k}_cos"] = (cos_acc[k] / max(1, cos_n[k])) if has_clean else float("nan")
    return res, per_snr, clean_out


def run_jamming_interpretability_probe(
    model: tf.keras.Model,
    arch: str,
    test_data: DataDict,
    config: Dict[str, Any],
    out_dir: Path,
    jsr_values: Sequence[float],
    jammer_types: Sequence[str],
    ret_subset: int = 3000,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    
    from src.experiments.run_jamming import apply_jamming

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(test_data["x"])
    bit, tau = np.asarray(test_data["bit"]), np.asarray(test_data["tau"])
    f_d = np.asarray(test_data["f_d"])
    snr_db = np.asarray(test_data["snr_db"])
    x_ref = np.asarray(test_data["x_ref"])
    n = x.shape[0]
    ret_idx = np.linspace(0, n - 1, max(1, min(ret_subset, n))).astype(np.int64)

    probe, cap = _build_probe(model, arch)
    logger.info("[jamming_interpretability %s] probe ready: %d captured layers", arch, len(cap))

    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])
    base_seed = int(config["general"].get("seed", 42))

    cond_rows: List[Dict[str, Any]] = []
    per_snr_rows: List[Dict[str, Any]] = []
    clean_store: Optional[Dict[str, np.ndarray]] = None

    def _save() -> None:
        pd.DataFrame(cond_rows).to_csv(out_dir / "conditions.csv", index=False)
        pd.DataFrame(per_snr_rows).to_csv(out_dir / "per_snr.csv", index=False)

    def _go(jammer: str, jsr_db: float, is_clean: bool) -> None:
        nonlocal clean_store
        t0c = time.time()
        if is_clean:
            xx, tag = x, "clean"
        else:
            seed = base_seed + (sum(ord(ch) for ch in jammer) % 10000) + int(jsr_db) * 7
            xx = apply_jamming(x, jammer, float(jsr_db), np.random.default_rng(seed))
            tag = jammer
        res, per_snr, clean_out = _run_condition(
            model, probe, cap, arch, xx, bit, tau, f_d, snr_db, x_ref,
            clean_store, ret_idx, tau_max, fd_max, capture_clean=is_clean,
        )
        if is_clean:
            clean_store = clean_out
        res.update({"arch": arch, "jammer": tag,
                    "jsr_db": float("nan") if is_clean else float(jsr_db)})
        cond_rows.append(res)
        for r in per_snr:
            r.update({"arch": arch, "jammer": tag, "jsr_db": res["jsr_db"]})
            per_snr_rows.append(r)
        _save()
        ent = res.get("attn_entropy", float("nan"))
        logger.info("[jamming_interpretability %s] %-11s JSR=%6s | BER=%.4f corr_tau=%.3f "
                    "margin=%.2f entr_attn=%.3f (%.0f s)",
                    arch, tag, "clean" if is_clean else f"{jsr_db:g}dB",
                    res["ber"], res["corr_tau"], res["margin_mean"],
                    ent if np.isfinite(ent) else float("nan"), time.time() - t0c)

    _go("", float("nan"), True)
    for jammer in jammer_types:
        for jsr_db in jsr_values:
            _go(jammer, jsr_db, False)

    metadata = {
        "experiment": "jamming_interpretability", "arch": arch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "jsr_grid": [float(v) for v in jsr_values],
        "jammer_types": list(jammer_types),
        "n_test_samples": int(n), "ret_subset": int(ret_subset),
        "batch_size": _BATCH,
        "n_conditions": len(cond_rows),
        "tag": tag,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[jamming_interpretability %s] completed -> %s (conditions=%d)", arch, out_dir, len(cond_rows))
    return {"conditions_csv": str(out_dir / "conditions.csv"),
            "per_snr_csv": str(out_dir / "per_snr.csv"),
            "n_conditions": len(cond_rows)}


if __name__ == "__main__":
    raise SystemExit("Module not directly executable: use the runner (jamming_interpretability).")
