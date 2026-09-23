"""Cross-channel generalization of the frozen receivers.

The receivers are trained on the nominal aerial channel of the paper and then
evaluated without any retraining on held-out channel models that differ in the
echo fading law, in the direct-path Rician factor, or in the number of echoes.
The transmitted waveforms, the SNRs and the sensing labels keep the same
semantics, so the measured degradation is attributable to the channel mismatch
only.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.data_loader import _build_feature_matrix
from src.data.dataset_generator import generate_test_batch
from src.evaluation.metrics import bit_error_count
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VARIANT_CHANNEL_KEYS = (
    "echo_fading",
    "echo_fading_kappa_db",
    "direct_fading_kappa_db",
    "num_echoes_model",
    "poisson_echoes_mean",
)
_PREDICT_BATCH = 1024
_OPERATING_SNR_DB = 5.0
_TARGET_BER = 1e-4


def variant_config(config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    channel = dict(cfg.get("channel") or {})
    for key in _VARIANT_CHANNEL_KEYS:
        if key in variant:
            channel[key] = variant[key]
    cfg["channel"] = channel
    return cfg


def _predict(
    model: tf.keras.Model,
    feat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    logits_parts: List[np.ndarray] = []
    sensing_parts: List[np.ndarray] = []
    for start in range(0, feat.shape[0], _PREDICT_BATCH):
        end = min(start + _PREDICT_BATCH, feat.shape[0])
        preds = model(tf.convert_to_tensor(feat[start:end]), training=False)
        logits_parts.append(preds["comm"].numpy())
        sensing_parts.append(preds["sensing"].numpy())
    if not logits_parts:
        raise RuntimeError("empty prediction batch")
    return np.concatenate(logits_parts, axis=0), np.concatenate(sensing_parts, axis=0)


def _batch_seed(base_seed: int, variant_index: int, snr_db: float, batch_idx: int) -> int:
    value = (
        int(base_seed) * 1000003
        + int(variant_index) * 15485863
        + int(round(float(snr_db) * 100.0)) * 7919
        + int(batch_idx) * 104729
    )
    return int(abs(value) % (2**31 - 1))


def _evaluate_variant(
    model: tf.keras.Model,
    arch: str,
    variant_cfg: Dict[str, Any],
    variant_name: str,
    variant_index: int,
    snr_range: List[float],
    echoes: List[int],
    num_symbols: int,
    error_threshold: int,
    max_symbols: int,
    batch_symbols: int,
    feature_mode: str,
    feature_norm: str,
    base_seed: int,
    tau_max: float,
    fd_max: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for snr in snr_range:
        accum_err = 0
        accum_sym = 0
        batch_idx = 0
        tau_true: List[np.ndarray] = []
        tau_pred_parts: List[np.ndarray] = []
        fd_true: List[np.ndarray] = []
        fd_pred_parts: List[np.ndarray] = []
        while accum_sym < max_symbols and accum_err < error_threshold:
            k = echoes[batch_idx % len(echoes)]
            rng = np.random.default_rng(
                _batch_seed(base_seed, variant_index, snr, batch_idx)
            )
            n_batch = min(batch_symbols, max_symbols - accum_sym)
            batch = generate_test_batch(variant_cfg, n_batch, float(snr), k, rng)
            feat = _build_feature_matrix(
                batch["x"], feature_mode, feature_norm, reference=batch["x_ref"]
            )
            logits, sensing = _predict(model, feat)
            bit = np.asarray(batch["bit"], dtype=np.int64)
            n_err, _ = bit_error_count(logits, bit)
            accum_err += int(n_err)
            accum_sym += int(bit.shape[0])
            tau_true.append(np.asarray(batch["tau"], dtype=np.float64))
            fd_true.append(np.asarray(batch["f_d"], dtype=np.float64))
            tau_pred_parts.append(np.clip(sensing[:, 0], 0.0, 1.0) * tau_max)
            fd_pred_parts.append(np.clip(sensing[:, 1], 0.0, 1.0) * fd_max)
            batch_idx += 1

        tau_t = np.concatenate(tau_true)
        tau_p = np.concatenate(tau_pred_parts)
        fd_t = np.concatenate(fd_true)
        fd_p = np.concatenate(fd_pred_parts)
        corr_tau = (
            float(np.corrcoef(tau_t, tau_p)[0, 1]) if np.std(tau_t) > 0.0 else float("nan")
        )
        corr_fd = (
            float(np.corrcoef(fd_t, fd_p)[0, 1]) if np.std(fd_t) > 0.0 else float("nan")
        )
        ber = accum_err / max(1, accum_sym)
        rows.append({
            "arch": arch,
            "variant": variant_name,
            "snr_db": float(snr),
            "ber": float(ber),
            "n_errors": int(accum_err),
            "n_symbols": int(accum_sym),
            "mse_tau": float(np.mean((tau_t - tau_p) ** 2)),
            "mse_fd": float(np.mean((fd_t - fd_p) ** 2)),
            "corr_tau": corr_tau,
            "corr_fd": corr_fd,
        })
        logger.info(
            "[channel_generalization] %-24s SNR=%6.1f dB | BER=%.6f (%d/%d) corr_tau=%.4f",
            variant_name, float(snr), ber, accum_err, accum_sym, corr_tau,
        )
    return rows


def _pooled_ber(rows: List[Dict[str, Any]], min_snr_db: float) -> float:
    errors = 0
    symbols = 0
    for row in rows:
        if float(row["snr_db"]) >= float(min_snr_db) - 1e-9:
            errors += int(row["n_errors"])
            symbols += int(row["n_symbols"])
    if symbols == 0:
        return float("nan")
    return errors / float(symbols)


def _min_snr_at_target(rows: List[Dict[str, Any]], target: float) -> float:
    for row in sorted(rows, key=lambda item: float(item["snr_db"])):
        if float(row["ber"]) <= float(target):
            return float(row["snr_db"])
    return float("nan")


def evaluate_channel_generalization(
    model: tf.keras.Model,
    config: Dict[str, Any],
    output_dir: Path,
    arch: str,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cg = (config.get("experiments") or {}).get("channel_generalization") or {}
    variants = list(cg.get("variants") or [{"name": "nominal"}])
    snr_range = [
        float(v)
        for v in (cg.get("snr_test_range") or config["evaluation"]["snr_test_range"])
    ]
    snr_range = sorted(snr_range)
    num_symbols = int(cg.get("num_symbols_per_snr", 2000))
    error_threshold = int(cg.get("bit_error_threshold", 100))
    max_symbols = int(cg.get("max_symbols_per_snr", 2000000))
    batch_symbols = int(cg.get("eval_batch_symbols", 100000))
    feature_mode = str(config["data"].get("feature_mode", "iq"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    echoes = [int(value) for value in config["data"]["echoes"]]
    if not echoes:
        raise ValueError("data.echoes is empty")
    base_seed = int(config["general"].get("seed", 42))
    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])

    logger.info(
        "[channel_generalization %s] variants=%s snr=%s symbols/snr>=%d",
        arch, [str(v.get("name")) for v in variants], snr_range, num_symbols,
    )

    all_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(
                f"variant {index} must be a dict, got: {type(variant).__name__}"
            )
        name = str(variant.get("name") or f"variant_{index}")
        variant_cfg = variant_config(config, variant)
        channel = dict(variant_cfg.get("channel") or {})
        rows = _evaluate_variant(
            model=model,
            arch=arch,
            variant_cfg=variant_cfg,
            variant_name=name,
            variant_index=index,
            snr_range=snr_range,
            echoes=echoes,
            num_symbols=num_symbols,
            error_threshold=error_threshold,
            max_symbols=max_symbols,
            batch_symbols=batch_symbols,
            feature_mode=feature_mode,
            feature_norm=feature_norm,
            base_seed=base_seed,
            tau_max=tau_max,
            fd_max=fd_max,
        )
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(variant_dir / "metrics.csv", index=False)
        all_rows.extend(rows)
        finite_corr = [float(r["corr_tau"]) for r in rows if np.isfinite(r["corr_tau"])]
        summary.append({
            "arch": arch,
            "variant": name,
            "echo_fading": str(channel.get("echo_fading", "none")),
            "echo_fading_kappa_db": float(channel.get("echo_fading_kappa_db") or 0.0),
            "direct_fading_kappa_db": float(
                channel.get("direct_fading_kappa_db")
                if channel.get("direct_fading_kappa_db") is not None
                else config["data"]["rician_kappa_db"]
            ),
            "num_echoes_model": str(channel.get("num_echoes_model", "fixed")),
            "poisson_echoes_mean": float(channel.get("poisson_echoes_mean") or 0.0),
            "pooled_ber": _pooled_ber(rows, _OPERATING_SNR_DB),
            "min_snr_at_1e-4": _min_snr_at_target(rows, _TARGET_BER),
            "corr_tau_top": max(finite_corr) if finite_corr else float("nan"),
            "mse_tau_top": min((float(r["mse_tau"]) for r in rows), default=float("nan")),
        })
        min_snr = summary[-1]["min_snr_at_1e-4"]
        logger.info(
            "[channel_generalization %s] %-24s pooled_ber=%.6f min_snr@1e-4=%s",
            arch, name, summary[-1]["pooled_ber"],
            f"{min_snr:.1f}" if np.isfinite(min_snr) else "n/a",
        )

    pd.DataFrame(all_rows).to_csv(output_dir / "metrics.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "channel_generalization",
        "arch": arch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tensorflow_version": tf.__version__,
        "variants": [str(v.get("name")) for v in variants],
        "snr_range": snr_range,
        "num_symbols_per_snr": num_symbols,
        "bit_error_threshold": error_threshold,
        "operating_snr_db": _OPERATING_SNR_DB,
        "target_ber": _TARGET_BER,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "metrics_csv": str(output_dir / "metrics.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "n_variants": len(variants),
    }
