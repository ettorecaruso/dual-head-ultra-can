"""Receiver evaluation over the SNR grid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import (
    DataDict,
    _build_feature_matrix,
    build_reference_matrix,
    load_npz_files,
    verify_snr_balance,
)
from src.data.dataset_generator import build_snr_grid
from src.evaluation.metrics import bit_error_count, mse_delay_doppler
from src.models.baselines import build_baseline
from src.models.ultra_can import build_dual_head_ultra_can
from src.models.ultra_can_qkv import build_dual_head_ultra_can_qkv
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, validate_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_SNR_ROUND_DECIMALS = 6
_RANGE_TOL = 1e-9
_MIN_BATCH_SIZE = 1
_MSE_NEG_TOL = 1e-9

def _validate_config(config: Dict[str, Any]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    general = config.get("general")
    data = config.get("data")
    training = config.get("training")
    evaluation = config.get("evaluation")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict")
    if not isinstance(data, dict):
        raise ValueError("sezione 'data' mancante o non dict")
    if not isinstance(training, dict):
        raise ValueError("sezione 'training' mancante o non dict")
    if not isinstance(evaluation, dict):
        raise ValueError("sezione 'evaluation' mancante o non dict")

    required_data = ("sequence_length", "max_delay", "max_doppler", "raw_dir", "echoes")
    for key in required_data:
        if key not in data:
            raise ValueError(f"chiave mancante in data: {key}")

    required_eval = ("snr_test_range", "bit_error_threshold", "max_symbols_per_snr")
    for key in required_eval:
        if key not in evaluation:
            raise ValueError(f"chiave mancante in evaluation: {key}")

    max_delay = data.get("max_delay")
    if not isinstance(max_delay, (int, float)) or max_delay <= 0:
        raise ValueError(f"data.max_delay deve essere > 0, ricevuto: {max_delay}")

    max_doppler = data.get("max_doppler")
    if not isinstance(max_doppler, (int, float)) or max_doppler <= 0:
        raise ValueError(f"data.max_doppler deve essere > 0, ricevuto: {max_doppler}")

    snr_test_range = evaluation.get("snr_test_range")
    if not isinstance(snr_test_range, (list, tuple)) or len(snr_test_range) == 0:
        raise ValueError(f"evaluation.snr_test_range deve essere una lista non vuota, ricevuto: {snr_test_range}")

    bit_error_threshold = evaluation.get("bit_error_threshold")
    if not isinstance(bit_error_threshold, int) or bit_error_threshold <= 0:
        raise ValueError(f"evaluation.bit_error_threshold deve essere un int > 0, ricevuto: {bit_error_threshold}")

    max_symbols_per_snr = evaluation.get("max_symbols_per_snr")
    if not isinstance(max_symbols_per_snr, int) or max_symbols_per_snr <= 0:
        raise ValueError(f"evaluation.max_symbols_per_snr deve essere un int > 0, ricevuto: {max_symbols_per_snr}")

    batch_size = training.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(f"training.batch_size deve essere un int >= {_MIN_BATCH_SIZE}, ricevuto: {batch_size}")

    logger.debug("Config evaluator validata")

def evaluate_model(
    model: tf.keras.Model,
    data: DataDict,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model deve essere tf.keras.Model, ricevuto: {type(model).__name__}")
    if not isinstance(data, dict):
        raise TypeError(f"data deve essere dict, ricevuto: {type(data).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    _validate_config(config)

    required_keys = ("x", "bit", "tau", "f_d", "snr_db", "seed")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"data manca delle chiavi: {missing}")

    x = data["x"]
    bit = data["bit"]
    tau = data["tau"]
    f_d = data["f_d"]
    snr_db = data["snr_db"]
    seed = data["seed"]

    if x.shape[0] == 0:
        raise ValueError("data non contiene campioni (N=0)")

    eval_cfg = config["evaluation"]
    snr_test_range = eval_cfg["snr_test_range"]
    bit_error_threshold = eval_cfg["bit_error_threshold"]
    max_symbols_per_snr = eval_cfg["max_symbols_per_snr"]
    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])
    batch_size = int(config["training"]["batch_size"])

    feature_mode = str(config["data"].get("feature_mode", "real"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    x_ref = data.get("x_ref")

    snr_values: List[float] = []
    ber_list: List[float] = []
    mse_tau_list: List[float] = []
    mse_fd_list: List[float] = []
    n_errors_list: List[int] = []
    n_symbols_list: List[int] = []
    corr_tau_list: List[float] = []
    corr_fd_list: List[float] = []

    logger.info("Avvio valutazione su %d punti SNR", len(snr_test_range))

    for snr_target in snr_test_range:
        mask = np.isclose(snr_db, snr_target, rtol=0, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            logger.warning("Nessun campione per SNR=%.1f dB, salto", snr_target)
            continue

        x_snr = None
        bit_snr = bit[idx]
        tau_snr = tau[idx]
        f_d_snr = f_d[idx]

        n_total = len(bit_snr)
        accum_errors = 0
        accum_symbols = 0
        mse_tau_acc = 0.0
        mse_fd_acc = 0.0
        tau_pred_all: List[float] = []
        fd_pred_all: List[float] = []
        tau_true_all: List[float] = []
        fd_true_all: List[float] = []
        n_out_range = 0
        n_out_range_total = 0
        n_out_range_min: Optional[float] = None
        n_out_range_max: Optional[float] = None

        start = 0
        while start < n_total and accum_symbols < max_symbols_per_snr and accum_errors < bit_error_threshold:
            end = min(start + batch_size, n_total)
            raw_idx = idx[start:end]
            batch_bit = bit[raw_idx]
            batch_tau = tau[raw_idx]
            batch_fd = f_d[raw_idx]
            batch_size_actual = len(batch_bit)

            if x_ref is not None:
                batch_ref = x_ref[raw_idx]
            else:
                batch_ref = build_reference_matrix(batch_bit, seed[raw_idx], config)
            batch_x = _build_feature_matrix(
                x[raw_idx], feature_mode, feature_norm, batch_ref
            )

            pred = model(batch_x, training=False)
            comm_logits = pred["comm"].numpy()
            sensing_pred = pred["sensing"].numpy()

            if comm_logits.shape[0] != batch_size_actual or comm_logits.ndim != 2:
                raise ValueError(f"comm_logits shape {comm_logits.shape} non valida")
            if sensing_pred.shape[0] != batch_size_actual or sensing_pred.ndim != 2 or sensing_pred.shape[1] != 2:
                raise ValueError(f"sensing_pred shape {sensing_pred.shape} non valida")
            if not np.all(np.isfinite(comm_logits)):
                raise RuntimeError("comm_logits contiene NaN/Inf")
            if not np.all(np.isfinite(sensing_pred)):
                raise RuntimeError("sensing_pred contiene NaN/Inf")

            n_errors_batch, _ = bit_error_count(
                comm_logits, np.asarray(batch_bit, dtype=np.int64)
            )
            accum_errors += int(n_errors_batch)
            accum_symbols += batch_size_actual

            n_out_batch = int(np.count_nonzero((sensing_pred < 0.0) | (sensing_pred > 1.0)))
            if n_out_batch:
                n_out_range += n_out_batch
                n_out_range_total += sensing_pred.size
                batch_min = float(np.min(sensing_pred))
                batch_max = float(np.max(sensing_pred))
                if n_out_range_min is None or batch_min < n_out_range_min:
                    n_out_range_min = batch_min
                if n_out_range_max is None or batch_max > n_out_range_max:
                    n_out_range_max = batch_max
                logger.debug(
                    "SNR=%.1f dB, batch: %d/%d elementi sensing fuori da [0,1] -> clamp",
                    snr_target, n_out_batch, sensing_pred.size,
                )
            sensing_pred_clamped = np.clip(sensing_pred, 0.0, 1.0)
            tau_pred = sensing_pred_clamped[:, 0] * tau_max
            fd_pred = sensing_pred_clamped[:, 1] * fd_max

            tau_min = np.min(tau_pred)
            tau_max_pred = np.max(tau_pred)
            fd_min = np.min(fd_pred)
            fd_max_pred = np.max(fd_pred)

            if tau_min < 0.0 or tau_max_pred > tau_max + _RANGE_TOL:
                raise RuntimeError(
                    f"tau_pred fuori range [0, {tau_max}] per SNR={snr_target}: "
                    f"min={tau_min:.4f}, max={tau_max_pred:.4f}"
                )
            if fd_min < 0.0 or fd_max_pred > fd_max + _RANGE_TOL:
                raise RuntimeError(
                    f"fd_pred fuori range [0, {fd_max}] per SNR={snr_target}: "
                    f"min={fd_min:.4e}, max={fd_max_pred:.4e}"
                )

            mse_tau_b, mse_fd_b = mse_delay_doppler(
                tau_pred,
                fd_pred,
                np.asarray(batch_tau, dtype=np.float64),
                np.asarray(batch_fd, dtype=np.float64),
            )
            mse_tau_acc += mse_tau_b * batch_size_actual
            mse_fd_acc += mse_fd_b * batch_size_actual
            tau_pred_all.extend(float(v) for v in tau_pred)
            fd_pred_all.extend(float(v) for v in fd_pred)
            tau_true_all.extend(float(v) for v in batch_tau)
            fd_true_all.extend(float(v) for v in batch_fd)

            start = end

        if n_out_range > 0:
            logger.warning(
                "SNR=%.1f dB: %d/%d elementi sensing fuori da [0,1] "
                "(min=%.4f, max=%.4f) -> clamp",
                snr_target, n_out_range, n_out_range_total,
                n_out_range_min, n_out_range_max,
            )

        if accum_symbols == 0:
            logger.warning("Nessun simbolo valutato per SNR=%.1f dB, salto", snr_target)
            continue

        ber = accum_errors / accum_symbols
        mse_tau = mse_tau_acc / accum_symbols
        mse_fd = mse_fd_acc / accum_symbols

        if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
            raise RuntimeError(f"BER non valida per SNR {snr_target}: {ber}")
        if not np.isfinite(mse_tau) or not np.isfinite(mse_fd):
            raise RuntimeError(f"MSE non finito per SNR {snr_target}: mse_tau={mse_tau}, mse_fd={mse_fd}")
        if mse_tau < -_MSE_NEG_TOL or mse_fd < -_MSE_NEG_TOL:
            logger.warning("MSE negativo per SNR=%.1f: mse_tau=%.6f, mse_fd=%.6f", snr_target, mse_tau, mse_fd)
            mse_tau = max(0.0, mse_tau)
            mse_fd = max(0.0, mse_fd)

        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_tau_snr = _pearson(
            np.asarray(tau_pred_all), np.asarray(tau_true_all)
        )
        corr_fd_snr = _pearson(
            np.asarray(fd_pred_all), np.asarray(fd_true_all)
        )

        snr_values.append(float(snr_target))
        ber_list.append(float(ber))
        mse_tau_list.append(float(mse_tau))
        mse_fd_list.append(float(mse_fd))
        n_errors_list.append(accum_errors)
        n_symbols_list.append(accum_symbols)
        corr_tau_list.append(corr_tau_snr)
        corr_fd_list.append(corr_fd_snr)

        logger.debug(
            "SNR=%.1f dB: BER=%.6f, MSE_tau=%.6f, MSE_fd=%.6e, "
            "corr(tau)=%.3f, corr(fD)=%.3f, n_errors=%d, n_symbols=%d",
            snr_target, ber, mse_tau, mse_fd, corr_tau_snr, corr_fd_snr,
            accum_errors, accum_symbols,
        )

    if not snr_values:
        raise RuntimeError("Nessun SNR valutato (dataset vuoto o nessuna corrispondenza)")

    results = {
        "snr_db": np.array(snr_values, dtype=np.float64),
        "ber": np.array(ber_list, dtype=np.float64),
        "mse_tau": np.array(mse_tau_list, dtype=np.float64),
        "mse_fd": np.array(mse_fd_list, dtype=np.float64),
        "n_errors": np.array(n_errors_list, dtype=np.int64),
        "n_symbols": np.array(n_symbols_list, dtype=np.int64),
        "corr_tau": np.array(corr_tau_list, dtype=np.float64),
        "corr_fd": np.array(corr_fd_list, dtype=np.float64),
    }

    logger.info("Valutazione completata per %d punti SNR", len(snr_values))
    return results

def _online_batch_seed(seed_base: int, snr_db: float, batch_idx: int) -> int:
    
    digest = hashlib.sha256(f"{int(seed_base)}:{float(snr_db)}:{int(batch_idx)}".encode())
    return int.from_bytes(digest.digest()[:4], "big")


def _predict_online_chunked(
    model: tf.keras.Model,
    batch_x: np.ndarray,
    predict_batch: int,
    min_batch: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    
    n = int(batch_x.shape[0])
    if isinstance(predict_batch, bool) or int(predict_batch) < 1:
        raise ValueError(
            f"predict_batch deve essere un int >= 1, ricevuto: {predict_batch!r}"
        )
    if isinstance(min_batch, bool) or int(min_batch) < 1:
        raise ValueError(f"min_batch deve essere un int >= 1, ricevuto: {min_batch!r}")
    chunk = int(predict_batch)
    min_batch = int(min_batch)

    comm_parts: List[np.ndarray] = []
    sensing_parts: List[np.ndarray] = []
    start = 0
    while start < n:
        end = min(start + chunk, n)
        try:
            pred = model(batch_x[start:end], training=False)
        except tf.errors.ResourceExhaustedError:
            if chunk <= min_batch:
                logger.error(
                    "OOM persistente nel forward online anche a chunk=%d "
                    "(batch=%d campioni): interrompo la valutazione",
                    min_batch, n,
                )
                raise
            chunk = max(min_batch, chunk // 2)
            logger.warning(
                "OOM nel forward online (chunk=%d campioni): dimezzo a %d e riprovo",
                end - start, chunk,
            )
            gc.collect()
            continue
        comm_parts.append(pred["comm"].numpy())
        sensing_parts.append(pred["sensing"].numpy())
        start = end
    return np.concatenate(comm_parts, axis=0), np.concatenate(sensing_parts, axis=0)


def evaluate_model_online(
    model: tf.keras.Model,
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model deve essere tf.keras.Model, ricevuto: {type(model).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    _validate_config(config)

    eval_cfg = config["evaluation"]
    snr_test_range = eval_cfg["snr_test_range"]
    bit_error_threshold = eval_cfg["bit_error_threshold"]
    max_symbols_per_snr = eval_cfg["max_symbols_per_snr"]
    eval_batch_symbols = int(eval_cfg.get("eval_batch_symbols", 100_000))
    predict_batch = int(eval_cfg.get("eval_predict_batch_size", 1024))
    tau_max = float(config["data"]["max_delay"])
    fd_max = float(config["data"]["max_doppler"])
    echoes = [int(k) for k in config["data"]["echoes"]]
    if not echoes:
        raise ValueError("data.echoes vuoto: impossibile generare i batch online")
    feature_mode = str(config["data"].get("feature_mode", "real"))
    feature_norm = str(config["data"].get("feature_norm", "none"))
    seed_base = int(config["general"].get("seed", 42))

    from src.data.dataset_generator import generate_test_batch

    snr_values: List[float] = []
    ber_list: List[float] = []
    mse_tau_list: List[float] = []
    mse_fd_list: List[float] = []
    n_errors_list: List[int] = []
    n_symbols_list: List[int] = []
    corr_tau_list: List[float] = []
    corr_fd_list: List[float] = []

    logger.info(
        "Valutazione ONLINE su %d punti SNR (batch=%d, max_symbols=%d, "
        "floor BER=%.1e)",
        len(snr_test_range), eval_batch_symbols, max_symbols_per_snr,
        1.0 / (2.0 * max_symbols_per_snr),
    )

    for snr_target in snr_test_range:
        accum_errors = 0
        accum_symbols = 0
        mse_tau_acc = 0.0
        mse_fd_acc = 0.0
        n_corr = 0
        tau_sum_t = 0.0; tau_sum_p = 0.0; tau_sum_t2 = 0.0; tau_sum_p2 = 0.0; tau_sum_tp = 0.0
        fd_sum_t = 0.0; fd_sum_p = 0.0; fd_sum_t2 = 0.0; fd_sum_p2 = 0.0; fd_sum_tp = 0.0
        n_out_range = 0
        n_out_range_total = 0
        n_out_range_min: Optional[float] = None
        n_out_range_max: Optional[float] = None

        batch_idx = 0
        while (
            accum_symbols < max_symbols_per_snr
            and accum_errors < bit_error_threshold
        ):
            k = echoes[batch_idx % len(echoes)]
            rng = np.random.default_rng(
                _online_batch_seed(seed_base, snr_target, batch_idx)
            )
            batch_symbols = min(eval_batch_symbols, max_symbols_per_snr - accum_symbols)
            batch = generate_test_batch(
                config, batch_symbols, snr_target, k, rng
            )
            batch_bit = batch["bit"]
            batch_tau = batch["tau"]
            batch_fd = batch["f_d"]
            batch_size_actual = int(batch_bit.shape[0])

            batch_x = _build_feature_matrix(
                batch["x"], feature_mode, feature_norm, batch["x_ref"]
            )
            comm_logits, sensing_pred = _predict_online_chunked(
                model, batch_x, predict_batch
            )

            if comm_logits.shape[0] != batch_size_actual or comm_logits.ndim != 2:
                raise ValueError(f"comm_logits shape {comm_logits.shape} non valida")
            if (
                sensing_pred.shape[0] != batch_size_actual
                or sensing_pred.ndim != 2
                or sensing_pred.shape[1] != 2
            ):
                raise ValueError(f"sensing_pred shape {sensing_pred.shape} non valida")
            if not np.all(np.isfinite(comm_logits)):
                raise RuntimeError("comm_logits contiene NaN/Inf")
            if not np.all(np.isfinite(sensing_pred)):
                raise RuntimeError("sensing_pred contiene NaN/Inf")

            n_errors_batch, _ = bit_error_count(
                comm_logits, np.asarray(batch_bit, dtype=np.int64)
            )
            accum_errors += int(n_errors_batch)
            accum_symbols += batch_size_actual

            n_out_batch = int(
                np.count_nonzero((sensing_pred < 0.0) | (sensing_pred > 1.0))
            )
            if n_out_batch:
                n_out_range += n_out_batch
                n_out_range_total += sensing_pred.size
                batch_min = float(np.min(sensing_pred))
                batch_max = float(np.max(sensing_pred))
                if n_out_range_min is None or batch_min < n_out_range_min:
                    n_out_range_min = batch_min
                if n_out_range_max is None or batch_max > n_out_range_max:
                    n_out_range_max = batch_max

            sensing_pred_clamped = np.clip(sensing_pred, 0.0, 1.0)
            tau_pred = sensing_pred_clamped[:, 0] * tau_max
            fd_pred = sensing_pred_clamped[:, 1] * fd_max

            tau_min = np.min(tau_pred)
            tau_max_pred = np.max(tau_pred)
            fd_min = np.min(fd_pred)
            fd_max_pred = np.max(fd_pred)
            if tau_min < 0.0 or tau_max_pred > tau_max + _RANGE_TOL:
                raise RuntimeError(
                    f"tau_pred fuori range [0, {tau_max}] per SNR={snr_target}: "
                    f"min={tau_min:.4f}, max={tau_max_pred:.4f}"
                )
            if fd_min < 0.0 or fd_max_pred > fd_max + _RANGE_TOL:
                raise RuntimeError(
                    f"fd_pred fuori range [0, {fd_max}] per SNR={snr_target}: "
                    f"min={fd_min:.4e}, max={fd_max_pred:.4e}"
                )

            mse_tau_b, mse_fd_b = mse_delay_doppler(
                tau_pred,
                fd_pred,
                np.asarray(batch_tau, dtype=np.float64),
                np.asarray(batch_fd, dtype=np.float64),
            )
            mse_tau_acc += mse_tau_b * batch_size_actual
            mse_fd_acc += mse_fd_b * batch_size_actual
            tau_t = np.asarray(batch_tau, dtype=np.float64)
            tau_p = np.asarray(tau_pred, dtype=np.float64)
            fd_t = np.asarray(batch_fd, dtype=np.float64)
            fd_p = np.asarray(fd_pred, dtype=np.float64)
            n_corr += batch_size_actual
            tau_sum_t += float(tau_t.sum()); tau_sum_p += float(tau_p.sum())
            tau_sum_t2 += float((tau_t * tau_t).sum()); tau_sum_p2 += float((tau_p * tau_p).sum())
            tau_sum_tp += float((tau_t * tau_p).sum())
            fd_sum_t += float(fd_t.sum()); fd_sum_p += float(fd_p.sum())
            fd_sum_t2 += float((fd_t * fd_t).sum()); fd_sum_p2 += float((fd_p * fd_p).sum())
            fd_sum_tp += float((fd_t * fd_p).sum())

            batch_idx += 1

        if n_out_range > 0:
            logger.warning(
                "SNR=%.1f dB: %d/%d elementi sensing fuori da [0,1] "
                "(min=%.4f, max=%.4f) -> clamp",
                snr_target, n_out_range, n_out_range_total,
                n_out_range_min, n_out_range_max,
            )

        if accum_symbols == 0:
            logger.warning("Nessun simbolo valutato per SNR=%.1f dB, salto", snr_target)
            continue

        ber = accum_errors / accum_symbols
        mse_tau = mse_tau_acc / accum_symbols
        mse_fd = mse_fd_acc / accum_symbols

        if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
            raise RuntimeError(f"BER non valida per SNR {snr_target}: {ber}")
        if not np.isfinite(mse_tau) or not np.isfinite(mse_fd):
            raise RuntimeError(
                f"MSE non finito per SNR {snr_target}: mse_tau={mse_tau}, mse_fd={mse_fd}"
            )
        if mse_tau < -_MSE_NEG_TOL or mse_fd < -_MSE_NEG_TOL:
            mse_tau = max(0.0, mse_tau)
            mse_fd = max(0.0, mse_fd)

        def _pearson_from_sums(
            n: int, sa: float, sa2: float, sb: float, sb2: float, sab: float
        ) -> float:
            """Pearson da somme incrementali (equivalente a np.corrcoef senza
            materializzare le liste; identico a parita' di floating point)."""
            if n < 2:
                return 0.0
            cov = n * sab - sa * sb
            va = n * sa2 - sa * sa
            vb = n * sb2 - sb * sb
            if va <= 1e-20 or vb <= 1e-20:
                return 0.0
            return float(cov / ((va * vb) ** 0.5))

        corr_tau_snr = _pearson_from_sums(
            n_corr, tau_sum_t, tau_sum_t2, tau_sum_p, tau_sum_p2, tau_sum_tp
        )
        corr_fd_snr = _pearson_from_sums(
            n_corr, fd_sum_t, fd_sum_t2, fd_sum_p, fd_sum_p2, fd_sum_tp
        )

        snr_values.append(float(snr_target))
        ber_list.append(float(ber))
        mse_tau_list.append(float(mse_tau))
        mse_fd_list.append(float(mse_fd))
        n_errors_list.append(accum_errors)
        n_symbols_list.append(accum_symbols)
        corr_tau_list.append(corr_tau_snr)
        corr_fd_list.append(corr_fd_snr)

        logger.debug(
            "SNR=%.1f dB: BER=%.3e (err=%d/%d), MSE_tau=%.6f, corr(tau)=%.3f",
            snr_target, ber, accum_errors, accum_symbols, mse_tau, corr_tau_snr,
        )

    if not snr_values:
        raise RuntimeError("Nessun SNR valutato (snr_test_range vuoto)")

    results = {
        "snr_db": np.array(snr_values, dtype=np.float64),
        "ber": np.array(ber_list, dtype=np.float64),
        "mse_tau": np.array(mse_tau_list, dtype=np.float64),
        "mse_fd": np.array(mse_fd_list, dtype=np.float64),
        "n_errors": np.array(n_errors_list, dtype=np.int64),
        "n_symbols": np.array(n_symbols_list, dtype=np.int64),
        "corr_tau": np.array(corr_tau_list, dtype=np.float64),
        "corr_fd": np.array(corr_fd_list, dtype=np.float64),
    }

    logger.info("Valutazione online completata per %d punti SNR", len(snr_values))
    return results


def compute_ber_curve(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    
    if not isinstance(results, dict):
        raise TypeError(f"results deve essere dict, ricevuto: {type(results).__name__}")
    required = ("snr_db", "ber", "mse_tau", "mse_fd", "n_errors", "n_symbols")
    missing = [k for k in required if k not in results]
    if missing:
        raise ValueError(f"results manca delle chiavi: {missing}")

    df = pd.DataFrame({
        "snr_db": results["snr_db"],
        "ber": results["ber"],
        "mse_tau": results["mse_tau"],
        "mse_fd": results["mse_fd"],
        "n_errors": results["n_errors"],
        "n_symbols": results["n_symbols"],
        "corr_tau": results.get("corr_tau", np.zeros(len(results["snr_db"]))),
        "corr_fd": results.get("corr_fd", np.zeros(len(results["snr_db"]))),
    })

    ber = df["ber"].to_numpy(dtype=np.float64)
    n_symbols = df["n_symbols"].to_numpy(dtype=np.float64)
    zero_mask = (ber == 0.0) & (n_symbols > 0)
    if np.any(zero_mask):
        n_floored = int(np.count_nonzero(zero_mask))
        df["ber"] = np.where(zero_mask, 1.0 / (2.0 * n_symbols), ber)
        logger.info(
            "BER floor applicato a %d punti (nessun errore osservato): "
            "1/(2*n_symbols)", n_floored,
        )

    df = df.sort_values("snr_db").reset_index(drop=True)
    logger.debug("Curve generate: %d punti", len(df))
    return df

def compute_sensing_rmse(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    
    if not isinstance(results, dict):
        raise TypeError(f"results deve essere dict, ricevuto: {type(results).__name__}")
    required = ("snr_db", "mse_tau", "mse_fd")
    missing = [k for k in required if k not in results]
    if missing:
        raise ValueError(f"results manca delle chiavi: {missing}")

    rmse_tau = np.sqrt(np.maximum(results["mse_tau"], 0.0))
    rmse_fd = np.sqrt(np.maximum(results["mse_fd"], 0.0))

    df = pd.DataFrame({
        "snr_db": results["snr_db"],
        "rmse_tau": rmse_tau,
        "rmse_fd": rmse_fd,
    })
    df = df.sort_values("snr_db").reset_index(drop=True)
    logger.debug("RMSE sensing calcolati per %d punti", len(df))
    return df

def plot_ber_vs_snr(
    df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df deve essere pd.DataFrame, ricevuto: {type(df).__name__}")
    if "snr_db" not in df.columns or "ber" not in df.columns:
        raise ValueError("df deve contenere le colonne 'snr_db' e 'ber'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(df["snr_db"], df["ber"], marker="o", linestyle="-", linewidth=2, label=model_name)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"BER vs SNR - {model_name}")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()

    min_ber = df["ber"].min()
    if min_ber > 0:
        y_lower = max(1e-6, min_ber / 10)
    else:
        y_lower = 1e-6
    ax.set_ylim([y_lower, 1.0])

    output_path = output_dir / f"ber_vs_snr_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Plot BER salvato in %s", output_path)
    return output_path

def plot_loss_curves(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(history, tf.keras.callbacks.History):
        raise TypeError(f"history deve essere tf.keras.callbacks.History, ricevuto: {type(history).__name__}")
    if not history.history:
        raise ValueError("history.history vuoto")
    if "loss" not in history.history:
        raise ValueError("history.history must contain the key 'loss'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], label="Train Loss", marker="o", linestyle="-")
    if "val_loss" in history.history:
        ax.plot(epochs, history.history["val_loss"], label="Val Loss", marker="s", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves - {model_name}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    output_path = output_dir / f"loss_curves_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Plot loss salvato in %s", output_path)
    return output_path

def plot_sensing_error(
    df: pd.DataFrame,
    output_dir: Path,
    plot_format: str,
    model_name: str,
) -> Path:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df deve essere pd.DataFrame, ricevuto: {type(df).__name__}")
    required = ("snr_db", "rmse_tau", "rmse_fd")
    for col in required:
        if col not in df.columns:
            raise ValueError(f"df deve contenere la colonna '{col}'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(df["snr_db"], df["rmse_tau"], marker="o", linestyle="-", color="blue")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("RMSE τ (samples)")
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.semilogy(df["snr_db"], df["rmse_fd"], marker="s", linestyle="-", color="orange")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("RMSE fD (cycles/sample)")
    ax2.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(f"Sensing Error vs SNR - {model_name}")

    output_path = output_dir / f"sensing_rmse_{model_name}.{plot_format}"
    plt.savefig(output_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Plot sensing salvato in %s", output_path)
    return output_path

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Valutazione modello Dual-Head Ultra-CAN (ISAC in IoD)"
    )
    parser.add_argument("--config", required=True, help="path della config esperimento (YAML)")
    parser.add_argument(
        "--model",
        choices=["conv1d", "qkv", "lstm", "mc_dlsk"],
        default="conv1d",
        help="modello da valutare (default: conv1d)",
    )
    parser.add_argument(
        "--plot-format",
        default="pdf",
        help="formato dei plot (default: pdf)",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="se specificato, costruisce il modello da zero se il checkpoint non esiste (altrimenti lancia FileNotFoundError)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    _validate_config(config)

    general = config.get("general", {})
    experiment_name = str(general.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    from src.utils.dataset_utils import get_dataset_dir

    data_cfg = config["data"]
    data_dir = get_dataset_dir(config)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset non trovato in {data_dir}. Generare i dati tramite il runner "
            "(src/experiments/runner.py) oppure scripts/generate_all_datasets.sh."
        )
    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])

    logger.info("Caricamento dataset di test da %s", data_dir)
    data = load_npz_files(data_dir, snr_grid, echoes, "test", config)
    logger.info("Dataset test caricato: %d campioni", data["x"].shape[0])
    data["x_ref"] = build_reference_matrix(data["bit"], data["seed"], config)

    verify_snr_balance(data, snr_grid, echoes)

    model_dir = _REPO_ROOT / "results" / experiment_name / args.model / "models"
    checkpoint_candidates: List[Path] = [
        model_dir / "best_model.keras",
        model_dir / "best_model.h5",
    ]
    training_cfg = config.get("training")
    if isinstance(training_cfg, dict) and training_cfg.get("checkpoint_path"):
        ckpt_from_config = Path(str(training_cfg["checkpoint_path"]))
        if not ckpt_from_config.is_absolute():
            ckpt_from_config = _REPO_ROOT / ckpt_from_config
        checkpoint_candidates.append(ckpt_from_config)
    model_path = next(
        (p for p in checkpoint_candidates if p.is_file()),
        checkpoint_candidates[0],
    )
    if model_path.is_file():
        logger.info("Caricamento modello da %s", model_path)
        model = load_model(model_path)
    else:
        if args.allow_missing_model:
            logger.warning("Modello non trovato in %s, costruisco da zero senza training", model_path)
            if args.model == "conv1d":
                model = build_dual_head_ultra_can(config)
            elif args.model == "qkv":
                model = build_dual_head_ultra_can_qkv(config)
            elif args.model in ("lstm", "mc_dlsk"):
                model = build_baseline(config, args.model)
            else:
                raise ValueError(f"Modello non supportato: {args.model}")
        else:
            raise FileNotFoundError(
                f"Modello non trovato: {model_path}. "
                "Addestra il modello prima di valutarlo oppure usa --allow-missing-model per costruirlo da zero."
            )

    results = evaluate_model(model, data, config)

    df_ber = compute_ber_curve(results)
    df_sensing = compute_sensing_rmse(results)

    plot_dir = _REPO_ROOT / "results" / experiment_name / args.model / "plots"
    plot_ber_vs_snr(df_ber, plot_dir, args.plot_format, args.model)
    plot_sensing_error(df_sensing, plot_dir, args.plot_format, args.model)

    history_csv = _REPO_ROOT / "results" / experiment_name / "logs" / "history.csv"
    if history_csv.exists():
        try:
            history_df = pd.read_csv(history_csv)
            if "loss" in history_df.columns:
                history = tf.keras.callbacks.History()
                history.history = {col: history_df[col].tolist() for col in history_df.columns}
                plot_loss_curves(history, plot_dir, args.plot_format, args.model)
            else:
                logger.warning("history.csv non contiene la colonna 'loss', salto plot loss curves")
        except Exception as e:
            logger.warning("Impossibile caricare history per loss curves: %s", e)
    else:
        logger.info("history.csv non trovato, salto plot loss curves")

    csv_path = plot_dir / "metrics.csv"
    df_ber.to_csv(csv_path, index=False)
    logger.info("Metriche salvate in %s", csv_path)

    npz_path = plot_dir / "evaluation_results.npz"
    np.savez(npz_path, **results)
    logger.info("Risultati salvati in %s", npz_path)

    logger.info("Valutazione completata. Risultati in %s", plot_dir)

if __name__ == "__main__":
    main()

