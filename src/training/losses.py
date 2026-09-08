"""Multi-task losses: communication cross-entropy and sensing MSE."""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

def assert_finite(t: tf.Tensor, name: str) -> None:
    
    if not t.dtype.is_floating:
        logger.debug(
            "assert_finite: dtype %s non floating, check saltato per %s",
            t.dtype,
            name,
        )
        return
    tf.debugging.assert_all_finite(t, f"{name} contains NaN/Inf")

def comm_ce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    
    tf.debugging.assert_shapes([
        (y_true, ('B',)),
        (y_pred, ('B', 'M'))
    ])
    assert_finite(y_true, "comm_labels")
    assert_finite(y_pred, "comm_logits")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    ce_loss = loss_fn(y_true, y_pred)

    assert_finite(ce_loss, "comm_ce_loss")
    ce_loss = tf.maximum(ce_loss, 0.0)

    logger.debug("comm_ce_loss = %.6f", ce_loss)
    return ce_loss

def comm_ce_loss_factory(label_smoothing: float = 0.0) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    
    if not isinstance(label_smoothing, (int, float)) or not (0.0 <= float(label_smoothing) < 1.0):
        raise ValueError(f"label_smoothing deve essere in [0, 1), ricevuto: {label_smoothing!r}")
    smoothing = float(label_smoothing)

    def _comm_ce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_shapes([
            (y_true, ('B',)),
            (y_pred, ('B', 'M')),
        ])
        assert_finite(y_true, "comm_labels")
        assert_finite(y_pred, "comm_logits")
        y_true_int = tf.cast(y_true, tf.int32)
        log_probs = tf.nn.log_softmax(y_pred, axis=-1)
        if smoothing > 0.0:
            m_float = tf.cast(tf.shape(y_pred)[1], tf.float32)
            y_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[1])
            y_smoothed = (1.0 - smoothing) * y_one_hot + smoothing / m_float
            per_sample = -tf.reduce_sum(y_smoothed * log_probs, axis=-1)
        else:
            per_sample = -tf.reduce_sum(
                tf.one_hot(y_true_int, depth=tf.shape(y_pred)[1]) * log_probs, axis=-1,
            )
        ce_loss = tf.reduce_mean(per_sample)
        assert_finite(ce_loss, "comm_ce_loss")
        ce_loss = tf.maximum(ce_loss, 0.0)
        logger.debug("comm_ce_loss (smoothing=%.3f) = %.6f", smoothing, ce_loss)
        return ce_loss

    return _comm_ce_loss

def _mse_sensing_loss_core(y_true: tf.Tensor, y_pred: tf.Tensor, penalty: float) -> tf.Tensor:
    
    tf.debugging.assert_shapes([
        (y_true, ('B', 2)),
        (y_pred, ('B', 2))
    ])
    assert_finite(y_true, "sensing_labels")
    assert_finite(y_pred, "sensing_pred")

    if tf.executing_eagerly():
        y_true_np = y_true.numpy()
        if np.any(y_true_np < -1e-7) or np.any(y_true_np > 1.0 + 1e-7):
            logger.warning(
                "sensing labels fuori da [0,1]: min=%.4f, max=%.4f",
                np.min(y_true_np), np.max(y_true_np)
            )

    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    mse_loss = loss_fn(y_true, y_pred)

    if penalty > 0.0:
        out_low = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))
        out_high = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - 1.0)))
        mse_loss = mse_loss + penalty * (out_low + out_high)

    assert_finite(mse_loss, "mse_sensing_loss")
    mse_loss = tf.maximum(mse_loss, 0.0)

    logger.debug("mse_sensing_loss (range_penalty=%.2f) = %.6f", penalty, mse_loss)
    return mse_loss

def mse_sensing_loss_factory(
    range_penalty: float = 0.0,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    
    if not isinstance(range_penalty, (int, float)) or not math.isfinite(float(range_penalty)) or float(range_penalty) < 0.0:
        raise ValueError(
            f"range_penalty deve essere un numero finito >= 0, ricevuto: {range_penalty!r}"
        )
    penalty = float(range_penalty)

    def _mse_sensing_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return _mse_sensing_loss_core(y_true, y_pred, penalty)

    _mse_sensing_loss.__name__ = "mse_sensing_loss"
    _mse_sensing_loss.__qualname__ = "mse_sensing_loss"
    return _mse_sensing_loss

def mse_sensing_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    
    return _mse_sensing_loss_core(y_true, y_pred, 0.0)

def combined_loss_factory(lambda_mse: float) -> Callable[[Dict[str, tf.Tensor], Dict[str, tf.Tensor]], tf.Tensor]:
    
    if not isinstance(lambda_mse, (int, float)):
        raise TypeError(f"lambda_mse deve essere un numero, ricevuto: {type(lambda_mse).__name__}")
    if not tf.math.is_finite(lambda_mse):
        raise ValueError(f"lambda_mse deve essere finito, ricevuto: {lambda_mse}")
    if lambda_mse < 0.0:
        raise ValueError(f"lambda_mse deve essere >= 0, ricevuto: {lambda_mse}")

    logger.info("combined_loss_factory: lambda_mse = %s", lambda_mse)

    def _inner_loss(y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        
        try:
            y_true_comm = y_true["comm"]
            y_true_sensing = y_true["sensing"]
            y_pred_comm = y_pred["comm"]
            y_pred_sensing = y_pred["sensing"]
        except KeyError as exc:
            raise ValueError(f"Chiave mancante nei dizionari y_true/y_pred: {exc}")

        assert_finite(y_pred_comm, "y_pred_comm")
        assert_finite(y_pred_sensing, "y_pred_sensing")
        assert_finite(y_true_comm, "y_true_comm")
        assert_finite(y_true_sensing, "y_true_sensing")

        L_comm = comm_ce_loss(y_true_comm, y_pred_comm)
        L_sensing = mse_sensing_loss(y_true_sensing, y_pred_sensing)

        L_total = L_comm + lambda_mse * L_sensing

        assert_finite(L_total, "total_loss")
        L_total = tf.maximum(L_total, 0.0)

        logger.debug("L_comm=%.6f, L_sensing=%.6f, L_total=%.6f", L_comm, L_sensing, L_total)
        return L_total

    return _inner_loss

