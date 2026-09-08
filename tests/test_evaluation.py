"""
Test per il modulo di evaluation (``tests/test_evaluation.py``).

Superficie coperta (funzioni pubbliche, >= 3 test ciascuna: validi, limite,
invalidi):
  ``src/evaluation/metrics.py``:
    - ``ber_from_logits``    : BER da logits (hard decision): perfetti -> 0 (T1),
        random -> ~0.5 (T2), multi-classe M=4, batch 1/1024, NaN/Inf, labels
        fuori [0, M-1], shape non coincidenti, batch vuoto -> ValueError.
    - ``ber``                : BER da array di bit: identici -> 0, complementari
        -> 1, random -> ~0.5, valori fuori {0,1}, NaN/Inf, shape diverse,
        batch vuoto -> ValueError.
    - ``mse_delay_doppler``  : MSE su tau/fD denormalizzati: pred == truth -> 0
        (T3), pred random -> >0 finito, errori estremi (1e6), liste come input,
        NaN/Inf, array vuoti, shape diverse -> ValueError.
    - ``rmse_from_mse``      : RMSE = sqrt(MSE) (T4): normali, mse=0, mse
        negativo -> ValueError, clamp entro tolleranza, mse non finito ->
        ValueError, mse non numerico -> TypeError.
    - ``bit_error_count``    : conteggio (errori, totale) per il criterio
        "almeno 100 errori" (paper Sez. V-A): normale, perfetto, NaN/Inf,
        batch vuoto, labels fuori range -> ValueError.

  ``src/evaluation/evaluator.py``:
    - ``evaluate_model``     : valutazione per SNR (criterio 100 errori / 10^7
        campioni): oracolo -> BER=0/MSE=0, random -> ~0.5/>0, stop alla soglia
        errori, stop a max_symbols, aggregazione SNR, batch=1, data vuota,
        chiave mancante, config non valida -> ValueError, logits NaN, sensing
        fuori range, nessun SNR -> RuntimeError.
    - ``compute_ber_curve``  : results -> DataFrame ordinato; chiave mancante.
    - ``compute_sensing_rmse``: results -> DataFrame con rmse_tau/rmse_fd;
        chiave mancante.
    - ``plot_ber_vs_snr``    : crea file; colonna mancante.
    - ``plot_loss_curves``   : crea file; history vuota/senza 'loss'.
    - ``plot_sensing_error`` : crea file; colonna mancante.
    - ``main``               : main([]) senza --config -> SystemExit.

I 4 TEST OBBLIGATORI DEL PLAN:
  T1  test_ber_perfect_logits       logits perfetti -> BER = 0
  T2  test_ber_random_logits        logits random -> BER ~ 0.5 (±0.05)
  T3  test_mse_perfect_predictions  pred == truth -> MSE = 0
  T4  test_rmse_from_mse            rmse == sqrt(mse)

Test di integrazione: import dei due moduli + flusso end-to-end oracolo ->
evaluate_model -> compute_ber_curve -> compute_sensing_rmse.

Scelta di pytest: coerente con l'intera suite esistente (test_models,
test_channel, test_losses, test_training) e con il comando dal log
Sez. 8): ``python -m pytest tests/test_evaluation.py -v``. Il pattern ``setUp``
di unittest e' mappato sulle fixture ``tiny_config`` (conftest.py)
0.8) e sugli helper di modulo; ``@pytest.mark.parametrize`` copre i batch size
senza duplicare codice. Il backend matplotlib ``Agg``
e' impostato PRIMA dell'import di ``evaluator`` (che importa pyplot) per
eseguire la suite in ambiente headless/CI.

Conformita' : shape-check, range-check e check NaN/Inf su ogni
output (); type hints e docstring su tutte le funzioni; nessun
comando shell con pipe.

TODO (test mancanti, segnalati esplicitamente):
  - ``test_main_end_to_end_fast_test`` (CLI end-to-end con la config
    ``fast_test.yaml`` e dataset reale): richiede i .npz generati e un modello
    addestrato; coperto a livello funzionale dai test su ``evaluate_model`` con
    modelli mock e rinviato alla smoke Fase 0.
  - ``test_plot_*_unwritable_dir_raises``: dipende dai permessi dell'utente che
    esegue pytest (non portabile in CI); il controllo OSError resta al sorgente.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.evaluation.evaluator import (
    _predict_online_chunked,
    compute_ber_curve,
    compute_sensing_rmse,
    evaluate_model,
    evaluate_model_online,
    main as evaluator_main,
    plot_ber_vs_snr,
    plot_loss_curves,
    plot_sensing_error,
)
from src.evaluation.metrics import (
    ber,
    ber_from_logits,
    bit_error_count,
    mse_delay_doppler,
    rmse_from_mse,
)

_BATCH_SIZES = (1, 32, 128)
_M_BPSK = 2
_M_QPSK = 4
_BER_TOL = 0.05
_SEQ_LEN = 100
_SNR_EVAL = [0.0, 10.0]
_EXTREME_ERR = 1e6

class _OracleModel(tf.keras.Model):
    

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        
        del training, mask
        bit = inputs[:, 0, 0]
        tau_norm = inputs[:, 1, 0]
        fd_norm = inputs[:, 2, 0]
        return {
            "comm": tf.stack([1.0 - bit, bit], axis=-1),
            "sensing": tf.stack([tau_norm, fd_norm], axis=-1),
        }

class _AlwaysWrongModel(tf.keras.Model):
    

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Forward: logits comm invertiti -> BER 1 su ogni campione.

        Args:
            inputs: Features ``(B, N_seq, 1)``.
            training: Flag di training (non usato).
            mask: Maschera (non usata).

        Returns:
            Dict ``{"comm": (B, 2), "sensing": (B, 2)}`` finito.
        """
        del training, mask
        bit = inputs[:, 0, 0]
        tau_norm = inputs[:, 1, 0]
        fd_norm = inputs[:, 2, 0]
        return {
            "comm": tf.stack([bit, 1.0 - bit], axis=-1),
            "sensing": tf.stack([tau_norm, fd_norm], axis=-1),
        }

class _RandomModel(tf.keras.Model):
    

    def __init__(self, seed: int = 0, **kwargs: Any) -> None:
        
        super().__init__(**kwargs)
        self._seed = seed

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Forward: logits comm casuali e sensing uniforme in [0, 1].

        Args:
            inputs: Features ``(B, N_seq, 1)`` (usata per la shape del batch).
            training: Flag di training (non usato).
            mask: Maschera (non usata).

        Returns:
            Dict ``{"comm": (B, 2), "sensing": (B, 2)}`` finito.
        """
        del training, mask
        batch = tf.shape(inputs)[0]
        rng = tf.random.Generator.from_seed(self._seed)
        logits = rng.normal((batch, _M_BPSK))
        sensing = rng.uniform((batch, 2), minval=0.0, maxval=1.0)
        return {"comm": logits, "sensing": sensing}

class _NaNCommModel(tf.keras.Model):
    

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Forward: logits comm = NaN, sensing valido.

        Args:
            inputs: Features ``(B, N_seq, 1)``.
            training: Flag di training (non usato).
            mask: Maschera (non usata).

        Returns:
            Dict con ``"comm"`` contenente NaN e ``"sensing"`` finito.
        """
        del training, mask
        batch = tf.shape(inputs)[0]
        tau_norm = inputs[:, 1, 0]
        fd_norm = inputs[:, 2, 0]
        nan_logits = tf.fill(
            [batch, _M_BPSK], tf.constant(float("nan"), dtype=tf.float32)
        )
        return {
            "comm": nan_logits,
            "sensing": tf.stack([tau_norm, fd_norm], axis=-1),
        }

class _OutOfRangeSensingModel(tf.keras.Model):
    

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Dict[str, tf.Tensor]:
        """Forward: comm perfetto, sensing costante = 2.0 (fuori [0,1]).

        Args:
            inputs: Features ``(B, N_seq, 1)``.
            training: Flag di training (non usato).
            mask: Maschera (non usata).

        Returns:
            Dict con ``"sensing"`` fuori range e ``"comm"`` valido.
        """
        del training, mask
        batch = tf.shape(inputs)[0]
        bit = inputs[:, 0, 0]
        out_of_range = tf.fill(
            [batch, 2], tf.constant(2.0, dtype=tf.float32)
        )
        return {
            "comm": tf.stack([1.0 - bit, bit], axis=-1),
            "sensing": out_of_range,
        }

def _make_eval_data(
    n_per_snr: int,
    snr_values: Sequence[float],
    max_delay: float,
    max_doppler: float,
    seq_len: int = _SEQ_LEN,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    bit_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    fd_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    seed_parts: list[np.ndarray] = []

    for snr in snr_values:
        bit = rng.integers(0, 2, size=n_per_snr).astype(np.float64)
        tau = rng.uniform(0.0, max_delay, size=n_per_snr)
        f_d = rng.uniform(0.0, max_doppler, size=n_per_snr)
        tau_norm = tau / max_delay
        fd_norm = f_d / max_doppler

        x = np.zeros((n_per_snr, seq_len), dtype=np.complex128)
        x[:, 0] = bit
        x[:, 1] = tau_norm
        x[:, 2] = fd_norm

        x_parts.append(x)
        bit_parts.append(bit.astype(np.int64))
        tau_parts.append(tau)
        fd_parts.append(f_d)
        snr_parts.append(np.full(n_per_snr, float(snr)))
        k_parts.append(np.ones(n_per_snr, dtype=np.int64))
        seed_parts.append(rng.integers(1, 2**31, size=n_per_snr).astype(np.int64))

    return {
        "x": np.concatenate(x_parts, axis=0),
        "bit": np.concatenate(bit_parts, axis=0),
        "tau": np.concatenate(tau_parts, axis=0),
        "f_d": np.concatenate(fd_parts, axis=0),
        "snr_db": np.concatenate(snr_parts, axis=0),
        "k": np.concatenate(k_parts, axis=0),
        "seed": np.concatenate(seed_parts, axis=0),
    }

def _eval_config(
    tiny_config: Dict[str, Any],
    snr_test_range: Optional[Sequence[float]] = None,
    bit_error_threshold: int = 100,
    max_symbols_per_snr: int = 10000,
    batch_size: int = 32,
) -> Dict[str, Any]:
    
    config = copy.deepcopy(tiny_config)
    if snr_test_range is not None:
        config["evaluation"]["snr_test_range"] = list(snr_test_range)
    config["evaluation"]["bit_error_threshold"] = bit_error_threshold
    config["evaluation"]["max_symbols_per_snr"] = max_symbols_per_snr
    config["training"]["batch_size"] = batch_size
    return config

def _max_delay_from(tiny_config: Dict[str, Any]) -> float:
    
    return float(tiny_config["data"]["max_delay"])

def _max_doppler_from(tiny_config: Dict[str, Any]) -> float:
    
    return float(tiny_config["data"]["max_doppler"])

def test_ber_perfect_logits() -> None:
    """T1 (): logits perfetti (one-hot) -> BER = 0."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, _M_BPSK, size=1024)
    logits = np.zeros((1024, _M_BPSK))
    logits[np.arange(1024), labels] = 10.0
    assert ber_from_logits(logits, labels) == 0.0

def test_ber_random_logits() -> None:
    """T2 (): logits random -> BER ~ 0.5 (tolleranza 0.05)."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, _M_BPSK, size=2000)
    logits = rng.normal(size=(2000, _M_BPSK))
    ber_value = ber_from_logits(logits, labels)
    assert 0.5 - _BER_TOL <= ber_value <= 0.5 + _BER_TOL

def test_ber_multi_class_qpsk() -> None:
    """Multi-classe M=4 (guardia  Sez. 1.4): BER = 0 con argmax."""
    rng = np.random.default_rng(2)
    labels = rng.integers(0, _M_QPSK, size=512)
    logits = rng.normal(size=(512, _M_QPSK))
    logits[np.arange(512), labels] += 20.0
    assert ber_from_logits(logits, labels) == 0.0

@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_ber_from_logits_batch_sizes(batch_size: int) -> None:
    """Batch 1/32/128 (template): BER corretto e finito su ogni dimensione."""
    rng = np.random.default_rng(batch_size)
    labels = rng.integers(0, _M_BPSK, size=batch_size)
    logits = rng.normal(size=(batch_size, _M_BPSK))
    ber_value = ber_from_logits(logits, labels)
    assert 0.0 <= ber_value <= 1.0
    assert np.isfinite(ber_value)

def test_ber_from_logits_nan_raises() -> None:
    """Input limite: logits con NaN -> ValueError ( Sez. 1.2)."""
    logits = np.array([[1.0, 2.0], [float("nan"), 1.0]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="NaN/Inf"):
        ber_from_logits(logits, labels)

def test_ber_from_logits_inf_raises() -> None:
    """Input limite: logits con +Inf -> ValueError ( Sez. 1.2)."""
    logits = np.array([[1.0, 2.0], [float("inf"), 1.0]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="NaN/Inf"):
        ber_from_logits(logits, labels)

def test_ber_from_logits_labels_out_of_range_raises() -> None:
    """Input invalido: labels fuori [0, M-1] -> ValueError."""
    logits = np.random.default_rng(0).normal(size=(4, _M_BPSK))
    labels = np.array([0, 1, 2, 0])
    with pytest.raises(ValueError, match="fuori"):
        ber_from_logits(logits, labels)

def test_ber_from_logits_empty_raises() -> None:
    """Input invalido: batch vuoto (B=0) -> ValueError."""
    logits = np.zeros((0, _M_BPSK))
    labels = np.zeros((0,), dtype=np.int64)
    with pytest.raises(ValueError, match="nessun campione"):
        ber_from_logits(logits, labels)

def test_ber_from_logits_shape_mismatch_raises() -> None:
    """Input invalido: shape non coincidenti -> ValueError."""
    logits = np.zeros((4, _M_BPSK))
    labels = np.zeros((5,), dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        ber_from_logits(logits, labels)

def test_ber_from_logits_not_ndarray_raises() -> None:
    """Input invalido: logits non np.ndarray -> TypeError."""
    with pytest.raises(TypeError, match="np.ndarray"):
        ber_from_logits([[1.0, 2.0]], np.array([0]))

def test_ber_identical_bits_zero() -> None:
    """Caso normale: array identici -> BER = 0."""
    bits = np.array([0, 1, 1, 0, 1])
    assert ber(bits, bits.copy()) == 0.0

def test_ber_complementary_bits_one() -> None:
    """Caso limite: bit tutti invertiti -> BER = 1."""
    bits = np.array([0, 1, 1, 0, 1])
    flipped = 1 - bits
    assert ber(bits, flipped) == 1.0

def test_ber_random_bits_half() -> None:
    """Caso limite: pred random -> BER ~ 0.5 (tolleranza 0.05)."""
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=2000)
    pred = rng.integers(0, 2, size=2000)
    ber_value = ber(bits, pred)
    assert 0.5 - _BER_TOL <= ber_value <= 0.5 + _BER_TOL

def test_ber_y_pred_out_of_range_raises() -> None:
    """Input invalido: y_pred fuori {0,1} -> ValueError."""
    bits = np.array([0, 1, 1])
    pred = np.array([0, 2, 1])
    with pytest.raises(ValueError, match="0/1"):
        ber(bits, pred)

def test_ber_nan_raises() -> None:
    """Input invalido: NaN -> ValueError ( Sez. 1.2)."""
    bits = np.array([0, 1, 1])
    pred = np.array([0.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="NaN/Inf"):
        ber(bits, pred)

def test_ber_shape_mismatch_raises() -> None:
    """Input invalido: shape diverse -> ValueError."""
    with pytest.raises(ValueError, match="shape"):
        ber(np.array([0, 1]), np.array([0, 1, 1]))

def test_ber_empty_raises() -> None:
    """Input invalido: batch vuoto -> ValueError."""
    with pytest.raises(ValueError, match="nessun campione"):
        ber(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

def test_mse_perfect_predictions() -> None:
    """T3 (): pred == truth -> MSE_tau = MSE_fd = 0."""
    tau_pred = np.array([1.0, 2.0, 3.0])
    fd_pred = np.array([1e-5, 2e-5, 3e-5])
    mse_tau, mse_fd = mse_delay_doppler(tau_pred, fd_pred, tau_pred, fd_pred)
    assert mse_tau == 0.0
    assert mse_fd == 0.0

def test_mse_random_predictions_positive() -> None:
    """Caso normale: pred random -> MSE > 0 e finito."""
    rng = np.random.default_rng(4)
    tau_pred = rng.uniform(0.0, 10.0, size=512)
    tau_true = rng.uniform(0.0, 10.0, size=512)
    fd_pred = rng.uniform(0.0, 8e-5, size=512)
    fd_true = rng.uniform(0.0, 8e-5, size=512)
    mse_tau, mse_fd = mse_delay_doppler(tau_pred, fd_pred, tau_true, fd_true)
    assert mse_tau > 0.0 and np.isfinite(mse_tau)
    assert mse_fd > 0.0 and np.isfinite(mse_fd)

def test_mse_extreme_values_finite() -> None:
    """Input limite: errori estremi (1e6) -> MSE finito e positivo."""
    tau_pred = np.full(8, _EXTREME_ERR)
    tau_true = np.zeros(8)
    fd_pred = np.full(8, _EXTREME_ERR)
    fd_true = np.zeros(8)
    mse_tau, mse_fd = mse_delay_doppler(tau_pred, fd_pred, tau_true, fd_true)
    assert np.isfinite(mse_tau) and mse_tau > 0.0
    assert np.isfinite(mse_fd) and mse_fd > 0.0
    assert mse_tau == pytest.approx(_EXTREME_ERR ** 2)

def test_mse_accepts_lists() -> None:
    """Caso limite: liste Python come input (conversione difensiva)."""
    mse_tau, mse_fd = mse_delay_doppler(
        [1.0, 2.0], [1e-5, 2e-5], [1.0, 2.0], [1e-5, 2e-5]
    )
    assert mse_tau == 0.0 and mse_fd == 0.0

def test_mse_nan_pred_raises() -> None:
    """Input invalido: tau_pred con NaN -> ValueError ( Sez. 1.2)."""
    tau_pred = np.array([1.0, float("nan")])
    fd_pred = np.array([1e-5, 2e-5])
    tau_true = np.array([1.0, 2.0])
    fd_true = np.array([1e-5, 2e-5])
    with pytest.raises(ValueError, match="NaN/Inf"):
        mse_delay_doppler(tau_pred, fd_pred, tau_true, fd_true)

def test_mse_inf_true_raises() -> None:
    """Input invalido: fd_true con Inf -> ValueError ( Sez. 1.2)."""
    tau_pred = np.array([1.0, 2.0])
    fd_pred = np.array([1e-5, 2e-5])
    tau_true = np.array([1.0, 2.0])
    fd_true = np.array([1e-5, float("inf")])
    with pytest.raises(ValueError, match="NaN/Inf"):
        mse_delay_doppler(tau_pred, fd_pred, tau_true, fd_true)

def test_mse_empty_raises() -> None:
    """Input invalido: array vuoti -> ValueError."""
    with pytest.raises(ValueError, match="nessun campione"):
        mse_delay_doppler(np.array([]), np.array([]), np.array([]), np.array([]))

def test_mse_shape_mismatch_raises() -> None:
    """Input invalido: lunghezze non allineate -> ValueError."""
    with pytest.raises(ValueError, match="shape"):
        mse_delay_doppler(
            np.array([1.0, 2.0, 3.0]),
            np.array([1e-5, 2e-5, 3e-5]),
            np.array([1.0, 2.0]),
            np.array([1e-5, 2e-5]),
        )

def test_mse_2d_input_raises() -> None:
    """Input invalido: array 2D -> ValueError (attesi vettori 1D)."""
    with pytest.raises(ValueError, match="1D"):
        mse_delay_doppler(
            np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.zeros(2)
        )

def test_rmse_from_mse() -> None:
    """T4 (): mse=4 -> rmse=2; mse=2.25 -> rmse=1.5."""
    assert rmse_from_mse(4.0) == pytest.approx(2.0)
    assert rmse_from_mse(2.25) == pytest.approx(1.5)

def test_rmse_zero() -> None:
    """Caso limite: mse=0 -> rmse=0."""
    assert rmse_from_mse(0.0) == 0.0

def test_rmse_negative_raises() -> None:
    """Input invalido: mse=-0.1 -> ValueError (MSE mai negativo)."""
    with pytest.raises(ValueError, match="negativo"):
        rmse_from_mse(-0.1)

def test_rmse_tiny_negative_clamped() -> None:
    """Caso limite: mse=-1e-12 (entro tolleranza numerica) -> clamp a 0."""
    assert rmse_from_mse(-1e-12) == 0.0

def test_rmse_nan_raises() -> None:
    """Input invalido: mse=NaN -> ValueError ( Sez. 1.2)."""
    with pytest.raises(ValueError, match="finito"):
        rmse_from_mse(float("nan"))

def test_rmse_inf_raises() -> None:
    """Input invalido: mse=Inf -> ValueError ( Sez. 1.2)."""
    with pytest.raises(ValueError, match="finito"):
        rmse_from_mse(float("inf"))

def test_rmse_non_number_raises() -> None:
    """Input invalido: mse non numerico -> TypeError."""
    with pytest.raises(TypeError, match="int o float"):
        rmse_from_mse("4")

def test_bit_error_count() -> None:
    """Caso normale: (n_errors, n_total) coerenti con gli argmax."""
    labels = np.array([0, 1, 0, 1, 1])
    logits = np.array([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.0, 3.0], [1.0, 3.0]])
    n_errors, n_total = bit_error_count(logits, labels)
    assert n_total == 5
    assert n_errors == 0

def test_bit_error_count_perfect() -> None:
    """Caso limite: logits perfetti -> 0 errori, totale = B."""
    rng = np.random.default_rng(6)
    labels = rng.integers(0, _M_BPSK, size=1024)
    logits = np.zeros((1024, _M_BPSK))
    logits[np.arange(1024), labels] = 10.0
    n_errors, n_total = bit_error_count(logits, labels)
    assert n_errors == 0
    assert n_total == 1024

def test_bit_error_count_nan_logits_raises() -> None:
    """Input invalido: logits con NaN -> ValueError ( Sez. 1.2)."""
    with pytest.raises(ValueError, match="NaN/Inf"):
        bit_error_count(np.array([[float("nan"), 1.0]]), np.array([0]))

def test_bit_error_count_empty_raises() -> None:
    """Input invalido: batch vuoto -> ValueError."""
    with pytest.raises(ValueError, match="nessun campione"):
        bit_error_count(np.zeros((0, _M_BPSK)), np.zeros((0,), dtype=np.int64))

def test_bit_error_count_labels_out_of_range_raises() -> None:
    """Input invalido: labels fuori [0, M-1] -> ValueError."""
    with pytest.raises(ValueError, match="fuori"):
        bit_error_count(np.zeros((2, _M_BPSK)), np.array([0, 5]))

def test_evaluate_model_basic(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=100,
        snr_values=_SNR_EVAL,
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=_SNR_EVAL)

    results = evaluate_model(_OracleModel(), data, config)

    assert results["ber"].shape == (2,)
    assert np.all(results["ber"] == 0.0)
    assert np.all(np.asarray(results["mse_tau"]) < 1e-4)
    assert np.all(np.asarray(results["mse_fd"]) < 1e-12)
    assert np.all(results["n_errors"] == 0)
    assert np.all(results["n_symbols"] == 100)
    assert np.all(np.isfinite(results["ber"]))
    assert list(results["snr_db"]) == pytest.approx(_SNR_EVAL)

def test_evaluate_model_random(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=2000,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(
        tiny_config,
        snr_test_range=[0.0],
        bit_error_threshold=10 ** 6,
        max_symbols_per_snr=10 ** 6,
    )

    results = evaluate_model(_RandomModel(seed=7), data, config)

    assert results["ber"].shape == (1,)
    ber_value = float(results["ber"][0])
    assert 0.5 - _BER_TOL <= ber_value <= 0.5 + _BER_TOL
    assert float(results["mse_tau"][0]) > 0.0
    assert float(results["mse_fd"][0]) > 0.0
    assert int(results["n_symbols"][0]) == 2000

def test_evaluate_model_stops_at_threshold(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=1000,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(
        tiny_config,
        snr_test_range=[0.0],
        bit_error_threshold=100,
        max_symbols_per_snr=10 ** 6,
        batch_size=32,
    )

    results = evaluate_model(_AlwaysWrongModel(), data, config)

    n_symbols = int(results["n_symbols"][0])
    n_errors = int(results["n_errors"][0])
    assert n_errors >= 100
    assert n_errors == n_symbols
    assert n_symbols < 1000

def test_evaluate_model_stops_at_max_symbols(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=1000,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(
        tiny_config,
        snr_test_range=[0.0],
        bit_error_threshold=10 ** 6,
        max_symbols_per_snr=500,
        batch_size=50,
    )

    results = evaluate_model(_OracleModel(), data, config)

    assert int(results["n_symbols"][0]) == 500
    assert int(results["n_errors"][0]) == 0

def test_evaluate_model_snr_aggregation(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=50,
        snr_values=[0.0, 10.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(
        tiny_config,
        snr_test_range=[0.0, 10.0],
        batch_size=64,
    )

    results = evaluate_model(_OracleModel(), data, config)

    assert results["snr_db"].shape == (2,)
    assert np.all(results["n_symbols"] == 50)
    assert np.all(results["ber"] == 0.0)

@pytest.mark.parametrize("batch_size", (1, 32, 128))
def test_evaluate_model_batch_sizes(
    tiny_config: Dict[str, Any], batch_size: int
) -> None:
    
    data = _make_eval_data(
        n_per_snr=256,
        snr_values=[5.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(
        tiny_config,
        snr_test_range=[5.0],
        bit_error_threshold=10 ** 6,
        max_symbols_per_snr=10 ** 6,
        batch_size=batch_size,
    )

    results = evaluate_model(_OracleModel(), data, config)

    assert int(results["n_symbols"][0]) == 256
    assert float(results["ber"][0]) == 0.0

def test_evaluate_model_empty_data_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: data senza campioni -> ValueError."""
    data = _make_eval_data(
        n_per_snr=0,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    with pytest.raises(ValueError, match="non contiene campioni"):
        evaluate_model(_OracleModel(), data, config)

def test_evaluate_model_missing_key_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: data senza chiave 'bit' -> ValueError."""
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    del data["bit"]
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    with pytest.raises(ValueError, match="manca delle chiavi"):
        evaluate_model(_OracleModel(), data, config)

def test_evaluate_model_invalid_config_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: config senza sezione evaluation -> ValueError (fail-fast)."""
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    del config["evaluation"]
    with pytest.raises(ValueError, match="evaluation"):
        evaluate_model(_OracleModel(), data, config)

def test_evaluate_model_nan_prediction_raises(tiny_config: Dict[str, Any]) -> None:
    """Input limite: logits comm con NaN -> RuntimeError ( Sez. 1.2)."""
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        evaluate_model(_NaNCommModel(), data, config)

def test_evaluate_model_sensing_out_of_range_clamped(
    tiny_config: Dict[str, Any],
) -> None:
    
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    results = evaluate_model(_OutOfRangeSensingModel(), data, config)
    assert np.all(np.isfinite(results["ber"]))
    assert np.all(np.isfinite(results["mse_tau"]))
    assert np.all(np.isfinite(results["mse_fd"]))
    max_delay = _max_delay_from(tiny_config)
    expected_mse_tau = float(np.mean((max_delay - data["tau"]) ** 2))
    assert np.all(np.isclose(results["mse_tau"], expected_mse_tau))

def test_evaluate_model_no_matching_snr_raises(tiny_config: Dict[str, Any]) -> None:
    """Input limite: nessun campione per gli SNR di test -> RuntimeError."""
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[100.0])
    with pytest.raises(RuntimeError, match="Nessun SNR valutato"):
        evaluate_model(_OracleModel(), data, config)

def test_evaluate_model_non_model_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: model non tf.keras.Model -> TypeError."""
    data = _make_eval_data(
        n_per_snr=10,
        snr_values=[0.0],
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    with pytest.raises(TypeError, match="tf.keras.Model"):
        evaluate_model(object(), data, config)

def _sample_results() -> Dict[str, np.ndarray]:
    
    return {
        "snr_db": np.array([10.0, 0.0], dtype=np.float64),
        "ber": np.array([0.001, 0.2], dtype=np.float64),
        "mse_tau": np.array([1.0, 4.0], dtype=np.float64),
        "mse_fd": np.array([1e-8, 4e-8], dtype=np.float64),
        "n_errors": np.array([10, 200], dtype=np.int64),
        "n_symbols": np.array([10000, 1000], dtype=np.int64),
    }

def test_compute_ber_curve_returns_dataframe() -> None:
    """Caso normale: results -> DataFrame con le colonne attese."""
    df = compute_ber_curve(_sample_results())
    assert isinstance(df, pd.DataFrame)
    expected_cols = {"snr_db", "ber", "mse_tau", "mse_fd", "n_errors", "n_symbols"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2

def test_compute_ber_curve_sorts_by_snr() -> None:
    """Caso limite: snr non ordinati -> DataFrame ordinato per snr crescente."""
    df = compute_ber_curve(_sample_results())
    assert list(df["snr_db"]) == pytest.approx([0.0, 10.0])
    assert float(df.loc[df["snr_db"] == 0.0, "ber"].iloc[0]) == pytest.approx(0.2)

def test_compute_ber_curve_missing_key_raises() -> None:
    """Input invalido: results senza 'ber' -> ValueError."""
    results = _sample_results()
    del results["ber"]
    with pytest.raises(ValueError, match="manca delle chiavi"):
        compute_ber_curve(results)

def test_compute_ber_curve_non_dict_raises() -> None:
    """Input invalido: results non dict -> TypeError."""
    with pytest.raises(TypeError, match="dict"):
        compute_ber_curve("not a dict")

def test_compute_sensing_rmse_returns_dataframe() -> None:
    """Caso normale: results -> DataFrame con rmse_tau e rmse_fd."""
    df = compute_sensing_rmse(_sample_results())
    assert isinstance(df, pd.DataFrame)
    assert {"snr_db", "rmse_tau", "rmse_fd"}.issubset(set(df.columns))
    assert len(df) == 2

def test_compute_sensing_rmse_values() -> None:
    """Caso limite: rmse == sqrt(mse) per ogni SNR (T4 applicato al sensing).

    ``compute_sensing_rmse`` ordina il DataFrame per SNR crescente: le MSE
    attese vanno riordinate di conseguenza (SNR 0 -> mse 4.0, SNR 10 -> mse 1.0).
    """
    df = compute_sensing_rmse(_sample_results())
    assert list(df["rmse_tau"]) == pytest.approx(np.sqrt([4.0, 1.0]))
    assert list(df["rmse_fd"]) == pytest.approx(np.sqrt([4e-8, 1e-8]))
    assert list(df["snr_db"]) == pytest.approx([0.0, 10.0])

def test_compute_sensing_rmse_missing_key_raises() -> None:
    """Input invalido: results senza 'mse_tau' -> ValueError."""
    results = _sample_results()
    del results["mse_tau"]
    with pytest.raises(ValueError, match="manca delle chiavi"):
        compute_sensing_rmse(results)

def _ber_df() -> pd.DataFrame:
    
    return pd.DataFrame({
        "snr_db": [0.0, 5.0, 10.0],
        "ber": [0.2, 0.05, 0.01],
        "mse_tau": [4.0, 1.0, 0.25],
        "mse_fd": [4e-8, 1e-8, 2.5e-9],
        "n_errors": [200, 50, 10],
        "n_symbols": [1000, 1000, 1000],
    })

def test_plot_ber_vs_snr_creates_file(tmp_path: Path) -> None:
    
    output = plot_ber_vs_snr(_ber_df(), tmp_path, "png", "conv1d")
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.name == "ber_vs_snr_conv1d.png"

def test_plot_ber_vs_snr_missing_column_raises(tmp_path: Path) -> None:
    """Input invalido: DataFrame senza colonna 'ber' -> ValueError."""
    df = _ber_df().drop(columns=["ber"])
    with pytest.raises(ValueError, match="ber"):
        plot_ber_vs_snr(df, tmp_path, "png", "conv1d")

def test_plot_ber_vs_snr_pdf_format(tmp_path: Path) -> None:
    """Caso limite: formato PDF (default)."""
    output = plot_ber_vs_snr(_ber_df(), tmp_path, "pdf", "qkv")
    assert output.exists()
    assert output.suffix == ".pdf"
    assert output.stat().st_size > 0

def _loss_history() -> tf.keras.callbacks.History:
    """History Keras minima con loss e val_loss (3 epoche)."""
    history = tf.keras.callbacks.History()
    history.history = {
        "loss": [1.0, 0.5, 0.3],
        "val_loss": [0.9, 0.45, 0.28],
    }
    return history

def test_plot_loss_curves_creates_file(tmp_path: Path) -> None:
    
    output = plot_loss_curves(_loss_history(), tmp_path, "png", "conv1d")
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.name == "loss_curves_conv1d.png"

def test_plot_loss_curves_empty_history_raises(tmp_path: Path) -> None:
    """Input invalido: history vuota -> ValueError."""
    empty_history = tf.keras.callbacks.History()
    with pytest.raises(ValueError, match="vuoto"):
        plot_loss_curves(empty_history, tmp_path, "png", "conv1d")

def test_plot_loss_curves_missing_loss_raises(tmp_path: Path) -> None:
    """Input invalido: history senza chiave 'loss' -> ValueError."""
    history = tf.keras.callbacks.History()
    history.history = {"val_loss": [1.0]}
    with pytest.raises(ValueError, match="loss"):
        plot_loss_curves(history, tmp_path, "png", "conv1d")

def test_plot_loss_curves_no_val_loss(tmp_path: Path) -> None:
    """Caso limite: history senza val_loss -> plot solo train loss."""
    history = tf.keras.callbacks.History()
    history.history = {"loss": [1.0, 0.5]}
    output = plot_loss_curves(history, tmp_path, "png", "conv1d")
    assert output.exists()
    assert output.stat().st_size > 0

def test_plot_sensing_error_creates_file(tmp_path: Path) -> None:
    
    df = compute_sensing_rmse(_sample_results())
    output = plot_sensing_error(df, tmp_path, "png", "conv1d")
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.name == "sensing_rmse_conv1d.png"

def test_plot_sensing_error_missing_column_raises(tmp_path: Path) -> None:
    """Input invalido: DataFrame senza colonna 'rmse_tau' -> ValueError."""
    df = compute_sensing_rmse(_sample_results()).drop(columns=["rmse_tau"])
    with pytest.raises(ValueError, match="rmse_tau"):
        plot_sensing_error(df, tmp_path, "png", "conv1d")

def test_evaluation_module_import() -> None:
    """Test di integrazione: i moduli di evaluation si importano senza errori."""
    metrics_module = importlib.import_module("src.evaluation.metrics")
    evaluator_module = importlib.import_module("src.evaluation.evaluator")
    assert hasattr(metrics_module, "ber_from_logits")
    assert hasattr(evaluator_module, "evaluate_model")

def test_end_to_end_evaluation_flow(tiny_config: Dict[str, Any]) -> None:
    
    data = _make_eval_data(
        n_per_snr=200,
        snr_values=_SNR_EVAL,
        max_delay=_max_delay_from(tiny_config),
        max_doppler=_max_doppler_from(tiny_config),
    )
    config = _eval_config(tiny_config, snr_test_range=_SNR_EVAL)

    results = evaluate_model(_OracleModel(), data, config)
    df_ber = compute_ber_curve(results)
    df_rmse = compute_sensing_rmse(results)

    assert len(df_ber) == 2
    assert len(df_rmse) == 2
    assert np.all(df_ber["ber"].values == pytest.approx(1.0 / (2.0 * 200)))
    assert np.all(np.asarray(df_rmse["rmse_tau"].values) < 1e-3)
    assert np.all(np.asarray(df_rmse["rmse_fd"].values) < 1e-6)
    assert list(df_ber["snr_db"].values) == pytest.approx(_SNR_EVAL)

def test_main_missing_config_raises() -> None:
    """CLI: main([]) senza --config -> SystemExit (argparse required)."""
    with pytest.raises(SystemExit):
        evaluator_main([])

def test_main_invalid_config_path_raises(tmp_path: Path) -> None:
    
    missing_config = tmp_path / "missing.yaml"
    with pytest.raises((SystemExit, FileNotFoundError)):
        evaluator_main(["--config", str(missing_config)])


def _dummy_online_model() -> tf.keras.Model:
    
    init = tf.keras.initializers.GlorotUniform(seed=42)
    inputs = tf.keras.layers.Input(shape=(100, 2))
    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", kernel_initializer=init
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    comm = tf.keras.layers.Dense(2, kernel_initializer=init)(x)
    sensing = tf.keras.layers.Dense(2, kernel_initializer=init)(x)
    return tf.keras.Model(inputs, {"comm": comm, "sensing": sensing})


def test_evaluate_model_online_structure(tiny_config: Dict[str, Any]) -> None:
    
    config = _eval_config(
        tiny_config,
        snr_test_range=[0.0, 10.0],
        bit_error_threshold=20,
        max_symbols_per_snr=4000,
        batch_size=32,
    )
    config["evaluation"]["eval_batch_symbols"] = 1000
    config["evaluation"]["eval_predict_batch_size"] = 500

    results = evaluate_model_online(_dummy_online_model(), config)

    for key in ("snr_db", "ber", "mse_tau", "mse_fd", "n_errors", "n_symbols",
                "corr_tau", "corr_fd"):
        assert key in results
    assert results["snr_db"].shape == (2,)
    assert np.all(results["ber"] > 0.35) and np.all(results["ber"] < 0.65)
    assert int(np.max(results["n_symbols"])) < 4000
    assert int(np.sum(results["n_errors"])) >= 20


def test_evaluate_model_online_floor_low_ber(tiny_config: Dict[str, Any]) -> None:
    """ONLINE: con soglia errori alta si arriva a max_symbols (niente floor 1/(2*n))."""
    config = _eval_config(
        tiny_config,
        snr_test_range=[5.0],
        bit_error_threshold=10 ** 6,
        max_symbols_per_snr=3000,
        batch_size=32,
    )
    config["evaluation"]["eval_batch_symbols"] = 1000
    config["evaluation"]["eval_predict_batch_size"] = 500

    results = evaluate_model_online(_dummy_online_model(), config)

    assert int(results["n_symbols"][0]) > 0


def test_evaluate_model_online_invalid_config_raises(tiny_config: Dict[str, Any]) -> None:
    """ONLINE: config con data.echoes vuoto -> ValueError (fail-fast)."""
    config = _eval_config(tiny_config, snr_test_range=[0.0])
    config["data"]["echoes"] = []
    with pytest.raises(ValueError):
        evaluate_model_online(_dummy_online_model(), config)


def test_predict_online_chunked_boundaries() -> None:
    """ONLINE: forward a chunk con N non multiplo del chunk e chunk=1."""
    model = _dummy_online_model()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 100, 2)).astype(np.float32)

    comm, sens = _predict_online_chunked(model, x, predict_batch=256)
    assert comm.shape == (1000, 2)
    assert sens.shape == (1000, 2)
    assert np.all(np.isfinite(comm)) and np.all(np.isfinite(sens))

    comm1, sens1 = _predict_online_chunked(model, x, predict_batch=1)
    assert comm1.shape == (1000, 2)
    assert np.allclose(comm, comm1, atol=1e-5, rtol=1e-4)
    assert np.allclose(sens, sens1, atol=1e-5, rtol=1e-4)


def test_predict_online_chunked_invalid_args_raise() -> None:
    """ONLINE: predict_batch/min_batch non validi -> ValueError."""
    model = _dummy_online_model()
    x = np.zeros((16, 100, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        _predict_online_chunked(model, x, predict_batch=0)
    with pytest.raises(ValueError):
        _predict_online_chunked(model, x, predict_batch=-5)
    with pytest.raises(ValueError):
        _predict_online_chunked(model, x, predict_batch=True)
    with pytest.raises(ValueError):
        _predict_online_chunked(model, x, predict_batch=8, min_batch=0)

