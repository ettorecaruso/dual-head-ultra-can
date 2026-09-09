"""
Test per ``src/training/losses.py`` (``tests/test_losses.py``).

Questi test verificano:
  T1  test_lambda_0               lambda=0: L_total == L_comm (entro 1e-6)
  T2  test_lambda_100             lambda=100: L_total finita e positiva; il
                                  termine MSE domina
  T3  test_loss_finite            batch reale (tiny_dataset) -> tutte le loss finite
  T4  test_mse_non_negative       MSE >= 0 (tolleranza 1e-7)
  T5  test_ce_positive            CE > 0 con logits random
  T6  test_targets_normalized_01  target sensing in [0,1] (tolleranza 1e-7);
                                  guardia eager -> WARNING se outside range
  T7  test_lambda_from_yaml       lambda letto dalla config; fail-fast se mancante

Superficie coperta (funzioni pubbliche di ``src/training/losses.py``):
  - ``assert_finite``        : tensore finite OK; NaN/Inf -> InvalidArgumentError.
  - ``comm_ce_loss``         : CE sparsa (paper Sez. IV-D, Eq. (14)): caso
      normale, minimo con logits perfetti, CE > 0 con logits random (T5),
      batch 1/32/128, logits NaN/Inf -> errore, shape errate -> ValueError,
      etichette outside da {0, M-1} -> errore.
  - ``mse_sensing_loss``     : MSE su target normalizzati (Sez. IV-D): caso
      normale, pred == truth -> 0, MSE >= 0 (T4), batch 1/32/128, shape
      errate -> ValueError, pred NaN -> errore, target outside [0,1] -> WARNING (T6).
  - ``combined_loss_factory``: L_total = L_comm + lambda_mse * L_sensing
      (Eq. (14)): lambda 0/100/da YAML/mancante/negativo/NaN/Inf/non-number,
      keys mancanti nei dict -> ValueError, shape errate -> ValueError, batch
      1/32/128.

Ogni funzione pubblica ha ALMENO 3 test: input validi, input limite e input
invalidi.

Test di integrazione: import del modulo, flusso end-to-end (config ->
normalizzazione target sensing -> factory -> loss) sul dataset reale
``tiny_dataset`` e ``FileNotFoundError`` su config experiment mancante.

Le casistiche di correttezza matematica su sequenze caotiche (mappa logistica
r=4.0/r=3.57, Bernoulli), canale (conservazione energia, Eq. (4)) e range di
tau/fD NON sono replicate qui: appartengono a ``tests/test_chaotic_maps.py`` e ``tests/test_channel.py``, perche' il modulo
losses non genera sequenze ne' applica canali (consuma solo tensori).

Scelta di pytest: coerente con l'intera suite esistente (``test_models``,
``test_channel``, ``test_chaotic_maps`` usano pytest con fixture e caplog) e con
il comando dal log: ``python -m pytest tests/test_losses.py -v``.
Il template utente mostrava ``unittest``; il pattern ``setUp`` e' mappato sulle
fixture ``tiny_config``/``tiny_dataset`` (``tests/conftest.py``)
e sugli helper di modulo, mentre ``@pytest.mark.parametrize`` copre i batch size
(1, 32, 128) senza duplicare codice.

Conformita' : shape-check, range-check e check NaN/Inf su ogni
output (); type hints e docstring su tutte le funzioni; nessun
comando shell con pipe; costanti documentate con riferimento al paper.

TODO (test mancanti, segnalati esplicitamente):
  - SNR estremi (-20/+30 dB) su una loss: NON applicabile al modulo losses
    (non genera ne' riceve segnali, solo tensori); la robustezza SNR e' coperta
    da ``tests/test_channel.py``.
  - ``FileNotFoundError`` per dataset mancante: NON applicabile a losses;
    spetta a ``tests/test_input_validation.py`` (``data_loader.load_npz_files``,
    ).
  - ``test_comm_ce_loss_out_of_range_labels_raises`` dipende dal backend TF
    (``sparse_softmax_cross_entropy_with_logits``): se una futura versione non
    solleva piu' InvalidArgumentError, il controllo va spostato a livello di
    data loader (il contratto richiede etichette in {0, M-1}).

FIX P5 (correzione sorgente evidenziata da questa suite):
  - ``assert_finite`` crashava su label comm int32 (CheckNumerics di TF accetta
    solo float): corretto in ``src/training/losses.py`` saltando il check per
    dtype non floating (gli interi sono sempre finiti).
  - Le shape errate sollevano ``ValueError`` (``assert_shapes`` in eager), NON
    ``InvalidArgumentError``: i test shape sono allineati al contratto
    documentato nelle docstring di ``losses.py``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from src.training.losses import (
    assert_finite,
    combined_loss_factory,
    comm_ce_loss,
    mse_sensing_loss,
    mse_sensing_loss_factory,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    validate_config,
)

_BATCH_SIZES = (1, 32, 128)
_M_BPSK = 2
_TOL_EQ = 1e-6
_LAMBDA_YAML = 2.0

def _make_comm_batch(
    batch_size: int, modulation_order: int = _M_BPSK, seed: int = 0
) -> tuple[tf.Tensor, tf.Tensor]:
    
    rng = np.random.default_rng(seed)
    labels = tf.constant(
        rng.integers(0, modulation_order, size=batch_size), dtype=tf.int32
    )
    logits = tf.constant(
        rng.normal(size=(batch_size, modulation_order)), dtype=tf.float32
    )
    return labels, logits

def _make_sensing_batch(batch_size: int, seed: int = 1) -> tuple[tf.Tensor, tf.Tensor]:
    
    rng = np.random.default_rng(seed)
    targets = tf.constant(rng.uniform(0.0, 1.0, size=(batch_size, 2)), dtype=tf.float32)
    preds = tf.constant(rng.uniform(0.0, 1.0, size=(batch_size, 2)), dtype=tf.float32)
    return targets, preds

def _make_loss_batch(
    batch_size: int = 32, modulation_order: int = _M_BPSK, seed: int = 2
) -> tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    
    rng = np.random.default_rng(seed)
    y_true = {
        "comm": tf.constant(
            rng.integers(0, modulation_order, size=batch_size), dtype=tf.int32
        ),
        "sensing": tf.constant(
            rng.uniform(0.0, 1.0, size=(batch_size, 2)), dtype=tf.float32
        ),
    }
    y_pred = {
        "comm": tf.constant(
            rng.normal(size=(batch_size, modulation_order)), dtype=tf.float32
        ),
        "sensing": tf.constant(
            rng.uniform(0.0, 1.0, size=(batch_size, 2)), dtype=tf.float32
        ),
    }
    return y_true, y_pred

def test_assert_finite_accepts_finite_tensor() -> None:
    
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    assert assert_finite(tensor, "tensor") is None

def test_assert_finite_nan_raises() -> None:
    """Input limite: NaN -> InvalidArgumentError ( Sez. 1.2)."""
    tensor = tf.constant([[1.0, float("nan")]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        assert_finite(tensor, "tensor")

def test_assert_finite_inf_raises() -> None:
    """Input limite: +Inf -> InvalidArgumentError ( Sez. 1.2)."""
    tensor = tf.constant([[1.0, float("inf")]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        assert_finite(tensor, "tensor")

def test_assert_finite_int_dtype_skipped() -> None:
    
    tensor = tf.constant([0, 1], dtype=tf.int32)
    assert assert_finite(tensor, "comm_labels") is None

def test_comm_ce_loss_valid_batch() -> None:
    """Caso normale: batch valido -> loss scalare, finita e positiva."""
    labels, logits = _make_comm_batch(batch_size=32)
    loss_val = float(comm_ce_loss(labels, logits))
    assert np.isfinite(loss_val)
    assert loss_val > 0.0

def test_comm_ce_loss_perfect_logits_minimum() -> None:
    """Correttezza: logits perfettamente separati -> CE tende al minimo (0)."""
    labels = tf.constant([0, 1, 0, 1], dtype=tf.int32)
    logits = tf.constant(
        [[20.0, -20.0], [-20.0, 20.0], [20.0, -20.0], [-20.0, 20.0]]
    )
    loss_val = float(comm_ce_loss(labels, logits))
    assert 0.0 <= loss_val < 1e-3

def test_ce_positive_random_logits() -> None:
    """ (T5): CE > 0 con logits random (equiv. ~ -log(0.5) per M=2)."""
    labels, logits = _make_comm_batch(batch_size=128, seed=5)
    loss_val = float(comm_ce_loss(labels, logits))
    assert loss_val > 0.0

@pytest.mark.parametrize("batch_size", list(_BATCH_SIZES))
def test_comm_ce_loss_batch_sizes(batch_size: int) -> None:
    """Consistenza dimensioni: batch 1/32/128 -> output scalare e finite."""
    labels, logits = _make_comm_batch(batch_size=batch_size)
    loss_val = comm_ce_loss(labels, logits)
    assert tf.rank(loss_val) == 0
    assert np.isfinite(float(loss_val))

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_comm_ce_loss_nonfinite_logits_raises(bad_value: float) -> None:
    """Stabilita' numerica: logits NaN/Inf -> InvalidArgumentError (Sez. 1.2)."""
    labels = tf.constant([0, 1], dtype=tf.int32)
    logits = tf.constant([[bad_value, 0.0], [0.0, 1.0]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        comm_ce_loss(labels, logits)

def test_comm_ce_loss_wrong_labels_shape_raises() -> None:
    """Input invalido: etichette 2D (B,1) invece di (B,) -> shape error."""
    labels = tf.constant([[0], [1]], dtype=tf.int32)
    logits = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        comm_ce_loss(labels, logits)

def test_comm_ce_loss_wrong_logits_shape_raises() -> None:
    """Input invalido: logits 1D (B,) invece di (B, M) -> shape error."""
    labels = tf.constant([0, 1], dtype=tf.int32)
    logits = tf.constant([0.0, 1.0])
    with pytest.raises(ValueError):
        comm_ce_loss(labels, logits)

def test_comm_ce_loss_out_of_range_labels_raises() -> None:
    """Input invalido: etichetta outside da {0, M-1} -> errore del backend TF."""
    labels = tf.constant([0, 5], dtype=tf.int32)
    logits = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        comm_ce_loss(labels, logits)

def test_mse_sensing_loss_valid_batch() -> None:
    """Caso normale: target/pred (B,2) -> loss scalare, finita e >= 0."""
    targets, preds = _make_sensing_batch(batch_size=32)
    loss_val = float(mse_sensing_loss(targets, preds))
    assert np.isfinite(loss_val)
    assert loss_val >= 0.0

def test_mse_sensing_loss_perfect_predictions_zero() -> None:
    """Correttezza: pred == truth -> MSE = 0 (minimo, tolleranza 1e-7)."""
    targets = tf.constant([[0.1, 0.2], [0.8, 0.9]], dtype=tf.float32)
    loss_val = float(mse_sensing_loss(targets, targets))
    assert loss_val == pytest.approx(0.0, abs=1e-7)

def test_mse_non_negative() -> None:
    """ (T4): MSE >= 0 con batch random (tolleranza 1e-7)."""
    targets, preds = _make_sensing_batch(batch_size=128, seed=11)
    loss_val = float(mse_sensing_loss(targets, preds))
    assert loss_val >= -1e-7

@pytest.mark.parametrize("batch_size", list(_BATCH_SIZES))
def test_mse_sensing_loss_batch_sizes(batch_size: int) -> None:
    """Consistenza dimensioni: batch 1/32/128 -> output scalare e finite."""
    targets, preds = _make_sensing_batch(batch_size=batch_size)
    loss_val = mse_sensing_loss(targets, preds)
    assert tf.rank(loss_val) == 0
    assert np.isfinite(float(loss_val))

def test_mse_shape_raises() -> None:
    """Input invalido: y_true (B,3) vs expected (B,2) -> shape error (header losses.py)."""
    targets = tf.ones((4, 3), dtype=tf.float32)
    preds = tf.ones((4, 2), dtype=tf.float32)
    with pytest.raises(ValueError):
        mse_sensing_loss(targets, preds)

def test_mse_1d_targets_raises() -> None:
    """Input invalido: target 1D (B,) invece di (B,2) -> shape error."""
    targets = tf.ones((4,), dtype=tf.float32)
    preds = tf.ones((4, 2), dtype=tf.float32)
    with pytest.raises(ValueError):
        mse_sensing_loss(targets, preds)

def test_mse_nonfinite_preds_raises() -> None:
    """Stabilita' numerica: predizioni NaN -> InvalidArgumentError (Sez. 1.2)."""
    targets = tf.ones((2, 2), dtype=tf.float32)
    preds = tf.constant([[float("nan"), 0.0], [0.0, 1.0]])
    with pytest.raises(tf.errors.InvalidArgumentError):
        mse_sensing_loss(targets, preds)

def test_mse_sensing_loss_factory_zero_matches_legacy() -> None:
    """Retro-compatibilita': factory con penalty=0 equivale a mse_sensing_loss."""
    targets, preds = _make_sensing_batch(batch_size=32)
    loss_legacy = float(mse_sensing_loss(targets, preds))
    loss_factory = float(mse_sensing_loss_factory(0.0)(targets, preds))
    assert loss_factory == pytest.approx(loss_legacy, rel=1e-6)

def test_mse_sensing_loss_factory_penalty_activates_only_out_of_range() -> None:
    
    targets = tf.constant([[0.2, 0.4], [0.6, 0.8]], dtype=tf.float32)
    preds_in = tf.constant([[0.1, 0.3], [0.5, 0.7]], dtype=tf.float32)
    loss_plain = float(mse_sensing_loss(targets, preds_in))
    loss_penalized = float(mse_sensing_loss_factory(5.0)(targets, preds_in))
    assert loss_penalized == pytest.approx(loss_plain, rel=1e-6)
    preds_out = tf.constant([[1.2, 0.3], [-0.1, 0.7]], dtype=tf.float32)
    loss_out_plain = float(mse_sensing_loss(targets, preds_out))
    loss_out_pen = float(mse_sensing_loss_factory(5.0)(targets, preds_out))
    assert loss_out_pen > loss_out_plain

def test_mse_sensing_loss_factory_invalid_penalty_raises() -> None:
    """Penale negativa / non finita -> ValueError (fail-fast, Sez. 1.2)."""
    with pytest.raises(ValueError):
        mse_sensing_loss_factory(-1.0)
    with pytest.raises(ValueError):
        mse_sensing_loss_factory(float("nan"))

def test_targets_normalized_01(caplog: pytest.LogCaptureFixture) -> None:
    
    targets_ok = tf.constant([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=tf.float32)
    preds = tf.constant([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]], dtype=tf.float32)
    loss_val = float(mse_sensing_loss(targets_ok, preds))
    assert np.isfinite(loss_val)
    assert loss_val >= 0.0

    raw_targets = tf.constant([[0.0, 0.0], [33.0, 0.0]], dtype=tf.float32)
    with caplog.at_level(logging.WARNING, logger="src.training.losses"):
        mse_sensing_loss(raw_targets, tf.zeros_like(raw_targets))
    assert "outside [0,1]" in caplog.text

def test_lambda_0() -> None:
    """ (T1): lambda=0 -> L_total == L_comm (sensing disattivato)."""
    loss_fn = combined_loss_factory(0.0)
    y_true, y_pred = _make_loss_batch()
    L_total = float(loss_fn(y_true, y_pred))
    L_comm = float(comm_ce_loss(y_true["comm"], y_pred["comm"]))
    assert np.isfinite(L_total)
    assert L_total == pytest.approx(L_comm, abs=_TOL_EQ)

def test_lambda_100() -> None:
    """ (T2): lambda=100 -> finita, positiva e il termine MSE domina."""
    loss_fn = combined_loss_factory(100.0)
    y_true, y_pred = _make_loss_batch(seed=3)
    L_total = float(loss_fn(y_true, y_pred))
    L_comm = float(comm_ce_loss(y_true["comm"], y_pred["comm"]))
    L_sensing = float(mse_sensing_loss(y_true["sensing"], y_pred["sensing"]))
    assert np.isfinite(L_total)
    assert L_total > 0.0
    assert L_total > L_comm
    assert L_total == pytest.approx(L_comm + 100.0 * L_sensing, rel=1e-5)

def test_loss_finite(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray]
) -> None:
    """ (T3): batch reale (tiny_dataset) -> tutte le loss finite."""
    data = tiny_config["data"]
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    lambda_mse = float(tiny_config["training"]["lambda_mse"])

    batch = tiny_dataset["comm_labels"].shape[0]
    y_true = {
        "comm": tf.constant(tiny_dataset["comm_labels"], dtype=tf.int32),
        "sensing": tf.constant(
            np.stack(
                [
                    tiny_dataset["sensing_labels"][:, 0] / max_delay,
                    tiny_dataset["sensing_labels"][:, 1] / max_doppler,
                ],
                axis=-1,
            ),
            dtype=tf.float32,
        ),
    }
    rng = np.random.default_rng(7)
    y_pred = {
        "comm": tf.constant(rng.normal(size=(batch, _M_BPSK)), dtype=tf.float32),
        "sensing": tf.constant(rng.uniform(size=(batch, 2)), dtype=tf.float32),
    }

    loss_fn = combined_loss_factory(lambda_mse)
    L_total = float(loss_fn(y_true, y_pred))
    L_comm = float(comm_ce_loss(y_true["comm"], y_pred["comm"]))
    L_sensing = float(mse_sensing_loss(y_true["sensing"], y_pred["sensing"]))
    assert np.isfinite(L_total)
    assert np.isfinite(L_comm)
    assert np.isfinite(L_sensing)
    assert L_total == pytest.approx(L_comm + lambda_mse * L_sensing, rel=1e-5)

def test_lambda_from_yaml(tiny_config: Dict[str, Any]) -> None:
    
    validate_config(tiny_config, ["training.lambda_mse"])
    lambda_mse = float(tiny_config["training"]["lambda_mse"])
    assert lambda_mse == pytest.approx(_LAMBDA_YAML)

    loss_fn = combined_loss_factory(lambda_mse)
    y_true, y_pred = _make_loss_batch()
    L_total = float(loss_fn(y_true, y_pred))
    L_comm = float(comm_ce_loss(y_true["comm"], y_pred["comm"]))
    L_sensing = float(mse_sensing_loss(y_true["sensing"], y_pred["sensing"]))
    assert np.isfinite(L_total)
    assert L_total == pytest.approx(L_comm + lambda_mse * L_sensing, rel=1e-5)

def test_lambda_missing_from_config_raises(tiny_config: Dict[str, Any]) -> None:
    
    del tiny_config["training"]["lambda_mse"]
    with pytest.raises(ValueError, match="training.lambda_mse"):
        validate_config(tiny_config, ["training.lambda_mse"])

def test_lambda_negative_raises() -> None:
    """Input invalido: lambda negativo -> ValueError (header di losses.py)."""
    with pytest.raises(ValueError, match=">= 0"):
        combined_loss_factory(-0.1)

def test_lambda_nan_raises() -> None:
    
    with pytest.raises(ValueError, match="finite"):
        combined_loss_factory(float("nan"))

def test_lambda_inf_raises() -> None:
    
    with pytest.raises(ValueError, match="finite"):
        combined_loss_factory(float("inf"))

def test_lambda_type_raises() -> None:
    """Input invalido: lambda non numerico -> TypeError."""
    with pytest.raises(TypeError, match="number"):
        combined_loss_factory("0.1")

@pytest.mark.parametrize(
    "missing_key",
    ["y_true.comm", "y_pred.comm", "y_true.sensing", "y_pred.sensing"],
)
def test_missing_key_raises(missing_key: str) -> None:
    
    y_true, y_pred = _make_loss_batch()
    owner, key = missing_key.split(".")
    del (y_true if owner == "y_true" else y_pred)[key]

    loss_fn = combined_loss_factory(0.1)
    with pytest.raises(ValueError, match="Missing key"):
        loss_fn(y_true, y_pred)

def test_combined_wrong_comm_shape_raises() -> None:
    
    y_true, y_pred = _make_loss_batch()
    y_true["comm"] = tf.expand_dims(y_true["comm"], axis=-1)
    loss_fn = combined_loss_factory(0.1)
    with pytest.raises(ValueError):
        loss_fn(y_true, y_pred)

def test_combined_wrong_sensing_shape_raises() -> None:
    """Input invalido: y_true['sensing'] (B,3) invece di (B,2) -> shape error."""
    y_true, y_pred = _make_loss_batch()
    y_true["sensing"] = tf.concat(
        [y_true["sensing"], y_true["sensing"][:, :1]], axis=1
    )
    loss_fn = combined_loss_factory(0.1)
    with pytest.raises(ValueError):
        loss_fn(y_true, y_pred)

@pytest.mark.parametrize("batch_size", list(_BATCH_SIZES))
def test_combined_loss_batch_sizes(batch_size: int) -> None:
    """Consistenza dimensioni: batch 1/32/128 -> L_total scalare, finita, >= 0."""
    loss_fn = combined_loss_factory(0.1)
    y_true, y_pred = _make_loss_batch(batch_size=batch_size)
    L_total = loss_fn(y_true, y_pred)
    assert tf.rank(L_total) == 0
    assert np.isfinite(float(L_total))
    assert float(L_total) >= 0.0

def test_modules_importable() -> None:
    
    module = importlib.import_module("src.training.losses")
    for name in (
        "assert_finite",
        "comm_ce_loss",
        "mse_sensing_loss",
        "combined_loss_factory",
    ):
        assert hasattr(module, name), f"funzione pubblica mancante: {name}"

def test_end_to_end_combined_loss(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray]
) -> None:
    """Flusso end-to-end: config -> normalizzazione target -> factory -> loss."""
    data = tiny_config["data"]
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    lambda_mse = float(tiny_config["training"]["lambda_mse"])

    sensing_labels = tiny_dataset["sensing_labels"].astype(np.float32)
    sensing_norm = np.stack(
        [sensing_labels[:, 0] / max_delay, sensing_labels[:, 1] / max_doppler],
        axis=-1,
    ).astype(np.float32)
    assert np.all(sensing_norm >= 0.0)
    assert np.all(sensing_norm <= 1.0 + 1e-7)

    rng = np.random.default_rng(42)
    batch = sensing_norm.shape[0]
    y_true = {
        "comm": tf.constant(tiny_dataset["comm_labels"], dtype=tf.int32),
        "sensing": tf.constant(sensing_norm),
    }
    y_pred = {
        "comm": tf.constant(rng.normal(size=(batch, _M_BPSK)), dtype=tf.float32),
        "sensing": tf.constant(
            sensing_norm + 0.1 * rng.normal(size=(batch, 2)), dtype=tf.float32
        ),
    }

    loss_fn = combined_loss_factory(lambda_mse)
    L_total = float(loss_fn(y_true, y_pred))
    assert np.isfinite(L_total)
    assert L_total >= 0.0

def test_config_file_missing_raises(tmp_path: Path) -> None:
    """File di input mancante -> FileNotFoundError (load_config, fail-fast)."""
    missing = tmp_path / "missing_experiment.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing, DEFAULT_BASE_CONFIG_PATH)

