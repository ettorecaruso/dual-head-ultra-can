"""
Test per ``src/training/trainer.py`` (``tests/test_training.py``).

Verificano:

  - ``Trainer.__init__``: costruzione valida (attributi, ``lambda_mse``
    loggato), lambda mancante/not finite -> ValueError, training mancante ->
    ValueError, train_ds empty -> ValueError, val_ds empty -> WARNING, tipi
    invalidi -> TypeError.
  - ``_validate_loss_weights``: allineamento comm->1.0 e sensing->lambda
    (Eq. (14) Sez. IV-D) con WARNING; loss_weights mancante -> ValueError.
  - ``_build_callbacks``: EarlyStopping (val_ds + patience>0, monitor
    "val_loss", restore_best_weights), ModelCheckpoint, CSVLogger,
    TensorBoard; checkpoint assente -> WARNING; path non creabile -> OSError.
  - ``verify_finite_gradients``: gradienti finiti (batch 1/32/128); gradiente
    NaN simulato -> ValueError ( Sez. 1.2).
  - ``save_artifacts``: history.csv + run_metadata.json con lambda_mse
    ( Sez. 4.2); history invalida -> TypeError; dir di log non
    scrivibile -> OSError.
  - ``train()``: un passo reale -> loss/gradienti finiti (batch 1/32/128);
    log INFO con lambda_mse; early stopping con patience dal config;
    checkpoint salvato; senza val_ds; lambda 0 e 100 stabili; loss_weights
    mancante -> ValueError; label senza keys attese -> RuntimeError.

Test di integrazione: import del modulo e flusso end-to-end (config ->
dataset -> Trainer -> train -> artefatti finiti e coerenti con lambda_mse).

Le casistiche su sequenze caotiche/canale/SNR (-20/+30 dB) NON sono replicate
qui: appartengono a ``test_chaotic_maps.py`` (1.4) e ``test_channel.py`` (1.5).

Scelta di pytest: coerente con l'intera suite (``test_models``/``test_losses``
usano fixture ``tiny_config``/``tiny_dataset``/``tiny_model``, caplog,
monkeypatch, parametrize) e con il comando dal log:
``python -m pytest tests/test_training.py -v``. Il ``setUp`` del template
unittest e' mappato sulle fixture di ``tests/conftest.py``.

Conformita' : shape-check e check NaN/Inf ();
nessuna pipe in shell; no valid .npz file scritto outside da ``paper/`` (log e
checkpoint dei test in ``tmp_path`` via monkeypatch di ``Trainer._get_log_dir``).

TODO (test mancanti, segnalati esplicitamente):
  - SNR estremi (-20/+30 dB): NON applicabile al trainer (consuma solo
    tensori/dataset); coperto da ``test_channel.py``.
  - ``FileNotFoundError`` per config mancante: NON applicabile (la config e'
    un dict via ``load_config``); coperto da ``test_config_loader.py``;
    l'analogo filesystem e' coperto qui come ``OSError``.
  - Branch ``gradients is None`` di ``verify_finite_gradients``: non
    simulabile deterministicamente con un modello reale (branch documentata
    nel sorgente).
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from src.training.trainer import Trainer

_LAMBDA_YAML = 2.0
_M_BPSK = 2
_BATCH_SIZES = (1, 32, 128)
_EPOCHS_TEST = 1
_TOL = 1e-6

def _make_fake_model(num_classes: int = _M_BPSK, seq_len: int = 100) -> tf.keras.Model:
    
    inp = tf.keras.layers.Input(shape=(seq_len, 2), name="r_input")
    x = tf.keras.layers.Flatten(name="flatten")(inp)
    comm = tf.keras.layers.Dense(num_classes, activation="linear", name="comm")(x)
    sensing = tf.keras.layers.Dense(2, name="sensing")(x)
    return tf.keras.Model(inputs=inp, outputs={"comm": comm, "sensing": sensing})

def _make_tf_dataset(
    tiny_dataset: Dict[str, np.ndarray],
    config: Dict[str, Any],
    batch_size: int,
) -> tf.data.Dataset:
    
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size deve essere un int >= 1, ricevuto: {batch_size!r}")
    data = config["data"]
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    x = tiny_dataset["x"].astype(np.float32)
    comm = tiny_dataset["comm_labels"].astype(np.int32)
    sensing = np.stack(
        [
            tiny_dataset["sensing_labels"][:, 0] / max_delay,
            tiny_dataset["sensing_labels"][:, 1] / max_doppler,
        ],
        axis=-1,
    ).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (x, {"comm": comm, "sensing": sensing})
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _trainer_config(
    tiny_config: Dict[str, Any],
    tmp_path: Path,
    **overrides: Any,
) -> Dict[str, Any]:
    
    config = copy.deepcopy(tiny_config)
    config["training"]["epochs"] = _EPOCHS_TEST
    config["training"]["batch_size"] = 32
    config["training"]["checkpoint_path"] = str(tmp_path / "models" / "checkpoint.h5")
    config["training"].update(overrides)
    return config

def _patch_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    
    monkeypatch.setattr(Trainer, "_get_log_dir", lambda self: tmp_path)

def _empty_dataset() -> tf.data.Dataset:
    """Dataset empty con il contratto features/labels del Trainer.

    Returns:
        ``tf.data.Dataset`` con cardinalita' 0 e labels ``{"comm", "sensing"}``.
    """
    x = np.zeros((0, 100, 1), dtype=np.float32)
    labels = {
        "comm": np.zeros((0,), dtype=np.int32),
        "sensing": np.zeros((0, 2), dtype=np.float32),
    }
    return tf.data.Dataset.from_tensor_slices((x, labels))

def test_trainer_init_valid(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    assert trainer.config is config
    assert trainer.model is model
    assert trainer.train_ds is ds
    assert trainer.val_ds is ds
    assert trainer.lambda_mse == pytest.approx(_LAMBDA_YAML, abs=_TOL)
    assert trainer.training_cfg["checkpoint_path"] == str(
        tmp_path / "models" / "checkpoint.h5"
    )

def test_trainer_init_logs_lambda(
    caplog: pytest.LogCaptureFixture,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        Trainer(config, model, ds, ds)
    assert any("lambda_mse" in record.getMessage() for record in caplog.records)
    assert "lambda_mse = 2" in caplog.text

def test_trainer_init_missing_lambda(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: config senza ``training.lambda_mse`` -> ValueError."""
    config = _trainer_config(tiny_config, tmp_path)
    del config["training"]["lambda_mse"]
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with pytest.raises(ValueError, match="lambda_mse"):
        Trainer(config, model, ds, ds)

@pytest.mark.parametrize("bad_lambda", [float("nan"), float("inf"), "abc"])
def test_trainer_init_lambda_non_finite(
    bad_lambda: Any,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    """Input limite: lambda NaN/Inf/non numerico -> ValueError (fail-fast)."""
    config = _trainer_config(tiny_config, tmp_path)
    config["training"]["lambda_mse"] = bad_lambda
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with pytest.raises(ValueError, match="lambda_mse"):
        Trainer(config, model, ds, ds)

def test_trainer_init_missing_training_section(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: sezione ``training`` assente -> ValueError."""
    config = _trainer_config(tiny_config, tmp_path)
    del config["training"]
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with pytest.raises(ValueError, match="training"):
        Trainer(config, model, ds, ds)

def test_trainer_init_empty_train_ds(
    tiny_config: Dict[str, Any], tmp_path: Path
) -> None:
    """Input limite: ``train_ds`` empty -> ValueError (nessun training su dati vuoti)."""
    config = _trainer_config(tiny_config, tmp_path)
    model = _make_fake_model()
    with pytest.raises(ValueError, match="train_ds"):
        Trainer(config, model, _empty_dataset(), None)

def test_trainer_init_empty_val_ds(
    caplog: pytest.LogCaptureFixture,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    """Input limite: ``val_ds`` empty -> WARNING e disabilitato (self.val_ds = None)."""
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        trainer = Trainer(config, model, ds, _empty_dataset())
    assert trainer.val_ds is None
    assert "val_ds" in caplog.text

def test_trainer_init_invalid_config_type(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: config non dict -> TypeError."""
    ds = _make_tf_dataset(tiny_dataset, _trainer_config(tiny_config, tmp_path), 32)
    model = _make_fake_model()
    with pytest.raises(TypeError, match="config"):
        Trainer(42, model, ds, ds)

def test_trainer_init_invalid_model_type(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: model non ``tf.keras.Model`` -> TypeError."""
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    with pytest.raises(TypeError, match="model"):
        Trainer(config, "not a model", ds, ds)

def test_trainer_init_invalid_train_ds_type(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: ``train_ds`` non ``tf.data.Dataset`` -> TypeError."""
    config = _trainer_config(tiny_config, tmp_path)
    model = _make_fake_model()
    with pytest.raises(TypeError, match="train_ds"):
        Trainer(config, model, "not a dataset", None)

def test_trainer_init_invalid_val_ds_type(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: ``val_ds`` non Dataset e non None -> TypeError."""
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    with pytest.raises(TypeError, match="val_ds"):
        Trainer(config, model, ds, "not a dataset")

def test_validate_loss_weights_already_aligned(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    trainer._validate_loss_weights()
    assert trainer.training_cfg["loss_weights"]["comm"] == pytest.approx(1.0)
    assert trainer.training_cfg["loss_weights"]["sensing"] == pytest.approx(
        _LAMBDA_YAML
    )

def test_loss_weights_alignment(
    caplog: pytest.LogCaptureFixture,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    """Input limite: loss_weights divergenti -> WARNING e allineamento a Eq. (14)."""
    config = _trainer_config(tiny_config, tmp_path)
    config["training"]["loss_weights"] = {"comm": 0.5, "sensing": 0.7}
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        trainer._validate_loss_weights()
    assert trainer.training_cfg["loss_weights"]["comm"] == pytest.approx(1.0)
    assert trainer.training_cfg["loss_weights"]["sensing"] == pytest.approx(
        _LAMBDA_YAML
    )
    assert "forced to 1.0" in caplog.text
    assert "aligning to lambda_mse" in caplog.text

def test_loss_weights_missing_raises(
    tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Input invalido: sezione ``loss_weights`` mancante -> ValueError."""
    config = _trainer_config(tiny_config, tmp_path)
    del config["training"]["loss_weights"]
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    with pytest.raises(ValueError, match="loss_weights"):
        trainer._validate_loss_weights()

def _build_callbacks_trainer(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> Trainer:
    """Helper: Trainer con callback pronti (log su tmp_path)."""
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path, **overrides)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    return Trainer(config, model, ds, ds)

def test_early_stopping_configured(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    trainer = _build_callbacks_trainer(
        tiny_config, tiny_dataset, tmp_path, monkeypatch, early_stopping_patience=4
    )
    callbacks = trainer._build_callbacks()
    early = [
        c for c in callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping)
    ]
    assert len(early) == 1
    assert early[0].patience == 4
    assert early[0].monitor == "val_loss"
    assert early[0].restore_best_weights is True

def test_build_callbacks_no_early_stopping_patience0(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input limite: patience=0 -> nessun EarlyStopping."""
    trainer = _build_callbacks_trainer(
        tiny_config, tiny_dataset, tmp_path, monkeypatch, early_stopping_patience=0
    )
    callbacks = trainer._build_callbacks()
    assert not any(
        isinstance(c, tf.keras.callbacks.EarlyStopping) for c in callbacks
    )

def test_build_callbacks_no_early_stopping_no_val_ds(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input limite: ``val_ds=None`` -> nessun EarlyStopping (niente da monitorare)."""
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, None)
    callbacks = trainer._build_callbacks()
    assert not any(
        isinstance(c, tf.keras.callbacks.EarlyStopping) for c in callbacks
    )

def test_build_callbacks_checkpoint_and_loggers(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso normale: ModelCheckpoint, CSVLogger e TensorBoard presenti."""
    trainer = _build_callbacks_trainer(tiny_config, tiny_dataset, tmp_path, monkeypatch)
    callbacks = trainer._build_callbacks()
    names = [type(c).__name__ for c in callbacks]
    assert "ModelCheckpoint" in names
    assert "CSVLogger" in names
    assert "TensorBoard" in names

def test_build_callbacks_no_checkpoint_warns(
    caplog: pytest.LogCaptureFixture,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input limite: ``checkpoint_path`` assente -> WARNING e nessun ModelCheckpoint."""
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    del config["training"]["checkpoint_path"]
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    with caplog.at_level(logging.WARNING, logger="src.training.trainer"):
        callbacks = trainer._build_callbacks()
    assert "checkpoint_path not set" in caplog.text
    assert not any(
        isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks
    )

def test_build_callbacks_checkpoint_parent_is_file_raises(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("x", encoding="utf-8")
    config = _trainer_config(tiny_config, tmp_path)
    config["training"]["checkpoint_path"] = str(blocked / "checkpoint.h5")
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    with pytest.raises(OSError):
        trainer._build_callbacks()

@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_verify_finite_gradients_batch_sizes(
    batch_size: int,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path, batch_size=batch_size)
    ds = _make_tf_dataset(tiny_dataset, config, batch_size)
    trainer = Trainer(config, tiny_model, ds, ds)
    trainer.verify_finite_gradients()

def test_verify_finite_gradients_nan_raises(
    monkeypatch: pytest.MonkeyPatch,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    
    from src.training import trainer as trainer_module

    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()

    def _nan_comm_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        del y_true
        return tf.reduce_sum(y_pred * tf.constant(float("nan"), dtype=y_pred.dtype))

    monkeypatch.setattr(trainer_module, "comm_ce_loss", _nan_comm_loss)
    trainer = Trainer(config, model, ds, ds)
    with pytest.raises(ValueError, match="Non-finite gradient"):
        trainer.verify_finite_gradients()

def test_verify_finite_gradients_requires_dataset(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    batch_features = next(iter(ds))[0]
    predictions = trainer.model(batch_features, training=False)
    tf.debugging.assert_all_finite(predictions["comm"], "comm not finite")
    tf.debugging.assert_all_finite(predictions["sensing"], "sensing not finite")
    assert predictions["comm"].shape == (32, _M_BPSK)
    assert predictions["sensing"].shape == (32, 2)

def test_trainer_save_artifacts(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)

    history = tf.keras.callbacks.History()
    history.history = {"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
    trainer.save_artifacts(history)

    assert (tmp_path / "history.csv").exists()
    metadata_path = tmp_path / "run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["lambda_mse"] == pytest.approx(_LAMBDA_YAML, abs=_TOL)
    assert metadata["model_name"] == model.name

def test_save_artifacts_invalid_history_raises(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    """Input invalido: ``history`` non e' un History Keras -> TypeError."""
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)
    with pytest.raises(TypeError, match="history"):
        trainer.save_artifacts("not a history")

def test_save_artifacts_log_dir_is_file_raises(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input invalido: directory di log non scrivibile -> OSError.

    Analogo filesystem di un "file di output mancante/bloccato": se la
    directory di log e' in realta' un file, la scrittura di ``history.csv``
    solleva ``OSError`` (fail-fast,).
    """
    blocked = tmp_path / "logs"
    blocked.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Trainer, "_get_log_dir", lambda self: blocked)

    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    model = _make_fake_model()
    trainer = Trainer(config, model, ds, ds)

    history = tf.keras.callbacks.History()
    history.history = {"loss": [0.5]}
    with pytest.raises(OSError):
        trainer.save_artifacts(history)

@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_train_step_gradients_finite(
    batch_size: int,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path, batch_size=batch_size)
    ds = _make_tf_dataset(tiny_dataset, config, batch_size)
    trainer = Trainer(config, tiny_model, ds, ds)
    history = trainer.train()

    loss = np.asarray(history.history["loss"], dtype=np.float64)
    assert loss.size >= 1
    assert np.all(np.isfinite(loss))
    assert np.all(loss >= -_TOL)
    if "val_loss" in history.history:
        val_loss = np.asarray(history.history["val_loss"], dtype=np.float64)
        assert np.all(np.isfinite(val_loss))

def test_trainer_logs_lambda(
    caplog: pytest.LogCaptureFixture,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    with caplog.at_level(logging.INFO, logger="src.training.trainer"):
        trainer = Trainer(config, tiny_model, ds, ds)
        trainer.train()
    assert any("lambda_mse" in record.getMessage() for record in caplog.records)
    assert "lambda_mse = 2" in caplog.text

def test_trainer_checkpoint_saved(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso normale: dopo ``train()`` il checkpoint file esiste (e non e' empty)."""
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, ds)
    trainer.train()
    checkpoint = Path(config["training"]["checkpoint_path"]).with_suffix(".weights.h5")
    assert checkpoint.exists()
    assert checkpoint.stat().st_size > 0

def test_trainer_train_no_val_ds(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, None)
    history = trainer.train()
    assert "loss" in history.history
    assert "val_loss" not in history.history
    assert np.all(np.isfinite(np.asarray(history.history["loss"], dtype=np.float64)))

@pytest.mark.parametrize("lambda_mse", [0.0, 100.0])
def test_trainer_extreme_lambda(
    lambda_mse: float,
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input limite: lambda=0 e lambda=100 -> training stabile (nessun NaN).

    Come in ``tests/test_losses.py`` (T1/T2): con λ=0 il sensing e'
    disattivato, con λ=100 domina la loss MSE; in entrambi i casi la loss
    totale del run resta finita.
    """
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path, lambda_mse=lambda_mse)
    config["training"]["loss_weights"] = {"comm": 1.0, "sensing": lambda_mse}
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, ds)
    history = trainer.train()
    loss = np.asarray(history.history["loss"], dtype=np.float64)
    assert np.all(np.isfinite(loss))
    assert np.all(loss >= -_TOL)

def test_train_missing_loss_weights_raises(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
) -> None:
    """Input invalido: ``loss_weights`` mancante -> ValueError (fail-fast)."""
    config = _trainer_config(tiny_config, tmp_path)
    del config["training"]["loss_weights"]
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, ds)
    with pytest.raises(ValueError, match="loss_weights"):
        trainer.train()

def test_train_missing_label_keys_raises(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    
    config = _trainer_config(tiny_config, tmp_path)
    x = tiny_dataset["x"].astype(np.float32)
    comm = tiny_dataset["comm_labels"].astype(np.int32)
    bad_ds = tf.data.Dataset.from_tensor_slices((x, {"foo": comm})).batch(32)
    model = _make_fake_model()
    trainer = Trainer(config, model, bad_ds, bad_ds)
    with pytest.raises(RuntimeError, match="Gradient verification failed"):
        trainer.train()

def test_trainer_restores_best_weights(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path, early_stopping_patience=2)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, ds)
    early = [
        c
        for c in trainer._build_callbacks()
        if isinstance(c, tf.keras.callbacks.EarlyStopping)
    ]
    assert len(early) == 1
    assert early[0].restore_best_weights is True

    trainer.train()
    batch_features = next(iter(ds))[0]
    predictions = trainer.model(batch_features, training=False)
    tf.debugging.assert_all_finite(predictions["comm"], "comm not finite")
    tf.debugging.assert_all_finite(predictions["sensing"], "sensing not finite")

def test_trainer_module_import() -> None:
    
    module = importlib.import_module("src.training.trainer")
    assert hasattr(module, "Trainer")
    assert callable(module.Trainer)

def test_end_to_end_trainer(
    tiny_config: Dict[str, Any],
    tiny_dataset: Dict[str, np.ndarray],
    tiny_model: tf.keras.Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    _patch_log_dir(monkeypatch, tmp_path)
    config = _trainer_config(tiny_config, tmp_path, epochs=2)
    ds = _make_tf_dataset(tiny_dataset, config, 32)
    trainer = Trainer(config, tiny_model, ds, ds)

    trainer.verify_finite_gradients()

    history = trainer.train()
    assert "loss" in history.history
    assert np.all(np.isfinite(np.asarray(history.history["loss"], dtype=np.float64)))

    assert Path(config["training"]["checkpoint_path"]).with_suffix(".weights.h5").exists()
    assert (tmp_path / "history.csv").exists()
    metadata_path = tmp_path / "run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["lambda_mse"] == pytest.approx(_LAMBDA_YAML, abs=_TOL)

    batch_features = next(iter(ds))[0]
    predictions = trainer.model(batch_features, training=False)
    assert predictions["comm"].shape == (32, _M_BPSK)
    assert predictions["sensing"].shape == (32, 2)
    tf.debugging.assert_all_finite(predictions["comm"], "comm not finite")
    tf.debugging.assert_all_finite(predictions["sensing"], "sensing not finite")
