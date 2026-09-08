"""
Test per ``src/models/heads.py``, ``src/models/ultra_can.py`` e
``src/models/ultra_can_qkv.py`` (``tests/test_models.py``).

Questi test verificano:
  T1  test_forward_batch_1        batch size=1: comm (1,2), sensing (1,2), finito
  T2  test_forward_batch_1024     batch size=1024: shape corrette, nessun NaN/Inf
  T3  test_model_output_shapes    tf.debugging.assert_shapes su input (B,100,1),
                                  H^att (B,96,64), logits (B,2), sensing (B,2)
  T4  test_qkv_dim_compatibility  attention_dim % num_heads == 0 e
                                  Q==K==V==attention_dim
  T5  test_param_count_expected   conteggio parametri attesi (8578 comm + 2146
                                  sensing + backbone 6401 conv1d / 22976 qkv)

Superficie coperta (funzioni pubbliche dei tre moduli):
  ``heads.build_communication_head`` / ``heads.build_sensing_head`` /
  ``heads.num_params`` / ``heads._head_input_dim`` / ``heads.main`` :
      shape di output (B, M) e (B, 2), conteggi parametri 8578/2146/8836,
      input estremi |v| = 1e-6/1e6 (finito, logits stabili), input nullo
      (= bias del layer finale per comm — Fase 4: attivazione lineare,
      logits — e bias per sensing), NaN -> guardia,
      feature dim errata -> shape error, config invalida -> ValueError,
      dropout train vs inferenza, conteggio su modello non buildato,
      derivazione di ``v`` da ``conv_filters[1]``/``attention_dim``.
  ``ultra_can.build_backbone`` / ``ultra_can.build_dual_head_ultra_can`` /
  ``ultra_can.count_trainable_params`` / ``ultra_can._expected_att_length`` /
  ``ultra_can.main`` :
      forward batch 1/1024, H^att (B, 100, 64), alpha softmax in (0, 1), conteggi
      parametri (10496 / 26501 real), input nullo e estremi, NaN -> guardia,
      dimensione input errata -> shape error, config backbone invalida ->
      ValueError, deterministicita' in inferenza, logits comm (softmax dei
      logits ~1),
      feature_mode "iq" (10592), sequence_length e conv_kernel variabili,
      size="micro" -> WARNING, config non dict -> TypeError, attention-pooling,
      ordine colonne sensing [tau, fD], CLI senza --config -> SystemExit.
  ``ultra_can_qkv.build_qkv_attention_layer`` /
  ``ultra_can_qkv.build_qkv_backbone`` /
  ``ultra_can_qkv.build_dual_head_ultra_can_qkv`` /
  ``ultra_can_qkv.count_trainable_params`` /
  ``ultra_can_qkv._log_attention_saturation`` / ``ultra_can_qkv.main`` :
      compatibilita' dimensionale QKV (64/4 ok, 64/3, 64/0, dim<=0 ->
      ValueError; non int -> TypeError), forward batch 1/1024, input nullo
      ed estremi, conteggi parametri (22976 / 43076, "iq" 23072 / 43172),
      NaN -> errore interno, config QKV invalida -> ValueError, att_len
      dinamico, WARNING saturazione softmax (Sez. 4.3), WARNING micro,
      get_config round-trip, deterministicita', GAP condiviso, logits comm.

Test di integrazione: import dei tre moduli e flusso end-to-end
(build -> forward su 128 sequenze reali di ``tiny_dataset``) per conv1d e QKV.

Le casistiche di correttezza matematica su sequenze caotiche, canale e SNR
NON sono replicate qui: appartengono a ``tests/test_chaotic_maps.py`` (1.4) e
``tests/test_channel.py`` (1.5) e operano a valle dei modelli.

Scelta di pytest: coerente con l'intera suite esistente (``test_chaotic_maps``,
``test_channel``, ``test_logger`` usano pytest con fixtures e caplog) e con il
comando dal log: ``python -m pytest tests/test_models.py -v``.
Il template utente mostrava ``unittest``; il pattern ``setUp``/``tearDown`` e'
mappato sulle fixture ``tiny_config``/``tiny_model`` (``tests/conftest.py``,
e su helper di modulo con ``pytest.raises``.

Conformita' : shape-check, range-check e check NaN/Inf su ogni
output (); type hints e docstring su tutte le funzioni; nessun
comando shell con pipe; valori attesi documentati con riferimento al paper.

TODO (test mancanti, segnalati esplicitamente):
  - ``build_dual_head_ultra_can`` (conv1d) NON ha guardia interna anti NaN/Inf
    nel forward (il controllo e' nel modulo QKV): il test innesca l'errore con
    ``tf.debugging.assert_all_finite`` a valle. Da valutare una guardia interna
    nel sorgente (difesa in profondita',).
  - ``_MultiHeadQKVAttention.from_config`` di default fallisce: ``__init__``
    non accetta ``trainable``/``dtype`` (verificato in fase di scrittura del
    test). Il round-trip e' testato sui soli parametri di ``get_config``;
    se serve ``model.save``/deserializzazione, va aggiunto ``**kwargs`` o un
    override di ``from_config`` nel layer sorgente.
  - ``FileNotFoundError`` per file mancante: NON applicabile ai modelli
    (non leggono file); spetta a ``tests/test_input_validation.py``
    (``data_loader.load_npz_files``, ).
  - SNR estremi (-20/+30 dB) su INPUT AI MODELLI: non applicabile a shape fixa
    (B, 100, 1) indipendente dall'SNR; la robustezza SNR e' coperta da
    ``tests/test_channel.py`` e dal trainer (Fase 3).
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Callable, Dict

import numpy as np
import pytest
import tensorflow as tf

from src.models.heads import (
    _head_input_dim,
    build_communication_head,
    build_sensing_head,
    main as heads_main,
    num_params,
)
from src.models.ultra_can import (
    _expected_att_length,
    build_backbone,
    build_dual_head_ultra_can,
    count_trainable_params,
    main as ultra_can_main,
)
from src.models.ultra_can_qkv import (
    _log_attention_saturation,
    build_qkv_attention_layer,
    build_qkv_backbone,
    build_dual_head_ultra_can_qkv,
    count_trainable_params as qkv_count_trainable_params,
    main as qkv_main,
)

_N_SEQ = 100
_ATT_LEN = 100
_F2 = 64
_M_BPSK = 2
_M_QPSK = 4
_HEAD_INPUT_DIM = 64
_COMM_PARAMS_M2 = 8578
_COMM_PARAMS_M4 = 8836
_SENSING_PARAMS = 2146
_SENSING_PARAMS_VMAX = 7362
_SENSING_PARAMS_MC = 8514
_BACKBONE_PARAMS = 10496
_BACKBONE_IN_MODEL = 10561
_TOTAL_PARAMS = 26501
_BACKBONE_PARAMS_IQ = 10592
_QKV_LAYER_PARAMS = 16640
_QKV_BACKBONE_PARAMS = 22976
_QKV_BACKBONE_IN_MODEL = 27136
_QKV_TOTAL_PARAMS = 43076
_QKV_BACKBONE_PARAMS_IQ = 23072
_QKV_TOTAL_PARAMS_IQ = 43172
_SMALL_BATCH = 1
_LARGE_BATCH = 1024
_MEDIUM_BATCH = 32

def _with_override(config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    
    cfg = copy.deepcopy(config)
    node: Any = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return cfg

def _qkv_config(tiny_config: Dict[str, Any]) -> Dict[str, Any]:
    
    cfg = _with_override(tiny_config, "model.backbone_type", "qkv_attention")
    return _with_override(cfg, "model.attention_checks", True)

def _random_input(batch: int, seq_len: int = _N_SEQ, channels: int = 2) -> np.ndarray:
    
    return np.random.default_rng(0).uniform(size=(batch, seq_len, channels)).astype(
        np.float32
    )

@pytest.mark.parametrize("batch", [_SMALL_BATCH, _LARGE_BATCH])
def test_comm_head_output_shapes(tiny_config: Dict[str, Any], batch: int) -> None:
    """Output comm: (B, M) logits, finito, softmax(logits) ~1, argmax in {0,1} (M=2)."""
    comm_head = build_communication_head(tiny_config)
    v = np.zeros((batch, _HEAD_INPUT_DIM), dtype=np.float32)
    out = comm_head(v, training=False).numpy()
    assert out.shape == (batch, _M_BPSK)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(
        tf.nn.softmax(out).numpy().sum(axis=-1), np.ones(batch), rtol=0, atol=1e-6
    )
    assert np.all(np.isin(out.argmax(axis=-1), [0, 1]))

@pytest.mark.parametrize("modulation_order", [_M_BPSK, _M_QPSK])
def test_comm_head_modulation_order(
    tiny_config: Dict[str, Any], modulation_order: int
) -> None:
    """M=4 (QPSK): output (B, 4) logits e softmax(logits) ~1 (Sez. 1.4)."""
    cfg = _with_override(
        tiny_config, "model.communication_head.modulation_order", modulation_order
    )
    comm_head = build_communication_head(cfg)
    v = np.zeros((_SMALL_BATCH, _HEAD_INPUT_DIM), dtype=np.float32)
    out = comm_head(v, training=False).numpy()
    assert out.shape == (_SMALL_BATCH, modulation_order)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(
        tf.nn.softmax(out).numpy().sum(axis=-1),
        np.ones(_SMALL_BATCH),
        rtol=0,
        atol=1e-6,
    )

@pytest.mark.parametrize("batch", [_SMALL_BATCH, _LARGE_BATCH])
def test_sensing_head_output_shapes(tiny_config: Dict[str, Any], batch: int) -> None:
    """Output sensing: (B, 2) [tau, fD], finito (Sez. IV-C)."""
    sensing_head = build_sensing_head(tiny_config)
    v = np.zeros((batch, _HEAD_INPUT_DIM), dtype=np.float32)
    out = sensing_head(v, training=False).numpy()
    assert out.shape == (batch, 2)
    assert np.all(np.isfinite(out))

def test_head_param_counts(tiny_config: Dict[str, Any]) -> None:
    """Conteggi attesi: comm=8578, sensing=2146, TOT=10724; M=4 -> comm=8836."""
    comm_head = build_communication_head(tiny_config)
    sensing_head = build_sensing_head(tiny_config)
    assert num_params(comm_head) == _COMM_PARAMS_M2
    assert num_params(sensing_head) == _SENSING_PARAMS
    assert num_params(comm_head) + num_params(sensing_head) == 8578 + 2146
    cfg4 = _with_override(
        tiny_config, "model.communication_head.modulation_order", _M_QPSK
    )
    comm4 = build_communication_head(cfg4)
    assert num_params(comm4) == _COMM_PARAMS_M4

@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_heads_extreme_input(tiny_config: Dict[str, Any], scale: float) -> None:
    """|v| = 1e-6 e 1e6: output finiti, softmax stabile (nessun NaN/Inf)."""
    comm_head = build_communication_head(tiny_config)
    sensing_head = build_sensing_head(tiny_config)
    v = np.full((4, _HEAD_INPUT_DIM), scale, dtype=np.float32)
    c_out = comm_head(v, training=False).numpy()
    s_out = sensing_head(v, training=False).numpy()
    assert np.all(np.isfinite(c_out))
    assert np.all(np.isfinite(s_out))
    np.testing.assert_allclose(
        tf.nn.softmax(c_out).numpy().sum(axis=-1), np.ones(4), rtol=0, atol=1e-5
    )

def test_heads_zero_input(tiny_config: Dict[str, Any]) -> None:
    """v=0: comm = bias del layer finale (logits, Fase 4), sensing = bias layer finale (bias=0 default)."""
    comm_head = build_communication_head(tiny_config)
    sensing_head = build_sensing_head(tiny_config)
    v = np.zeros((2, _HEAD_INPUT_DIM), dtype=np.float32)
    c_out = comm_head(v, training=False).numpy()
    s_out = sensing_head(v, training=False).numpy()
    expected_comm = comm_head.get_layer("comm_logits").bias.numpy()
    np.testing.assert_allclose(
        c_out, np.broadcast_to(expected_comm, c_out.shape), rtol=0, atol=1e-6
    )
    expected_sensing = np.zeros((2, 2), dtype=np.float32)
    np.testing.assert_allclose(
        s_out, expected_sensing, rtol=0, atol=1e-6
    )

def test_heads_nan_input_raises(tiny_config: Dict[str, Any]) -> None:
    """v con NaN: output NaN -> la guardia assert_all_finite solleva (Sez. 1.2)."""
    comm_head = build_communication_head(tiny_config)
    v = np.full((2, _HEAD_INPUT_DIM), np.nan, dtype=np.float32)
    out = comm_head(v, training=False)
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.debugging.assert_all_finite(out, message="guardia NaN heads")

def test_heads_wrong_feature_dim_raises(tiny_config: Dict[str, Any]) -> None:
    """v (B, 32) vs atteso (B, 64): shape error all'forward."""
    comm_head = build_communication_head(tiny_config)
    v = np.zeros((2, 32), dtype=np.float32)
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        comm_head(v, training=False)

@pytest.mark.parametrize(
    "path, value, builder",
    [
        ("model.communication_head.modulation_order", 1, "comm"),
        ("model.sensing_head.output_units", 3, "sensing"),
        ("model.dropout_rate", 1.5, "comm"),
    ],
)
def test_heads_invalid_config_raises(
    tiny_config: Dict[str, Any], path: str, value: Any, builder: str
) -> None:
    """Config testa invalida (M=1, output_units=3, dropout=1.5) -> ValueError."""
    cfg = _with_override(tiny_config, path, value)
    build_fn = build_communication_head if builder == "comm" else build_sensing_head
    with pytest.raises(ValueError):
        build_fn(cfg)

def test_dropout_train_vs_inference(tiny_config: Dict[str, Any]) -> None:
    """Contratto inferenza: training=False deterministico, training=True randomico."""
    comm_head = build_communication_head(tiny_config)
    v = np.random.default_rng(0).uniform(size=(_MEDIUM_BATCH, _HEAD_INPUT_DIM)).astype(
        np.float32
    )
    train_1 = comm_head(v, training=True).numpy()
    train_2 = comm_head(v, training=True).numpy()
    infer_1 = comm_head(v, training=False).numpy()
    infer_2 = comm_head(v, training=False).numpy()
    assert not np.array_equal(train_1, train_2)
    assert np.array_equal(infer_1, infer_2)

def test_num_params_unbuilt_model() -> None:
    """num_params: TypeError su oggetto non keras.Model; build esplicita al bisogno."""
    with pytest.raises(TypeError):
        num_params("not_a_model")
    with pytest.raises(ValueError):
        num_params(tf.keras.Sequential())
    seq = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(_HEAD_INPUT_DIM,)),
            tf.keras.layers.Dense(10),
        ]
    )
    assert num_params(seq) == 10 * _HEAD_INPUT_DIM + 10

def test_head_input_dim_qkv_uses_attention_dim(tiny_config: Dict[str, Any]) -> None:
    """_head_input_dim: conv1d -> conv_filters[1]; qkv -> attention_dim (64)."""
    assert _head_input_dim(tiny_config) == _HEAD_INPUT_DIM
    cfg_qkv = _qkv_config(tiny_config)
    assert _head_input_dim(cfg_qkv) == 64
    cfg_qkv_dim32 = _with_override(cfg_qkv, "model.attention_dim", 32)
    assert _head_input_dim(cfg_qkv_dim32) == 32

def test_forward_batch_1(tiny_model: tf.keras.Model) -> None:
    """T1 (): batch 1 -> comm (1,2), sensing (1,2), tutto finito."""
    x = _random_input(_SMALL_BATCH)
    out = tiny_model(x, training=False)
    assert out["comm"].shape == (_SMALL_BATCH, _M_BPSK)
    assert out["sensing"].shape == (_SMALL_BATCH, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

def test_forward_batch_1024(tiny_model: tf.keras.Model) -> None:
    """T2 (): batch 1024 -> shape corrette, nessun NaN/Inf."""
    x = _random_input(_LARGE_BATCH)
    out = tiny_model(x, training=False)
    assert out["comm"].shape == (_LARGE_BATCH, _M_BPSK)
    assert out["sensing"].shape == (_LARGE_BATCH, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

def test_model_output_shapes(tiny_model: tf.keras.Model) -> None:
    """T3 (): tf.debugging.assert_shapes su (B,100,1), H^att, logits, sensing."""
    mid = tf.keras.Model(
        inputs=tiny_model.input,
        outputs={
            "h_att": tiny_model.get_layer("pool_projection").output,
            "comm": tiny_model.output["comm"],
            "sensing": tiny_model.output["sensing"],
        },
    )
    x = _random_input(8)
    out = mid(tf.constant(x), training=False)
    tf.debugging.assert_shapes(
        [
            (tf.constant(x), ("B", _N_SEQ, 2)),
            (out["h_att"], ("B", _ATT_LEN, _F2)),
            (out["comm"], ("B", _M_BPSK)),
            (out["sensing"], ("B", 2)),
        ]
    )
    assert tuple(out["h_att"].shape) == (8, _ATT_LEN, _F2)

def test_param_count_expected(tiny_model: tf.keras.Model) -> None:
    
    comm_head = tiny_model.get_layer("communication_head")
    sensing_head = tiny_model.get_layer("sensing_head")
    assert num_params(comm_head) == _COMM_PARAMS_M2
    assert num_params(sensing_head) == _SENSING_PARAMS_VMAX
    backbone_params = (
        count_trainable_params(tiny_model)
        - num_params(comm_head)
        - num_params(sensing_head)
    )
    assert backbone_params == _BACKBONE_IN_MODEL
    assert count_trainable_params(tiny_model) == _TOTAL_PARAMS

@pytest.mark.parametrize("batch", [_SMALL_BATCH, _LARGE_BATCH])
def test_backbone_h_att_shape(tiny_config: Dict[str, Any], batch: int) -> None:
    """Backbone conv1d: H^att (B, 96, 64), finito, per batch 1 e 1024."""
    backbone = build_backbone(tiny_config)
    x = _random_input(batch, channels=1)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (batch, _ATT_LEN, _F2)
    assert np.all(np.isfinite(h_att))

def test_backbone_param_count(tiny_config: Dict[str, Any]) -> None:
    """Backbone conv1d = 10496 params (128 conv1 + 6208 conv2 + 4160 pool_proj)."""
    backbone = build_backbone(tiny_config)
    assert count_trainable_params(backbone) == _BACKBONE_PARAMS

def test_attention_weights_in_unit_interval(tiny_model: tf.keras.Model) -> None:
    """Pesi softmax 'attn_weights' dell'attention-pooling: in (0, 1), somma 1.

    (Guardia saturazione: alpha = softmax_temporale su Dense(1)(h_att).)
    """
    alpha_model = tf.keras.Model(
        inputs=tiny_model.input, outputs=tiny_model.get_layer("attn_weights").output
    )
    x = _random_input(4)
    alpha = alpha_model(x, training=False).numpy()
    assert alpha.shape == (4, _ATT_LEN, 1)
    assert np.all(alpha > 0.0)
    assert np.all(alpha < 1.0)
    np.testing.assert_allclose(
        alpha.sum(axis=1), np.ones((4, 1)), rtol=1e-5, atol=1e-5
    )

def test_ultra_can_zero_input_finite(tiny_model: tf.keras.Model) -> None:
    """Input zero: comm = bias del layer finale (logits, Fase 4), sensing = bias, finito."""
    x0 = np.zeros((2, _N_SEQ, 2), dtype=np.float32)
    out = tiny_model(x0, training=False)
    comm = out["comm"].numpy()
    sensing = out["sensing"].numpy()
    assert np.all(np.isfinite(comm))
    assert np.all(np.isfinite(sensing))
    expected_comm = (
        tiny_model.get_layer("communication_head").get_layer("comm_logits").bias.numpy()
    )
    np.testing.assert_allclose(
        comm, np.broadcast_to(expected_comm, comm.shape), rtol=0, atol=1e-6
    )
    expected_sensing = np.zeros((2, 2), dtype=np.float32)
    np.testing.assert_allclose(
        sensing, expected_sensing, rtol=0, atol=1e-6
    )

@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_ultra_can_extreme_input_finite(
    tiny_model: tf.keras.Model, scale: float
) -> None:
    """|x| = 1e-6 e 1e6: output finiti, nessun NaN/Inf (Sez. 1.2)."""
    x = np.full((4, _N_SEQ, 2), scale, dtype=np.float32)
    out = tiny_model(x, training=False)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

def test_ultra_can_nan_input_raises(tiny_model: tf.keras.Model) -> None:
    """Input con NaN: output NaN -> guardia assert_all_finite solleva (Sez. 1.2)."""
    x_nan = np.full((2, _N_SEQ, 2), np.nan, dtype=np.float32)
    out = tiny_model(x_nan, training=False)
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.debugging.assert_all_finite(out["comm"], message="guardia NaN ultra_can")

def test_ultra_can_wrong_input_dim_raises(tiny_model: tf.keras.Model) -> None:
    
    x_wrong = np.zeros((2, 50, 1), dtype=np.float32)
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        tiny_model(x_wrong, training=False)

@pytest.mark.parametrize(
    "path, value",
    [
        ("model.conv_padding", "causal"),
        ("model.conv_kernel", 100),
        ("model.conv_filters", [32]),
        ("model.backbone_type", "qkv_attention"),
    ],
)
def test_invalid_backbone_config_raises(
    tiny_config: Dict[str, Any], path: str, value: Any
) -> None:
    
    cfg = _with_override(tiny_config, path, value)
    with pytest.raises(ValueError):
        build_backbone(cfg)

def test_inference_deterministic(tiny_model: tf.keras.Model) -> None:
    """Due forward training=False -> output identici (contratto inferenza)."""
    x = _random_input(4)
    o1 = tiny_model(x, training=False)
    o2 = tiny_model(x, training=False)
    assert np.array_equal(o1["comm"].numpy(), o2["comm"].numpy())
    assert np.array_equal(o1["sensing"].numpy(), o2["sensing"].numpy())

def test_comm_outputs_logits(tiny_model: tf.keras.Model) -> None:
    
    x = _random_input(4)
    out = tiny_model(x, training=False)
    comm = out["comm"].numpy()
    assert np.all(np.isfinite(comm))
    np.testing.assert_allclose(
        tf.nn.softmax(comm).numpy().sum(axis=-1), np.ones(4), rtol=0, atol=1e-6
    )

def test_backbone_feature_mode_iq(tiny_config: Dict[str, Any]) -> None:
    """feature_mode='iq' : input (B, 100, 2), H^att (B, 96, 64), 6497 params."""
    cfg = _with_override(tiny_config, "data.feature_mode", "iq")
    backbone = build_backbone(cfg)
    x = _random_input(2, channels=2)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (2, _ATT_LEN, _F2)
    assert np.all(np.isfinite(h_att))
    assert count_trainable_params(backbone) == _BACKBONE_PARAMS_IQ

def test_backbone_seq_len_variation(tiny_config: Dict[str, Any]) -> None:
    """sequence_length=50 -> H^att (B, 50, 64) (padding 'same' conserva N)."""
    cfg = _with_override(tiny_config, "data.sequence_length", 50)
    backbone = build_backbone(cfg)
    x = np.zeros((2, 50, 1), dtype=np.float32)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (2, 50, _F2)
    assert np.all(np.isfinite(h_att))

@pytest.mark.parametrize("kernel, att_len", [(1, 100), (5, 100)])
def test_backbone_kernel_variation(
    tiny_config: Dict[str, Any], kernel: int, att_len: int
) -> None:
    
    cfg = _with_override(tiny_config, "model.conv_kernel", kernel)
    backbone = build_backbone(cfg)
    x = np.zeros((2, _N_SEQ, 1), dtype=np.float32)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (2, att_len, _F2)
    assert np.all(np.isfinite(h_att))

def test_ultra_can_micro_size_warning(
    tiny_config: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """size='micro' (Exp 2): build OK + WARNING via caplog (non un errore)."""
    cfg = _with_override(tiny_config, "model.size", "micro")
    with caplog.at_level(logging.WARNING, logger="src.models.ultra_can"):
        model = build_dual_head_ultra_can(cfg)
    assert count_trainable_params(model) > 0
    assert any("micro" in r.getMessage() for r in caplog.records)

def test_build_backbone_config_not_dict() -> None:
    """build_backbone(42): config non dict -> TypeError (fail-fast)."""
    with pytest.raises(TypeError):
        build_backbone(42)

def test_count_trainable_params_type_error() -> None:
    """count_trainable_params su oggetto non keras.Model -> TypeError."""
    with pytest.raises(TypeError):
        count_trainable_params("not_a_model")

def test_gap_shared_no_duplicate_weights(tiny_model: tf.keras.Model) -> None:
    """Un solo 'attn_scores'; TOT == backbone + comm + sensing (nessun ramo duplicato)."""
    attn = [layer for layer in tiny_model.layers if layer.name == "attn_scores"]
    assert len(attn) == 1
    total = count_trainable_params(tiny_model)
    comm_params = num_params(tiny_model.get_layer("communication_head"))
    sensing_params = num_params(tiny_model.get_layer("sensing_head"))
    backbone_params = total - comm_params - sensing_params
    assert total == backbone_params + comm_params + sensing_params

def test_sensing_output_column_order(tiny_model: tf.keras.Model) -> None:
    """Colonne output sensing = [tau, fD]: layer finale 'sensing_out' con 2 unit."""
    sensing_head = tiny_model.get_layer("sensing_head")
    out_layer = sensing_head.get_layer("sensing_out")
    assert int(out_layer.units) == 2
    x = np.zeros((4, _N_SEQ, 2), dtype=np.float32)
    out = tiny_model(x, training=False)
    assert out["sensing"].shape == (4, 2)

def test_expected_att_length_normal() -> None:
    """_expected_att_length: default 'valid' (100/3->96, 50/3->46, ...); 'same'->100."""
    assert _expected_att_length(100, 3) == 96
    assert _expected_att_length(50, 3) == 46
    assert _expected_att_length(100, 1) == 100
    assert _expected_att_length(100, 5) == 92
    assert _expected_att_length(100, 3, padding="same") == 100
    assert _expected_att_length(100, 5, padding="same", dilations=(1, 16)) == 100
    assert _expected_att_length(100, 3, padding="valid", dilations=(1, 16)) == 100 - 2 * 17

@pytest.mark.parametrize("seq_len, kernel", [(5, 5), (4, 5), (0, 3), (100, -2)])
def test_expected_att_length_invalid(seq_len: int, kernel: int) -> None:
    """_expected_att_length: lunghezza risultante <= 0 o argomenti non positivi."""
    with pytest.raises(ValueError):
        _expected_att_length(seq_len, kernel)

def test_qkv_dim_compatibility() -> None:
    """T4 (): 64/4 ok; Q==K==V==attention_dim; casi invalidi -> errori."""
    layer = build_qkv_attention_layer(64, 4)
    assert layer._dense_q.units == 64
    assert layer._dense_k.units == 64
    assert layer._dense_v.units == 64
    assert layer._out_proj.units == 64
    assert 64 % 4 == 0
    for dim, heads in [(64, 3), (64, 0), (0, 4), (-8, 4)]:
        with pytest.raises(ValueError):
            build_qkv_attention_layer(dim, heads)
    for dim, heads in [("64", 4), (64, "4"), (True, 4), (64, True)]:
        with pytest.raises(TypeError):
            build_qkv_attention_layer(dim, heads)

@pytest.mark.parametrize("batch", [_SMALL_BATCH, _LARGE_BATCH])
def test_qkv_forward_batch(tiny_config: Dict[str, Any], batch: int) -> None:
    """QKV forward batch 1/1024: comm (B,2), sensing (B,2), H^att (B,96,64), finiti."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = _random_input(batch)
    out = model(x, training=False)
    assert out["comm"].shape == (batch, _M_BPSK)
    assert out["sensing"].shape == (batch, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

def test_qkv_backbone_h_att_shape(tiny_config: Dict[str, Any]) -> None:
    
    backbone = build_qkv_backbone(_qkv_config(tiny_config))
    x = _random_input(4, channels=1)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (4, _ATT_LEN, _F2)
    assert np.all(np.isfinite(h_att))

def test_qkv_backbone_zero_input(tiny_config: Dict[str, Any]) -> None:
    
    qkv_cfg = _qkv_config(tiny_config)
    qkv_cfg["model"]["positional_encoding"] = False
    model = build_dual_head_ultra_can_qkv(qkv_cfg)
    x0 = np.zeros((2, _N_SEQ, 2), dtype=np.float32)
    out = model(x0, training=False)
    comm = out["comm"].numpy()
    sensing = out["sensing"].numpy()
    assert np.all(np.isfinite(comm))
    assert np.all(np.isfinite(sensing))
    expected_comm = (
        model.get_layer("communication_head").get_layer("comm_logits").bias.numpy()
    )
    np.testing.assert_allclose(
        comm, np.broadcast_to(expected_comm, comm.shape), rtol=0, atol=1e-6
    )
    expected_sensing = np.zeros((2, 2), dtype=np.float32)
    np.testing.assert_allclose(
        sensing, expected_sensing, rtol=0, atol=1e-6
    )

@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_qkv_extreme_input(tiny_config: Dict[str, Any], scale: float) -> None:
    """|x| = 1e-6 e 1e6: output finiti (saturazione ammessa, valori finiti)."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = np.full((4, _N_SEQ, 2), scale, dtype=np.float32)
    out = model(x, training=False)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

def test_qkv_param_count_expected(tiny_config: Dict[str, Any]) -> None:
    """QKV: backbone=22976, comm=8578, sensing=7362, proj=4160, TOT=43076; iq -> 43172."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    comm_params = num_params(model.get_layer("communication_head"))
    sensing_params = num_params(model.get_layer("sensing_head"))
    assert comm_params == _COMM_PARAMS_M2
    assert sensing_params == _SENSING_PARAMS_VMAX
    backbone_params = count_trainable_params(model) - comm_params - sensing_params
    assert backbone_params == _QKV_BACKBONE_IN_MODEL
    assert count_trainable_params(model) == _QKV_TOTAL_PARAMS
    cfg_iq = _with_override(_qkv_config(tiny_config), "data.feature_mode", "iq")
    model_iq = build_dual_head_ultra_can_qkv(cfg_iq)
    assert count_trainable_params(model_iq) == _QKV_TOTAL_PARAMS_IQ

def test_qkv_build_reproducible_with_seed(tiny_config: Dict[str, Any]) -> None:
    
    cfg = _qkv_config(tiny_config)

    def _seed() -> None:
        random.seed(7)
        np.random.seed(7)
        tf.random.set_seed(7)

    _seed()
    m1 = build_dual_head_ultra_can_qkv(cfg)
    _seed()
    m2 = build_dual_head_ultra_can_qkv(cfg)
    w1, w2 = m1.get_weights(), m2.get_weights()
    assert len(w1) == len(w2)
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)

def test_qkv_nan_input_raises(tiny_config: Dict[str, Any]) -> None:
    """Input con NaN: la guardia interna del layer QKV solleva (Sez. 1.2)."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x_nan = np.full((2, _N_SEQ, 2), np.nan, dtype=np.float32)
    with pytest.raises(tf.errors.InvalidArgumentError):
        model(x_nan, training=False)

def test_qkv_gap_shared_no_duplicate_weights(tiny_config: Dict[str, Any]) -> None:
    """QKV: un solo 'shared_gap' e TOT == backbone + comm + sensing."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    gaps = [layer for layer in model.layers if layer.name == "shared_gap"]
    assert len(gaps) == 1
    total = count_trainable_params(model)
    comm_params = num_params(model.get_layer("communication_head"))
    sensing_params = num_params(model.get_layer("sensing_head"))
    backbone_params = total - comm_params - sensing_params
    assert total == backbone_params + comm_params + sensing_params

def test_qkv_inference_deterministic(tiny_config: Dict[str, Any]) -> None:
    """QKV: due forward training=False -> output identici."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = _random_input(4)
    o1 = model(x, training=False)
    o2 = model(x, training=False)
    assert np.array_equal(o1["comm"].numpy(), o2["comm"].numpy())
    assert np.array_equal(o1["sensing"].numpy(), o2["sensing"].numpy())

@pytest.mark.parametrize(
    "path, value",
    [
        ("model.attention_dim", 32),
        ("model.attention_heads", 3),
        ("model.conv_padding", "causal"),
    ],
)
def test_invalid_qkv_config_raises(
    tiny_config: Dict[str, Any], path: str, value: Any
) -> None:
    """Config QKV invalida (dim, heads, padding) -> ValueError (fail-fast)."""
    cfg = _with_override(_qkv_config(tiny_config), path, value)
    with pytest.raises(ValueError):
        build_qkv_backbone(cfg)

def test_qkv_backbone_feature_mode_iq(tiny_config: Dict[str, Any]) -> None:
    """QKV feature_mode='iq': input (B, 100, 2), H^att (B, 100, 64), 23072 params."""
    cfg = _with_override(_qkv_config(tiny_config), "data.feature_mode", "iq")
    backbone = build_qkv_backbone(cfg)
    x = _random_input(2, channels=2)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (2, _ATT_LEN, _F2)
    assert np.all(np.isfinite(h_att))
    assert qkv_count_trainable_params(backbone) == _QKV_BACKBONE_PARAMS_IQ

@pytest.mark.parametrize("seq_len, kernel, att_len", [(50, 3, 50), (100, 5, 100)])
def test_qkv_att_len_dynamic(
    tiny_config: Dict[str, Any], seq_len: int, kernel: int, att_len: int
) -> None:
    
    cfg = _with_override(_qkv_config(tiny_config), "data.sequence_length", seq_len)
    cfg = _with_override(cfg, "model.conv_kernel", kernel)
    backbone = build_qkv_backbone(cfg)
    x = np.zeros((2, seq_len, 1), dtype=np.float32)
    h_att = backbone(x, training=False).numpy()
    assert h_att.shape == (2, att_len, _F2)
    assert np.all(np.isfinite(h_att))

def test_attention_saturation_warning(
    tiny_config: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    
    backbone = build_qkv_backbone(_qkv_config(tiny_config))
    qkv_layer = backbone.get_layer("qkv_attention")
    x_big = np.full((2, _N_SEQ, 1), 1e6, dtype=np.float32)
    h_att = backbone(x_big, training=False).numpy()
    assert np.all(np.isfinite(h_att))
    qkv_layer._last_attention_weights = tf.ones((2, 96, 96))
    with caplog.at_level(logging.WARNING, logger="src.models.ultra_can_qkv"):
        _log_attention_saturation(qkv_layer)
    assert any("satur" in r.getMessage() for r in caplog.records)

def test_qkv_serialization_get_config() -> None:
    
    layer = build_qkv_attention_layer(64, 4)
    cfg = layer.get_config()
    assert cfg["attention_dim"] == 64
    assert cfg["num_heads"] == 4
    assert cfg["name"] == "qkv_attention"
    restored = type(layer)(
        attention_dim=cfg["attention_dim"],
        num_heads=cfg["num_heads"],
        name=cfg["name"],
    )
    rcfg = restored.get_config()
    assert rcfg["attention_dim"] == cfg["attention_dim"]
    assert rcfg["num_heads"] == cfg["num_heads"]

def test_qkv_model_save_load_roundtrip(
    tiny_config: Dict[str, Any], tmp_path: Any
) -> None:
    
    from src.utils.model_io import load_model

    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = _random_input(4)
    out_before = model(x, training=False)

    ckpt = tmp_path / "qkv_model.keras"
    model.save(str(ckpt))

    loaded = load_model(ckpt)
    out_after = loaded(x, training=False)

    np.testing.assert_allclose(
        out_before["comm"].numpy(), out_after["comm"].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        out_before["sensing"].numpy(), out_after["sensing"].numpy(), atol=1e-6
    )
    assert loaded.count_params() == model.count_params()

def test_qkv_micro_size_warning(
    tiny_config: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """QKV size='micro': build OK + WARNING via caplog (non un errore)."""
    cfg = _with_override(_qkv_config(tiny_config), "model.size", "micro")
    with caplog.at_level(logging.WARNING, logger="src.models.ultra_can_qkv"):
        model = build_dual_head_ultra_can_qkv(cfg)
    assert count_trainable_params(model) > 0
    assert any("micro" in r.getMessage() for r in caplog.records)

def test_qkv_comm_outputs_logits(tiny_config: Dict[str, Any]) -> None:
    """QKV: la testa comm emette LOGITS grezzi (Fase 4: attivazione lineare)."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = _random_input(4)
    out = model(x, training=False)
    comm = out["comm"].numpy()
    assert np.all(np.isfinite(comm))
    np.testing.assert_allclose(
        tf.nn.softmax(comm).numpy().sum(axis=-1), np.ones(4), rtol=0, atol=1e-6
    )

def test_qkv_sensing_output_column_order(tiny_config: Dict[str, Any]) -> None:
    """QKV: colonne output sensing = [tau, fD] (layer 'sensing_out', 2 unit)."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    sensing_head = model.get_layer("sensing_head")
    out_layer = sensing_head.get_layer("sensing_out")
    assert int(out_layer.units) == 2
    x = np.zeros((4, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    assert out["sensing"].shape == (4, 2)

def test_qkv_wrong_input_dim_raises(tiny_config: Dict[str, Any]) -> None:
    """QKV input con seq_len incompatibile (B, 50, 1) vs atteso (B, 100, 1)."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x_wrong = np.zeros((2, 50, 1), dtype=np.float32)
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        model(x_wrong, training=False)

def test_modules_importable() -> None:
    """I tre moduli modelli si importano ed espongono le funzioni pubbliche."""
    import src.models.heads
    import src.models.ultra_can
    import src.models.ultra_can_qkv

    assert callable(src.models.heads.build_communication_head)
    assert callable(src.models.heads.build_sensing_head)
    assert callable(src.models.ultra_can.build_dual_head_ultra_can)
    assert callable(src.models.ultra_can_qkv.build_dual_head_ultra_can_qkv)

def test_end_to_end_conv1d(tiny_model: tf.keras.Model, tiny_dataset: Dict[str, np.ndarray]) -> None:
    """Flusso end-to-end conv1d: forward su 128 sequenze reali del tiny_dataset."""
    x = tiny_dataset["x"]
    assert x.shape == (128, _N_SEQ, 2)
    out = tiny_model(x, training=False)
    assert out["comm"].shape == (128, _M_BPSK)
    assert out["sensing"].shape == (128, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))
    np.testing.assert_allclose(
        tf.nn.softmax(out["comm"].numpy()).numpy().sum(axis=-1),
        np.ones(128),
        rtol=0,
        atol=1e-6,
    )

def test_end_to_end_qkv(tiny_config: Dict[str, Any], tiny_dataset: Dict[str, np.ndarray]) -> None:
    """Flusso end-to-end QKV: forward su 128 sequenze reali del tiny_dataset."""
    model = build_dual_head_ultra_can_qkv(_qkv_config(tiny_config))
    x = tiny_dataset["x"]
    out = model(x, training=False)
    assert out["comm"].shape == (128, _M_BPSK)
    assert out["sensing"].shape == (128, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

@pytest.mark.parametrize("batch", [_SMALL_BATCH, _MEDIUM_BATCH, 128])
def test_forward_batch_variation(tiny_model: tf.keras.Model, batch: int) -> None:
    """Batch 1/32/128: shape corrette e output finiti (template utente)."""
    x = _random_input(batch)
    out = tiny_model(x, training=False)
    assert out["comm"].shape == (batch, _M_BPSK)
    assert out["sensing"].shape == (batch, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

@pytest.mark.parametrize("main_fn", [heads_main, ultra_can_main, qkv_main])
def test_main_cli_missing_config(main_fn: Callable[..., None]) -> None:
    """main([]) senza --config -> SystemExit (argparse, --config required)."""
    with pytest.raises(SystemExit):
        main_fn([])

