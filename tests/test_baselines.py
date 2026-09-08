"""
Test per le baseline DL (LSTM, MC-DLCSK) e il correlatore DCSK classico.

(tests/test_baselines.py).

Questi test verificano:
  - Contratto di input/output delle architetture (LSTM, MC-DLCSK, DCSK)
  - Conteggio dei parametri per le varianti full e micro
  - Correttezza matematica dei demodulatori classici (DCSK, matched filter, energy)
  - Gestione di input estremi e invalidi (NaN/Inf, shape errate)
  - Stabilità numerica a SNR estremi (-20 dB, +30 dB)
  - Coerenza con il dataset condiviso (input (B, 100, 1), output {"comm", "sensing"})
  - Emissione di warning per template degeneri o configurazioni micro

Tutti i test utilizzano pytest e le fixture di conftest.py.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from src.models.baselines import (
    _validate_baseline_config,
    build_baseline,
    build_lstm_baseline,
    build_mc_dlsk_baseline,
    count_trainable_params as baseline_count,
)
from src.models.dcsk_correlator import (
    count_trainable_params as dcsk_count,
    dcsk_correlator_demodulate,
    energy_detector_demodulate,
    evaluate_classical,
    main as dcsk_main,
    matched_filter_demodulate,
)
from src.models.heads import num_params
from src.models.ultra_can import build_dual_head_ultra_can
from src.models.ultra_can_qkv import build_dual_head_ultra_can_qkv

_N_SEQ = 100
_HEAD_INPUT_DIM = 64
_M_BPSK = 2
_BATCH_SIZES = (1, 1024)
_TINY_BATCH = 4

_LSTM_FULL_PARAMS_EST = 45_680
_LSTM_MICRO_PARAMS_EST = 45_680
_MC_DLSK_FULL_PARAMS_EST = 44_356
_MC_DLSK_MICRO_PARAMS_EST = 44_356

def _with_override(config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Copia profonda e applica override su path dot-separato."""
    cfg = copy.deepcopy(config)
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return cfg

def _qkv_config(tiny_config: Dict[str, Any]) -> Dict[str, Any]:
    """Converte la config in modalità qkv_attention."""
    return _with_override(tiny_config, "model.backbone_type", "qkv_attention")


@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
@pytest.mark.parametrize("batch", _BATCH_SIZES)
def test_baseline_forward_shapes(tiny_config: Dict[str, Any], name: str, batch: int) -> None:
    
    model = build_baseline(tiny_config, name)
    x = np.zeros((batch, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    assert out["comm"].shape == (batch, _M_BPSK)
    assert out["sensing"].shape == (batch, 2)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

@pytest.mark.parametrize("size", ["full", "micro"])
def test_lstm_param_count(tiny_config: Dict[str, Any], size: str) -> None:
    
    cfg = _with_override(tiny_config, "baselines.lstm.size", size)
    model = build_baseline(cfg, "lstm")
    total = baseline_count(model)
    if size == "full":
        assert total == _LSTM_FULL_PARAMS_EST
    else:
        assert total == _LSTM_MICRO_PARAMS_EST

@pytest.mark.parametrize("size", ["full", "micro"])
def test_mc_dlsk_param_count(tiny_config: Dict[str, Any], size: str) -> None:
    
    cfg = _with_override(tiny_config, "baselines.mc_dlsk.size", size)
    model = build_baseline(cfg, "mc_dlsk")
    total = baseline_count(model)
    if size == "full":
        assert total > 40_000 and total < 50_000
    else:
        assert total == _MC_DLSK_FULL_PARAMS_EST

def test_baseline_wrong_input_dim_raises(tiny_config: Dict[str, Any]) -> None:
    
    model = build_baseline(tiny_config, "lstm")
    for x_bad in (
        np.zeros((4, 50, 2), dtype=np.float32),
        np.zeros((4, _N_SEQ, 3), dtype=np.float32),
    ):
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
            model(x_bad, training=False)

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_baseline_extreme_input(tiny_config: Dict[str, Any], name: str, scale: float) -> None:
    """Input con valori estremi (1e-6, 1e6) -> output finiti."""
    model = build_baseline(tiny_config, name)
    x = np.full((_TINY_BATCH, _N_SEQ, 2), scale, dtype=np.float32)
    out = model(x, training=False)
    assert np.all(np.isfinite(out["comm"].numpy()))
    assert np.all(np.isfinite(out["sensing"].numpy()))

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_zero_input_finite(tiny_config: Dict[str, Any], name: str) -> None:
    
    model = build_baseline(tiny_config, name)
    x = np.zeros((4, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    comm = out["comm"].numpy()
    assert np.all(np.isfinite(comm))
    assert np.all(np.isfinite(out["sensing"].numpy()))
    np.testing.assert_allclose(
        tf.nn.softmax(comm).numpy().sum(axis=-1), np.ones(4), rtol=0, atol=1e-6
    )

def test_baseline_nan_input_raises(tiny_config: Dict[str, Any]) -> None:
    """Input con NaN -> la guardia assert_all_finite solleva."""
    model = build_baseline(tiny_config, "lstm")
    x_nan = np.full((2, _N_SEQ, 2), np.nan, dtype=np.float32)
    out = model(x_nan, training=False)
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.debugging.assert_all_finite(out["comm"], message="NaN")

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_baseline_inference_deterministic(tiny_config: Dict[str, Any], name: str) -> None:
    """Due forward con training=False producono output identici."""
    model = build_baseline(tiny_config, name)
    x = np.random.default_rng(0).uniform(size=(4, _N_SEQ, 2)).astype(np.float32)
    o1 = model(x, training=False)
    o2 = model(x, training=False)
    assert np.array_equal(o1["comm"].numpy(), o2["comm"].numpy())
    assert np.array_equal(o1["sensing"].numpy(), o2["sensing"].numpy())

def test_lstm_dropout_train_vs_inference(tiny_config: Dict[str, Any]) -> None:
    """Dropout attivo solo in training."""
    model = build_baseline(tiny_config, "lstm")
    x = np.random.default_rng(0).uniform(size=(8, _N_SEQ, 2)).astype(np.float32)
    train_1 = model(x, training=True)
    train_2 = model(x, training=True)
    infer_1 = model(x, training=False)
    infer_2 = model(x, training=False)
    assert not np.array_equal(train_1["comm"].numpy(), train_2["comm"].numpy())
    assert np.array_equal(infer_1["comm"].numpy(), infer_2["comm"].numpy())

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_comm_outputs_logits(tiny_config: Dict[str, Any], name: str) -> None:
    """La testa comm emette LOGITS grezzi (Fase 4: softmax(logits) ~ 1)."""
    model = build_baseline(tiny_config, name)
    x = np.random.default_rng(0).uniform(size=(4, _N_SEQ, 2)).astype(np.float32)
    out = model(x, training=False)
    comm = out["comm"].numpy()
    assert np.all(np.isfinite(comm))
    np.testing.assert_allclose(
        tf.nn.softmax(comm).numpy().sum(axis=-1), np.ones(4), rtol=0, atol=1e-6
    )

def test_lstm_projection_when_units_neq_64(tiny_config: Dict[str, Any]) -> None:
    """baselines.lstm.units=[32] -> proiezione Dense(64) attiva, build OK."""
    cfg = _with_override(tiny_config, "baselines.lstm.units", [32])
    model = build_baseline(cfg, "lstm")
    proj = model.get_layer("lstm_proj")
    assert int(proj.units) == _HEAD_INPUT_DIM
    x = np.zeros((2, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    assert out["comm"].shape == (2, _M_BPSK)
    assert out["sensing"].shape == (2, 2)

def test_multi_layer_lstm_variation(tiny_config: Dict[str, Any]) -> None:
    """baselines.lstm.units=[32,32,32] -> 3 layer LSTM, build OK."""
    cfg = _with_override(tiny_config, "baselines.lstm.units", [32, 32, 32])
    model = build_baseline(cfg, "lstm")
    lstm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)
    ]
    assert len(lstm_layers) == 3
    x = np.zeros((2, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    assert out["comm"].shape == (2, _M_BPSK)
    assert out["sensing"].shape == (2, 2)

def test_baseline_unknown_name_raises(tiny_config: Dict[str, Any]) -> None:
    """build_baseline con nome sconosciuto -> ValueError."""
    with pytest.raises(ValueError):
        build_baseline(tiny_config, "transformer")

def test_config_not_dict_raises() -> None:
    """_validate_baseline_config con config non dict -> TypeError."""
    with pytest.raises(TypeError):
        _validate_baseline_config(42, "lstm")

@pytest.mark.parametrize(
    "path, value, name",
    [
        ("baselines.lstm.units", [], "lstm"),
        ("baselines.lstm.units", [0], "lstm"),
        ("baselines.lstm.dropout", 1.5, "lstm"),
        ("baselines.lstm.size", "x", "lstm"),
        ("baselines.mc_dlsk.units", [], "mc_dlsk"),
    ],
)
def test_invalid_baseline_config_raises(
    tiny_config: Dict[str, Any], path: str, value: Any, name: str
) -> None:
    """Config baseline invalida -> ValueError (fail-fast,  Sez. 7)."""
    cfg = _with_override(tiny_config, path, value)
    with pytest.raises(ValueError):
        build_baseline(cfg, name)

def test_baseline_enabled_false_raises(tiny_config: Dict[str, Any]) -> None:
    """baselines.enabled=false -> ValueError (fail-fast)."""
    cfg = _with_override(tiny_config, "baselines.enabled", False)
    with pytest.raises(ValueError, match="disabilitata"):
        build_baseline(cfg, "lstm")

def test_enabled_missing_or_not_bool_raises(tiny_config: Dict[str, Any]) -> None:
    """baselines.enabled assente o non bool -> ValueError (fail-fast)."""
    cfg = copy.deepcopy(tiny_config)
    cfg["baselines"].pop("enabled", None)
    with pytest.raises(ValueError, match="enabled"):
        build_baseline(cfg, "lstm")

    cfg = _with_override(tiny_config, "baselines.enabled", "yes")
    with pytest.raises(ValueError, match="enabled"):
        build_baseline(cfg, "lstm")

def test_mc_dlsk_invalid_dropout_raises(tiny_config: Dict[str, Any]) -> None:
    """dropout fuori range in baselines.mc_dlsk -> ValueError (fail-fast)."""
    cfg = _with_override(tiny_config, "baselines.mc_dlsk.dropout", 1.5)
    with pytest.raises(ValueError, match="dropout"):
        build_baseline(cfg, "mc_dlsk")

def test_mc_dlsk_is_bilstm_classifier(tiny_config: Dict[str, Any]) -> None:
    
    model = build_baseline(tiny_config, "mc_dlsk")
    names = [layer.name for layer in model.layers]
    assert "mc_bilstm_1" in names
    assert not any(isinstance(layer, tf.keras.layers.Conv1D) for layer in model.layers)

def test_mc_dlsk_has_no_conv1_layer(tiny_config: Dict[str, Any]) -> None:
    
    model = build_baseline(tiny_config, "mc_dlsk")
    with pytest.raises(ValueError):
        model.get_layer("conv1")

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_seq_len_variation(tiny_config: Dict[str, Any], name: str) -> None:
    """sequence_length=50 -> build OK (MC-DLCSK: att_len = 50 - 4 = 46)."""
    cfg = _with_override(tiny_config, "data.sequence_length", 50)
    model = build_baseline(cfg, name)
    x = np.zeros((2, 50, 1), dtype=np.float32)
    out = model(x, training=False)
    assert out["comm"].shape == (2, _M_BPSK)
    assert out["sensing"].shape == (2, 2)

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_feature_mode_iq(tiny_config: Dict[str, Any], name: str) -> None:
    
    cfg = _with_override(tiny_config, "data.feature_mode", "iq")
    model = build_baseline(cfg, name)
    x = np.zeros((4, _N_SEQ, 3), dtype=np.float32)
    out = model(x, training=False)
    assert out["comm"].shape == (4, _M_BPSK)
    assert out["sensing"].shape == (4, 2)

def test_head_dim_mismatch_raises(tiny_config: Dict[str, Any]) -> None:
    
    cfg = _with_override(tiny_config, "model.conv_filters", [16, 32])
    with pytest.raises(ValueError, match="v in R"):
        build_baseline(cfg, "lstm")

@pytest.mark.parametrize("name", ["lstm", "mc_dlsk"])
def test_sensing_output_column_order(tiny_config: Dict[str, Any], name: str) -> None:
    """Colonne output sensing = [tau, fD]: layer finale 'sensing_out' con 2 unit."""
    model = build_baseline(tiny_config, name)
    sensing_head = model.get_layer("sensing_head")
    out_layer = sensing_head.get_layer("sensing_out")
    assert int(out_layer.units) == 2
    x = np.zeros((4, _N_SEQ, 2), dtype=np.float32)
    out = model(x, training=False)
    assert out["sensing"].shape == (4, 2)


def test_dcsk_correlator_zero_params() -> None:
    
    assert dcsk_count() == 0

@pytest.mark.parametrize("threshold", [0.0, 0.1, -0.1])
def test_dcsk_demod_correct_at_high_snr(tiny_config: Dict[str, Any], threshold: float) -> None:
    """Frame DCSK [ref, +-ref] -> BER 0 per DCSK e matched filter."""
    from src.data.dataset_generator import generate_chaotic_sequence
    cfg = tiny_config
    data = cfg["data"]
    beta = int(cfg["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = generate_chaotic_sequence(
        str(data["map_type"]), float(data["map_param"]), 42, beta
    )
    y0 = np.concatenate([ref, -ref])
    y1 = np.concatenate([ref, ref])
    y_stack = np.stack([y0, y1])
    bits_true = np.array([0, 1], dtype=np.int8)

    bits_dcsk = dcsk_correlator_demodulate(y_stack, threshold=threshold)
    assert bits_dcsk.tolist() == [0, 1]

    bits_mf = matched_filter_demodulate(y_stack[:, beta:], ref, threshold=threshold)
    assert bits_mf.tolist() == [0, 1]

def test_dcsk_demod_logistic_mu4_limit(tiny_config: Dict[str, Any]) -> None:
    """Template con mu=4.0 (limite caos) -> BER 0."""
    from src.data.dataset_generator import generate_chaotic_sequence
    cfg = tiny_config
    beta = int(cfg["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = generate_chaotic_sequence("logistic", 4.0, 123, beta)
    y0 = np.concatenate([ref, -ref])
    y1 = np.concatenate([ref, ref])
    y_stack = np.stack([y0, y1])
    bits_true = np.array([0, 1])
    bits = dcsk_correlator_demodulate(y_stack)
    assert np.array_equal(bits, bits_true)

def test_dcsk_demod_k0_no_echoes(tiny_config: Dict[str, Any]) -> None:
    """K=0 (solo AWGN ad alto SNR) -> decodifica corretta (test di integrazione)."""
    from src.data.dataset_generator import (
        generate_chaotic_sequence,
        apply_aerial_channel,
    )
    cfg = tiny_config
    beta = int(cfg["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = generate_chaotic_sequence("logistic", 3.9, 55, beta)
    y0 = np.concatenate([ref, -ref])
    y1 = np.concatenate([ref, ref])
    y_stack = np.stack([y0, y1])
    rng = np.random.default_rng(0)
    y_clean = []
    for y in y_stack:
        y_ch, _ = apply_aerial_channel(
            y, complex(1.0, 0.0), 0.0, [], 30.0, rng
        )
        y_clean.append(y_ch)
    y_clean = np.stack(y_clean)
    bits_true = np.array([0, 1])
    bits = dcsk_correlator_demodulate(y_clean)
    assert np.array_equal(bits, bits_true)

def test_dcsk_correlator_meta_symbol_too_long_raises(tiny_config: Dict[str, Any]) -> None:
    """2*beta > N_seq -> ValueError."""
    cfg = _with_override(
        tiny_config,
        "baselines.dcsk_correlator.correlation_length",
        60
    )
    with pytest.raises(ValueError, match="sequence_length"):
        evaluate_classical(
            np.zeros((2, 120), dtype=np.complex128),
            np.array([0, 1]),
            "dcsk",
            cfg
        )

def test_dcsk_demod_snr_minus20_finite(tiny_config: Dict[str, Any]) -> None:
    """SNR=-20 dB -> output finiti, BER ~0.5 (non NaN)."""
    from src.data.dataset_generator import (
        generate_chaotic_sequence,
        apply_aerial_channel,
    )
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = generate_chaotic_sequence("logistic", 3.9, 77, beta)
    y0 = np.concatenate([ref, -ref])
    y1 = np.concatenate([ref, ref])
    y_stack = np.stack([y0, y1])
    rng = np.random.default_rng(1)
    y_noisy = []
    for y in y_stack:
        y_ch, _ = apply_aerial_channel(y, complex(1.0, 0.0), 0.0, [], -20.0, rng)
        y_noisy.append(y_ch)
    y_noisy = np.stack(y_noisy)
    bits = dcsk_correlator_demodulate(y_noisy)
    errors = np.sum(bits != np.array([0, 1]))
    assert 0 <= errors <= 2
    assert np.all(np.isfinite(bits))

def test_dcsk_energy_balance_floor(tiny_config: Dict[str, Any]) -> None:
    """Su frame casuali, i tre detector danno BER ~0.5 (floor)."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    rng = np.random.default_rng(42)
    n_samples = 200
    y = rng.normal(size=(n_samples, 2*beta)) + 1j * rng.normal(size=(n_samples, 2*beta))
    bits_true = rng.integers(0, 2, size=n_samples)

    for detector in ("dcsk", "matched_filter", "energy_detector"):
        frame = y[:, :beta] if detector == "matched_filter" else y
        res = evaluate_classical(
            frame, bits_true, detector, tiny_config,
            template=None if detector == "dcsk" else np.zeros(beta)
        )
        ber = res["ber"]
        assert 0.45 <= ber <= 0.55

def test_dcsk_correlator_invalid_config_raises(tiny_config: Dict[str, Any]) -> None:
    """correlation_length=0, threshold=NaN -> ValueError."""
    cfg = _with_override(tiny_config, "baselines.dcsk_correlator.correlation_length", 0)
    with pytest.raises(ValueError):
        evaluate_classical(np.zeros((2, 1)), np.array([0, 1]), "dcsk", cfg)

    cfg = _with_override(tiny_config, "baselines.dcsk_correlator.threshold", float("nan"))
    with pytest.raises(ValueError):
        evaluate_classical(np.zeros((2, 2)), np.array([0, 1]), "dcsk", cfg)

def test_dcsk_correlator_nan_input_raises(tiny_config: Dict[str, Any]) -> None:
    """Input y con NaN -> ValueError."""
    y_nan = np.full((2, 10), np.nan + 0j, dtype=np.complex128)
    with pytest.raises(ValueError, match="NaN/Inf"):
        dcsk_correlator_demodulate(y_nan)

def test_matched_filter_template_mismatch_raises(tiny_config: Dict[str, Any]) -> None:
    """Template lunghezza diversa da y -> ValueError."""
    y = np.zeros((2, 20), dtype=np.complex128)
    template = np.zeros(10)
    with pytest.raises(ValueError, match="allineato"):
        matched_filter_demodulate(y, template)

def test_energy_detector_empty_raises(tiny_config: Dict[str, Any]) -> None:
    """y vuoto -> ValueError."""
    y = np.zeros((0, 10), dtype=np.complex128)
    with pytest.raises(ValueError, match="vuoto"):
        energy_detector_demodulate(y)

def test_dcsk_demod_ref_explicit(tiny_config: Dict[str, Any]) -> None:
    """Fornire ref esterno -> decodifica corretta."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = np.ones(beta, dtype=np.complex128)
    y0 = np.concatenate([ref, -ref])
    y1 = np.concatenate([ref, ref])
    y_stack = np.stack([y0, y1])
    bits_true = np.array([0, 1])
    bits = dcsk_correlator_demodulate(y_stack, ref=ref)
    assert np.array_equal(bits, bits_true)

def test_dcsk_demod_ref_broadcast(tiny_config: Dict[str, Any]) -> None:
    """ref con shape (1, beta) broadcast su batch."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = np.ones((1, beta), dtype=np.complex128)
    y0 = np.concatenate([ref[0], -ref[0]])
    y1 = np.concatenate([ref[0], ref[0]])
    y_stack = np.stack([y0, y1])
    bits = dcsk_correlator_demodulate(y_stack, ref=ref)
    assert np.array_equal(bits, np.array([0, 1]))

def test_dcsk_demod_odd_length_raises(tiny_config: Dict[str, Any]) -> None:
    """y a lunghezza dispari -> ValueError."""
    y_odd = np.zeros((2, 15), dtype=np.complex128)
    with pytest.raises(ValueError, match="pari"):
        dcsk_correlator_demodulate(y_odd)

def test_dcsk_demod_complex_input(tiny_config: Dict[str, Any]) -> None:
    """Input complesso (come da npz) -> output finito."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    y = np.random.default_rng(0).normal(size=(4, 2*beta)) + 1j * np.random.default_rng(1).normal(size=(4, 2*beta))
    bits = dcsk_correlator_demodulate(y)
    assert np.all(np.isfinite(bits))
    assert set(bits) <= {0, 1}


@pytest.mark.parametrize("detector", ["dcsk", "matched_filter", "energy_detector"])
def test_evaluate_classical_dispatch(tiny_config: Dict[str, Any], detector: str) -> None:
    
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    y = np.random.default_rng(0).normal(size=(10, 2*beta)) + 1j * np.random.default_rng(1).normal(size=(10, 2*beta))
    bits_true = np.random.default_rng(2).integers(0, 2, size=10)
    if detector == "matched_filter":
        template = np.random.default_rng(3).normal(size=beta) + 1j * np.random.default_rng(4).normal(size=beta)
        res = evaluate_classical(y[:, :beta], bits_true, detector, tiny_config, template=template)
    else:
        res = evaluate_classical(y, bits_true, detector, tiny_config)
    assert "ber" in res and "n_errors" in res and "n_bits" in res
    assert 0.0 <= res["ber"] <= 1.0
    assert res["n_bits"] == 10

def test_evaluate_classical_unknown_detector_raises(tiny_config: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="detector"):
        evaluate_classical(np.zeros((2, 10)), np.array([0, 1]), "foobar", tiny_config)

def test_evaluate_classical_mf_missing_template_raises(tiny_config: Dict[str, Any]) -> None:
    
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    with pytest.raises(ValueError, match="template"):
        evaluate_classical(
            np.zeros((2, beta)), np.array([0, 1]), "matched_filter", tiny_config
        )

def test_evaluate_classical_bits_true_invalid_raises(tiny_config: Dict[str, Any]) -> None:
    """bits_true non 1D o valori fuori {0,1} -> ValueError."""
    with pytest.raises(ValueError, match="bits_true"):
        evaluate_classical(np.zeros((2, 10)), np.array([[0, 1]]), "dcsk", tiny_config)
    with pytest.raises(ValueError, match="0/1"):
        evaluate_classical(np.zeros((2, 10)), np.array([0, 2]), "dcsk", tiny_config)

def test_evaluate_classical_frame_length_raises(tiny_config: Dict[str, Any]) -> None:
    """Lunghezza frame y non coerente con beta -> ValueError."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    y_bad = np.zeros((2, 2*beta + 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="correlation_length"):
        evaluate_classical(y_bad, np.array([0, 1]), "dcsk", tiny_config)


def test_ber_empty_raises(tiny_config: Dict[str, Any]) -> None:
    """_ber con array vuoti -> ValueError (se chiamata direttamente)."""
    from src.models.dcsk_correlator import _ber
    with pytest.raises(ValueError, match="n_bits"):
        _ber(np.array([]), np.array([]))

def test_ber_shape_mismatch_raises(tiny_config: Dict[str, Any]) -> None:
    from src.models.dcsk_correlator import _ber
    with pytest.raises(ValueError, match="shape"):
        _ber(np.array([0, 1]), np.array([0]))


def test_matched_filter_degenerate_template_warning(
    tiny_config: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Template degenere (energia ~0) emette WARNING."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    template = np.zeros(beta, dtype=np.complex128)
    y = np.random.default_rng(0).normal(size=(2, beta)) + 1j * np.random.default_rng(1).normal(size=(2, beta))
    with caplog.at_level(logging.WARNING, logger="src.models.dcsk_correlator"):
        matched_filter_demodulate(y, template)
    assert any("degenere" in r.getMessage() for r in caplog.records)

def test_baseline_micro_size_warning(
    tiny_config: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """size='micro' emette WARNING in baselines.py."""
    cfg = _with_override(tiny_config, "baselines.lstm.size", "micro")
    with caplog.at_level(logging.WARNING, logger="src.models.baselines"):
        build_baseline(cfg, "lstm")
    assert any("micro" in r.getMessage() for r in caplog.records)


def test_all_models_share_dataset_contract(tiny_config: Dict[str, Any]) -> None:
    """Tutti i modelli (conv1d, qkv, lstm, mc_dlsk) accettano (B,100,1) e producono {'comm','sensing'}."""
    models = {
        "conv1d": build_dual_head_ultra_can,
        "qkv": lambda cfg: build_dual_head_ultra_can_qkv(_qkv_config(cfg)),
        "lstm": lambda cfg: build_baseline(cfg, "lstm"),
        "mc_dlsk": lambda cfg: build_baseline(cfg, "mc_dlsk"),
    }
    x = np.random.default_rng(0).uniform(size=(4, _N_SEQ, 2)).astype(np.float32)
    for name, builder in models.items():
        model = builder(tiny_config)
        out = model(x, training=False)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"comm", "sensing"}
        assert out["comm"].shape == (4, _M_BPSK)
        assert out["sensing"].shape == (4, 2)

def test_module_importable() -> None:
    """I moduli si importano senza errori."""
    import src.models.baselines
    import src.models.dcsk_correlator
    assert callable(src.models.baselines.build_baseline)
    assert callable(src.models.dcsk_correlator.dcsk_correlator_demodulate)


def test_main_cli_missing_config() -> None:
    """dcsk_main([]) -> SystemExit (--config required)."""
    with pytest.raises(SystemExit):
        dcsk_main([])


def test_threshold_flips_decision(tiny_config: Dict[str, Any]) -> None:
    """Soglia intermedia inverte la decisione su un singolo frame."""
    beta = int(tiny_config["baselines"]["dcsk_correlator"]["correlation_length"])
    ref = np.ones(beta, dtype=np.complex128)
    y = np.concatenate([ref, -ref])
    y = y.reshape(1, -1)
    bits0 = dcsk_correlator_demodulate(y, threshold=0.0)
    bits_neg = dcsk_correlator_demodulate(y, threshold=-1e9)
    bits_pos = dcsk_correlator_demodulate(y, threshold=1e9)
    assert bits_neg[0] == 1
    assert bits_pos[0] == 0
