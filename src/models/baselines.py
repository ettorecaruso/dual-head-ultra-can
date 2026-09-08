"""LSTM-OFDM-DCSK and MC-DLCSK baseline receivers."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.heads import (
    _assert_config_finite,
    _as_positive_int,
    _head_input_dim,
    _position_feature,
    _slice_received,
    build_communication_head,
    build_sensing_features,
    build_sensing_head,
)
from src.models.ultra_can import (
    _SMOKE_BATCH_SIZES,
    _expected_att_length,
    _smoke_check_forward,
    count_trainable_params,
)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_SENSING_OUTPUT_UNITS = 2
_HEAD_INPUT_DIM = 64
_MODEL_LSTM_NAME = "lstm_baseline"
_MODEL_MC_DLSK_NAME = "mc_dlsk_baseline"
_VALID_BASELINE_NAMES: Tuple[str, ...] = ("lstm", "mc_dlsk")
_VALID_SIZES: Tuple[str, ...] = ("full", "micro")
_VALID_FEATURE_MODES: Tuple[str, ...] = ("real", "iq")
_LSTM_REQUIRED_KEYS: Tuple[str, ...] = ("units", "dropout", "size")
_MC_DLSK_REQUIRED_KEYS: Tuple[str, ...] = ("units", "dropout", "size")

def _validate_baseline_config(
    config: Dict[str, Any],
    name: str,
) -> Tuple[Dict[str, Any], int, int]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    baselines_cfg = config.get("baselines")
    if not isinstance(baselines_cfg, dict):
        raise ValueError("sezione 'baselines' mancante o non dict nella config")

    enabled = baselines_cfg.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"baselines.enabled deve essere bool, ricevuto: {enabled!r}")
    if not enabled:
        logger.warning(
            "baselines.enabled e' false: la build delle baseline e' disabilitata"
        )
        raise ValueError(
            "baselines.enabled e' false: la build delle baseline e' disabilitata"
        )

    if name not in _VALID_BASELINE_NAMES:
        raise ValueError(
            f"baseline non valida: {name!r} (attese: {sorted(_VALID_BASELINE_NAMES)})"
        )

    baseline_cfg = baselines_cfg.get(name)
    if not isinstance(baseline_cfg, dict):
        raise ValueError(f"sezione 'baselines.{name}' mancante o non dict nella config")

    _assert_config_finite(baseline_cfg)

    required = _LSTM_REQUIRED_KEYS if name == "lstm" else _MC_DLSK_REQUIRED_KEYS
    missing = [key for key in required if key not in baseline_cfg]
    if missing:
        raise ValueError(f"chiavi mancanti in baselines.{name}: {missing}")

    size = baseline_cfg["size"]
    if not isinstance(size, str) or size not in _VALID_SIZES:
        raise ValueError(
            f"baselines.{name}.size deve essere uno di {list(_VALID_SIZES)}, "
            f"ricevuto: {size!r}"
        )

    if name in ("lstm", "mc_dlsk"):
        units = baseline_cfg["units"]
        if not isinstance(units, (list, tuple)) or len(units) == 0:
            raise ValueError(
                f"baselines.{name}.units deve essere una lista non vuota di "
                "interi positivi (layer ricorrenti), ricevuto: %r" % (units,)
            )
        for index, unit in enumerate(units):
            _as_positive_int(unit, f"baselines.{name}.units[{index}]")
        dropout = baseline_cfg["dropout"]
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(float(dropout))
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError(
                f"baselines.{name}.dropout deve essere in [0, 1), ricevuto: {dropout!r}"
            )

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("sezione 'data' mancante o non dict nella config")
    _assert_config_finite(data_cfg)

    seq_len = _as_positive_int(data_cfg.get("sequence_length"), "data.sequence_length")
    feature_mode = data_cfg.get("feature_mode", "real")
    if not isinstance(feature_mode, str) or feature_mode not in _VALID_FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode deve essere uno di {list(_VALID_FEATURE_MODES)}, "
            f"ricevuto: {feature_mode!r}"
        )
    num_features = 1 if feature_mode == "real" else 2

    head_input_dim = _head_input_dim(config)
    if head_input_dim != _HEAD_INPUT_DIM:
        raise ValueError(
            f"le teste condivise attendono v in R^{_HEAD_INPUT_DIM}, ma "
            f"heads._head_input_dim deriva {head_input_dim} dal backbone "
            "(model.conv_filters[1]/model.attention_dim): allineare la config "
            "prima di buildare le baseline"
        )

    logger.debug(
        "config baseline '%s' validata: size=%s, seq_len=%d, num_features=%d",
        name,
        size,
        seq_len,
        num_features,
    )
    return baseline_cfg, seq_len, num_features

def build_lstm_baseline(config: Dict[str, Any]) -> tf.keras.Model:
    
    baseline_cfg, seq_len, num_features = _validate_baseline_config(config, "lstm")
    size = str(baseline_cfg["size"])
    units = [int(unit) for unit in baseline_cfg["units"]]
    dropout = float(baseline_cfg["dropout"])

    if size == "micro":
        logger.warning(
            "baselines.lstm.size='micro' (~100 params, Exp 2): le teste condivise "
            "da heads.py sono fisse a 10724 params -> il target NON e' "
            "raggiungibile da questo modulo (ALERTA-3, decisione utente pendente)"
        )

    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    lstm_out = received
    for index, unit in enumerate(units):
        lstm_out = tf.keras.layers.LSTM(
            units=unit,
            return_sequences=True,
            name=f"lstm_{index + 1}",
        )(lstm_out)

    lstm_out = tf.keras.layers.Dropout(rate=dropout, name="dropout_lstm")(lstm_out)

    v, v_sensing = build_sensing_features(lstm_out, config, received_input=model_input)

    lstm_dim = int(units[-1])
    if lstm_dim != _HEAD_INPUT_DIM:
        logger.warning(
            "baselines.lstm.units[-1]=%d != contratto v in R^%d (Eq. (gap_comm)): "
            "aggiungo proiezione Dense(%d) prima delle teste condivise",
            lstm_dim,
            _HEAD_INPUT_DIM,
            _HEAD_INPUT_DIM,
        )
        v = tf.keras.layers.Dense(
            units=_HEAD_INPUT_DIM,
            activation="relu",
            kernel_initializer="he_normal",
            name="lstm_proj",
        )(v)
        sensing_cfg = config.get("model", {}).get("sensing_head") or {}
        use_position = bool(sensing_cfg.get("use_position_feature", False))
        use_profile = bool(sensing_cfg.get("use_reference_profile", True))
        parts: list = [v]
        if use_position:
            v_max = tf.keras.layers.Lambda(_position_feature, name="sensing_position")(lstm_out)
            parts.append(v_max)
        else:
            v_max = tf.keras.layers.GlobalMaxPooling1D(name="sensing_max_pool")(lstm_out)
            parts.append(v_max)
        if use_profile:
            from src.models.heads import _SENSING_PROFILE_LAG_KEY, _dca_mf_profile

            profile_max_lag = int(sensing_cfg.get(
                _SENSING_PROFILE_LAG_KEY, int(config.get("data", {}).get("max_delay", 33))
            ))
            nf = 2 if str(config.get("data", {}).get("feature_mode", "iq")) == "iq" else 1
            parts.append(tf.keras.layers.Lambda(
                _dca_mf_profile,
                arguments={"max_delay": profile_max_lag, "num_features": nf},
                name="sensing_delay_profile",
            )(model_input))
        v_sensing = tf.keras.layers.Concatenate(name="sensing_features")(parts)

    comm_head = build_communication_head(config)
    sensing_head = build_sensing_head(config, input_dim=int(v_sensing.shape[-1]))
    comm = comm_head(v)
    sensing = sensing_head(v_sensing)
    modulation_order = int(comm_head.output_shape[-1])

    if tuple(v.shape) != (None, _HEAD_INPUT_DIM):
        raise ValueError(
            f"v shape attesa (None, {_HEAD_INPUT_DIM}), ricevuta: {tuple(v.shape)}"
        )
    if tuple(comm.shape) != (None, modulation_order):
        raise ValueError(
            f"comm shape attesa (None, {modulation_order}), "
            f"ricevuta: {tuple(comm.shape)}"
        )
    if tuple(sensing.shape) != (None, _SENSING_OUTPUT_UNITS):
        raise ValueError(
            f"sensing shape attesa (None, {_SENSING_OUTPUT_UNITS}), "
            f"ricevuta: {tuple(sensing.shape)}"
        )

    model = tf.keras.Model(
        inputs=model_input,
        outputs={"comm": comm, "sensing": sensing},
        name=_MODEL_LSTM_NAME,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, _SENSING_OUTPUT_UNITS),
        },
    )
    comm_params = count_trainable_params(comm_head)
    sensing_params = count_trainable_params(sensing_head)
    total_params = count_trainable_params(model)
    logger.info(
        "modello '%s' costruito: backbone=%d params, comm=%d params, "
        "sensing=%d params, TOT=%d params",
        _MODEL_LSTM_NAME,
        total_params - comm_params - sensing_params,
        comm_params,
        sensing_params,
        total_params,
    )
    return model

def build_mc_dlsk_baseline(config: Dict[str, Any]) -> tf.keras.Model:
    
    baseline_cfg, seq_len, num_features = _validate_baseline_config(config, "mc_dlsk")
    size = str(baseline_cfg["size"])
    units = [int(unit) for unit in baseline_cfg["units"]]
    dropout = float(baseline_cfg["dropout"])

    if size == "micro":
        logger.warning(
            "baselines.mc_dlsk.size='micro' (~100 params, Exp 2): le teste "
            "condivise da heads.py sono fisse -> il target NON e' "
            "raggiungibile da questo modulo (ALERTA-3, decisione utente pendente)"
        )

    model_input = tf.keras.layers.Input(
        shape=(seq_len, num_features + 1), name="isac_input"
    )
    received = tf.keras.layers.Lambda(
        _slice_received,
        arguments={"num_features": num_features},
        name="received_slice",
    )(model_input)

    mc_out = received
    for index, unit in enumerate(units):
        mc_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=unit, return_sequences=True),
            name=f"mc_bilstm_{index + 1}",
        )(mc_out)

    mc_out = tf.keras.layers.Dropout(rate=dropout, name="dropout_mc")(mc_out)

    v, v_sensing = build_sensing_features(mc_out, config, received_input=model_input)

    mc_dim = int(2 * units[-1])
    if mc_dim != _HEAD_INPUT_DIM:
        logger.warning(
            "baselines.mc_dlsk: 2*units[-1]=%d != contratto v in R^%d: "
            "aggiungo proiezione Dense(%d) prima delle teste condivise",
            mc_dim,
            _HEAD_INPUT_DIM,
            _HEAD_INPUT_DIM,
        )
        v = tf.keras.layers.Dense(
            units=_HEAD_INPUT_DIM,
            activation="relu",
            kernel_initializer="he_normal",
            name="mc_proj",
        )(v)
        sensing_cfg = config.get("model", {}).get("sensing_head") or {}
        use_position = bool(sensing_cfg.get("use_position_feature", False))
        use_profile = bool(sensing_cfg.get("use_reference_profile", True))
        parts: list = [v]
        if use_position:
            v_max = tf.keras.layers.Lambda(_position_feature, name="sensing_position")(mc_out)
            parts.append(v_max)
        else:
            v_max = tf.keras.layers.GlobalMaxPooling1D(name="sensing_max_pool")(mc_out)
            parts.append(v_max)
        if use_profile:
            from src.models.heads import _SENSING_PROFILE_LAG_KEY, _dca_mf_profile

            profile_max_lag = int(sensing_cfg.get(
                _SENSING_PROFILE_LAG_KEY, int(config.get("data", {}).get("max_delay", 33))
            ))
            nf = 2 if str(config.get("data", {}).get("feature_mode", "iq")) == "iq" else 1
            parts.append(tf.keras.layers.Lambda(
                _dca_mf_profile,
                arguments={"max_delay": profile_max_lag, "num_features": nf},
                name="sensing_delay_profile",
            )(model_input))
        v_sensing = tf.keras.layers.Concatenate(name="sensing_features")(parts)

    comm_head = build_communication_head(config)
    sensing_head = build_sensing_head(config, input_dim=int(v_sensing.shape[-1]))
    comm = comm_head(v)
    sensing = sensing_head(v_sensing)
    modulation_order = int(comm_head.output_shape[-1])

    if tuple(v.shape) != (None, _HEAD_INPUT_DIM):
        raise ValueError(
            f"v shape attesa (None, {_HEAD_INPUT_DIM}), ricevuta: {tuple(v.shape)}"
        )
    if tuple(comm.shape) != (None, modulation_order):
        raise ValueError(
            f"comm shape attesa (None, {modulation_order}), "
            f"ricevuta: {tuple(comm.shape)}"
        )
    if tuple(sensing.shape) != (None, _SENSING_OUTPUT_UNITS):
        raise ValueError(
            f"sensing shape attesa (None, {_SENSING_OUTPUT_UNITS}), "
            f"ricevuta: {tuple(sensing.shape)}"
        )

    model = tf.keras.Model(
        inputs=model_input,
        outputs={"comm": comm, "sensing": sensing},
        name=_MODEL_MC_DLSK_NAME,
    )
    _smoke_check_forward(
        model,
        _SMOKE_BATCH_SIZES,
        expected={
            "comm": (None, modulation_order),
            "sensing": (None, _SENSING_OUTPUT_UNITS),
        },
    )
    comm_params = count_trainable_params(comm_head)
    sensing_params = count_trainable_params(sensing_head)
    total_params = count_trainable_params(model)
    logger.info(
        "modello '%s' costruito: backbone=%d params, comm=%d params, "
        "sensing=%d params, TOT=%d params",
        _MODEL_MC_DLSK_NAME,
        total_params - comm_params - sensing_params,
        comm_params,
        sensing_params,
        total_params,
    )
    return model

def build_baseline(config: Dict[str, Any], name: str) -> tf.keras.Model:
    
    if not isinstance(name, str) or name not in _VALID_BASELINE_NAMES:
        raise ValueError(
            f"baseline non valida: {name!r} (attese: {list(_VALID_BASELINE_NAMES)})"
        )

    builders: Dict[str, Any] = {
        "lstm": build_lstm_baseline,
        "mc_dlsk": build_mc_dlsk_baseline,
    }
    model = builders[name](config)

    input_shape = tuple(model.input_shape)
    if len(input_shape) != 3 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(
            f"shape input attesa (None, L, F), ricevuta: {input_shape!r}"
        )

    if isinstance(model.output, dict):
        output_keys = set(model.output.keys())
    else:
        output_keys = {getattr(model.output, "name", str(model.output))}
    if output_keys != {"comm", "sensing"}:
        raise ValueError(
            f"output attesi {{'comm', 'sensing'}}, ricevuti: {sorted(output_keys)}"
        )

    dummy_input = tf.ones(
        (1,) + tuple(model.input_shape[1:]),
        dtype=model.inputs[0].dtype,
    )
    outputs = model(dummy_input, training=False)
    for output_name in ("comm", "sensing"):
        tf.debugging.assert_all_finite(
            outputs[output_name],
            f"output '{output_name}' non finito su input non degenere (build)",
        )

    logger.info(
        "baseline '%s' costruita: input=%s, output=%s, params=%d",
        name,
        input_shape,
        sorted(output_keys),
        count_trainable_params(model),
    )
    return model

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Smoke test baseline DL (LSTM e MC-DLCSK), Fase 2.5"
    )
    parser.add_argument(
        "--config", required=True, help="path della config esperimento (YAML)"
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
    )

    general = config.get("general")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict nella config")
    experiment_name = str(general.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)

    lstm = build_baseline(config, "lstm")
    mc_dlsk = build_baseline(config, "mc_dlsk")
    logger.info(
        "baseline pronte: lstm=%d params, mc_dlsk=%d params",
        count_trainable_params(lstm),
        count_trainable_params(mc_dlsk),
    )

if __name__ == "__main__":
    main()

