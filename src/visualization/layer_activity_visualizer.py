#!/usr/bin/env python3

"""Layer activity/retention visualisation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.data_loader import load_npz_files
from src.data.dataset_generator import build_snr_grid
from src.experiments.run_jamming import apply_jamming
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, validate_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model
from src.visualization.attention_visualizer import extract_attention_weights

logger = get_logger(__name__)

_REQUIRED_KEYS: tuple[str, ...] = (
    "general.experiment_name",
    "general.seed",
    "general.log_level",
    "data.sequence_length",
    "data.feature_mode",
    "data.snr_range",
    "data.snr_step",
    "data.echoes",
    "data.max_delay",
    "data.max_doppler",
    "data.raw_dir",
    "visualization.plot_format",
)


def _build_model_if_possible(model: tf.keras.Model) -> None:
    
    if model.input_shape is not None:
        return
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        raise ValueError(
            "Il modello non è buildato e non ha un attributo 'input_shape' noto. "
            "Impossibile procedere."
        )
    try:
        model.build(input_shape=tuple(input_shape))
        logger.debug("Modello buildato con input_shape=%s", input_shape)
    except Exception as e:
        raise ValueError(f"Impossibile buildare il modello con input_shape={input_shape}: {e}")


def layer_weight_norms(model: tf.keras.Model) -> Dict[str, float]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model deve essere tf.keras.Model, ricevuto: {type(model).__name__}")

    _build_model_if_possible(model)

    activity: Dict[str, float] = {}
    for layer in model.layers:
        if not layer.trainable_variables:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        squared_sum = 0.0
        for w in weights:
            if not np.all(np.isfinite(w)):
                logger.warning(
                    "Il layer '%s' contiene valori non finiti nei pesi, saltato",
                    layer.name
                )
                continue
            squared_sum += np.sum(w ** 2)
        norm = float(np.sqrt(squared_sum))
        activity[layer.name] = norm
        logger.debug("layer '%s': norma L2 = %.6f", layer.name, norm)

    if not activity:
        logger.warning("Nessun layer con pesi addestrabili trovato nel modello")
    return activity

def layer_activation_variance(model: tf.keras.Model, sample: np.ndarray) -> Dict[str, float]:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model deve essere tf.keras.Model, ricevuto: {type(model).__name__}")
    if not isinstance(sample, np.ndarray):
        raise TypeError(f"sample deve essere np.ndarray, ricevuto: {type(sample).__name__}")
    if not np.all(np.isfinite(sample)):
        raise ValueError("sample contiene NaN/Inf ( Sez. 1.2)")

    _build_model_if_possible(model)

    layer_outputs = []
    layer_names = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, tf.keras.Model):
            logger.debug("Layer '%s' è un sub-modello, saltato", layer.name)
            continue
        if hasattr(layer, "output") and isinstance(layer.output, tf.keras.KerasTensor):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
        else:
            logger.debug("Layer '%s' non ha un output singolo, saltato", layer.name)

    if not layer_outputs:
        raise ValueError("Nessun layer con output singolo trovato nel modello")

    intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    logger.debug("Modello intermedio creato con %d layer di output", len(layer_outputs))

    outputs = intermediate_model(sample, training=False)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    variance: Dict[str, float] = {}
    for name, out in zip(layer_names, outputs):
        out_np = out.numpy()
        if not np.all(np.isfinite(out_np)):
            raise RuntimeError(f"L'output del layer '{name}' contiene NaN/Inf")
        var = float(np.var(out_np))
        variance[name] = var
        logger.debug("layer '%s': varianza attivazioni = %.6f", name, var)

    return variance

def plot_layer_activity(
    activity: Dict[str, float],
    output_path: Path,
    title_prefix: str = "Layer Activity",
) -> Path:
    
    if not activity:
        raise ValueError("activity è vuoto, impossibile generare il plot")
    for name, val in activity.items():
        if not np.isfinite(val):
            raise ValueError(f"Valore non finito per layer '{name}': {val}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(activity.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(labels))))
    ax.bar(range(len(values)), values, color="skyblue", edgecolor="navy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Valore")
    ax.set_title(f"{title_prefix} - ordinato per attività")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, format=output_path.suffix[1:], bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Plot layer activity salvato in %s", output_path)
    return output_path

def plot_attention_vs_jamming(
    alpha_clean: np.ndarray,
    alpha_jammed: np.ndarray,
    output_path: Path,
) -> Path:
    
    if alpha_clean.shape != alpha_jammed.shape:
        raise ValueError(
            f"Shape diverse: alpha_clean {alpha_clean.shape}, alpha_jammed {alpha_jammed.shape}"
        )
    if not np.all(np.isfinite(alpha_clean)) or not np.all(np.isfinite(alpha_jammed)):
        raise ValueError("alpha_clean o alpha_jammed contiene NaN/Inf")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(alpha_clean))
    ax.plot(x, alpha_clean, "o-", label="Pulito", color="blue", linewidth=2, markersize=6)
    ax.plot(x, alpha_jammed, "s-", label="Jammed", color="red", linewidth=2, markersize=6)

    ax.set_xlabel("Indice temporale")
    ax.set_ylabel("Peso di attenzione α[i]")
    ax.set_title("Confronto pesi di attenzione: pulito vs jamming")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, format=output_path.suffix[1:], bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Plot attenzione vs jamming salvato in %s", output_path)
    return output_path


def _validate_main_config(config: Dict[str, Any]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    general = config.get("general")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict")
    if "experiment_name" not in general:
        raise ValueError("chiave mancante: general.experiment_name")
    if "seed" not in general:
        raise ValueError("chiave mancante: general.seed")

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("sezione 'data' mancante o non dict")
    for key in ("sequence_length", "feature_mode", "snr_range", "snr_step",
                "echoes", "max_delay", "max_doppler", "raw_dir"):
        if key not in data:
            raise ValueError(f"chiave mancante in data: {key}")

    vis = config.get("visualization")
    if not isinstance(vis, dict):
        raise ValueError("sezione 'visualization' mancante o non dict")
    if "plot_format" not in vis:
        raise ValueError("chiave mancante: visualization.plot_format")

    logger.debug("Config validata per layer_activity_visualizer")

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Genera plot di layer activity e attenzione per l'Esperimento 4"
    )
    parser.add_argument("--config", required=True, help="path della config esperimento (YAML)")
    parser.add_argument("--model-path", required=True, help="path del modello Keras (SavedModel o .h5)")
    parser.add_argument("--jamming", action="store_true", help="genera anche il confronto attenzione pulito vs jamming")
    parser.add_argument(
        "--output-dir",
        default="results/full_experiment/jamming/plots",
        help="directory di output per i plot (default: results/full_experiment/jamming/plots)",
    )
    parser.add_argument(
        "--plot-format",
        default=None,
        help="formato dei plot (default: da config visualization.plot_format)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"File di config non trovato: {config_path}")

    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    _validate_main_config(config)

    general = config["general"]
    experiment_name = str(general.get("experiment_name", "jamming"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    logger.info("Caricamento modello da %s", model_path)
    model = load_model(model_path)
    logger.info("Modello caricato con successo")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = _REPO_ROOT / raw_dir

    snr_grid = build_snr_grid(data_cfg["snr_range"], data_cfg["snr_step"])
    echoes = list(data_cfg["echoes"])

    logger.info("Caricamento dataset di test da %s", raw_dir)
    data = load_npz_files(raw_dir, snr_grid, echoes, "test", config)
    if data["x"].shape[0] == 0:
        raise RuntimeError("Dataset di test vuoto, impossibile estrarre campioni")

    from src.data.data_loader import _build_feature_matrix, build_reference_matrix
    feature_mode = str(data_cfg.get("feature_mode", "real"))
    sample_raw = data["x"][0:1]
    sample_ref = build_reference_matrix(data["bit"][0:1], data["seed"][0:1], config)
    sample = _build_feature_matrix(sample_raw, feature_mode, reference=sample_ref)
    logger.debug("Campione estratto: shape=%s", sample.shape)

    output_dir = Path(args.output_dir)
    plot_format = args.plot_format or config["visualization"]["plot_format"]

    logger.info("Calcolo norme L2 dei pesi per layer...")
    weight_norms = layer_weight_norms(model)
    if weight_norms:
        out_path = output_dir / f"layer_activity_weights.{plot_format}"
        plot_layer_activity(weight_norms, out_path, title_prefix="Norma L2 dei pesi per layer")
    else:
        logger.warning("Nessuna norma L2 dei pesi estratta (modello senza layer con pesi)")

    logger.info("Calcolo varianza delle attivazioni per layer...")
    try:
        activation_variance = layer_activation_variance(model, sample)
        if activation_variance:
            out_path = output_dir / f"layer_activity_variance.{plot_format}"
            plot_layer_activity(activation_variance, out_path, title_prefix="Varianza delle attivazioni per layer")
        else:
            logger.warning("Nessuna varianza delle attivazioni estratta")
    except Exception as e:
        logger.error("Errore nel calcolo della varianza delle attivazioni: %s", e)

    if args.jamming:
        logger.info("Generazione confronto attenzione pulito vs jamming...")
        try:
            jamming_cfg = config.get("jamming", {})
            jamming_types = jamming_cfg.get("jamming_types", ["cw"])
            jsr_range = jamming_cfg.get("jsr_range", [0, 10])

            jamming_type = jamming_types[0] if jamming_types else "cw"
            if isinstance(jsr_range, (list, tuple)) and len(jsr_range) >= 2:
                jsr_db = (jsr_range[0] + jsr_range[1]) / 2.0
            else:
                jsr_db = 5.0

            logger.info("Jamming utilizzato: tipo='%s', JSR=%.1f dB", jamming_type, jsr_db)

            rng = np.random.default_rng(int(general["seed"]))
            sample_jammed_raw = apply_jamming(sample_raw, jamming_type, jsr_db, rng)
            sample_jammed = _build_feature_matrix(
                sample_jammed_raw, feature_mode, reference=sample_ref
            )

            alpha_clean = extract_attention_weights(model, sample)
            alpha_jammed = extract_attention_weights(model, sample_jammed)

            out_path = output_dir / f"attention_compare.{plot_format}"
            plot_attention_vs_jamming(alpha_clean, alpha_jammed, out_path)
        except Exception as e:
            logger.error("Errore durante la generazione del confronto attenzione: %s", e)

    logger.info("Generazione layer activity plots completata. Output in %s", output_dir)

if __name__ == "__main__":
    main()

