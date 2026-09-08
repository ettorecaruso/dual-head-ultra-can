"""SWaP-C weight table generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.baselines import build_baseline
from src.models.dcsk_correlator import count_trainable_params as dcsk_count
from src.models.ultra_can import build_dual_head_ultra_can
from src.models.ultra_can_qkv import build_dual_head_ultra_can_qkv
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, validate_config
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model

logger = get_logger(__name__)

_BYTES_PER_PARAM_FLOAT32 = 4
_KB_FACTOR = 1024.0

def model_param_count(model: tf.keras.Model) -> int:
    
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"model deve essere tf.keras.Model, ricevuto: {type(model).__name__}")
    try:
        return int(model.count_params())
    except ValueError as exc:
        if hasattr(model, "input_shape") and model.input_shape is not None:
            try:
                model.build(input_shape=tuple(model.input_shape))
                return int(model.count_params())
            except Exception as build_exc:
                raise ValueError(
                    f"Impossibile buildare il modello per contare i parametri: {build_exc}"
                ) from build_exc
        raise ValueError(
            f"Modello non buildato e senza input_shape definita: {exc}"
        ) from exc

def model_size_kb(model: tf.keras.Model) -> float:
    
    params = model_param_count(model)
    return params * _BYTES_PER_PARAM_FLOAT32 / _KB_FACTOR

def _generate_latex_table(rows: list[dict]) -> str:
    
    latex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Confronto dei parametri e del footprint di memoria "
        "per le architetture considerate (pesi in float32).}",
        "\\label{tab:weight_comparison}",
        "\\begin{tabular}{l r r l}",
        "\\toprule",
        "\\textbf{Architettura} & \\textbf{Parametri} & \\textbf{Memoria (kB)} & \\textbf{Note} \\\\",
        "\\midrule",
    ]

    for row in rows:
        name = row["name"]
        params = f"{row['params']:,}" if row["params"] > 0 else "0"
        size_kb = f"{row['size_kb']:.1f}"
        note = row.get("note", "")
        latex.append(f"{name} & {params} & {size_kb} & {note} \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(latex)

def generate_weight_table(
    models: Dict[str, Optional[tf.keras.Model]],
    output_path: Path,
    include_classical: bool = True,
) -> Path:
    
    if not isinstance(models, dict):
        raise TypeError(f"models deve essere dict, ricevuto: {type(models).__name__}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_keys = {"conv1d", "qkv", "lstm", "mc_dlsk", "classical"}

    required_keys = expected_keys if include_classical else expected_keys - {"classical"}
    missing = required_keys - set(models.keys())
    if missing:
        raise ValueError(f"Chiavi mancanti in models: {sorted(missing)}")

    rows = []
    for name in sorted(models.keys()):
        model = models[name]
        if name == "classical":
            params = 0
            size_kb = 0.0
            note = "Solo equazioni (0 parametri)"
        else:
            if model is None:
                raise ValueError(f"Modello per '{name}' è None, ma è richiesto per la tabella")
            params = model_param_count(model)
            size_kb = model_size_kb(model)
            note = ""

        display_name = {
            "conv1d": "Ultra-CAN (Conv1D+Att. 1D)",
            "qkv": "Ultra-CAN (QKV Attention)",
            "lstm": "LSTM (LSTM-OFDM-DCSK)",
            "mc_dlsk": "MC-DLCSK (BiLSTM)",
            "classical": "Ricevitori classici",
        }.get(name, name)

        rows.append({
            "name": display_name,
            "params": params,
            "size_kb": size_kb,
            "note": note,
        })

    latex_content = _generate_latex_table(rows)

    try:
        output_path.write_text(latex_content, encoding="utf-8")
    except OSError as e:
        logger.error("Errore di scrittura del file %s: %s", output_path, e)
        raise

    logger.info("Tabella pesi salvata in %s", output_path)
    return output_path

def _try_load_model_from_checkpoint(experiment_name: str, arch: str) -> Optional[tf.keras.Model]:
    
    ckpt_candidates = [
        _REPO_ROOT / "results" / experiment_name / arch / "models" / "best_model.keras",
        _REPO_ROOT / "results" / experiment_name / arch / "models" / "best_model.h5",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        logger.debug(
            "Checkpoint non trovato per '%s' (cercati: %s)",
            arch, [str(p) for p in ckpt_candidates],
        )
        return None

    logger.info("Caricamento modello da checkpoint: %s", ckpt_path)
    try:
        model = load_model(ckpt_path)
        logger.debug("Modello caricato con successo da %s", ckpt_path)
        return model
    except Exception as e:
        logger.warning("Caricamento del checkpoint %s fallito: %s", ckpt_path, e)
        return None

def _validate_main_config(config: Dict[str, Any]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")

    general = config.get("general")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict")
    if "experiment_name" not in general:
        raise ValueError("chiave mancante: general.experiment_name")

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("sezione 'model' mancante o non dict")

    baselines_cfg = config.get("baselines")
    if not isinstance(baselines_cfg, dict):
        raise ValueError("sezione 'baselines' mancante o non dict")

    exp5_cfg = config.get("experiments", {}).get("final_report")
    if not isinstance(exp5_cfg, dict):
        raise ValueError("sezione 'experiments.final_report' mancante o non dict")
    if "output" not in exp5_cfg:
        raise ValueError("chiave mancante: experiments.final_report.output")

    logger.debug("Config validata per weight_comparison")

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description="Genera tabella comparativa dei pesi e footprint (Sez. V-D, Exp 5)"
    )
    parser.add_argument("--config", required=True, help="path della config esperimento (YAML)")
    parser.add_argument(
        "--output",
        default="results/full_experiment/final_report/weight_table.tex",
        help="percorso del file .tex di output (default: results/full_experiment/final_report/weight_table.tex)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"File di config non trovato: {config_path}")

    config = load_config(config_path, DEFAULT_BASE_CONFIG_PATH)
    _validate_main_config(config)

    general = config.get("general", {})
    experiment_name = str(general.get("experiment_name", "weight_comparison"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    log_file = setup_logging(
        log_dir=log_dir,
        level=str(general.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    logger.info("Log file: %s", log_file)

    models: Dict[str, Optional[tf.keras.Model]] = {}

    arch_builders = {
        "conv1d": build_dual_head_ultra_can,
        "qkv": build_dual_head_ultra_can_qkv,
        "lstm": lambda cfg: build_baseline(cfg, "lstm"),
        "mc_dlsk": lambda cfg: build_baseline(cfg, "mc_dlsk"),
    }

    for arch, builder in arch_builders.items():
        model = _try_load_model_from_checkpoint(experiment_name, arch)
        if model is None:
            logger.info("Costruzione del modello %s da zero (nessun checkpoint trovato)", arch)
            model = builder(config)
        else:
            logger.info("Modello %s caricato da checkpoint", arch)
        models[arch] = model

    models["classical"] = None

    output_path = Path(args.output)
    exp5_cfg = config.get("experiments", {}).get("final_report", {})
    include_classical = exp5_cfg.get("include_classical", True)

    try:
        generate_weight_table(models, output_path, include_classical=include_classical)
    except Exception as e:
        logger.error("Errore durante la generazione della tabella: %s", e)
        raise

    logger.info("Tabella pesi generata con successo in %s", output_path)

if __name__ == "__main__":
    main()

