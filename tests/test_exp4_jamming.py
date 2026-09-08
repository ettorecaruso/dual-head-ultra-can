"""
Test dell'Esperimento 4 (jamming) — pipeline fixata.

Verificano:
  1. la griglia JSR deriva dalla config (``jsr_range``/``jsr_step``), non dal
     default -10..20 (fix del run che partiva fuori dal range del paper);
  2. a basso JSR il jammer è una piccola perturbazione: la curva BER "jamata"
     si sovrappone bene alla baseline per conv1d e qkv;
  3. il BER cresce (non cala) all'aumentare del JSR (robustezza monotona).

Esecuzione:
  python -m pytest tests/test_jamming.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

def _tiny_exp4_config(tmp_path: Path) -> Dict[str, Any]:
    """Config minima per jamming (dataset minuscolo per CPU)."""
    from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

    config = load_config(_REPO_ROOT / "configs" / "experiments.yaml", DEFAULT_BASE_CONFIG_PATH)
    config["general"]["experiment_name"] = "test_jamming"
    config["general"]["seed"] = 42
    config["data"].update({
        "num_symbols_train": 600,
        "num_symbols_val": 12,
        "num_symbols_test": 60,
        "snr_range": [0, 10],
        "snr_step": 5,
        "echoes": [0, 1],
        "max_delay": 5,
    })
    config["data"]["raw_dir"] = str(tmp_path / "data")
    config["data"]["processed_dir"] = str(tmp_path / "data_processed")
    config["evaluation"]["snr_test_range"] = [0, 10]
    config["training"].update({"epochs": 30, "batch_size": 16})
    config["training"]["checkpoint_path"] = str(tmp_path / "ckpt.h5")
    config["jamming"] = {"jamming_types": ["cw", "barrage", "partial_band"],
                         "jsr_range": [0, 10], "jsr_step": 2}
    return config

def _train_and_evaluate(config: Dict[str, Any], model_type: str) -> float:
    
    from src.experiments import pipeline
    from src.evaluation.evaluator import compute_ber_curve, evaluate_model as eval_model

    data_dir = pipeline.prepare_dataset(config, no_regen=False)
    train_data, train_ds, val_ds, test_data = pipeline.load_datasets(config, data_dir)
    model = pipeline.build_model(config, model_type)
    pipeline.train_and_evaluate(config, model, train_ds, val_ds, test_data, config["general"]["run_output_dir"])
    results = eval_model(model, test_data, config)
    return float(np.mean(results["ber"]))

def test_jsr_grid_from_config() -> None:
    """La griglia JSR deriva da jsr_range/jsr_step (fix chiave jsr_values)."""
    from src.experiments.pipeline import _build_jsr_values

    assert _build_jsr_values({"jsr_range": [0, 10], "jsr_step": 2}) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert _build_jsr_values({"jsr_range": [0, 10], "jsr_step": 5}) == [0.0, 5.0, 10.0]
    assert _build_jsr_values({"jsr_values": [-5, 5], "jsr_range": [0, 10]}) == [-5.0, 5.0]
    assert _build_jsr_values({}) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    with pytest.raises(ValueError, match="jsr_step"):
        _build_jsr_values({"jsr_range": [0, 10], "jsr_step": 0})

@pytest.mark.parametrize("model_type", ["conv1d", "qkv"])
def test_jamming_low_jsr_overlaps_baseline(tmp_path: Path, model_type: str) -> None:
    
    from src.experiments import pipeline
    from src.experiments.run_jamming import apply_jamming
    from src.evaluation.evaluator import compute_ber_curve, evaluate_model as eval_model

    config = _tiny_exp4_config(tmp_path)
    config["general"]["run_output_dir"] = str(tmp_path / model_type)

    data_dir = pipeline.prepare_dataset(config, no_regen=False)
    train_data, train_ds, val_ds, test_data = pipeline.load_datasets(config, data_dir)
    model = pipeline.build_model(config, model_type)
    pipeline.train_and_evaluate(config, model, train_ds, val_ds, test_data, tmp_path / model_type)

    def ber_at_jsr(jsr_db: float) -> float:
        rng = np.random.default_rng(42)
        jammed = apply_jamming(test_data["x"], "cw", jsr_db, rng)
        data_j = dict(test_data)
        data_j["x"] = jammed
        return float(np.mean(eval_model(model, data_j, config)["ber"]))

    ber_base = float(np.mean(eval_model(model, test_data, config)["ber"]))
    ber_low = ber_at_jsr(-20.0)
    ber_high = ber_at_jsr(10.0)

    assert abs(ber_low - ber_base) <= 0.05, (
        f"{model_type}: basso JSR troppo invasivo: baseline={ber_base:.3f} jammed={ber_low:.3f}"
    )
    assert ber_high >= ber_low - 0.02, (
        f"{model_type}: BER non cresce col JSR: {ber_low:.3f} -> {ber_high:.3f}"
    )
