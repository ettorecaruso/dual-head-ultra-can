"""
Test dedicato: la Sensing Head IMPARA (fix del problema "sensing collassato").

Contesto (fix, ``src/models/heads.py`` + ``configs/base_config.yaml``):
  Prima del fix la loss di sensing restava ferma a 1/12 (varianza dei target
  normalizzati) e ``corr(tau_hat, tau) ~ 0.04``: la testa prediceva la media.
  Diagnosi (verificata empiricamente): con i parametri del paper l'eco e' ~25
  dB sotto il path diretto e la sola feature posizionale non e' sufficiente —
  il ritardo tau NON e' recuperabile dalle feature del solo GAP/MaxPool.

  Fix definitivo (profilo DCA+MF): la sensing head riceve anche il profilo di
  correlazione matched-filter con cancellazione del path diretto, calcolato
  sulla sequenza di riferimento nota al ricevitore ISAC (ultimo canale
  dell'input, zero-parametro). Verificato: P(picco del profilo == tau) = 100%
  a SNR 20 dB e ``corr(tau) > 0.5`` end-to-end con ``lambda_mse=2``.

Questo test verifica il comportamento END-TO-END sul modello reale (conv1d,
feature real, alpha reali di ber_vs_snr): la sensing loss deve scendere sotto il
floor 1/12 e la correlazione tra tau predetto e vero (su VALIDATION) deve
superare la soglia di "testa che impara".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CORR_MIN = 0.10
_SENSING_FLOOR = 1.0 / 12.0

@pytest.fixture(scope="module")
def real_config() -> Dict[str, Any]:
    """Config reale di ber_vs_snr (feature real, alpha reali), dataset ridotto per CPU."""
    from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

    cfg = load_config(_REPO_ROOT / "configs" / "experiments.yaml", DEFAULT_BASE_CONFIG_PATH)
    cfg["general"]["seed"] = 42
    cfg["data"].update({
        "num_symbols_train": 1680,
        "num_symbols_val": 120,
        "num_symbols_test": 120,
        "snr_range": [-5, 20],
        "snr_step": 5,
        "echoes": [1, 3],
        "max_delay": 33,
        "max_doppler": 8e-5,
        "alpha_min": 0.05,
        "alpha_max": 0.3,
        "feature_mode": "real",
    })
    cfg["training"].update({"epochs": 10, "batch_size": 64, "early_stopping_patience": 0})
    cfg["model"]["backbone_type"] = "conv1d"
    cfg["evaluation"]["snr_test_range"] = [-5, 20]
    return cfg

def _build_data(cfg: Dict[str, Any], tmp_path: Path):
    
    from src.data.data_loader import build_tf_dataset, load_npz_files, verify_snr_balance
    from src.data.dataset_generator import build_snr_grid, generate_dataset

    data_dir = tmp_path / "data"
    for split in ("train", "val", "test"):
        generate_dataset(cfg, split, data_dir)
    snr_grid = build_snr_grid(cfg["data"]["snr_range"], cfg["data"]["snr_step"])
    echoes = list(cfg["data"]["echoes"])
    data = load_npz_files(data_dir, snr_grid, echoes, "train", cfg)
    verify_snr_balance(data, snr_grid, echoes)
    val = load_npz_files(data_dir, snr_grid, echoes, "val", cfg)
    verify_snr_balance(val, snr_grid, echoes)
    train_ds = build_tf_dataset(data, batch_size=64, config=cfg, shuffle=True, seed=42)
    val_ds = build_tf_dataset(val, batch_size=64, config=cfg, shuffle=False, seed=42)
    return data, train_ds, val_ds

def test_sensing_head_learns(
    real_config: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La sensing loss scende sotto 1/12 e corr(tau) supera il collasso."""
    from src.models.ultra_can import build_dual_head_ultra_can
    from src.training.trainer import Trainer

    tf.random.set_seed(42)
    np.random.seed(42)

    real_config["training"]["checkpoint_path"] = str(tmp_path / "ckpt.h5")

    data, train_ds, val_ds = _build_data(real_config, tmp_path)
    real_config["general"]["run_output_dir"] = str(tmp_path)

    model = build_dual_head_ultra_can(real_config)
    trainer = Trainer(real_config, model, train_ds, val_ds)
    history = trainer.train()

    sensing_loss = history.history.get("sensing_loss", [])
    assert sensing_loss, "history senza sensing_loss"
    assert sensing_loss[-1] < _SENSING_FLOOR - 1e-3, (
        f"sensing_loss finale {sensing_loss[-1]:.4f} non sotto il floor "
        f"{_SENSING_FLOOR:.4f}: sensing head ancora collassata"
    )
    assert sensing_loss[-1] < sensing_loss[0] - 1e-4, (
        f"sensing_loss {sensing_loss[0]:.4f} -> {sensing_loss[-1]:.4f}: "
        "nessuna discesa (sensing head ancora collassata)"
    )

    from src.data.data_loader import (
        _build_feature_matrix,
        build_reference_matrix,
        load_npz_files,
    )
    from src.data.dataset_generator import build_snr_grid

    val = load_npz_files(
        tmp_path / "data",
        build_snr_grid(real_config["data"]["snr_range"], real_config["data"]["snr_step"]),
        list(real_config["data"]["echoes"]),
        "val",
        real_config,
    )
    n_eval = int(val["x"].shape[0])
    ref = build_reference_matrix(val["bit"][:n_eval], val["seed"][:n_eval], real_config)
    x_feat = _build_feature_matrix(
        val["x"][:n_eval], str(real_config["data"]["feature_mode"]), reference=ref
    )
    tau = val["tau"][:n_eval]
    out = model(x_feat, training=False)
    tau_hat = out["sensing"].numpy()[:, 0] * float(real_config["data"]["max_delay"])
    corr_tau = float(np.corrcoef(tau_hat, tau)[0, 1])
    assert corr_tau > _CORR_MIN, (
        f"corr(tau)={corr_tau:.3f} sotto la soglia {_CORR_MIN} "
        "(la testa sensing non sta imparando, predice ~la media)"
    )

def test_sensing_features_enabled_by_default(real_config: Dict[str, Any]) -> None:
    """La config abilita le feature posizionali, il profilo DCA+MF e lambda=2."""
    assert real_config["model"]["sensing_head"]["use_position_feature"] is True
    assert real_config["model"]["sensing_head"]["use_reference_profile"] is True
    assert real_config["model"]["sensing_head"]["output_activation"] == "linear"
    assert float(real_config["training"]["lambda_mse"]) == pytest.approx(2.0)
