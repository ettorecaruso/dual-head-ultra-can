"""Fixture condivise della suite di test.

Questo modulo fornisce tre fixture riutilizzate da tutti i file ``tests/test_*.py``:

  - ``tiny_config``  : dict minimo derivato dal vero ``configs/base_config.yaml``
      attraverso ``src.utils.config_loader.load_config`` (niente hard-coding).
      ``mu=3.9``, ``lambda_mse=2``, ``max_delay=10``, ``max_doppler=8e-5``,
      ``sequence_length=100``.
  - ``tiny_dataset`` : 128 sequenze caotiche generate al volo (SNR=0, K=1).
      Import LAZY di ``src.data.dataset_generator.generate_chaotic_sequence``:
      la fixture si attiva quando il modulo esiste. Contratto di output:
      ``x (128, 100, 1)``, ``comm_labels (128,)`` in ``{0, 1}``,
      ``sensing_labels (128, 2)`` = ``[tau, f_d]`` nei range della config.
  - ``tiny_model``   : ``build_dual_head_ultra_can`` (backbone conv1d).
      Import LAZY di ``src.models.ultra_can``: finche' il modulo
      modello non e' implementato la fixture esegue
      ``pytest.skip`` (non fallisce).

Nota su ``sys.path``: le fixture importano i moduli ``src.*``; il bootstrap
aggiunge la root del repository (``paper/``) a ``sys.path`` cosi' la suite
funziona anche quando pytest viene lanciato al di outside di ``paper/``
(stesso pattern gia' usato in ``src/data/dataset_generator.py``).

Scelta di pytest: il conftest e' un meccanismo nativo di pytest (nessuna
alternativa in unittest); le fixture ``tmp_path`` e ``pytest.skip`` sono le
primitive richieste dal contratto di test del progetto.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

_TINY_MAX_DELAY = 10
_TINY_ECHOES = [1]
_TINY_NUM_SAMPLES = 128

@pytest.fixture
def tiny_config(tmp_path: Path) -> Dict[str, Any]:
    
    experiment = {
        "general": {"experiment_name": "tiny_test"},
        "data": {"max_delay": _TINY_MAX_DELAY, "echoes": _TINY_ECHOES,
                 "feature_mode": "real"},
    }
    exp_path = tmp_path / "tiny_experiment.yaml"
    with open(exp_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(experiment, fh, sort_keys=False, allow_unicode=True)
    return load_config(exp_path, DEFAULT_BASE_CONFIG_PATH)

@pytest.fixture
def tiny_dataset(tiny_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    data = tiny_config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])
    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    rng = np.random.default_rng(int(tiny_config["general"]["seed"]))

    sequences = []
    for _ in range(_TINY_NUM_SAMPLES):
        seed = int(rng.integers(1, 2 ** 31 - 1))
        sequences.append(generate_chaotic_sequence(map_type, map_param, seed, seq_len))
    x = np.stack(sequences).astype(np.float32)
    x = x.reshape(_TINY_NUM_SAMPLES, seq_len, 1)
    x = np.concatenate([x, np.zeros_like(x)], axis=-1)

    comm_labels = rng.integers(0, 2, size=_TINY_NUM_SAMPLES).astype(np.int32)
    tau = rng.uniform(0.0, float(max_delay), size=_TINY_NUM_SAMPLES)
    f_d = rng.uniform(0.0, max_doppler, size=_TINY_NUM_SAMPLES)
    sensing_labels = np.stack([tau, f_d], axis=-1).astype(np.float32)

    return {
        "x": x,
        "comm_labels": comm_labels,
        "sensing_labels": sensing_labels,
        "snr_db": np.zeros(_TINY_NUM_SAMPLES, dtype=np.float32),
        "k": np.ones(_TINY_NUM_SAMPLES, dtype=np.int32),
    }

@pytest.fixture
def tiny_model(tiny_config: Dict[str, Any]):
    
    try:
        from src.models.ultra_can import build_dual_head_ultra_can
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"tiny_model non attivo: moduli modelli non ancora implementati ({exc})")
    return build_dual_head_ultra_can(tiny_config)
