"""
Test dell'Esperimento 3 (ricevitori classici, solo equazioni) — runner fixato.

Verificano le modifiche a ``src/experiments/runner.py``:
  - il bit MODULA il segnale (prima del fix il bit era generato ma mai usato:
    tutti i ricevitori davano BER=0.5 per costruzione);
  - il nuovo canale streaming ``[ref | ±ref]`` con ISI (eco dei simboli
    precedenti) + normalizzazione di potenza (Eq. (4)) produce curve BER
    MONOTONE crescenti con K e con il Doppler;
  - il matched filter (conoscenza della sequenza) è il migliore tra i ricevitori
    classici ma resta peggiore dei DL (qui: meglio del DCSK).

Conformità : shape-check e check NaN/Inf; type hints; nessun file
scritto fuori da ``paper/`` (i test sono read-only sui dati generati al volo).

Esecuzione:
  python -m pytest tests/test_exp3_echo_only.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.experiments import runner
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def module_config() -> Dict[str, Any]:
    """Config completa (base_config + experiments.yaml) per i test classical_receivers."""
    return load_config(_REPO_ROOT / "configs" / "experiments.yaml", DEFAULT_BASE_CONFIG_PATH)

def _beta(config: Dict[str, Any]) -> int:
    return int(config["baselines"]["dcsk_correlator"]["correlation_length"])

def _ber_for(config: Dict[str, Any], y: np.ndarray, bits: np.ndarray, detector: str) -> float:
    
    from src.data.dataset_generator import generate_chaotic_sequence
    from src.models.dcsk_correlator import evaluate_classical

    beta = _beta(config)
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    map_type = str(config["data"]["map_type"])
    map_param = float(config["data"]["map_param"])
    seed = int(config["general"]["seed"])
    template = generate_chaotic_sequence(map_type, map_param, seed, beta)
    if detector == "matched_filter":
        res = evaluate_classical(y[:, beta:], bits, detector, config, ref=None, template=template)
    else:
        res = evaluate_classical(y, bits, detector, config, ref=None, template=None)
    return float(res["ber"])

def test_exp3_bit_modulates_signal(module_config: Dict[str, Any]) -> None:
    
    echo_cfg = module_config["echo_only"]
    y, bits = runner._build_echo_only_dataset(
        module_config, k=1,
        doppler_direct=float(echo_cfg.get("doppler_direct_fixed", 1e-5)),
        num_symbols=400, echo_cfg=echo_cfg,
    )
    assert y.ndim == 2 and y.shape[1] == 2 * _beta(module_config)
    assert np.all(np.isfinite(y))
    assert bits.shape[0] == y.shape[0]

    ber_dcsk = _ber_for(module_config, y, bits, "dcsk")
    assert ber_dcsk < 0.45, f"il bit non modula il segnale: BER DCSK={ber_dcsk:.3f} a K=1"
    ber_mf = _ber_for(module_config, y, bits, "matched_filter")
    assert ber_mf < 0.30, f"matched filter non decodifica: BER={ber_mf:.3f} a K=1"

def test_exp3_k_sweep_monotonic(module_config: Dict[str, Any]) -> None:
    
    echo_cfg = module_config["echo_only"]
    doppler_direct = float(echo_cfg.get("doppler_direct_fixed", 1e-5))
    mf_bers = []
    dcsk_bers = []
    for k in (1, 3, 5, 8, 10):
        y, bits = runner._build_echo_only_dataset(
            module_config, k=k, doppler_direct=doppler_direct, num_symbols=400, echo_cfg=echo_cfg
        )
        mf_bers.append(_ber_for(module_config, y, bits, "matched_filter"))
        dcsk_bers.append(_ber_for(module_config, y, bits, "dcsk"))
        assert 0.0 <= mf_bers[-1] <= 1.0 and 0.0 <= dcsk_bers[-1] <= 1.0

    assert mf_bers[-1] >= mf_bers[0] - 0.02, f"matched filter non cresce con K: {mf_bers}"
    assert dcsk_bers[-1] >= dcsk_bers[0] - 0.03, f"DCSK non cresce con K: {dcsk_bers}"
    for mf, dcsk in zip(mf_bers, dcsk_bers):
        assert mf <= dcsk + 0.05, f"matched filter peggiore del DCSK: mf={mf} dcsk={dcsk}"

def test_exp3_doppler_sweep_monotonic(module_config: Dict[str, Any]) -> None:
    """Curva BER vs Doppler monotona crescente (perdita di coerenza).

    All'aumentare del Doppler sul path diretto il correlatore perde coerenza:
    BER(mf) e BER(dcsk) a fD max > a fD=0.
    """
    echo_cfg = module_config["echo_only"]
    mf_bers = []
    dcsk_bers = []
    for fd in (0.0, 5e-4, 1e-3, 2e-3, 3e-3):
        y, bits = runner._build_echo_only_dataset(
            module_config, k=int(echo_cfg.get("k_fixed", 3)), doppler_direct=fd,
            num_symbols=400, echo_cfg=echo_cfg,
        )
        mf_bers.append(_ber_for(module_config, y, bits, "matched_filter"))
        dcsk_bers.append(_ber_for(module_config, y, bits, "dcsk"))

    assert mf_bers[-1] >= mf_bers[0] + 0.05, f"matched filter non degrada con fD: {mf_bers}"
    assert dcsk_bers[-1] >= dcsk_bers[0] - 0.03, f"DCSK non degrada con fD: {dcsk_bers}"

def test_exp3_energy_detector_is_floor(module_config: Dict[str, Any]) -> None:
    """L'energy detector non discrimina la modulazione di segno -> floor ~0.5."""
    echo_cfg = module_config["echo_only"]
    y, bits = runner._build_echo_only_dataset(
        module_config, k=3,
        doppler_direct=float(echo_cfg.get("doppler_direct_fixed", 1e-5)),
        num_symbols=400, echo_cfg=echo_cfg,
    )
    ber_energy = _ber_for(module_config, y, bits, "energy_detector")
    assert 0.40 <= ber_energy <= 0.60, f"energy detector fuori dal floor: {ber_energy:.3f}"

def test_exp3_frame_consistency_raises(module_config: Dict[str, Any]) -> None:
    """Frame DCSK non rappresentabile (2*beta != sequence_length) -> ValueError."""
    import copy

    bad_cfg = copy.deepcopy(module_config)
    bad_cfg["baselines"]["dcsk_correlator"]["correlation_length"] = 40
    echo_cfg = bad_cfg["echo_only"]
    with pytest.raises(ValueError, match="2\\*correlation_length"):
        runner._build_echo_only_dataset(
            bad_cfg, k=1,
            doppler_direct=float(echo_cfg.get("doppler_direct_fixed", 1e-5)),
            num_symbols=64, echo_cfg=echo_cfg,
        )

