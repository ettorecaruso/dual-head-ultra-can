"""
Test per ``src/data/dataset_generator.py``: canale aereo (test_channel).

Questi test verificano:
  T1  test_channel_k10_echoes           K=10 (eco massimo): uscita finita,
                                        shape (100,), normalizzazione Eq. (4)
  T2  test_channel_k0_no_echoes         K=0: solo AWGN, output finito,
                                        label sensing (0, 0) ()
  T3  test_channel_extreme_delay_raises tau > sequence_length -> ValueError
                                        (ritardo fuori finestra di osservazione)
  T4  test_channel_snr_minus20          SNR=-20 dB: finito, rumore dominante
  T5  test_channel_snr_plus30           SNR=+30 dB: finito, segnale dominante
  T6  test_doppler_range                fD,k in [0, 8e-5] (Tab. I, Sez. III-B)
  T7  test_labels_ranges                tau in [0, max_delay]; alpha in
                                        [1e-6, 1e-2] (Tab. I, Sez. III-B)
  T8  test_power_normalization          E[|h_c|^2] + Somma alpha_k^2 = 1
                                        (Eq. (4), +-1e-6)

Oltre agli 8 test obbligatori, per rispettare la regola "almeno 3 test per
funzione" (input validi / limite / invalidi) il file copre anche:
  - ``apply_aerial_channel``   : shape preservata, riproducibilita' per seed,
    conservazione energia ad alto SNR, lunghezze variabili (N=2..512), SNR
    estremi (-40 dB documentato), tau == N (boundary) e 7 gruppi di input
    invalidi (x, h_c, f_dc, snr_db, echoes, tau, potenza echi);
  - ``sample_echo_parameters`` : K=1/K=10/K=0, K == max_delay, K > max_delay,
    k invalidi (negativi, float, bool), config incoerente;
  - ``sample_direct_path``     : range di f_Dc, potenza unitaria statistica
    E[|h_c|^2] = 1, K=0 dB (Rayleigh) e K=30 dB (LOS dominante), kappa
    invalidi e chiave mancante;
  - ``_dominant_echo_index``   : eco dominante = alpha massimo (),
    lista singola e lista vuota (ValueError);
  - ``EchoParams``/``DirectPathParams``: dataclass frozen (immutabilita');
  - proprieta' caotiche del segnale nel canale (r=3.8 non convergente,
    r=4.0 e r=3.57 a SNR estremi, output non degenere);
  - test di integrazione: import del modulo, pipeline end-to-end
    (mappa -> echi -> path diretto -> canale) e ``generate_dataset``
    (batch shape (8, 100), label (0,0) per K=0).

Scelta di pytest: coerente con la suite esistente (``tests/test_chaotic_maps.py``
usa pytest con fixtures e parametrize) e con il comando dal log: ``python -m pytest tests/test_channel.py -v``.

Conformita' : shape-check, range-check e check NaN/Inf su ogni
output; niente pipe nei comandi shell; type hints e docstring su tutte le
funzioni pubbliche di questo modulo di test.

TODO (test mancanti, segnalati esplicitamente):
  - ``apply_aerial_channel`` NON impone un limite inferiore a ``snr_db``
    (accetta qualsiasi valore finito): il template utente suggeriva
    "SNR < -30 -> ValueError", NON implementato nella sorgente. Testato qui
    solo il comportamento reale (SNR=-40 dB: finito). Se il paper richiede
    una soglia minima, va aggiunta la validazione nel sorgente (ALERT).
  - ``sample_direct_path`` NON valida ``doppler_direct_max`` nel proprio corpo
    (range controllato solo da ``_validate_config``): test mancante per
    ``doppler_direct_max`` negativo/non finito a livello di funzione.
  - ``FileNotFoundError`` per file di input mancante: NON applicabile al
    generatore (il canale non legge file); spetta a ``tests/test_data``
    (data_loader).
  - ``generate_dataset``/``main`` con dimensioni zero/negative (seq_len,
    num_symbols) -> ``tests/test_input_validation.py``;
    smoke end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from src.data.dataset_generator import (
    DirectPathParams,
    EchoParams,
    _dominant_echo_index,
    apply_aerial_channel,
    generate_chaotic_sequence,
    generate_dataset,
    generate_test_batch,
    generate_transmitted_batch,
    sample_direct_path,
    sample_echo_parameters,
)

_MU_PAPER = 3.9
_MU_NON_CONV = 3.8
_MU_MIN = 3.57
_MU_MAX = 4.0
_N_SEQ = 100
_SEED = 42
_MAX_DELAY = 33
_MAX_DOPPLER = 8e-5
_ALPHA_MIN = 1e-6
_ALPHA_MAX = 0.01
_DOPPLER_DIRECT_MAX = 1e-5

def _tiny_config(data_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
    data: Dict[str, Any] = {
        "sequence_length": _N_SEQ,
        "map_type": "logistic",
        "map_param": _MU_PAPER,
        "fc_hz": 2.4e9,
        "fs_hz": 1.0e7,
        "snr_range": [-5.0, 20.0],
        "snr_step": 2.0,
        "echoes": [1, 3],
        "max_delay": _MAX_DELAY,
        "max_doppler": _MAX_DOPPLER,
        "alpha_min": _ALPHA_MIN,
        "alpha_max": _ALPHA_MAX,
        "num_symbols_train": 5040,
        "num_symbols_val": 1000,
        "num_symbols_test": 2000,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "rician_kappa_db": 10.0,
        "doppler_direct_max": _DOPPLER_DIRECT_MAX,
    }
    if data_overrides:
        data.update(data_overrides)
    return {"general": {"seed": _SEED}, "data": data}

def _unit_direct_path() -> DirectPathParams:
    
    return DirectPathParams(h_c=complex(1.0, 0.0), f_dc=0.0)

def test_channel_k10_echoes() -> None:
    """T1: K=10 (eco massimo) -> uscita finita, shape (100,), Eq. (4) rispettata."""
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(10, rng, cfg)
    assert len(echoes) == 10
    echo_power = float(sum(e.alpha ** 2 for e in echoes))
    assert echo_power < 1.0 - 1e-9

    direct = sample_direct_path(rng, cfg)
    y, noise_var = apply_aerial_channel(x, direct.h_c, direct.f_dc, echoes, 10.0, rng)

    assert y.shape == (_N_SEQ,)
    assert y.dtype == np.complex128
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var)
    assert noise_var > 0.0
    assert float(np.mean(np.abs(y) ** 2)) > 0.0
    assert float(np.max(np.abs(y)) - np.min(np.abs(y))) > 0.0

def test_channel_k0_no_echoes(tmp_path: Path) -> None:
    """T2: K=0 -> solo path diretto + AWGN, output finito, label sensing (0, 0)."""
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)

    echoes = sample_echo_parameters(0, rng, cfg)
    assert echoes == []

    y, noise_var = apply_aerial_channel(x, complex(1.0, 0.0), 0.0, [], 10.0, rng)
    assert y.shape == (_N_SEQ,)
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0

    cfg["data"]["echoes"] = [0]
    cfg["data"]["snr_range"] = [0.0, 2.0]
    cfg["data"]["snr_step"] = 2.0
    cfg["data"]["num_symbols_test"] = 4
    paths = generate_dataset(cfg, "test", tmp_path)
    data = np.load(paths[0])
    assert np.all(data["tau"] == 0.0)
    assert np.all(data["f_d"] == 0.0)
    assert np.all(np.isfinite(data["x"]))
    assert data["x"].shape == (4, _N_SEQ)

def test_channel_extreme_delay_raises() -> None:
    """T3: tau > sequence_length -> ValueError (ritardo fuori finestra)."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echo_out_of_window = EchoParams(tau=_N_SEQ + 1, f_doppler=1e-5, alpha=1e-3)
    with pytest.raises(ValueError, match="finestra"):
        apply_aerial_channel(
            x, complex(1.0, 0.0), 0.0, [echo_out_of_window], 10.0,
            np.random.default_rng(0),
        )

    echo_at_window = EchoParams(tau=_N_SEQ, f_doppler=1e-5, alpha=1e-3)
    y, _ = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [echo_at_window], 10.0, np.random.default_rng(0)
    )
    assert y.shape == (_N_SEQ,)
    assert np.all(np.isfinite(y))

def test_channel_snr_minus20() -> None:
    
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, rng, cfg)

    y, noise_var = apply_aerial_channel(x, complex(1.0, 0.0), 0.0, echoes, -20.0, rng)

    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0
    received = float(np.mean(np.abs(y) ** 2))
    assert abs(received - noise_var) / noise_var < 0.35
    assert received > 0.0

def test_channel_snr_plus30() -> None:
    
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, rng, cfg)

    y, noise_var = apply_aerial_channel(x, complex(1.0, 0.0), 0.0, echoes, 30.0, rng)

    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0
    received = float(np.mean(np.abs(y) ** 2))
    assert received > 100.0 * noise_var
    assert received > 0.0

def test_doppler_range() -> None:
    """T6: fD,k in [0, 8e-5] e f_Dc in [0, 1e-5] (Tab. I, Sez. III-B)."""
    cfg = _tiny_config()
    rng = np.random.default_rng(7)
    for k in (1, 3, 10):
        echoes = sample_echo_parameters(k, rng, cfg)
        for echo in echoes:
            assert 0.0 <= echo.f_doppler <= _MAX_DOPPLER
            assert echo.f_doppler < 0.5

    direct = sample_direct_path(rng, cfg)
    assert 0.0 <= direct.f_dc <= _DOPPLER_DIRECT_MAX
    assert direct.f_dc < 0.5

def test_labels_ranges() -> None:
    """T7: tau in [1, max_delay], alpha in [1e-6, 1e-2], fD in [0, 8e-5]."""
    cfg = _tiny_config()
    rng = np.random.default_rng(3)
    echoes = sample_echo_parameters(10, rng, cfg)
    for echo in echoes:
        assert isinstance(echo.tau, int)
        assert 1 <= echo.tau <= _MAX_DELAY
        assert _ALPHA_MIN <= echo.alpha <= _ALPHA_MAX
        assert 0.0 <= echo.f_doppler <= _MAX_DOPPLER

    dominant = echoes[_dominant_echo_index(echoes)]
    assert 1 <= dominant.tau <= _MAX_DELAY
    assert _ALPHA_MIN <= dominant.alpha <= _ALPHA_MAX
    assert 0.0 <= dominant.f_doppler <= _MAX_DOPPLER

def test_power_normalization() -> None:
    """T8: E[|h_c|^2] + Somma alpha_k^2 = 1 (Eq. (4), tolleranza +-1e-6)."""
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    echoes = sample_echo_parameters(10, rng, cfg)
    echo_power = float(sum(e.alpha ** 2 for e in echoes))
    assert echo_power < 1.0 - 1e-9

    h_c_eff_power = 1.0 * (1.0 - echo_power)
    assert h_c_eff_power + echo_power == pytest.approx(1.0, abs=1e-6)

    rng = np.random.default_rng(1234)
    powers = np.array(
        [abs(sample_direct_path(rng, cfg).h_c) ** 2 for _ in range(20000)]
    )
    assert float(np.mean(powers)) == pytest.approx(1.0, abs=5e-2)

def test_apply_channel_shape_and_dtype() -> None:
    """Caso normale: shape (N,) preservata e output complex128."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    y, noise_var = apply_aerial_channel(
        x, complex(0.8, 0.6), 2e-5, [], 10.0, np.random.default_rng(1)
    )
    assert y.shape == (_N_SEQ,)
    assert y.dtype == np.complex128
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0

def test_apply_channel_reproducible_seed() -> None:
    """Caso normale: stesso seed -> stesso output esatto (riproducibilita')."""
    cfg = _tiny_config()
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, np.random.default_rng(5), cfg)
    direct = _unit_direct_path()

    y1, nv1 = apply_aerial_channel(
        x, direct.h_c, direct.f_dc, echoes, 10.0, np.random.default_rng(3)
    )
    y2, nv2 = apply_aerial_channel(
        x, direct.h_c, direct.f_dc, echoes, 10.0, np.random.default_rng(3)
    )
    assert np.array_equal(y1, y2)
    assert nv1 == nv2

def test_apply_channel_energy_conservation_high_snr() -> None:
    """Conservazione energia: a SNR alto mean(|y|^2) ~= mean(|x|^2) (Eq. (3)/(4))."""
    cfg = _tiny_config()
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, np.random.default_rng(5), cfg)
    direct = _unit_direct_path()

    y, _ = apply_aerial_channel(
        x, direct.h_c, direct.f_dc, echoes, 30.0, np.random.default_rng(5)
    )
    ratio = float(np.mean(np.abs(y) ** 2) / np.mean(x ** 2))
    assert ratio == pytest.approx(1.0, rel=0.10)

@pytest.mark.parametrize("n", [2, 32, 100, 128, 512])
def test_apply_channel_variable_sequence_length(n: int) -> None:
    
    x = generate_chaotic_sequence("logistic", _MU_PAPER, 11, n)
    y, noise_var = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [], 10.0, np.random.default_rng(0)
    )
    assert y.shape == (n,)
    assert y.dtype == np.complex128
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0

def test_apply_channel_direct_only_doppler() -> None:
    """Caso normale: solo path diretto con Doppler (K=0) -> |y| ~= |h_c_eff*x|."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    direct = _unit_direct_path()
    y, _ = apply_aerial_channel(x, direct.h_c, 2e-5, [], 30.0, np.random.default_rng(2))
    ratio = float(np.mean(np.abs(y) ** 2) / np.mean(x ** 2))
    assert ratio == pytest.approx(1.0, rel=0.05)

def test_apply_channel_snr_extreme_low_minus40_finite() -> None:
    
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    y, noise_var = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [], -40.0, np.random.default_rng(0)
    )
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0
    assert noise_var > float(np.mean(x ** 2))

def test_apply_channel_tau_equal_length_allowed() -> None:
    """Caso limite: tau == N (boundary) ammesso, contributo eco nullo, finito."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echo = EchoParams(tau=_N_SEQ, f_doppler=1e-5, alpha=1e-3)
    y, _ = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [echo], 10.0, np.random.default_rng(0)
    )
    assert y.shape == (_N_SEQ,)
    assert np.all(np.isfinite(y))
    y_ref, _ = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [], 10.0, np.random.default_rng(0)
    )
    assert float(np.mean(np.abs(y) ** 2)) == pytest.approx(
        float(np.mean(np.abs(y_ref) ** 2)), rel=0.02
    )

@pytest.mark.parametrize(
    "bad_x",
    [
        np.zeros((2, _N_SEQ)),
        np.zeros(0),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_apply_channel_invalid_x(bad_x: np.ndarray) -> None:
    """Input invalidi: x non 1D/vuoto/NaN/Inf -> ValueError."""
    with pytest.raises(ValueError):
        apply_aerial_channel(bad_x, complex(1.0, 0.0), 0.0, [], 10.0, np.random.default_rng(0))

@pytest.mark.parametrize("bad_h_c", [complex(float("nan"), 0.0), complex(0.0, float("inf"))])
def test_apply_channel_invalid_h_c(bad_h_c: complex) -> None:
    """Input invalidi: h_c con NaN/Inf -> ValueError (messaggio su h_c)."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    with pytest.raises(ValueError, match="h_c"):
        apply_aerial_channel(x, bad_h_c, 0.0, [], 10.0, np.random.default_rng(0))

@pytest.mark.parametrize("bad_f_dc", [-0.1, 0.5, float("nan")])
def test_apply_channel_invalid_f_dc(bad_f_dc: float) -> None:
    """Input invalidi: f_dc fuori da [0, 0.5) o non finito -> ValueError."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    with pytest.raises(ValueError, match="f_dc"):
        apply_aerial_channel(x, complex(1.0, 0.0), bad_f_dc, [], 10.0, np.random.default_rng(0))

@pytest.mark.parametrize("bad_snr", [float("nan"), float("inf")])
def test_apply_channel_invalid_snr(bad_snr: float) -> None:
    """Input invalidi: snr_db non finito -> ValueError."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    with pytest.raises(ValueError, match="snr_db"):
        apply_aerial_channel(x, complex(1.0, 0.0), 0.0, [], bad_snr, np.random.default_rng(0))

def test_apply_channel_echoes_wrong_type() -> None:
    """Input invalidi: echoes non list -> TypeError; elemento non EchoParams -> TypeError."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    with pytest.raises(TypeError, match="echoes"):
        apply_aerial_channel(x, complex(1.0, 0.0), 0.0, (), 10.0, np.random.default_rng(0))
    with pytest.raises(TypeError, match="EchoParams"):
        apply_aerial_channel(
            x, complex(1.0, 0.0), 0.0, [("non", "eco")], 10.0, np.random.default_rng(0)
        )

def test_apply_channel_echo_power_too_high() -> None:
    """Input invalidi: Somma alpha_k^2 >= 1 -> ValueError (Eq. (4) impossibile)."""
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    strong_echoes = [
        EchoParams(tau=5, f_doppler=1e-5, alpha=0.8),
        EchoParams(tau=10, f_doppler=1e-5, alpha=0.7),
    ]
    with pytest.raises(ValueError, match=r"Eq\. \(4\)"):
        apply_aerial_channel(
            x, complex(1.0, 0.0), 0.0, strong_echoes, 10.0, np.random.default_rng(0)
        )

def test_sample_echoes_k1_normal() -> None:
    """Caso normale: K=1 -> un solo eco con campi nei range (Tab. I)."""
    cfg = _tiny_config()
    echoes = sample_echo_parameters(1, np.random.default_rng(1), cfg)
    assert len(echoes) == 1
    echo = echoes[0]
    assert isinstance(echo, EchoParams)
    assert isinstance(echo.tau, int)
    assert 1 <= echo.tau <= _MAX_DELAY
    assert 0.0 <= echo.f_doppler <= _MAX_DOPPLER
    assert _ALPHA_MIN <= echo.alpha <= _ALPHA_MAX

def test_sample_echoes_k10_ranges() -> None:
    """Caso normale: K=10 -> 10 echi, tau distinti, range di Tab. I rispettati."""
    cfg = _tiny_config()
    echoes = sample_echo_parameters(10, np.random.default_rng(3), cfg)
    assert len(echoes) == 10
    taus = [echo.tau for echo in echoes]
    assert len(set(taus)) == 10
    for echo in echoes:
        assert 1 <= echo.tau <= _MAX_DELAY
        assert 0.0 <= echo.f_doppler <= _MAX_DOPPLER
        assert _ALPHA_MIN <= echo.alpha <= _ALPHA_MAX

def test_sample_echoes_k0_empty() -> None:
    """Caso normale: K=0 -> lista vuota (, nessun eco)."""
    cfg = _tiny_config()
    assert sample_echo_parameters(0, np.random.default_rng(1), cfg) == []

def test_sample_echoes_k_max_delay_edge() -> None:
    """Caso limite: K == max_delay -> tutti i ritardi di [1, max_delay] usati."""
    cfg = _tiny_config()
    echoes = sample_echo_parameters(_MAX_DELAY, np.random.default_rng(2), cfg)
    assert len(echoes) == _MAX_DELAY
    assert sorted(e.tau for e in echoes) == list(range(1, _MAX_DELAY + 1))

def test_sample_echoes_k10_max_delay_10_edge() -> None:
    """Caso limite: K == max_delay=10 -> esattamente 10 ritardi in [1, 10]."""
    cfg = _tiny_config(data_overrides={"max_delay": 10})
    echoes = sample_echo_parameters(10, np.random.default_rng(2), cfg)
    assert len(echoes) == 10
    assert all(1 <= e.tau <= 10 for e in echoes)

@pytest.mark.parametrize("bad_k", [-1, 1.5, True, None, "3"])
def test_sample_echoes_invalid_k(bad_k: object) -> None:
    """Input invalidi: k non int >= 0 -> ValueError (messaggio su k)."""
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="k"):
        sample_echo_parameters(bad_k, np.random.default_rng(0), cfg)

def test_sample_echoes_k_exceeds_max_delay() -> None:
    """Input invalidi: k > max_delay -> ValueError (ritardi distinti impossibili)."""
    cfg = _tiny_config()
    with pytest.raises(ValueError, match="max_delay"):
        sample_echo_parameters(_MAX_DELAY + 1, np.random.default_rng(0), cfg)

def test_sample_echoes_config_inconsistent_max_delay() -> None:
    """Input invalidi: max_delay > sequence_length in config -> ValueError."""
    cfg = _tiny_config(data_overrides={"max_delay": _N_SEQ + 1})
    with pytest.raises(ValueError, match="max_delay"):
        sample_echo_parameters(1, np.random.default_rng(0), cfg)

def test_sample_direct_path_normal_ranges() -> None:
    """Caso normale: h_c finito, f_dc in [0, 1e-5], oggetto DirectPathParams."""
    cfg = _tiny_config()
    direct = sample_direct_path(np.random.default_rng(4), cfg)
    assert isinstance(direct, DirectPathParams)
    assert math.isfinite(direct.h_c.real) and math.isfinite(direct.h_c.imag)
    assert 0.0 <= direct.f_dc <= _DOPPLER_DIRECT_MAX
    assert direct.f_dc < 0.5

def test_sample_direct_path_unit_power_statistical() -> None:
    
    cfg = _tiny_config()
    rng = np.random.default_rng(1234)
    powers = np.array(
        [abs(sample_direct_path(rng, cfg).h_c) ** 2 for _ in range(20000)]
    )
    assert float(np.mean(powers)) == pytest.approx(1.0, abs=5e-2)
    assert np.all(np.isfinite(powers))

def test_sample_direct_path_kappa0_rayleigh_edge() -> None:
    
    cfg = _tiny_config(data_overrides={"rician_kappa_db": 0.0})
    rng = np.random.default_rng(6)
    powers = np.array(
        [abs(sample_direct_path(rng, cfg).h_c) ** 2 for _ in range(20000)]
    )
    assert float(np.mean(powers)) == pytest.approx(1.0, abs=5e-2)
    assert np.all(np.isfinite(powers))

def test_sample_direct_path_kappa30_los_edge() -> None:
    """Caso limite: K=30 dB (LOS dominante) -> |h_c| ~= 1 (componente diffusa trascurabile)."""
    cfg = _tiny_config(data_overrides={"rician_kappa_db": 30.0})
    rng = np.random.default_rng(5)
    for _ in range(20):
        h_c = sample_direct_path(rng, cfg).h_c
        assert abs(abs(h_c) - 1.0) < 0.1

@pytest.mark.parametrize("bad_kappa", [-1.0, float("nan"), float("inf")])
def test_sample_direct_path_invalid_kappa(bad_kappa: float) -> None:
    """Input invalidi: rician_kappa_db < 0 o non finito -> ValueError."""
    cfg = _tiny_config(data_overrides={"rician_kappa_db": bad_kappa})
    with pytest.raises(ValueError, match="rician_kappa_db"):
        sample_direct_path(np.random.default_rng(0), cfg)

def test_sample_direct_path_missing_key_raises() -> None:
    """Input invalidi: chiave rician_kappa_db mancante -> KeyError (fail-fast)."""
    cfg = _tiny_config()
    del cfg["data"]["rician_kappa_db"]
    with pytest.raises(KeyError, match="rician_kappa_db"):
        sample_direct_path(np.random.default_rng(0), cfg)

def test_dominant_echo_max_alpha() -> None:
    
    echoes = [
        EchoParams(tau=1, f_doppler=1e-5, alpha=1e-4),
        EchoParams(tau=2, f_doppler=2e-5, alpha=5e-3),
        EchoParams(tau=3, f_doppler=3e-5, alpha=1e-3),
    ]
    assert _dominant_echo_index(echoes) == 1

def test_dominant_echo_single_edge() -> None:
    """Caso limite: lista con un solo eco -> indice 0."""
    echoes = [EchoParams(tau=10, f_doppler=1e-5, alpha=1e-3)]
    assert _dominant_echo_index(echoes) == 0

def test_dominant_echo_empty_raises() -> None:
    """Input invalidi: lista vuota -> ValueError (nessun eco dominante)."""
    with pytest.raises(ValueError, match="echoes"):
        _dominant_echo_index([])

def test_echo_params_frozen() -> None:
    """Caso limite: EchoParams e' frozen -> assegnamento post-costruzione rifiutato."""
    echo = EchoParams(tau=1, f_doppler=1e-5, alpha=1e-3)
    with pytest.raises(FrozenInstanceError):
        echo.alpha = 0.5

def test_direct_path_params_frozen() -> None:
    """Caso limite: DirectPathParams e' frozen -> assegnamento post-costruzione rifiutato."""
    direct = DirectPathParams(h_c=complex(1.0, 0.0), f_dc=1e-5)
    with pytest.raises(FrozenInstanceError):
        direct.f_dc = 2e-5

def test_sample_echoes_returns_frozen_objects() -> None:
    
    cfg = _tiny_config()
    echoes = sample_echo_parameters(3, np.random.default_rng(9), cfg)
    assert all(isinstance(e, EchoParams) for e in echoes)
    with pytest.raises(FrozenInstanceError):
        echoes[0].tau = 99

def test_chaotic_logistic_r38_non_convergent() -> None:
    
    seq = generate_chaotic_sequence("logistic", _MU_NON_CONV, 7, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert np.all(np.isfinite(seq))
    assert np.all(seq > 0.0)
    assert np.all(seq < 1.0)
    spread = float(np.max(seq) - np.min(seq))
    assert spread > 0.3
    tail_spread = float(np.max(seq[-20:]) - np.min(seq[-20:]))
    assert tail_spread > 1e-3
    assert float(np.sum(seq ** 2)) > 0.0

@pytest.mark.parametrize("mu", [_MU_MIN, _MU_MAX])
@pytest.mark.parametrize("snr_db", [-20.0, 30.0])
def test_chaotic_map_limits_through_channel(mu: float, snr_db: float) -> None:
    """Stabilita' numerica: mappe limite (r=3.57, r=4.0) a SNR estremi -> no NaN/Inf."""
    x = generate_chaotic_sequence("logistic", mu, 13, _N_SEQ)
    y, noise_var = apply_aerial_channel(
        x, complex(1.0, 0.0), 1e-5, [], snr_db, np.random.default_rng(0)
    )
    assert y.shape == (_N_SEQ,)
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0

def test_chaotic_bernoulli_through_channel() -> None:
    
    x = generate_chaotic_sequence("bernoulli", 0.0, 17, _N_SEQ)
    y, noise_var = apply_aerial_channel(
        x, complex(1.0, 0.0), 0.0, [], 10.0, np.random.default_rng(0)
    )
    assert np.all(np.isfinite(y))
    assert math.isfinite(noise_var) and noise_var > 0.0

def test_channel_output_non_degenerate() -> None:
    
    cfg = _tiny_config()
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, np.random.default_rng(5), cfg)
    y, _ = apply_aerial_channel(
        x, complex(1.0, 0.0), 1e-5, echoes, 10.0, np.random.default_rng(5)
    )
    amplitude = np.abs(y)
    assert float(np.mean(amplitude ** 2)) > 0.0
    assert float(np.max(amplitude) - np.min(amplitude)) > 0.0

def test_integration_import_module() -> None:
    
    import src.data.dataset_generator as dg

    assert callable(dg.apply_aerial_channel)
    assert callable(dg.sample_echo_parameters)
    assert callable(dg.sample_direct_path)
    assert callable(dg.generate_dataset)

def test_integration_end_to_end_pipeline() -> None:
    
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    x = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    echoes = sample_echo_parameters(3, rng, cfg)
    direct = sample_direct_path(rng, cfg)
    y, noise_var = apply_aerial_channel(x, direct.h_c, direct.f_dc, echoes, 15.0, rng)

    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(noise_var))
    assert y.shape == (_N_SEQ,)
    dominant = echoes[_dominant_echo_index(echoes)]
    assert 1 <= dominant.tau <= _MAX_DELAY
    assert 0.0 <= dominant.f_doppler <= _MAX_DOPPLER

def test_integration_generate_dataset_batch_shape(tmp_path: Path) -> None:
    
    cfg = _tiny_config(
        data_overrides={
            "echoes": [1],
            "snr_range": [0.0, 2.0],
            "snr_step": 2.0,
            "num_symbols_test": 8,
        }
    )
    paths = generate_dataset(cfg, "test", tmp_path)
    assert len(paths) == 2

    for path in paths:
        data = np.load(path)
        x = data["x"]
        assert x.shape == (8, _N_SEQ)
        assert x.dtype == np.complex128
        assert np.all(np.isfinite(x))
        assert int(np.sum(data["bit"] == 0)) == 4
        assert int(np.sum(data["bit"] == 1)) == 4
        assert np.all(data["tau"] >= 1) and np.all(data["tau"] <= _MAX_DELAY)
        assert np.all(data["f_d"] >= 0.0) and np.all(data["f_d"] <= _MAX_DOPPLER)

def test_integration_main_cli_generates_npz(tmp_path: Path) -> None:
    
    import yaml

    from src.data.dataset_generator import main

    config_path = tmp_path / "tiny_test.yaml"
    config_path.write_text(
        "general:\n  seed: 42\n"
        "data:\n  echoes: [1]\n  snr_range: [0, 2]\n  snr_step: 2\n"
        "  num_symbols_test: 8\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    main(["--config", str(config_path), "--splits", "test", "--output-dir", str(out_dir)])

    npz_files = sorted(out_dir.glob("*.npz"))
    assert len(npz_files) == 2
    for path in npz_files:
        data = np.load(path)
        assert data["x"].shape == (8, _N_SEQ)
        assert data["x"].dtype == np.complex128
        assert np.all(np.isfinite(data["x"]))
        assert np.all(data["tau"] >= 1) and np.all(data["tau"] <= _MAX_DELAY)
        assert np.all(data["f_d"] >= 0.0) and np.all(data["f_d"] <= _MAX_DOPPLER)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["data"]["echoes"] == [1]



def test_generate_test_batch_shapes_and_ranges() -> None:
    
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    batch = generate_test_batch(cfg, 256, 10.0, 3, rng)

    assert set(batch.keys()) == {"x", "bit", "tau", "f_d", "seed", "x_ref"}
    assert batch["x"].shape == (256, _N_SEQ)
    assert batch["x"].dtype == np.complex128
    assert batch["x_ref"].shape == (256, _N_SEQ)
    assert batch["x_ref"].dtype == np.float32
    assert np.all(np.isfinite(batch["x"]))
    assert np.all(np.isin(batch["bit"], (0, 1)))
    assert np.all(batch["tau"] >= 1) and np.all(batch["tau"] <= _MAX_DELAY)
    assert np.all(batch["f_d"] >= 0.0) and np.all(batch["f_d"] <= _MAX_DOPPLER)
    x_ref = batch["x_ref"].astype(np.float64)
    assert np.allclose(x_ref.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(x_ref, axis=1), 1.0, atol=1e-4)


def test_generate_test_batch_reproducible() -> None:
    """BATCH: stesso seed -> stesso batch; seed diverso -> batch diverso."""
    cfg = _tiny_config()
    b1 = generate_test_batch(cfg, 128, 5.0, 1, np.random.default_rng(7))
    b2 = generate_test_batch(cfg, 128, 5.0, 1, np.random.default_rng(7))
    b3 = generate_test_batch(cfg, 128, 5.0, 1, np.random.default_rng(8))

    assert np.array_equal(b1["bit"], b2["bit"])
    assert np.array_equal(b1["x"], b2["x"])
    assert np.array_equal(b1["tau"], b2["tau"])
    assert not (np.array_equal(b1["bit"], b3["bit"]) and np.array_equal(b1["x"], b3["x"]))


def test_generate_test_batch_k0_labels() -> None:
    """BATCH: K=0 -> nessun eco, etichette sensing nulle (solo AWGN)."""
    cfg = _tiny_config()
    batch = generate_test_batch(cfg, 64, 5.0, 0, np.random.default_rng(_SEED))
    assert np.all(batch["tau"] == 0.0)
    assert np.all(batch["f_d"] == 0.0)


def test_generate_test_batch_invalid_args_raise() -> None:
    """BATCH: num_symbols/k/snr_db non validi -> ValueError."""
    cfg = _tiny_config()
    rng = np.random.default_rng(_SEED)
    with pytest.raises(ValueError):
        generate_test_batch(cfg, 0, 10.0, 1, rng)
    with pytest.raises(ValueError):
        generate_test_batch(cfg, -5, 10.0, 1, rng)
    with pytest.raises(ValueError):
        generate_test_batch(cfg, 64, 10.0, -1, rng)
    with pytest.raises(ValueError):
        generate_test_batch(cfg, 64, float("nan"), 1, rng)


def test_generate_transmitted_batch_matches_per_symbol() -> None:
    
    from src.data.dataset_generator import _center_normalize_symbol, _map_type_for_bit

    cfg = _tiny_config()
    bits = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    seeds = np.array([11, 22, 33, 44, 55, 66], dtype=np.int64)
    ref = generate_transmitted_batch(cfg, bits, seeds)

    map_type = str(cfg["data"]["map_type"])
    map_param = float(cfg["data"]["map_param"])
    for i in range(len(bits)):
        mt = _map_type_for_bit(map_type, int(bits[i]))
        x = generate_chaotic_sequence(mt, map_param, int(seeds[i]), _N_SEQ)
        x = _center_normalize_symbol(x)
        assert np.allclose(ref[i], x, atol=1e-14), f"riga {i} non coerente"

