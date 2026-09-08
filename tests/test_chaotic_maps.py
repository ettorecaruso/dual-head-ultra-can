"""
Test per ``src/data/dataset_generator.py``: mappe caotiche.

Questi test verificano:
  T1  test_logistic_map_r_400          mu=4.0 (limite del caos): no NaN/Inf,
                                       valori in (0,1), lunghezza 100
  T2  test_logistic_map_r_357          mu=3.57 (soglia del caos): limitata in [0,1]
  T3  test_logistic_map_invalid_mu     mu=3.0 -> ValueError (guardia regime caotico)
  T4  test_logistic_map_deterministic_seed  stesso seed -> stessa sequenza (1e-12)
  T5  test_bernoulli_map               Eq. (2): valori in [0,1], nessun collasso
  T6  test_autocorrelation_impulsive   lag 0 >> lag>0 (proprieta' Dirac-like, Sez. III-B)

Oltre ai 6 test obbligatori, per rispettare la regola "almeno 3 test per
funzione" (input validi / limite / invalidi) il file copre anche:
  - ``generate_chaotic_sequence`` : casi normali, limite (N=1/N=2, seed=0,
    mu=3.8 non convergenza) e invalidi (map_type, mu, seed, sequence_length);
  - ``_iterate_map``              : ricorrenza esatta Eq. (1)/(2), lunghezza
    dell'output, rami di Bernoulli ai bordi (x0 = 0.25 e x0 = 0.5);
  - ``_map_type_for_bit``         : CSK bit 0/1 (), separazione,
    input invalidi;
  - ``EchoParams`` / ``DirectPathParams`` : validazione dei range (Tab. I);
  - ``build_snr_grid``            : griglia aritmetica inclusiva sugli estremi (T13).

Scelta di pytest: coerente con la suite esistente (``tests/test_logger.py`` usa
pytest con fixtures) e con il comando dal log:
    python -m pytest tests/test_chaotic_maps.py -v

TODO (test NON inclusi qui, assegnati ad altre sottofasi della roadmap):
  - ``apply_aerial_channel`` / ``sample_echo_parameters`` / ``sample_direct_path``
    -> ``tests/test_channel.py``: SNR -20/+30 dB senza NaN/Inf,
    K=10, normalizzazione potenza Eq. (4), tau > sequence_length -> ValueError,
    range di fD/tau/alpha (Tab. I).
  - ``generate_dataset`` / ``main`` / validazioni di dimensione (seq_len,
    num_symbols = 0 o negativi) -> ``tests/test_input_validation.py`` e smoke end-to-end Fase 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data.dataset_generator import (
    DirectPathParams,
    EchoParams,
    _center_normalize_symbol,
    _iterate_map,
    _map_type_for_bit,
    build_snr_grid,
    generate_chaotic_sequence,
)

_LOGISTIC_MU_PAPER = 3.9
_LOGISTIC_MU_MAX = 4.0
_LOGISTIC_MU_MIN = 3.57
_N_SEQ = 100
_SEED = 42

def _normalized_autocorr(x: np.ndarray) -> np.ndarray:
    
    x_centered = np.asarray(x, dtype=np.float64) - np.mean(x)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0.0:
        return np.array([1.0])
    full = np.correlate(x_centered, x_centered, mode="full")
    mid = full.size // 2
    return full[mid:] / denom

def test_logistic_map_r_400() -> None:
    """T1: mu=4.0 (limite superiore del caos) -> no NaN/Inf, valori in (0,1)."""
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_MAX, _SEED, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert seq.dtype == np.float64
    assert np.all(np.isfinite(seq))
    assert np.all(seq > 0.0)
    assert np.all(seq < 1.0)

def test_logistic_map_r_357() -> None:
    
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_MIN, _SEED, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert np.all(np.isfinite(seq))
    assert np.all(seq >= 0.0)
    assert np.all(seq <= 1.0)
    assert float(np.max(seq) - np.min(seq)) > 1e-9

def test_logistic_map_invalid_mu() -> None:
    """T3: mu=3.0 fuori da [3.57, 4] -> ValueError (fail-fast,  Sez. 7)."""
    with pytest.raises(ValueError, match="regime caotico"):
        generate_chaotic_sequence("logistic", 3.0, _SEED, _N_SEQ)

def test_logistic_map_deterministic_seed() -> None:
    
    seq_a = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, 1234, _N_SEQ)
    seq_b = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, 1234, _N_SEQ)
    assert np.allclose(seq_a, seq_b, rtol=0.0, atol=1e-12)

def test_bernoulli_map() -> None:
    """T5: Eq. (2) -> valori in [0,1], statistiche uniformi, nessun collasso."""
    seq = generate_chaotic_sequence("bernoulli", 0.0, _SEED, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert np.all(np.isfinite(seq))
    assert np.all(seq >= 0.0)
    assert np.all(seq <= 1.0)
    assert np.mean(seq < 1e-12) < 0.2, "Bernoulli collassata a zero"
    assert 0.35 < np.mean(seq) < 0.65
    assert np.mean(seq ** 2) > 0.2

def test_autocorrelation_impulsive() -> None:
    
    seq4 = generate_chaotic_sequence("logistic", _LOGISTIC_MU_MAX, _SEED, 1000)
    r4 = _normalized_autocorr(seq4)
    assert r4[0] == pytest.approx(1.0, abs=1e-12)
    assert float(np.max(np.abs(r4[1 : r4.size // 2]))) < 0.35
    assert float(np.mean(np.abs(r4[1:20]))) < 0.1
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, 1000)
    r = _normalized_autocorr(seq)
    assert r[0] == pytest.approx(1.0, abs=1e-12)
    side_peak = float(np.max(np.abs(r[2 : r.size // 2])))
    assert side_peak < 0.35
    assert r[0] > 3.0 * side_peak

def test_generate_chaotic_sequence_normal() -> None:
    """Caso normale: logistic mu=3.9, N=100 -> shape, range ed energia attesi."""
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert seq.dtype == np.float64
    assert np.all(np.isfinite(seq))
    assert np.all(seq > 0.0)
    assert np.all(seq < 1.0)
    energy = float(np.sum(seq ** 2))
    assert 0.0 < energy < _N_SEQ

def test_generate_chaotic_sequence_seed_zero() -> None:
    """Caso limite: seed=0 e' un input valido (estremo inferiore ammesso)."""
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, 0, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert np.all(np.isfinite(seq))
    assert np.all(seq > 0.0)

def test_generate_chaotic_sequence_different_seeds() -> None:
    """Caso normale: seed diversi -> sequenze diverse (probabilita' di collisione ~0)."""
    seq_a = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, 1, _N_SEQ)
    seq_b = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, 2, _N_SEQ)
    assert not np.allclose(seq_a, seq_b, rtol=0.0, atol=1e-12)

def test_logistic_map_r_38_no_convergence() -> None:
    """Caso normale: mu=3.8 -> caotica, NON converge a un punto fisso."""
    seq = generate_chaotic_sequence("logistic", 3.8, _SEED, 500)
    assert np.all(np.isfinite(seq))
    assert float(np.max(seq) - np.min(seq)) > 0.5
    tail = seq[-20:]
    assert float(np.max(tail) - np.min(tail)) > 0.1

def test_generate_chaotic_sequence_length_2() -> None:
    
    seq = generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, 2)
    assert seq.shape == (2,)
    assert np.all(np.isfinite(seq))
    assert np.count_nonzero(seq) == 2

def test_generate_chaotic_sequence_length_1_raises() -> None:
    
    with pytest.raises(RuntimeError, match="degenerata"):
        generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, 1)

def test_bernoulli_map_ignores_map_param() -> None:
    """Caso limite: map_param fuori [3.57, 4] NON e' errore per Bernoulli ()."""
    seq = generate_chaotic_sequence("bernoulli", 1.0, _SEED, _N_SEQ)
    assert np.all(np.isfinite(seq))
    assert np.all(seq >= 0.0)
    assert np.all(seq <= 1.0)

def test_bernoulli_map_no_degenerate() -> None:
    
    seq = generate_chaotic_sequence("bernoulli", 0.0, _SEED + 1, _N_SEQ)
    assert float(np.max(seq) - np.min(seq)) > 1e-9
    assert float(np.sum(seq ** 2)) > 1e-12

def test_center_normalize_symbol() -> None:
    
    rng = np.random.default_rng(0)
    for map_type in ("logistic", "bernoulli"):
        for _ in range(20):
            seq = generate_chaotic_sequence(map_type, _LOGISTIC_MU_PAPER, int(rng.integers(0, 2**31 - 1)), _N_SEQ)
            norm = _center_normalize_symbol(seq)
            assert float(np.mean(norm)) == pytest.approx(0.0, abs=1e-9)
            assert float(np.sum(norm ** 2)) == pytest.approx(1.0, abs=1e-9)

def test_csk_classes_equienergetic() -> None:
    
    rng = np.random.default_rng(0)
    e_log, e_bern = [], []
    for _ in range(200):
        s_log = _center_normalize_symbol(
            generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, int(rng.integers(0, 2**31 - 1)), _N_SEQ))
        s_bern = _center_normalize_symbol(
            generate_chaotic_sequence("bernoulli", _LOGISTIC_MU_PAPER, int(rng.integers(0, 2**31 - 1)), _N_SEQ))
        e_log.append(float(np.sum(s_log ** 2)))
        e_bern.append(float(np.sum(s_bern ** 2)))
    assert float(np.mean(e_log)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.mean(e_bern)) == pytest.approx(1.0, abs=1e-9)

@pytest.mark.parametrize("map_type", ["tent", "", None, 3.7])
def test_generate_chaotic_sequence_invalid_map_type(map_type: object) -> None:
    """Input invalidi: map_type fuori da {logistic, bernoulli} -> ValueError."""
    with pytest.raises(ValueError, match="map_type"):
        generate_chaotic_sequence(map_type, _LOGISTIC_MU_PAPER, _SEED, _N_SEQ)

@pytest.mark.parametrize("map_param", [3.0, 4.1, 2.9, float("nan"), float("inf"), "3.9"])
def test_generate_chaotic_sequence_invalid_map_param(map_param: object) -> None:
    """Input invalidi: mu fuori [3.57, 4] o non finito -> ValueError (logistic)."""
    with pytest.raises(ValueError):
        generate_chaotic_sequence("logistic", map_param, _SEED, _N_SEQ)

@pytest.mark.parametrize("seed", [-1, -100, 3.5, True])
def test_generate_chaotic_sequence_invalid_seed(seed: object) -> None:
    """Input invalidi: seed non int >= 0 -> ValueError."""
    with pytest.raises(ValueError, match="seed"):
        generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, seed, _N_SEQ)

@pytest.mark.parametrize("sequence_length", [0, -5, 3.7])
def test_generate_chaotic_sequence_invalid_length(sequence_length: object) -> None:
    """Input invalidi: sequence_length non int > 0 -> ValueError."""
    with pytest.raises(ValueError, match="sequence_length"):
        generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, sequence_length)

def test_generate_chaotic_sequence_bool_length_regression() -> None:
    
    with pytest.raises(RuntimeError, match="degenerata"):
        generate_chaotic_sequence("logistic", _LOGISTIC_MU_PAPER, _SEED, True)

def test_iterate_map_logistic_recurrence() -> None:
    """Caso normale: x[0]=x0 e x[1]=mu*x0*(1-x0) (Eq. (1), Sez. III-A)."""
    x0, mu, n = 0.3, _LOGISTIC_MU_PAPER, 5
    seq = _iterate_map("logistic", mu, x0, n)
    assert seq[0] == pytest.approx(x0, abs=1e-15)
    assert seq[1] == pytest.approx(mu * x0 * (1.0 - x0), abs=1e-15)

def test_iterate_map_length_matches_sequence_length() -> None:
    """Caso limite: la lunghezza dell'output coincide con ``sequence_length``."""
    for n in (1, 2, 100, 1000):
        assert _iterate_map("logistic", 3.9, 0.3, n).shape == (n,)
        assert _iterate_map("bernoulli", 0.0, 0.3, n).shape == (n,)

def test_iterate_map_bernoulli_no_collapse() -> None:
    
    rng = np.random.default_rng(0)
    for _ in range(50):
        x0 = float(rng.uniform(0.01, 0.99))
        seq = _iterate_map("bernoulli", 0.0, x0, 100)
        assert np.all(seq >= 0.0) and np.all(seq <= 1.0)
        assert np.mean(seq < 1e-12) < 0.2, "Bernoulli collassata a zero"
        assert 0.3 < np.mean(seq) < 0.7, "statistica non uniforme"

def test_map_type_for_bit_zero() -> None:
    
    assert _map_type_for_bit("logistic", 0) == "logistic"
    assert _map_type_for_bit("bernoulli", 0) == "bernoulli"

def test_map_type_for_bit_one() -> None:
    
    assert _map_type_for_bit("logistic", 1) == "bernoulli"
    assert _map_type_for_bit("bernoulli", 1) == "logistic"

def test_map_type_for_bit_separation() -> None:
    """Caso limite: bit 0 e bit 1 usano SEMPRE mappe diverse (separazione CSK)."""
    for base in ("logistic", "bernoulli"):
        assert _map_type_for_bit(base, 0) != _map_type_for_bit(base, 1)

@pytest.mark.parametrize("bit", [-1, 2, 0.5, None])
def test_map_type_for_bit_invalid_bit(bit: object) -> None:
    """Input invalidi: bit non in {0, 1} -> ValueError."""
    with pytest.raises(ValueError, match="bit"):
        _map_type_for_bit("logistic", bit)

@pytest.mark.parametrize("map_type", ["tent", "", None])
def test_map_type_for_bit_invalid_map_type(map_type: object) -> None:
    """Input invalidi: map_type non valido -> ValueError."""
    with pytest.raises(ValueError, match="map_type"):
        _map_type_for_bit(map_type, 0)

def test_echo_params_valid() -> None:
    
    echo = EchoParams(tau=10, f_doppler=8e-5, alpha=1e-3)
    assert echo.tau == 10
    assert echo.f_doppler == pytest.approx(8e-5)
    assert echo.alpha == pytest.approx(1e-3)

@pytest.mark.parametrize("tau", [0, -1, 1.5, "10"])
def test_echo_params_invalid_tau(tau: object) -> None:
    """Input invalidi: tau non int >= 1 -> ValueError (policy)."""
    with pytest.raises(ValueError, match="tau"):
        EchoParams(tau=tau, f_doppler=1e-5, alpha=1e-3)

@pytest.mark.parametrize("f_doppler", [-0.1, 0.5, 1.0, float("nan")])
def test_echo_params_invalid_doppler(f_doppler: float) -> None:
    """Input invalidi: f_doppler fuori da [0, 0.5) o non finito -> ValueError."""
    with pytest.raises(ValueError, match="f_doppler"):
        EchoParams(tau=1, f_doppler=f_doppler, alpha=1e-3)

@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5, float("nan")])
def test_echo_params_invalid_alpha(alpha: float) -> None:
    """Input invalidi: alpha fuori da (0, 1) o non finito -> ValueError."""
    with pytest.raises(ValueError, match="alpha"):
        EchoParams(tau=1, f_doppler=1e-5, alpha=alpha)

def test_direct_path_params_valid() -> None:
    """Caso normale: DirectPathParams con h_c finito e f_dc in [0, 0.5)."""
    direct = DirectPathParams(h_c=complex(0.7, 0.3), f_dc=1e-5)
    assert math.isfinite(direct.h_c.real)
    assert direct.f_dc == pytest.approx(1e-5)

@pytest.mark.parametrize("h_c", [complex(float("nan"), 0.0), complex(0.0, float("inf"))])
def test_direct_path_params_invalid_hc(h_c: complex) -> None:
    """Input invalidi: h_c con NaN/Inf -> ValueError."""
    with pytest.raises(ValueError, match="h_c"):
        DirectPathParams(h_c=h_c, f_dc=1e-5)

@pytest.mark.parametrize("f_dc", [-0.1, 0.5, float("nan")])
def test_direct_path_params_invalid_fdc(f_dc: float) -> None:
    """Input invalidi: f_dc fuori da [0, 0.5) o non finito -> ValueError."""
    with pytest.raises(ValueError, match="f_dc"):
        DirectPathParams(h_c=complex(1.0, 0.0), f_dc=f_dc)

def test_build_snr_grid_normal() -> None:
    """Caso normale: [-5, 20] dB con step 2 -> 14 punti, estremi inclusi."""
    grid = build_snr_grid([-5.0, 20.0], 2.0)
    assert len(grid) == 14
    assert grid[0] == pytest.approx(-5.0)
    assert grid[-1] == pytest.approx(20.0)
    diffs = np.diff(grid)
    assert np.all(diffs[:-1] == pytest.approx(2.0))
    assert diffs[-1] == pytest.approx(1.0)

def test_build_snr_grid_includes_max() -> None:
    
    grid = build_snr_grid([0.0, 10.0], 3.0)
    assert grid == pytest.approx([0.0, 3.0, 6.0, 9.0, 10.0])

def test_build_snr_grid_single_point() -> None:
    """Caso limite: range minimo ammesso -> 2 punti (min e max)."""
    grid = build_snr_grid([5.0, 5.5], 0.5)
    assert grid == pytest.approx([5.0, 5.5])

@pytest.mark.parametrize(
    "snr_range, snr_step",
    [
        ([5.0, 5.0], 2.0),
        ([20.0, -5.0], 2.0),
        ([-5.0], 2.0),
        ([-5.0, 20.0], -2.0),
        ([-5.0, 20.0], 0.0),
        ([-5.0, 20.0], float("nan")),
    ],
)
def test_build_snr_grid_invalid_inputs(snr_range: object, snr_step: float) -> None:
    """Input invalidi: range/step non ammessi -> ValueError."""
    with pytest.raises(ValueError):
        build_snr_grid(snr_range, snr_step)

def test_integration_import_module() -> None:
    
    import src.data.dataset_generator as dg

    assert callable(dg.generate_chaotic_sequence)
    assert callable(dg.apply_aerial_channel)
    assert callable(dg.sample_echo_parameters)
    assert callable(dg.generate_dataset)

def test_integration_sequence_per_bit() -> None:
    
    sequences: dict = {}
    for bit in (0, 1):
        map_i = _map_type_for_bit("logistic", bit)
        seq = generate_chaotic_sequence(map_i, _LOGISTIC_MU_PAPER, _SEED + bit, _N_SEQ)
        assert seq.shape == (_N_SEQ,)
        assert np.all(np.isfinite(seq))
        sequences[bit] = seq
    assert not np.allclose(sequences[0], sequences[1], rtol=0.0, atol=1e-12)

def test_integration_map_bounds_aligned_with_paper() -> None:
    
    import src.data.dataset_generator as dg

    assert dg._LOGISTIC_MU_MIN == pytest.approx(_LOGISTIC_MU_MIN)
    assert dg._LOGISTIC_MU_MAX == pytest.approx(_LOGISTIC_MU_MAX)

