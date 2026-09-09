"""
Test di validazione degli input del pipeline dati

Questi test verificano:
  T1  test_zero_sequence_length_raises     sequence_length=0 -> ValueError
  T2  test_negative_sequence_length_raises sequence_length=-5 -> ValueError
  T3  test_zero_num_symbols_raises         num_symbols=0 -> ValueError
  T4  test_negative_num_symbols_raises     num_symbols=-10 -> ValueError

Superficie coperta (funzioni pubbliche e guardie dei moduli dati):
  ``dataset_generator._validate_config``  : validazione fail-fast della config
      del generatore (keys mancanti, NaN/Inf, map_param outside regime, policy
      max_delay <= sequence_length, dimensioni seq_len/num_symbols, echoes,
      snr_range/snr_step, seed, raw_dir/processed_dir).
  ``dataset_generator.generate_chaotic_sequence`` : validazione della
      lunghezza della sequenza (0, negativa, non-int) + caso valido N=100.
  ``dataset_generator.build_snr_grid``    : grid SNR inclusiva, input validi/
      limite/invalidi (range invertito, step <= 0, valori non finiti).
  ``dataset_generator.generate_dataset``  : guardie di bilanciamento
      (num_symbols_train not divisible, n_per_combo diseven) e split invalido.
  ``data_loader._validate_config``        : validazione fail-fast della config
      del loader (sezioni mancanti, seq_len/num_symbols/batch_size/seed/echoes/
      snr_range/feature_mode/max_doppler).
  ``data_loader.normalize_targets``       : Min-Max [0,1] (Sez. IV-D), shape
      allineate, tau_max/fd_max > 0, NaN/Inf, output outside range -> RuntimeError.
  ``data_loader.build_tf_dataset``        : batch size 1/32/128 con shape
      corrette ``(B, 100, 1)``, batch/seed/feature_mode/keys invalidi.
  ``data_loader._parse_npz_name``         : pattern ``<split>_snr<snr>_echo<k>``.
  ``data_loader._validate_npz_arrays``    : shape/range/finitezza dei blocchi.
  ``data_loader.load_npz_files``          : ``FileNotFoundError`` per directory
      mancante/vuota, grid (SNR, K) incomplete, argomenti invalidi.
  ``data_loader.verify_snr_balance``      : bilanciamento per (SNR, K) e ``expected_per_combo``.
  ``data_loader._expected_per_combo``     : conteggio expected per split.

Test di integrazione:
  flusso end-to-end generatore -> loader (generate_dataset -> load_npz_files ->
  verify_snr_balance -> normalize_targets -> build_tf_dataset) con dataset
  minimo e verifiche di shape/finitezza su ogni output.

Scelta di pytest: coerente con l'intera suite esistente (``test_chaotic_maps``,
``test_channel``, ``test_logger`` usano pytest con fixtures/parametrize) e con il
comando dal log: ``python -m pytest tests/test_input_validation.py -v``.
Il template utente mostrava ``unittest``; il pattern ``setUp``/``tearDown`` e'
stato mappato su helper di modulo e ``pytest.raises`` per uniformita' col resto.

Conformita' : shape-check, range-check e check NaN/Inf su ogni
output; type hints e docstring su tutte le funzioni di test.

TODO (test mancanti, segnalati esplicitamente):
  - ``apply_aerial_channel`` / ``sample_echo_parameters`` / ``sample_direct_path``
    con SNR estremi (-20/+30 dB) e mu limite (4.0/3.57): gia' coperti da
    ``tests/test_channel.py`` e ``tests/test_chaotic_maps.py`` -> non duplicati qui.
  - ``generate_chaotic_sequence`` con ``sequence_length=1`` (RuntimeError della
    guardia anti-degenerazione): coperto da ``test_generate_chaotic_sequence_length_1_raises``
    in ``tests/test_chaotic_maps.py``.
  - ``sequence_length=True`` (bool) NON solleva ValueError in
    ``generate_chaotic_sequence`` / ``_validate_config`` del generatore
    (``int(True) == 1``): incoerenza documentata da
    ``test_generate_chaotic_sequence_bool_length_regression`` in
    ``tests/test_chaotic_maps.py``; il loader la rifiuta (``isinstance(bool)``).
  - ramo ``n_per_combo <= 0`` di ``generate_dataset``: irraggiungibile perche'
    ``_validate_config`` impone ``num_symbols_* > 0`` prima (difesa in profondita').
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import yaml

from src.data.dataset_generator import (
    _validate_config as validate_generator_config,
    build_snr_grid,
    generate_chaotic_sequence,
    generate_dataset,
    resolve_n_per_combo,
)
from src.data.data_loader import (
    _expected_per_combo,
    _parse_npz_name,
    _validate_config as validate_loader_config,
    _validate_npz_arrays,
    build_tf_dataset,
    load_npz_files,
    normalize_targets,
    verify_snr_balance,
)

_N_SEQ = 100
_MU_PAPER = 3.9
_MU_MIN = 3.57
_MU_MAX = 4.0
_SEED = 42
_MAX_DELAY = 33
_MAX_DOPPLER = 8e-5
_ALPHA_MIN = 1e-6
_ALPHA_MAX = 1e-2
_DOPPLER_DIRECT_MAX = 1e-5

def _generator_config(data_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    
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

def _loader_config(
    data_overrides: Optional[Dict[str, Any]] = None,
    training_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    data: Dict[str, Any] = {
        "sequence_length": _N_SEQ,
        "feature_mode": "real",
        "map_type": "logistic",
        "map_param": 3.9,
        "max_delay": _MAX_DELAY,
        "max_doppler": _MAX_DOPPLER,
        "snr_range": [-5.0, 20.0],
        "snr_step": 2.0,
        "echoes": [1, 3],
        "num_symbols_train": 5040,
        "num_symbols_val": 1000,
        "num_symbols_test": 2000,
        "raw_dir": "data/raw",
    }
    if data_overrides:
        data.update(data_overrides)
    training: Dict[str, Any] = {"batch_size": 64}
    if training_overrides:
        training.update(training_overrides)
    return {
        "general": {"seed": _SEED, "experiment_name": "test_input_validation"},
        "data": data,
        "training": training,
    }

def _tiny_data_dict(n: int = 8, sequence_length: int = _N_SEQ) -> Dict[str, np.ndarray]:
    
    x = np.ones((n, sequence_length), dtype=np.complex128) + 0.5j
    bit = np.tile(np.array([0, 1], dtype=np.uint8), n // 2)
    tau = np.linspace(0.0, float(_MAX_DELAY), n, dtype=np.float64)
    f_d = np.linspace(0.0, float(_MAX_DOPPLER), n, dtype=np.float64)
    snr_db = np.zeros(n, dtype=np.float64)
    k = np.zeros(n, dtype=np.int64)
    seed = np.arange(n, dtype=np.int64)
    return {"x": x, "bit": bit, "tau": tau, "f_d": f_d, "snr_db": snr_db, "k": k, "seed": seed}

def _write_tiny_npz(
    data_dir: Path,
    split: str,
    snr: float,
    k: int,
    n: int = 4,
    sequence_length: int = _N_SEQ,
) -> Path:
    
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{split}_snr{snr:g}_echo{k}.npz"
    x = np.ones((n, sequence_length), dtype=np.complex128) + 0.3j
    bit = np.tile(np.array([0, 1], dtype=np.uint8), n // 2)
    tau = np.zeros(n, dtype=np.float64)
    f_d = np.zeros(n, dtype=np.float64)
    np.savez(
        path,
        x=x,
        bit=bit,
        tau=tau,
        f_d=f_d,
        snr_db=float(snr),
        k=k,
        seed=np.zeros(n, dtype=np.int64),
    )
    return path

def test_zero_sequence_length_raises() -> None:
    
    with pytest.raises(ValueError, match="sequence_length"):
        validate_generator_config(_generator_config({"sequence_length": 0}))
    with pytest.raises(ValueError, match="sequence_length"):
        generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, 0)
    with pytest.raises(ValueError, match="sequence_length"):
        validate_loader_config(_loader_config({"sequence_length": 0}))

def test_negative_sequence_length_raises() -> None:
    """T2: sequence_length=-5 -> ValueError (dimensioni negative)."""
    with pytest.raises(ValueError, match="sequence_length"):
        validate_generator_config(_generator_config({"sequence_length": -5}))
    with pytest.raises(ValueError, match="sequence_length"):
        generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, -5)
    with pytest.raises(ValueError, match="sequence_length"):
        validate_loader_config(_loader_config({"sequence_length": -5}))

def test_zero_num_symbols_raises() -> None:
    
    with pytest.raises(ValueError, match="num_symbols"):
        validate_generator_config(_generator_config({"num_symbols_val": 0}))
    with pytest.raises(ValueError, match="num_symbols"):
        validate_loader_config(_loader_config({"num_symbols_test": 0}))

def test_negative_num_symbols_raises() -> None:
    """T4: num_symbols=-10 -> ValueError (dimensioni negative)."""
    with pytest.raises(ValueError, match="num_symbols"):
        validate_generator_config(_generator_config({"num_symbols_train": -10}))
    with pytest.raises(ValueError, match="num_symbols"):
        validate_loader_config(_loader_config({"num_symbols_val": -10}))

def test_generate_chaotic_sequence_valid_length() -> None:
    """Caso normale: N=100 -> shape (100,), finita, valori in (0, 1)."""
    seq = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, _N_SEQ)
    assert seq.shape == (_N_SEQ,)
    assert seq.dtype == np.float64
    assert np.all(np.isfinite(seq))
    assert np.all(seq > 0.0) and np.all(seq < 1.0)

def test_generate_chaotic_sequence_min_length() -> None:
    
    seq = generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, 2)
    assert seq.shape == (2,)
    assert np.all(np.isfinite(seq))
    assert float(np.max(seq) - np.min(seq)) > 0.0

@pytest.mark.parametrize("sequence_length", [0, -5, 3.7, "100"])
def test_generate_chaotic_sequence_invalid_length(sequence_length: object) -> None:
    """Input invalidi: lunghezza zero/negativa/non-int -> ValueError."""
    with pytest.raises(ValueError, match="sequence_length"):
        generate_chaotic_sequence("logistic", _MU_PAPER, _SEED, sequence_length)

def test_build_snr_grid_normal() -> None:
    """Caso normale: [-5, 20] dB con step 2 -> 14 punti, estremi inclusi."""
    grid = build_snr_grid([-5.0, 20.0], 2.0)
    assert grid[0] == pytest.approx(-5.0)
    assert grid[-1] == pytest.approx(20.0)
    assert len(grid) == 14

def test_build_snr_grid_edge_includes_max() -> None:
    
    grid = build_snr_grid([0.0, 10.0], 3.0)
    assert grid == pytest.approx([0.0, 3.0, 6.0, 9.0, 10.0])

@pytest.mark.parametrize(
    ("snr_range", "snr_step"),
    [
        ([10.0, 0.0], 2.0),
        ([0.0, 10.0], 0.0),
        ([0.0, 10.0], -1.0),
        ([0.0, float("nan")], 2.0),
        ([0.0, 10.0], float("inf")),
        ([0.0, 1.0, 2.0], 0.5),
        ("0,10", 2.0),
    ],
)
def test_build_snr_grid_invalid_inputs(snr_range: object, snr_step: object) -> None:
    """Input invalidi: range/step non ammessi -> ValueError (fail-fast)."""
    with pytest.raises(ValueError):
        build_snr_grid(snr_range, snr_step)

def test_generator_config_valid_passes() -> None:
    
    validate_generator_config(_generator_config())

def test_generator_config_not_dict_raises() -> None:
    """Input invalido: config non dict -> TypeError."""
    with pytest.raises(TypeError):
        validate_generator_config("non-dict")

def test_generator_config_missing_keys_raises() -> None:
    """Input invalido: chiave richiesta mancante -> ValueError (fail-fast)."""
    cfg = _generator_config()
    del cfg["data"]["max_doppler"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_generator_config(cfg)

def test_generator_config_nan_raises() -> None:
    
    cfg = _generator_config({"snr_range": [float("nan"), 20.0]})
    with pytest.raises(ValueError, match="non-finite"):
        validate_generator_config(cfg)

def test_generator_config_mu_out_of_regime_raises() -> None:
    """Input invalido: mu=3.0 outside [3.57, 4] -> ValueError (Eq. (1))."""
    cfg = _generator_config({"map_param": 3.0})
    with pytest.raises(ValueError, match="chaotic regime"):
        validate_generator_config(cfg)

def test_generator_config_max_delay_policy_raises() -> None:
    """Input invalido: max_delay > sequence_length -> ValueError (policy tau<=N)."""
    cfg = _generator_config({"max_delay": _N_SEQ + 1})
    with pytest.raises(ValueError, match="max_delay"):
        validate_generator_config(cfg)

def test_generator_config_invalid_echoes_raises() -> None:
    """Input invalido: echoes con k negativo -> ValueError (Tab. I)."""
    cfg = _generator_config({"echoes": [1, -3]})
    with pytest.raises(ValueError, match="echoes"):
        validate_generator_config(cfg)

def test_loader_config_valid_passes() -> None:
    
    validate_loader_config(_loader_config())

def test_loader_config_missing_section_raises() -> None:
    """Input invalido: sezione 'training' mancante -> ValueError."""
    cfg = _loader_config()
    del cfg["training"]
    with pytest.raises(ValueError, match="training"):
        validate_loader_config(cfg)

@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_loader_config_invalid_batch_size_raises(batch_size: object) -> None:
    """Input invalido: batch_size 0/negativo/bool -> ValueError."""
    with pytest.raises(ValueError, match="batch_size"):
        validate_loader_config(_loader_config(training_overrides={"batch_size": batch_size}))

def test_loader_config_invalid_feature_mode_raises() -> None:
    """Input invalido: feature_mode non ammesso -> ValueError ()."""
    cfg = _loader_config({"feature_mode": "magnitude"})
    with pytest.raises(ValueError, match="feature_mode"):
        validate_loader_config(cfg)

def test_loader_config_max_doppler_guard_raises() -> None:
    """Input invalido: max_doppler >= 0.5 -> ValueError (anti-aliasing)."""
    cfg = _loader_config({"max_doppler": 0.5})
    with pytest.raises(ValueError, match="max_doppler"):
        validate_loader_config(cfg)

def test_normalize_targets_normal_case() -> None:
    """Caso normale: tau/f_d in range -> shape (N, 2), valori in [0, 1]."""
    tau = np.array([0.0, _MAX_DELAY / 2.0, float(_MAX_DELAY)], dtype=np.float64)
    f_d = np.array([0.0, _MAX_DOPPLER / 2.0, float(_MAX_DOPPLER)], dtype=np.float64)
    targets = normalize_targets(tau, f_d, tau_max=_MAX_DELAY, fd_max=_MAX_DOPPLER)
    assert targets.shape == (3, 2)
    assert targets.dtype == np.float32
    assert np.all(targets >= 0.0) and np.all(targets <= 1.0)
    assert targets[0, 0] == pytest.approx(0.0)
    assert targets[2, 0] == pytest.approx(1.0)

def test_normalize_targets_edge_zero_inputs() -> None:
    """Caso limite: input nulli -> output nullo (normalizzazione degenere)."""
    tau = np.zeros(4, dtype=np.float64)
    f_d = np.zeros(4, dtype=np.float64)
    targets = normalize_targets(tau, f_d, tau_max=_MAX_DELAY, fd_max=_MAX_DOPPLER)
    assert targets.shape == (4, 2)
    assert np.all(targets == 0.0)

@pytest.mark.parametrize(
    ("tau", "f_d", "tau_max", "fd_max", "exc"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 10.0, 1e-4, ValueError),
        (np.zeros((2, 2)), np.zeros((2, 2)), 10.0, 1e-4, ValueError),
        (np.array([1.0]), np.array([1.0]), 0.0, 1e-4, ValueError),
        (np.array([1.0]), np.array([1.0]), 10.0, 0.0, ValueError),
        (np.array([1.0]), np.array([1.0]), -5.0, 1e-4, ValueError),
        (np.array([float("nan")]), np.array([1.0]), 10.0, 1e-4, ValueError),
        (np.array([1.0]), np.array([float("inf")]), 10.0, 1e-4, ValueError),
        (np.array([11.0]), np.array([1.0]), 10.0, 1e-4, RuntimeError),
    ],
)
def test_normalize_targets_invalid_inputs(
    tau: np.ndarray,
    f_d: np.ndarray,
    tau_max: float,
    fd_max: float,
    exc: type,
) -> None:
    """Input invalidi: shape/range/finitezza violati -> eccezione attesa."""
    with pytest.raises(exc):
        normalize_targets(tau, f_d, tau_max=tau_max, fd_max=fd_max)

@pytest.mark.parametrize("batch_size", [1, 32, 128])
def test_build_tf_dataset_variable_batch(batch_size: int) -> None:
    """Consistenza shape: batch 1/32/128 -> features (None,100,2) [Re + rif.], sensing (None,2)."""
    data = _tiny_data_dict(n=8, sequence_length=_N_SEQ)
    ds = build_tf_dataset(data, batch_size=batch_size, config=_loader_config())
    features_spec = ds.element_spec[0]
    comm_spec = ds.element_spec[1]["comm"]
    sensing_spec = ds.element_spec[1]["sensing"]
    assert features_spec.shape.as_list() == [None, _N_SEQ, 2]
    assert comm_spec.shape.as_list() == [None]
    assert sensing_spec.shape.as_list() == [None, 2]

def test_build_tf_dataset_iq_mode() -> None:
    
    data = _tiny_data_dict(n=8, sequence_length=_N_SEQ)
    ds = build_tf_dataset(
        data, batch_size=8, config=_loader_config(), feature_mode="iq"
    )
    assert ds.element_spec[0].shape.as_list() == [None, _N_SEQ, 3]

@pytest.mark.parametrize("batch_size", [0, -5, True])
def test_build_tf_dataset_invalid_batch_raises(batch_size: object) -> None:
    """Input invalido: batch_size 0/negativo/bool -> ValueError."""
    data = _tiny_data_dict()
    with pytest.raises(ValueError, match="batch_size"):
        build_tf_dataset(data, batch_size=batch_size, config=_loader_config())

def test_build_tf_dataset_missing_seed_raises() -> None:
    
    cfg = _loader_config()
    cfg["general"] = {"experiment_name": "no_seed"}
    data = _tiny_data_dict()
    with pytest.raises(ValueError, match="general.seed"):
        build_tf_dataset(data, batch_size=8, config=cfg, seed=None)

def test_build_tf_dataset_invalid_seed_raises() -> None:
    """Input invalido: seed negativo -> ValueError."""
    data = _tiny_data_dict()
    with pytest.raises(ValueError, match="seed"):
        build_tf_dataset(data, batch_size=8, config=_loader_config(), seed=-1)

def test_build_tf_dataset_invalid_feature_mode_raises() -> None:
    """Input invalido: feature_mode non ammesso -> ValueError."""
    data = _tiny_data_dict()
    with pytest.raises(ValueError, match="feature_mode"):
        build_tf_dataset(data, batch_size=8, config=_loader_config(), feature_mode="bad")

def test_build_tf_dataset_missing_data_keys_raises() -> None:
    
    data = _tiny_data_dict()
    del data["f_d"]
    with pytest.raises(ValueError, match="keys"):
        build_tf_dataset(data, batch_size=8, config=_loader_config())

@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("train_snr5_echo1.npz", ("train", 5.0, 1)),
        ("test_snr-5_echo0.npz", ("test", -5.0, 0)),
        ("val_snr5.5_echo3.npz", ("val", 5.5, 3)),
        ("train_snr1e-05_echo2.npz", ("train", 1e-5, 2)),
    ],
)
def test_parse_npz_name_valid(filename: str, expected: Tuple[str, float, int]) -> None:
    
    assert _parse_npz_name(Path(filename)) == expected

@pytest.mark.parametrize(
    "filename",
    [
        "foo.npz",
        "train_echo1.npz",
        "train_snr5_echoX.npz",
        "train_snr_echo1.npz",
        "train_snr5_echo1.txt",
        "train_snr5_echo-1.npz",
        "xtrain_snr5_echo1.npz",
        "train_snr5_echo1",
    ],
)
def test_parse_npz_name_invalid_raises(filename: str) -> None:
    """Input invalidi: nome outside pattern -> ValueError."""
    with pytest.raises(ValueError, match="file name not recognized"):
        _parse_npz_name(Path(filename))

def test_validate_npz_arrays_valid() -> None:
    
    data = _tiny_data_dict()
    _validate_npz_arrays(
        data["x"], data["bit"], data["tau"], data["f_d"],
        sequence_length=_N_SEQ, max_delay=_MAX_DELAY, max_doppler=_MAX_DOPPLER,
    )

def test_validate_npz_arrays_empty_block_raises() -> None:
    """Caso limite: blocco senza campioni (n=0) -> ValueError."""
    x = np.zeros((0, _N_SEQ), dtype=np.complex128)
    with pytest.raises(ValueError, match="n=0"):
        _validate_npz_arrays(
            x, np.array([], dtype=np.uint8), np.array([]), np.array([]),
            sequence_length=_N_SEQ, max_delay=_MAX_DELAY, max_doppler=_MAX_DOPPLER,
        )

@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        ("x_wrong_shape", "x must have shape"),
        ("x_nan", "NaN/Inf"),
        ("bit_invalid", "0/1"),
        ("tau_out_of_range", "tau out of range"),
        ("f_d_out_of_range", "f_d out of range"),
        ("lengths_mismatch", "lunghezze incoerenti"),
    ],
)
def test_validate_npz_arrays_invalid_raises(mutate: str, pattern: str) -> None:
    """Input invalidi: ogni violazione -> ValueError con messaggio dedicato."""
    data = _tiny_data_dict()
    if mutate == "x_wrong_shape":
        data["x"] = np.zeros((8, _N_SEQ - 1), dtype=np.complex128)
    elif mutate == "x_nan":
        data["x"] = np.full((8, _N_SEQ), np.nan + 0j, dtype=np.complex128)
    elif mutate == "bit_invalid":
        data["bit"] = np.full(8, 2, dtype=np.uint8)
    elif mutate == "tau_out_of_range":
        data["tau"] = np.full(8, float(_MAX_DELAY) + 1.0, dtype=np.float64)
    elif mutate == "f_d_out_of_range":
        data["f_d"] = np.full(8, float(_MAX_DOPPLER) * 2.0, dtype=np.float64)
    elif mutate == "lengths_mismatch":
        data["bit"] = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError, match=pattern):
        _validate_npz_arrays(
            data["x"], data["bit"], data["tau"], data["f_d"],
            sequence_length=_N_SEQ, max_delay=_MAX_DELAY, max_doppler=_MAX_DOPPLER,
        )

def test_load_npz_files_missing_dir_raises(tmp_path: Path) -> None:
    """File mancante: data_dir inesistente -> FileNotFoundError."""
    missing = tmp_path / "non_esiste"
    with pytest.raises(FileNotFoundError, match="data_dir"):
        load_npz_files(missing, [0.0], [0], "test", _loader_config())

def test_load_npz_files_empty_dir_raises(tmp_path: Path) -> None:
    """File mancante: directory senza file .npz -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no valid .npz file"):
        load_npz_files(tmp_path, [0.0], [0], "test", _loader_config())

def test_load_npz_files_incomplete_grid_raises(tmp_path: Path) -> None:
    """Input invalido: grid (SNR, K) incomplete -> ValueError."""
    _write_tiny_npz(tmp_path, "test", snr=0.0, k=0)
    with pytest.raises(ValueError, match="incomplete"):
        load_npz_files(tmp_path, [0.0, 2.0], [0], "test", _loader_config())

@pytest.mark.parametrize(
    ("split", "snr_values", "echoes"),
    [
        ("foo", [0.0], [0]),
        ("test", [], [0]),
        ("test", [0.0], []),
        ("test", [float("nan")], [0]),
        ("test", [0.0], [-1]),
    ],
)
def test_load_npz_files_invalid_args_raises(
    tmp_path: Path, split: str, snr_values: List[float], echoes: List[int]
) -> None:
    """Input invalidi: split/snr_values/echoes non ammessi -> ValueError."""
    with pytest.raises(ValueError):
        load_npz_files(tmp_path, snr_values, echoes, split, _loader_config())

def test_load_npz_files_roundtrip(tmp_path: Path) -> None:
    
    _write_tiny_npz(tmp_path, "test", snr=0.0, k=0, n=4)
    _write_tiny_npz(tmp_path, "test", snr=2.0, k=0, n=4)
    data = load_npz_files(tmp_path, [0.0, 2.0], [0], "test", _loader_config())
    assert data["x"].shape == (8, _N_SEQ)
    assert data["bit"].shape == (8,)
    assert data["tau"].shape == (8,)
    assert data["f_d"].shape == (8,)
    assert np.all(np.isfinite(data["x"]))

def test_verify_snr_balance_ok() -> None:
    
    data = _tiny_data_dict(n=8)
    counts = verify_snr_balance(data, [0.0], [0], expected_per_combo=8)
    assert counts == {(0.0, 0): 8}

def test_verify_snr_balance_unbalanced_raises() -> None:
    """Input invalido: conteggi non uniformi -> ValueError."""
    data = _tiny_data_dict(n=8)
    data["snr_db"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="unbalanced"):
        verify_snr_balance(data, [0.0, 2.0], [0])

def test_verify_snr_balance_expected_mismatch_raises() -> None:
    """Input invalido: conteggi uniformi ma != expected_per_combo -> ValueError."""
    data = _tiny_data_dict(n=8)
    with pytest.raises(ValueError, match="expected"):
        verify_snr_balance(data, [0.0], [0], expected_per_combo=4)

def test_verify_snr_balance_invalid_expected_raises() -> None:
    """Input invalido: expected_per_combo non int > 0 -> ValueError."""
    data = _tiny_data_dict()
    with pytest.raises(ValueError, match="expected_per_combo"):
        verify_snr_balance(data, [0.0], [0], expected_per_combo=0)

def test_verify_snr_balance_missing_key_raises() -> None:
    """Input invalido: chiave 'k' mancante -> ValueError."""
    data = _tiny_data_dict()
    del data["k"]
    with pytest.raises(ValueError, match="'k'"):
        verify_snr_balance(data, [0.0], [0])

def test_expected_per_combo_train_divisible() -> None:
    
    cfg = _generator_config()
    assert _expected_per_combo(cfg, "train", num_combos=28) == 5040 // 28

def test_expected_per_combo_val_test() -> None:
    
    cfg = _generator_config()
    assert _expected_per_combo(cfg, "val", num_combos=28) == 1000
    assert _expected_per_combo(cfg, "test", num_combos=28) == 2000

def test_expected_per_combo_not_divisible_raises() -> None:
    """Input invalido: num_symbols_train not divisible -> ValueError ()."""
    cfg = _generator_config({"num_symbols_train": 1000})
    with pytest.raises(ValueError, match="not divisible"):
        _expected_per_combo(cfg, "train", num_combos=28)

@pytest.mark.parametrize(
    ("split", "num_combos"),
    [
        ("foo", 28),
        ("train", 0),
        ("train", -1),
        ("train", True),
    ],
)
def test_expected_per_combo_invalid_args_raises(
    split: str, num_combos: object
) -> None:
    """Input invalidi: split/num_combos non ammessi -> ValueError."""
    with pytest.raises(ValueError):
        _expected_per_combo(_generator_config(), split, num_combos)

def test_expected_per_combo_odd_raises() -> None:
    """Input invalido: per-combo diseven -> ValueError (bit 0/1 bilanciati)."""
    cfg = _generator_config({"num_symbols_val": 5})
    with pytest.raises(ValueError, match="even"):
        _expected_per_combo(cfg, "val", num_combos=28)

def test_resolve_n_per_combo_train_108_even() -> None:
    
    cfg = _generator_config({"num_symbols_train": 108})
    assert resolve_n_per_combo(cfg, "train", num_combos=6) == 18
    assert resolve_n_per_combo(cfg, "val", num_combos=6) == 1000
    assert resolve_n_per_combo(cfg, "test", num_combos=6) == 2000

def test_resolve_n_per_combo_constructive_message() -> None:
    
    cfg = _generator_config({"num_symbols_train": 102})
    with pytest.raises(ValueError) as exc_info:
        resolve_n_per_combo(cfg, "train", num_combos=6)
    msg = str(exc_info.value)
    assert "n_per_combo (17)" in msg
    assert "2*num_combos=12" in msg
    assert "108" in msg

def test_resolve_n_per_combo_odd_val_hint() -> None:
    """Messaggio costruttivo anche per val/test diseven (suggerisce il even)."""
    cfg = _generator_config({"num_symbols_val": 5})
    with pytest.raises(ValueError) as exc_info:
        resolve_n_per_combo(cfg, "val", num_combos=6)
    msg = str(exc_info.value)
    assert "num_symbols_val even (e.g. 6)" in msg

def _even_per_combo_for_data(data: Dict[str, Any], split: str) -> int:
    """Risolve n_per_combo per un blocco ``data`` reale di experiments.yaml."""
    grid = build_snr_grid(data["snr_range"], data["snr_step"])
    combos = len(grid) * len(data["echoes"])
    return resolve_n_per_combo({"data": data}, split, combos)

def test_all_experiment_configs_even_per_combo() -> None:
    
    from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

    experiments_path = Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"
    base = load_config(experiments_path, DEFAULT_BASE_CONFIG_PATH)
    base_data = dict(base["data"])
    raw = yaml.safe_load(experiments_path.read_text(encoding="utf-8"))

    for exp_id in ("ber_vs_snr", "classical_receivers", "jamming",
                   "jamming_interpretability", "final_report"):
        for mode in ("full", "fast"):
            data = {**base_data, **raw[exp_id][mode]["data"]}
            for split in ("train", "val", "test"):
                n_per_combo = _even_per_combo_for_data(data, split)
                assert n_per_combo > 0 and n_per_combo % 2 == 0, (
                    f"{exp_id}.{mode}.{split}: n_per_combo={n_per_combo} non even"
                )

    for mode in ("fast", "full"):
        block = raw["ber_vs_snr"][mode]
        scenarios = (
            block.get("experiments", {})
            .get("ber_vs_snr", {})
            .get("scenarios", [])
        )
        for scenario in scenarios:
            if not isinstance(scenario, dict) or "name" not in scenario:
                continue
            data = {**base_data, **block["data"]}
            data["echoes"] = list(scenario["echoes"])
            data["max_doppler"] = float(scenario["max_doppler"])
            for split in ("train", "val", "test"):
                n_per_combo = _even_per_combo_for_data(data, split)
                assert n_per_combo > 0 and n_per_combo % 2 == 0, (
                    f"ber_vs_snr.{mode} scenario {scenario['name']}.{split}: "
                    f"n_per_combo={n_per_combo} non even"
                )

def test_generate_dataset_invalid_split_raises(tmp_path: Path) -> None:
    """Input invalido: split non ammesso -> ValueError."""
    with pytest.raises(ValueError, match="split"):
        generate_dataset(_generator_config(), "foo", tmp_path)

def test_generate_dataset_train_not_divisible_raises(tmp_path: Path) -> None:
    """Input invalido: num_symbols_train not divisible -> ValueError ()."""
    cfg = _generator_config({"num_symbols_train": 1000})
    with pytest.raises(ValueError, match="not divisible"):
        generate_dataset(cfg, "train", tmp_path)

def test_generate_dataset_odd_per_combo_raises(tmp_path: Path) -> None:
    """Input invalido: n_per_combo diseven -> ValueError (bit 0/1 bilanciati)."""
    cfg = _generator_config({"num_symbols_val": 5})
    with pytest.raises(ValueError, match="even"):
        generate_dataset(cfg, "val", tmp_path)

def test_integration_generator_to_loader(tmp_path: Path) -> None:
    
    cfg = _generator_config(
        {
            "snr_range": [0.0, 2.0],
            "snr_step": 2.0,
            "echoes": [0],
            "num_symbols_train": 4,
            "num_symbols_val": 4,
            "num_symbols_test": 4,
            "max_delay": 5,
        }
    )
    paths = generate_dataset(cfg, "test", tmp_path)
    assert len(paths) == 2

    loader_cfg = _loader_config(
        {
            "snr_range": [0.0, 2.0],
            "snr_step": 2.0,
            "echoes": [0],
            "num_symbols_test": 4,
            "max_delay": 5,
        }
    )
    data = load_npz_files(tmp_path, [0.0, 2.0], [0], "test", loader_cfg)
    assert data["x"].shape == (8, _N_SEQ)
    assert data["x"].dtype == np.complex128
    assert np.all(np.isfinite(data["x"]))

    counts = verify_snr_balance(data, [0.0, 2.0], [0], expected_per_combo=4)
    assert counts == {(0.0, 0): 4, (2.0, 0): 4}

    targets = normalize_targets(
        data["tau"], data["f_d"], tau_max=5.0, fd_max=_MAX_DOPPLER
    )
    assert targets.shape == (8, 2)
    assert np.all(targets >= 0.0) and np.all(targets <= 1.0)

    ds = build_tf_dataset(data, batch_size=4, config=loader_cfg)
    assert ds.element_spec[0].shape.as_list() == [None, _N_SEQ, 2]
    assert ds.element_spec[1]["sensing"].shape.as_list() == [None, 2]
    batch = next(iter(ds))
    features, labels = batch
    assert tuple(features.shape) == (4, _N_SEQ, 2)
    assert tuple(labels["comm"].shape) == (4,)
    assert tuple(labels["sensing"].shape) == (4, 2)
    assert np.all(np.isfinite(features.numpy()))
    assert np.all(np.isfinite(labels["sensing"].numpy()))

