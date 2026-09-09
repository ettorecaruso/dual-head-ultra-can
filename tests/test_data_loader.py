"""Test per ``src/data/data_loader.py`` (``tests/test_data_loader.py``).

Copre le funzioni pubbliche del modulo (12 test documentati in
``src/data/data_loader.py``): ``load_npz_files`` (roundtrip, combo mancante,
dir vuota), ``normalize_targets`` (estremi -> [0, 1], tau_max=0 -> ValueError),
``build_tf_dataset`` (shape, finitezza, batch invalido, seed riproducibile) e
``verify_snr_balance`` (uniforme OK, unbalanced/expected -> ValueError).

I file ``.npz`` sintetici replicano il formato del generatore
(``dataset_generator.generate_dataset``): ``<split>_snr<snr:g>_echo<k>.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from src.data.data_loader import (
    build_tf_dataset,
    load_npz_files,
    normalize_targets,
    verify_snr_balance,
)

_N_SEQ = 100
_MAX_DELAY = 10
_MAX_DOPPLER = 8e-5

def _write_npz(
    data_dir: Path,
    split: str,
    snr: float,
    k: int,
    n: int,
    config: Dict[str, Any],
    seed: int = 1,
) -> Path:
    
    data = config["data"]
    seq_len = int(data["sequence_length"])
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, seq_len)) + 1j * rng.standard_normal((n, seq_len))).astype(
        np.complex128
    )
    bit = rng.integers(0, 2, size=n).astype(np.uint8)
    tau = rng.uniform(0.0, max_delay, size=n).astype(np.float64)
    f_d = rng.uniform(0.0, max_doppler, size=n).astype(np.float64)
    per_sample_seed = rng.integers(1, 2**31, size=n).astype(np.int64)

    path = data_dir / f"{split}_snr{snr:g}_echo{k}.npz"
    np.savez(
        path,
        x=x,
        bit=bit,
        tau=tau,
        f_d=f_d,
        snr_db=float(snr),
        k=k,
        seed=per_sample_seed,
    )
    return path

def test_load_npz_files_roundtrip(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Caso normale: due file (SNR diversi, K=1) concatenati in un DataDict coerente."""
    _write_npz(tmp_path, "train", 0.0, 1, n=4, config=tiny_config, seed=10)
    _write_npz(tmp_path, "train", 5.0, 1, n=6, config=tiny_config, seed=20)

    data = load_npz_files(tmp_path, [0.0, 5.0], [1], "train", tiny_config)

    assert data["x"].shape == (10, _N_SEQ)
    assert data["x"].dtype == np.complex128
    assert data["bit"].shape == (10,)
    assert data["tau"].shape == (10,)
    assert data["f_d"].shape == (10,)
    assert set(np.unique(data["bit"])).issubset({0, 1})
    assert np.all(data["snr_db"][:4] == 0.0)
    assert np.all(data["snr_db"][4:] == 5.0)
    assert np.all(data["k"] == 1)
    assert np.all(np.isfinite(data["x"]))
    assert np.all(data["tau"] >= 0.0) and np.all(data["tau"] <= _MAX_DELAY)
    assert np.all(data["f_d"] >= 0.0) and np.all(data["f_d"] <= _MAX_DOPPLER)

def test_load_npz_files_missing_combo_raises(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Input invalido: combinazione (SNR, K) mancante -> ValueError."""
    _write_npz(tmp_path, "train", 0.0, 1, n=4, config=tiny_config)
    with pytest.raises(ValueError, match="grid"):
        load_npz_files(tmp_path, [0.0, 5.0], [1], "train", tiny_config)

def test_load_npz_files_empty_dir_raises(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Input invalido: directory senza file per lo split -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no valid .npz file"):
        load_npz_files(tmp_path, [0.0], [1], "train", tiny_config)

def test_load_npz_files_wrong_split_raises(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Input invalido: split inesistente -> ValueError."""
    with pytest.raises(ValueError, match="split"):
        load_npz_files(tmp_path, [0.0], [1], "nope", tiny_config)

def test_normalize_targets_edges(tiny_config: Dict[str, Any]) -> None:
    """Caso limite: estremi tau/f_d -> colonne esattamente [0, 1]."""
    tau = np.array([0.0, _MAX_DELAY], dtype=np.float64)
    f_d = np.array([0.0, _MAX_DOPPLER], dtype=np.float64)
    out = normalize_targets(tau, f_d, tau_max=_MAX_DELAY, fd_max=_MAX_DOPPLER)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0], atol=1e-7)

def test_normalize_targets_zero_max_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: tau_max = 0 -> ValueError (guardia 0-divisione)."""
    tau = np.array([0.0, 1.0])
    f_d = np.array([0.0, 1e-5])
    with pytest.raises(ValueError, match="tau_max"):
        normalize_targets(tau, f_d, tau_max=0.0, fd_max=_MAX_DOPPLER)

def test_normalize_targets_shape_mismatch_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: tau e f_d di lunghezza diversa -> ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        normalize_targets(np.zeros(3), np.zeros(2), tau_max=10.0, fd_max=_MAX_DOPPLER)

def _tiny_data_dict(n: int = 16, config: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
    """DataDict sintetico con ``n`` campioni (x complesso)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, _N_SEQ)) + 1j * rng.standard_normal((n, _N_SEQ))).astype(
        np.complex128
    )
    bit = rng.integers(0, 2, size=n).astype(np.uint8)
    tau = rng.uniform(0.0, _MAX_DELAY, size=n).astype(np.float64)
    f_d = rng.uniform(0.0, _MAX_DOPPLER, size=n).astype(np.float64)
    return {
        "x": x, "bit": bit, "tau": tau, "f_d": f_d,
        "snr_db": np.zeros(n, dtype=np.float64),
        "k": np.zeros(n, dtype=np.int64),
        "seed": rng.integers(1, 2**31, size=n).astype(np.int64),
    }

def test_build_tf_dataset_shapes(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    
    data = _tiny_data_dict(16, tiny_config)
    ds = build_tf_dataset(data, batch_size=4, config=tiny_config)
    features_spec = ds.element_spec[0]
    comm_spec = ds.element_spec[1]["comm"]
    sensing_spec = ds.element_spec[1]["sensing"]
    assert features_spec.shape.as_list() == [None, _N_SEQ, 2]
    assert comm_spec.shape.as_list() == [None]
    assert sensing_spec.shape.as_list() == [None, 2]

def test_build_tf_dataset_no_nan_inf(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Caso normale: un batch -> features e target finiti, sensing in [0, 1]."""
    data = _tiny_data_dict(16, tiny_config)
    ds = build_tf_dataset(data, batch_size=8, config=tiny_config)
    features, labels = next(iter(ds))
    assert np.all(np.isfinite(features.numpy()))
    assert np.all(np.isfinite(labels["comm"].numpy()))
    assert np.all(np.isfinite(labels["sensing"].numpy()))
    assert np.all(labels["sensing"].numpy() >= 0.0)
    assert np.all(labels["sensing"].numpy() <= 1.0 + 1e-6)

def test_build_tf_dataset_invalid_batch_raises(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Input invalidi: batch_size 0 o negativo -> ValueError."""
    data = _tiny_data_dict(16, tiny_config)
    with pytest.raises(ValueError, match="batch_size"):
        build_tf_dataset(data, batch_size=0, config=tiny_config)
    with pytest.raises(ValueError, match="batch_size"):
        build_tf_dataset(data, batch_size=-1, config=tiny_config)

def test_build_tf_dataset_reproducible_seed(tmp_path: Path, tiny_config: Dict[str, Any]) -> None:
    """Caso limite: stesso seed -> stesso primo batch (riproducibilità)."""
    data = _tiny_data_dict(32, tiny_config)
    ds1 = build_tf_dataset(data, batch_size=8, config=tiny_config, shuffle=True, seed=42)
    ds2 = build_tf_dataset(data, batch_size=8, config=tiny_config, shuffle=True, seed=42)
    f1, l1 = next(iter(ds1))
    f2, l2 = next(iter(ds2))
    np.testing.assert_array_equal(f1.numpy(), f2.numpy())
    np.testing.assert_array_equal(l1["comm"].numpy(), l2["comm"].numpy())

def test_verify_snr_balance_ok(tiny_config: Dict[str, Any]) -> None:
    """Caso normale: conteggi uniformi -> dict con conteggi attesi."""
    n_per_combo = 4
    data = {
        "snr_db": np.concatenate([np.full(n_per_combo, s) for s in (0.0, 5.0, 10.0)]),
        "k": np.concatenate([np.full(n_per_combo, 1) for _ in range(3)]),
    }
    counts = verify_snr_balance(data, [0.0, 5.0, 10.0], [1])
    assert len(counts) == 3
    assert all(c == n_per_combo for c in counts.values())

def test_verify_snr_balance_unbalanced_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: conteggi non uniformi -> ValueError."""
    data = {
        "snr_db": np.array([0.0, 0.0, 0.0, 5.0, 5.0]),
        "k": np.array([1, 1, 1, 1, 1]),
    }
    with pytest.raises(ValueError, match="unbalanced"):
        verify_snr_balance(data, [0.0, 5.0], [1])

def test_verify_snr_balance_expected_mismatch_raises(tiny_config: Dict[str, Any]) -> None:
    """Input invalido: conteggi uniformi ma != expected -> ValueError."""
    data = {
        "snr_db": np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]),
        "k": np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    }
    with pytest.raises(ValueError, match="expected"):
        verify_snr_balance(data, [0.0, 5.0], [1], expected_per_combo=5)

def test_verify_snr_balance_missing_combo_raises(tiny_config: Dict[str, Any]) -> None:
    
    data = {
        "snr_db": np.array([0.0, 0.0, 5.0, 5.0]),
        "k": np.array([1, 1, 1, 1]),
    }
    with pytest.raises(ValueError, match="unbalanced"):
        verify_snr_balance(data, [0.0, 5.0, 10.0], [1])


