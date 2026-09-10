"""Dataset generator for the Dual-Head Ultra-CAN architecture (ISAC in IoD).

Implements the chaotic maps and the multi-echo aerial channel used to build the
train/val/test .npz files:

  - Logistic map ``x[n+1] = mu x[n] (1 - x[n])``, ``mu = 3.9``.
  - Bernoulli map (two branches, threshold 0.5).
  - Multi-echo aerial channel.
  - Power normalization ``E[|h_c|^2] + sum(alpha_k^2) = 1``.
  - Sensing labels come from the echo with the largest ``alpha_k``.

Conventions:
  - Integer echo delays ``tau_k`` in ``[1, max_delay]``.
  - ``x0 = uniform(1e-9, 1 - 1e-9)`` derived from the seed stored in the .npz.
  - Bit 0 uses ``map_type``; bit 1 uses the complementary map.
  - ``num_symbols_val/test`` are per (SNR, K) point.
  - K=0 (smoke tests only): no echoes, null sensing labels.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, save_config_snapshot
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_LOGISTIC_MU_MIN = 3.57
_LOGISTIC_MU_MAX = 4.0
_ALPHA_COUPLING_FLOOR = 1e-3
_MAP_TYPES = frozenset({"logistic", "bernoulli"})
_MAP_RETRIES = 5
_X0_MIN = 1e-9
_X0_MAX = 1.0 - 1e-9
_ENERGY_EPS = 1e-12
_POWER_EPS = 1e-9
_FIXED_POINT_TOL = 1e-12
_SPREAD_EPS = 1e-9
_SPLITS = frozenset({"train", "val", "test"})
_SEED_HIGH = 2**63 - 1

_BERNOULLI_MULTIPLIER = 5
_BERNOULLI_MASK = (1 << 64) - 1
_BERNOULLI_SCALE = float(1 << 64)

@dataclass(frozen=True)
class EchoParams:

    tau: int
    f_doppler: float
    alpha: float

    def __post_init__(self) -> None:
        if not isinstance(self.tau, int) or self.tau < 1:
            raise ValueError(f"tau must be an int >= 1, got: {self.tau!r}")
        if not math.isfinite(self.f_doppler) or not (0.0 <= self.f_doppler < 0.5):
            raise ValueError(f"f_doppler must be in [0, 0.5), got: {self.f_doppler!r}")
        if not math.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got: {self.alpha!r}")

@dataclass(frozen=True)
class DirectPathParams:

    h_c: complex
    f_dc: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.h_c.real) and math.isfinite(self.h_c.imag)):
            raise ValueError(f"h_c not finite: {self.h_c!r}")
        if not math.isfinite(self.f_dc) or not (0.0 <= self.f_dc < 0.5):
            raise ValueError(f"f_dc must be in [0, 0.5), got: {self.f_dc!r}")

def _iter_config_leaves(config: Any, prefix: str = "config") -> Iterator[Tuple[str, Any]]:
    if isinstance(config, dict):
        for key, value in config.items():
            yield from _iter_config_leaves(value, f"{prefix}.{key}")
    elif isinstance(config, (list, tuple)):
        for index, value in enumerate(config):
            yield from _iter_config_leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, config

def _assert_config_finite(config: Any) -> None:
    for path, value in _iter_config_leaves(config):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite value in the config: {path} = {value!r}")

_REQUIRED_DATA_KEYS: Tuple[str, ...] = (
    "sequence_length", "map_type", "map_param", "fc_hz", "fs_hz",
    "snr_range", "snr_step", "echoes", "max_delay", "max_doppler",
    "alpha_min", "alpha_max", "num_symbols_train", "num_symbols_val",
    "num_symbols_test", "raw_dir", "processed_dir",
    "rician_kappa_db", "doppler_direct_max",
)

def _validate_config(config: Dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got: {type(config).__name__}")
    _assert_config_finite(config)
    data = config.get("data")
    general = config.get("general")
    if not isinstance(data, dict):
        raise ValueError("'data' section missing or not a dict in the config")
    if not isinstance(general, dict):
        raise ValueError("'general' section missing or not a dict in the config")
    missing = [key for key in _REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise ValueError(f"missing keys in config['data']: {missing}")
    if "seed" not in general:
        raise ValueError("missing key: general.seed")

    sequence_length = int(data["sequence_length"])
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got: {sequence_length}")

    map_type = str(data["map_type"])
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")

    map_param = float(data["map_param"])
    if not math.isfinite(map_param):
        raise ValueError(f"map_param not finite: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= map_param <= _LOGISTIC_MU_MAX):
        raise ValueError(
            f"map_param (mu) outside the chaotic regime "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )

    for key in ("fc_hz", "fs_hz", "max_doppler", "doppler_direct_max", "alpha_min", "alpha_max"):
        value = float(data[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"data.{key} must be finite and > 0, got: {value!r}")

    if float(data["max_doppler"]) >= 0.5:
        raise ValueError("data.max_doppler must be < 0.5 (guardia anti-aliasing)")
    if float(data["doppler_direct_max"]) > float(data["max_doppler"]):
        raise ValueError("data.doppler_direct_max must be <= data.max_doppler")

    if float(data["alpha_min"]) >= float(data["alpha_max"]):
        raise ValueError("data.alpha_min must be < data.alpha_max")

    rician_kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(rician_kappa_db) or rician_kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db must be >= 0, got: {rician_kappa_db!r}")

    snr_range = data["snr_range"]
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"data.snr_range must be [min, max], got: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min >= snr_max:
        raise ValueError(f"data.snr_range invalido: {snr_range!r}")

    snr_step = float(data["snr_step"])
    if not math.isfinite(snr_step) or snr_step <= 0.0:
        raise ValueError(f"data.snr_step must be > 0, got: {snr_step!r}")

    max_delay = int(data["max_delay"])
    if max_delay <= 0:
        raise ValueError(f"data.max_delay must be > 0, got: {max_delay}")
    if max_delay > sequence_length:
        raise ValueError(
            f"data.max_delay ({max_delay}) must be <= sequence_length ({sequence_length})"
        )

    echoes = data["echoes"]
    if not isinstance(echoes, (list, tuple)) or len(echoes) == 0:
        raise ValueError(f"data.echoes must be a non-empty list, got: {echoes!r}")
    for k in echoes:
        if not isinstance(k, (int,)) or isinstance(k, bool) or k < 0:
            raise ValueError(f"data.echoes must contain ints >= 0, got: {k!r}")
        if k > max_delay:
            raise ValueError(
                f"k={k} > max_delay={max_delay}: cannot sample {k} distinct delays"
            )

    for key in ("num_symbols_train", "num_symbols_val", "num_symbols_test"):
        value = data[key]
        if not isinstance(value, (int,)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"data.{key} must be an int > 0, got: {value!r}")

    for key in ("raw_dir", "processed_dir"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{key} must be a non-empty string, got: {value!r}")

    seed = general["seed"]
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"general.seed must be an int >= 0, got: {seed!r}")

    logger.debug(
        "config validata: seq_len=%d, map_type=%s, mu=%s, max_delay=%d, max_doppler=%s",
        sequence_length, map_type, map_param, max_delay, data["max_doppler"],
    )

def _iterate_map(map_type: str, map_param: float, x0: float, sequence_length: int) -> np.ndarray:
    seq = np.empty(sequence_length, dtype=np.float64)
    if map_type == "logistic":
        x = x0
        for i in range(sequence_length):
            seq[i] = x
            if i + 1 == sequence_length:
                break
            x = map_param * x * (1.0 - x)
    else:
        state = int(np.float64(x0).view(np.uint64)) & _BERNOULLI_MASK
        for i in range(sequence_length):
            seq[i] = state / _BERNOULLI_SCALE
            state = (state * _BERNOULLI_MULTIPLIER) & _BERNOULLI_MASK
    return seq

def generate_chaotic_sequence(
    map_type: str,
    map_param: float,
    seed: int,
    sequence_length: int,
) -> np.ndarray:
    if not isinstance(map_type, str) or map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")
    if not isinstance(map_param, (int, float)) or not math.isfinite(float(map_param)):
        raise ValueError(f"map_param must be a finite float, got: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= float(map_param) <= _LOGISTIC_MU_MAX):
        raise ValueError(
            f"map_param (mu) outside the chaotic regime "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be an int >= 0, got: {seed!r}")
    if not isinstance(sequence_length, (int,)) or int(sequence_length) <= 0:
        raise ValueError(f"sequence_length must be an int > 0, got: {sequence_length!r}")

    n = int(sequence_length)
    mu = float(map_param)
    x0 = _x0_from_seed(int(seed), map_type, mu)
    seq = _iterate_map(map_type, mu, x0, n)
    energy = float(np.sum(seq ** 2))
    spread = float(np.max(seq) - np.min(seq))
    valid = (
        np.all(np.isfinite(seq))
        and np.all(seq >= 0.0)
        and np.all(seq <= 1.0)
        and energy > _ENERGY_EPS
        and spread > _SPREAD_EPS
        and np.count_nonzero(seq) >= max(2, n // 10)
    )
    if valid:
        logger.debug(
            "map %s generated (x0=%.6f, energy=%.3e)",
            map_type, x0, energy,
        )
        return np.asarray(seq, dtype=np.float64)

    raise RuntimeError(
        f"map {map_type!r} degenerate: NaN/Inf or zero energy"
    )

def sample_echo_parameters(
    k: int,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> List[EchoParams]:
    if not isinstance(k, (int,)) or isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")
    k = int(k)
    if k == 0:
        return []
    data = config["data"]
    sequence_length = int(data["sequence_length"])
    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    alpha_min = float(data["alpha_min"])
    alpha_max = float(data["alpha_max"])
    if max_delay > sequence_length:
        raise ValueError(f"max_delay ({max_delay}) > sequence_length ({sequence_length})")
    if k > max_delay:
        raise ValueError(
            f"k ({k}) > max_delay ({max_delay}): cannot sample "
            f"{k} distinct integer delays in [1, max_delay]"
        )

    taus = rng.choice(np.arange(1, max_delay + 1, dtype=np.int64), size=k, replace=False)
    dopplers = rng.uniform(0.0, max_doppler, size=k)
    if bool(data.get("alpha_tau_coupling", False)):
        floor = float(data.get("alpha_floor", _ALPHA_COUPLING_FLOOR))
        alphas = np.clip(alpha_max / np.maximum(taus.astype(np.float64), 1.0),
                         floor, alpha_max)
    else:
        alphas = 10.0 ** rng.uniform(math.log10(alpha_min), math.log10(alpha_max), size=k)

    echoes = [
        EchoParams(tau=int(taus[i]), f_doppler=float(dopplers[i]), alpha=float(alphas[i]))
        for i in range(k)
    ]
    if not all(math.isfinite(e.f_doppler) and math.isfinite(e.alpha) for e in echoes):
        raise ValueError("echo parameters not finite during sampling")
    logger.debug(
        "campionati %d echi: tau=%s, fD=%s, alpha=%s",
        k, [e.tau for e in echoes], [e.f_doppler for e in echoes], [e.alpha for e in echoes],
    )
    return echoes

def sample_direct_path(rng: np.random.Generator, config: Dict[str, Any]) -> DirectPathParams:
    data = config["data"]
    kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(kappa_db) or kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db must be >= 0, got: {kappa_db!r}")
    doppler_direct_max = float(data["doppler_direct_max"])

    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    g = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = float(rng.uniform(0.0, doppler_direct_max))
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag) and math.isfinite(f_dc)):
        raise RuntimeError("direct path sampling not finite")
    logger.debug("path diretto: abs(h_c)=%.4f (kappa=%.1f dB), f_Dc=%.3e", abs(h_c), kappa_db, f_dc)
    return DirectPathParams(h_c=h_c, f_dc=f_dc)

def apply_aerial_channel(
    x: np.ndarray,
    h_c: complex,
    f_dc: float,
    echoes: List[EchoParams],
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    if not isinstance(echoes, list):
        raise TypeError(f"echoes must be list[EchoParams], got: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x must be a non-empty 1D vector, got: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contiene NaN/Inf")
    if not (math.isfinite(f_dc) and 0.0 <= f_dc < 0.5):
        raise ValueError(f"f_dc must be in [0, 0.5), got: {f_dc!r}")
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag)):
        raise ValueError(f"h_c not finite: {h_c!r}")
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db must be finite, got: {snr_db!r}")

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    echo_power = 0.0
    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"element of echoes is not an EchoParams: {echo!r}")
        if echo.tau > n_samples:
            raise ValueError(
                f"echo with tau={echo.tau} beyond the observation window (N={n_samples})"
            )
        echo_power += float(echo.alpha) ** 2
    if echo_power >= 1.0 - _POWER_EPS:
        raise ValueError(
            f"sum of alpha_k^2 = {echo_power:.6f} >= 1: normalization Eq. (4) is impossible"
        )

    h_c_eff = h_c * math.sqrt(1.0 - echo_power)

    y_clean = h_c_eff * x_arr * np.exp(1j * 2.0 * math.pi * f_dc * n_idx)
    for echo in echoes:
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo.alpha * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    signal_power = float(np.mean(np.abs(y_clean) ** 2))
    if not math.isfinite(signal_power) or signal_power <= 0.0:
        raise RuntimeError(f"invalid signal power: {signal_power!r} (degenerate symbol)")

    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    w = math.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    y = y_clean + w

    if not np.all(np.isfinite(y)) or not math.isfinite(noise_var):
        raise RuntimeError("channel output not finite (NaN/Inf)")

    realized_snr = 10.0 * math.log10(signal_power / noise_var) if noise_var > 0.0 else math.inf
    logger.debug(
        "channel: P_s=%.3e, noise_var=%.3e, realized SNR=%.2f dB (requested %.1f dB)",
        signal_power, noise_var, realized_snr, snr_db,
    )
    return y, noise_var

def apply_echo_only_channel(
    x: np.ndarray,
    echoes: List[EchoParams],
) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a np.ndarray, got: {type(x).__name__}")
    if not isinstance(echoes, list):
        raise TypeError(f"echoes must be list[EchoParams], got: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x must be a non-empty 1D vector, got: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contiene NaN/Inf")

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    y_clean = x_arr.astype(np.complex128)

    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"element of echoes is not an EchoParams: {echo!r}")
        if echo.tau > n_samples:
            raise ValueError(
                f"echo with tau={echo.tau} beyond the observation window (N={n_samples})"
            )
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo.alpha * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    if not np.all(np.isfinite(y_clean)):
        raise RuntimeError(
            "echo-only channel output not finite (NaN/Inf)"
        )

    logger.debug(
        "echo-only channel applied: N=%d, K=%d echoes, finite output",
        n_samples, len(echoes),
    )
    return y_clean

def build_snr_grid(snr_range: List[float], snr_step: float) -> List[float]:
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"snr_range must be [min, max], got: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    step = float(snr_step)
    if not (math.isfinite(snr_min) and math.isfinite(snr_max) and math.isfinite(step)):
        raise ValueError(f"non-finite values in snr_range/snr_step: {snr_range!r}, {snr_step!r}")
    if snr_min >= snr_max:
        raise ValueError(f"snr_min ({snr_min}) must be < snr_max ({snr_max})")
    if step <= 0.0:
        raise ValueError(f"snr_step must be > 0, got: {step}")

    n_points = int(math.ceil((snr_max - snr_min) / step - 1e-9))
    if n_points < 1:
        raise ValueError(f"empty SNR grid for {snr_range!r} with step {step}")
    points = [snr_min + i * step for i in range(n_points)]
    if abs(points[-1] - snr_max) > 1e-9:
        points.append(snr_max)
    logger.debug("inclusive SNR grid: %s", points)
    return points

def _map_type_for_bit(map_type: str, bit: int) -> str:
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}")
    if bit not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got: {bit!r}")
    if bit == 0:
        return map_type
    return "bernoulli" if map_type == "logistic" else "logistic"

def _dominant_echo_index(echoes: List[EchoParams]) -> int:
    if not echoes:
        raise ValueError("empty echoes: no dominant echo")
    return int(np.argmax([echo.alpha for echo in echoes]))

def resolve_n_per_combo(
    config: Dict[str, Any],
    split: str,
    num_combos: int,
) -> int:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got: {split!r}")
    if not isinstance(num_combos, int) or isinstance(num_combos, bool) or num_combos <= 0:
        raise ValueError(f"num_combos must be an int > 0, got: {num_combos!r}")

    data = config["data"]
    if split == "train":
        total = int(data["num_symbols_train"])
        if total % num_combos != 0:
            nearest_down = num_combos * (total // num_combos)
            nearest_up = nearest_down + num_combos
            raise ValueError(
                f"num_symbols_train ({total}) not divisible by num_combos ({num_combos}): "
                "balancing per (SNR, K) is impossible. "
                f"Valid nearby values: {nearest_down} or {nearest_up}."
            )
        n_per_combo = total // num_combos
    elif split == "val":
        n_per_combo = int(data["num_symbols_val"])
    else:
        n_per_combo = int(data["num_symbols_test"])

    if n_per_combo <= 0:
        raise ValueError(
            f"symbols per combination must be > 0, got: {n_per_combo}"
        )
    if n_per_combo % 2 != 0:
        if split == "train":
            unit = 2 * num_combos
            suggestion = unit * ((total + unit - 1) // unit)
            hint = (
                f"use a num_symbols_train multiple of "
                f"2*num_combos={unit} (valid nearby value: {suggestion})"
            )
        else:
            key = "num_symbols_val" if split == "val" else "num_symbols_test"
            hint = f"for split '{split}' use {key} even (e.g. {n_per_combo + 1})"
        raise ValueError(
            f"n_per_combo ({n_per_combo}) must be even to balance the 0/1 bits "
            f"for each (SNR, K) point: {hint}"
        )
    return n_per_combo

def _center_normalize_symbol(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    x_centered = x_arr - np.mean(x_arr)
    energy = float(np.sqrt(np.sum(x_centered ** 2)))
    if not math.isfinite(energy) or energy <= _ENERGY_EPS:
        raise RuntimeError(
            "degenerate symbol after centering/normalization (zero energy)"
        )
    return x_centered / energy



def _x0_from_seed(seed: int, map_type: str, map_param: float) -> float:
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}"
        )
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be an int >= 0, got: {seed!r}")
    seed = int(seed)
    mu = float(map_param)
    if map_type == "logistic":
        forbidden = (0.25, 0.5, 0.75, 1.0 - 1.0 / mu)
    else:
        forbidden = (0.0, 0.5, 1.0)

    rng_map = np.random.default_rng(seed)
    for _attempt in range(1, _MAP_RETRIES + 1):
        x0 = float(rng_map.uniform(_X0_MIN, _X0_MAX))
        if any(abs(x0 - point) < _FIXED_POINT_TOL for point in forbidden):
            continue
        return x0

    raise RuntimeError(
        f"map {map_type!r} degenerate after {_MAP_RETRIES} attempts: x0 on a fixed point"
    )


def _iterate_map_batch(
    map_type: str,
    map_param: float,
    x0: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    x0_arr = np.asarray(x0, dtype=np.float64)
    if x0_arr.ndim != 1:
        raise ValueError(f"x0 must be 1D, got: {x0_arr.shape}")
    n = int(x0_arr.shape[0])
    if n == 0:
        return np.empty((0, sequence_length), dtype=np.float64)
    seq = np.empty((n, sequence_length), dtype=np.float64)

    if map_type == "logistic":
        x = x0_arr.copy()
        for i in range(sequence_length):
            seq[:, i] = x
            if i + 1 == sequence_length:
                break
            x = map_param * x * (1.0 - x)
    else:
        state = np.asarray(x0_arr, dtype=np.float64).view(np.uint64) & _BERNOULLI_MASK
        for i in range(sequence_length):
            seq[:, i] = state / _BERNOULLI_SCALE
            state = (state * _BERNOULLI_MULTIPLIER) & _BERNOULLI_MASK
    return seq


def _map_types_for_bits(map_type: str, bits: np.ndarray) -> np.ndarray:
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type must be one of {sorted(_MAP_TYPES)}, got: {map_type!r}"
        )
    bits_arr = np.asarray(bits)
    if not np.all(np.isin(bits_arr, (0, 1))):
        raise ValueError(f"bits must contain only 0/1, got: {bits_arr!r}")
    if map_type == "logistic":
        return np.where(bits_arr == 0, "logistic", "bernoulli")
    return np.where(bits_arr == 0, "bernoulli", "logistic")


def _center_normalize_batch(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D (N, N_seq), got: {x_arr.shape}")
    x_centered = x_arr - np.mean(x_arr, axis=1, keepdims=True)
    energy = np.sqrt(np.sum(x_centered ** 2, axis=1, keepdims=True))
    if (not np.all(np.isfinite(x_centered))) or np.any(energy <= _ENERGY_EPS):
        raise RuntimeError(
            "degenerate batch after centering/normalization (zero energy or NaN/Inf)"
        )
    return x_centered / energy


def generate_transmitted_batch(
    config: Dict[str, Any],
    bits: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    if bits_arr.ndim != 1 or seeds_arr.ndim != 1 or bits_arr.shape != seeds_arr.shape:
        raise ValueError(
            f"bits/seeds must be aligned 1D arrays, got: "
            f"{bits_arr.shape}, {seeds_arr.shape}"
        )
    n = int(bits_arr.shape[0])
    if n == 0:
        return np.empty((0, seq_len), dtype=np.float64)

    maps = _map_types_for_bits(map_type, bits_arr)
    out = np.empty((n, seq_len), dtype=np.float64)
    for mt in sorted(_MAP_TYPES):
        idx = np.where(maps == mt)[0]
        if idx.size == 0:
            continue
        x0 = np.array(
            [_x0_from_seed(int(s), mt, map_param) for s in seeds_arr[idx]],
            dtype=np.float64,
        )
        out[idx] = _iterate_map_batch(mt, map_param, x0, seq_len)

    return _center_normalize_batch(out)


def generate_transmitted_batch_fast(
    config: Dict[str, Any],
    bits: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    if bits_arr.ndim != 1:
        raise ValueError(f"bits must be 1D, got: {bits_arr.shape}")
    n = int(bits_arr.shape[0])
    if n == 0:
        return np.empty((0, seq_len), dtype=np.float64)

    maps = _map_types_for_bits(map_type, bits_arr)
    out = np.empty((n, seq_len), dtype=np.float64)
    for mt in sorted(_MAP_TYPES):
        idx = np.where(maps == mt)[0]
        if idx.size == 0:
            continue
        x0 = rng.uniform(_X0_MIN, _X0_MAX, size=idx.size)
        if mt == "logistic":
            forbidden = (0.25, 0.5, 0.75, 1.0 - 1.0 / float(map_param))
        else:
            forbidden = (0.0, 0.5, 1.0)
        for point in forbidden:
            near = np.abs(x0 - point) < _FIXED_POINT_TOL
            if np.any(near):
                x0 = x0 + np.where(near, _FIXED_POINT_TOL * 10.0, 0.0)
        out[idx] = _iterate_map_batch(mt, map_param, x0, seq_len)

    return _center_normalize_batch(out)


def apply_channel_batch(
    x_norm: np.ndarray,
    k: int,
    snr_db: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_norm, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x_norm must be 2D (N, N_seq), got: {x_arr.shape}")
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")
    k = int(k)
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db must be finite, got: {snr_db!r}")

    data = config["data"]
    seq_len = int(data["sequence_length"])
    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])
    alpha_min = float(data["alpha_min"])
    alpha_max = float(data["alpha_max"])
    kappa_db = float(data["rician_kappa_db"])
    doppler_direct_max = float(data["doppler_direct_max"])
    if k > max_delay:
        raise ValueError(
            f"k ({k}) > max_delay ({max_delay}): cannot sample "
            f"{k} distinct integer delays in [1, max_delay]"
        )

    n = int(x_arr.shape[0])
    n_idx = np.arange(seq_len, dtype=np.float64)[None, :]
    n_idx_i = np.arange(seq_len, dtype=np.int64)[None, :]

    if k > 0:
        pool = np.tile(np.arange(1, max_delay + 1, dtype=np.int64)[None, :], (n, 1))
        taus = rng.permuted(pool, axis=1)[:, :k]
        dopplers = rng.uniform(0.0, max_doppler, size=(n, k))
        if bool(data.get("alpha_tau_coupling", False)):
            floor = float(data.get("alpha_floor", _ALPHA_COUPLING_FLOOR))
            alphas = np.clip(alpha_max / np.maximum(taus.astype(np.float64), 1.0),
                             floor, alpha_max)
        else:
            alphas = 10.0 ** rng.uniform(
                math.log10(alpha_min), math.log10(alpha_max), size=(n, k)
            )
        echo_power = np.sum(alphas ** 2, axis=1)
        if np.any(echo_power >= 1.0 - _POWER_EPS):
            raise ValueError("sum of alpha_k^2 >= 1: normalization Eq. (4) is impossible")
    else:
        taus = np.empty((n, 0), dtype=np.int64)
        dopplers = np.empty((n, 0), dtype=np.float64)
        alphas = np.empty((n, 0), dtype=np.float64)
        echo_power = np.zeros(n, dtype=np.float64)

    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = rng.uniform(0.0, doppler_direct_max, size=n)

    h_c_eff = h_c * np.sqrt(1.0 - echo_power)

    y = h_c_eff[:, None] * x_arr * np.exp(1j * 2.0 * math.pi * f_dc[:, None] * n_idx)
    for j in range(k):
        delay_idx = n_idx_i - taus[:, j, None]
        valid = delay_idx >= 0
        clip_idx = np.clip(delay_idx, 0, seq_len - 1)
        x_delayed = np.where(
            valid, np.take_along_axis(x_arr, clip_idx, axis=1), 0.0
        )
        phase = np.exp(1j * 2.0 * math.pi * dopplers[:, j, None] * n_idx)
        y = y + alphas[:, j, None] * x_delayed * phase

    signal_power = np.mean(np.abs(y) ** 2, axis=1)
    if np.any(signal_power <= 0.0) or not np.all(np.isfinite(signal_power)):
        raise RuntimeError("invalid signal power (degenerate symbol)")
    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    w = np.sqrt(noise_var[:, None] / 2.0) * (
        rng.standard_normal((n, seq_len)) + 1j * rng.standard_normal((n, seq_len))
    )
    y = y + w

    if not np.all(np.isfinite(y)):
        raise RuntimeError("channel output not finite (NaN/Inf)")

    if k > 0:
        dom_idx = np.argmax(alphas, axis=1)
        tau_labels = taus[np.arange(n), dom_idx].astype(np.float64)
        f_d_labels = dopplers[np.arange(n), dom_idx].astype(np.float64)
    else:
        tau_labels = np.zeros(n, dtype=np.float64)
        f_d_labels = np.zeros(n, dtype=np.float64)

    return y, tau_labels, f_d_labels


def generate_test_batch(
    config: Dict[str, Any],
    num_symbols: int,
    snr_db: float,
    k: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if isinstance(num_symbols, bool) or int(num_symbols) <= 0:
        raise ValueError(
            f"num_symbols must be an int > 0, got: {num_symbols!r}"
        )
    n = int(num_symbols)
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k must be an int >= 0, got: {k!r}")

    bits = rng.integers(0, 2, size=n, dtype=np.int64)
    seeds = rng.integers(0, _SEED_HIGH, size=n, dtype=np.int64)
    x_ref = generate_transmitted_batch_fast(config, bits, rng)
    y, tau, f_d = apply_channel_batch(x_ref, int(k), float(snr_db), config, rng)
    return {
        "x": y,
        "bit": bits,
        "tau": tau,
        "f_d": f_d,
        "seed": seeds,
        "x_ref": x_ref.astype(np.float32),
    }


def generate_dataset(
    config: Dict[str, Any],
    split: str,
    output_dir: Path,
) -> List[Path]:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}, got: {split!r}")
    _validate_config(config)
    data = config["data"]
    snr_grid = build_snr_grid(data["snr_range"], data["snr_step"])
    k_list = [int(k) for k in data["echoes"]]
    num_combos = len(snr_grid) * len(k_list)

    n_per_combo = resolve_n_per_combo(config, split, num_combos)

    max_delay = int(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    rng_root = np.random.default_rng(int(config["general"]["seed"]))
    combo_seeds = rng_root.integers(0, _SEED_HIGH, size=num_combos)

    paths: List[Path] = []
    combo_index = 0

    for snr in snr_grid:
        for k in k_list:
            rng = np.random.default_rng(int(combo_seeds[combo_index]))
            combo_index += 1

            bits = np.array([0] * (n_per_combo // 2) + [1] * (n_per_combo // 2), dtype=np.uint8)
            rng.shuffle(bits)

            seed_sym = rng.integers(0, _SEED_HIGH, size=n_per_combo, dtype=np.int64)
            x_ref = generate_transmitted_batch(config, bits, seed_sym)
            x_arr, tau, f_d = apply_channel_batch(x_ref, k, float(snr), config, rng)

            if not np.all(np.isfinite(x_arr)):
                raise RuntimeError(f"array x not finite for {split} snr={snr} k={k}")
            if np.any(tau < 0.0) or np.any(tau > float(max_delay)):
                raise RuntimeError(f"label tau out of range for {split} snr={snr} k={k}")
            if np.any(f_d < 0.0) or np.any(f_d > max_doppler):
                raise RuntimeError(f"label f_d out of range for {split} snr={snr} k={k}")
            if not np.all(np.isin(bits, (0, 1))):
                raise RuntimeError(f"bit outside 0/1 for {split} snr={snr} k={k}")

            file_name = f"{split}_snr{snr:g}_echo{k}.npz"
            out_path = Path(output_dir) / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                x=x_arr,
                bit=bits,
                tau=tau,
                f_d=f_d,
                snr_db=float(snr),
                k=k,
                seed=seed_sym,
            )
            logger.info(
                "saved %s: %d symbols, x shape=%s, tau in [%.0f, %.0f], f_d in [%.3e, %.3e]",
                out_path, n_per_combo, x_arr.shape,
                float(np.min(tau)), float(np.max(tau)),
                float(np.min(f_d)), float(np.max(f_d)),
            )
            paths.append(out_path)

    logger.info(
        "split '%s': %d files generated, %d symbols per combination (combos=%d)",
        split, len(paths), n_per_combo, num_combos,
    )
    return paths

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dataset generator for the Dual-Head Ultra-CAN (ISAC in IoD)"
    )
    parser.add_argument("--config", required=True, help="path to the experiment config (YAML)")
    parser.add_argument(
        "--splits", default="train,val,test", help="split da generare, separati da virgola"
    )
    parser.add_argument(
        "--output-dir", default=None, help="override della dir di output (default: data.raw_dir)"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path=config_path, base_config_path=DEFAULT_BASE_CONFIG_PATH)
    _validate_config(config)

    experiment_name = str(config["general"].get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(config["general"].get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)
    save_config_snapshot(config, log_dir)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits does not contain valid splits")
    for split in splits:
        if split not in _SPLITS:
            raise ValueError(f"invalid split: {split!r} (expected: {sorted(_SPLITS)})")

    from src.utils.dataset_utils import get_dataset_dir

    output_dir = Path(args.output_dir) if args.output_dir else get_dataset_dir(config)
    logger.info(
        "starting the dataset generation: splits=%s, output_dir=%s, seed=%s",
        splits, output_dir, config["general"]["seed"],
    )
    for split in splits:
        generate_dataset(config, split, output_dir)
    logger.info("dataset generation completed for %s", splits)

if __name__ == "__main__":
    main()

