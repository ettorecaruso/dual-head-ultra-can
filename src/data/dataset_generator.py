"""Generatore del dataset per la architettura Dual-Head Ultra-CAN (ISAC in IoD).

  - Docstring iniziale con lo scopo del modulo                     [questo blocco]
  - Type hints su tutti i parametri e i ritorni
  - Logging strutturato (INFO eventi principali, DEBUG shape e valori)
  - Docstring Google-style su ogni funzione pubblica
  - Formule del paper commentate con riferimento Sezione e Equazione
  - Shape-check e range-check (nessun NaN/Inf)
  - Nessun parametro hard-coded: tutto da YAML via ``src.utils.config_loader``
    con verifica fail-fast all'avvio (``_validate_config``)

Formule implementate (paper ``latex.txt``):
  - Eq. (1)  Sez. III-A : mappa logistica  x[n+1] = mu x[n] (1 - x[n]), mu = 3.9
  - Eq. (2)  Sez. III-A : mappa di Bernoulli (due rami, soglia 0.5)
  - Eq. (3)  Sez. III-B : canale aereo multi-eco
  - Eq. (4)  Sez. III-B : normalizzazione potenza E[abs(h_c)^2] + Somma alpha_k^2 = 1
  - Tab. I   Sez. III-C : range fisici (tau, f_D, alpha, SNR)
  - Sez. IV-C           : regressione sul "dominant target" (eco piu' forte)

Convenzioni adottate (risoluzione dei FLAG della pianificazione):
  - fail-fast su non-divisibilita' di num_symbols_train per le
    combinazioni (SNR, K) e parita' del conteggio per-combo; unica fonte di
    verita' in ``resolve_n_per_combo`` (config corrette: base=5040, fast=108).
  - ritardi interi tau_k in [1, max_delay] (Tab. I: 0.7 -> 1).
  - label sensing = parametri dell'eco con alpha_k massimo.
  - echi come copie scalate deterministiche (Eq. (3)/(4)).
  - f_Dc estratto in [0, data.doppler_direct_max] (chiave YAML).
  - SNR per bit = Es/N0 con Es misurata per simbolo:
             noise_var = P_s * 10^(-SNR_dB/10).
  - x0 = uniform(1e-9, 1-1e-9) derivato dal seed salvato nell'.npz.
  - bit 0 -> mappa ``map_type``, bit 1 -> mappa complementare.
  - num_symbols_val/test sono PER OGNI punto (SNR, K).
  - la validazione di map_param in [3.57, 4] vale solo per logistic.
  - raw_dir / processed_dir sono chiavi di base_config.yaml.
  - K=0 (solo smoke test): nessun eco, label sensing nulle.
  - l'.npz salva x complesso (complex128); real vs I/Q spetta al
             data_loader (sottofase 1.2).
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

# --- Bootstrap del path del repository (esecuzione come script) -------------
# Consente `python src/data/dataset_generator.py` da qualsiasi CWD: aggiunge la
# root del repo (paper/) a sys.path PRIMA degli import dei moduli interni.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config, save_config_snapshot
from src.utils.logger import log_config_summary, setup_logging

# Logger di modulo (`` Sez. 5).
logger = logging.getLogger(__name__)

# --- Costanti matematiche del paper (non parametri esperimento) -------------
_LOGISTIC_MU_MIN = 3.57          # Eq. (1) Sez. III-A: soglia del caos
_LOGISTIC_MU_MAX = 4.0           # Eq. (1) Sez. III-A: limite superiore
# Floor dell'ampiezza quando data.alpha_tau_coupling=true (alpha = alpha_max/tau):
# gli echi molto lontani restano deboli ma finiti (proxy radar ~1/range).
_ALPHA_COUPLING_FLOOR = 1e-3
_MAP_TYPES = frozenset({"logistic", "bernoulli"})
_MAP_RETRIES = 5                 # tentativi anti-degenerazione della mappa
_X0_MIN = 1e-9                   # esclude x0 = 0 (punto fisso)
_X0_MAX = 1.0 - 1e-9             # esclude x0 = 1 (punto fisso)
_ENERGY_EPS = 1e-12              # energia minima del simbolo (guardia NaN/Inf)
_POWER_EPS = 1e-9                # tolleranza normalizzazione potenza (Eq. (4))
_FIXED_POINT_TOL = 1e-12         # tolleranza sui punti fissi
_SPREAD_EPS = 1e-9               # variazione minima del segnale
_SPLITS = frozenset({"train", "val", "test"})
_SEED_HIGH = 2**63 - 1           # estremo esclusivo per i seed interi

# --- Mappa di Bernoulli (Eq. (2)) in precisione finita -----------------------
# La forma binaria x -> (2x mod 1) e' nilpotente su Z/2^64: ogni x0 in float64 e'
# un razionale diadico e l'orbita dello shift finisce a 0 in <= 64 passi (coda
# di zeri nel frame). Si usa la forma moltiplicativa equivalente in base a
# (dispari): x -> (a*x mod 1), una permutazione dello spazio di stato (shift di
# Bernoulli in base a, densita' invariante uniforme, nessun collasso).
_BERNOULLI_MULTIPLIER = 5        # moltiplicatore dispari (permutazione su Z/2^64)
_BERNOULLI_MASK = (1 << 64) - 1  # wrap mod 2^64
_BERNOULLI_SCALE = float(1 << 64)

@dataclass(frozen=True)
class EchoParams:
    """Parametri di un singolo eco del canale aereo (Eq. (3), Tab. I).

    Attributes:
        tau: Ritardo intero in campioni, in [1, max_delay] ().
        f_doppler: Doppler normalizzato (cycles/sample), in [0, 0.5).
        alpha: Attenuazione del percorso (free-space path loss), in (0, 1).
    """

    tau: int
    f_doppler: float
    alpha: float

    def __post_init__(self) -> None:
        if not isinstance(self.tau, int) or self.tau < 1:
            raise ValueError(f"tau deve essere int >= 1, ricevuto: {self.tau!r}")
        if not math.isfinite(self.f_doppler) or not (0.0 <= self.f_doppler < 0.5):
            raise ValueError(f"f_doppler deve essere in [0, 0.5), ricevuto: {self.f_doppler!r}")
        if not math.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha deve essere in (0, 1), ricevuto: {self.alpha!r}")

@dataclass(frozen=True)
class DirectPathParams:
    """Parametri del path diretto del canale (Eq. (3), Sez. V-A: Rician).

    Attributes:
        h_c: Coefficiente di fading Rician complesso, con E[abs(h_c)^2] = 1.
        f_dc: Doppler normalizzato del path diretto, in [0, 0.5).
    """

    h_c: complex
    f_dc: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.h_c.real) and math.isfinite(self.h_c.imag)):
            raise ValueError(f"h_c non finito: {self.h_c!r}")
        if not math.isfinite(self.f_dc) or not (0.0 <= self.f_dc < 0.5):
            raise ValueError(f"f_dc deve essere in [0, 0.5), ricevuto: {self.f_dc!r}")

def _iter_config_leaves(config: Any, prefix: str = "config") -> Iterator[Tuple[str, Any]]:
    """Itera ricorsivamente le foglie della config (dict, liste, scalari)."""
    if isinstance(config, dict):
        for key, value in config.items():
            yield from _iter_config_leaves(value, f"{prefix}.{key}")
    elif isinstance(config, (list, tuple)):
        for index, value in enumerate(config):
            yield from _iter_config_leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, config

def _assert_config_finite(config: Any) -> None:
    """Fail-fast su valori numerici non finiti (NaN/Inf) dentro la config."""
    for path, value in _iter_config_leaves(config):
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"valore non finito nella config: {path} = {value!r}")

_REQUIRED_DATA_KEYS: Tuple[str, ...] = (
    "sequence_length", "map_type", "map_param", "fc_hz", "fs_hz",
    "snr_range", "snr_step", "echoes", "max_delay", "max_doppler",
    "alpha_min", "alpha_max", "num_symbols_train", "num_symbols_val",
    "num_symbols_test", "raw_dir", "processed_dir",
    "rician_kappa_db", "doppler_direct_max",
)

def _validate_config(config: Dict[str, Any]) -> None:
    """Validazione fail-fast della config usata dal generatore.

    Args:
        config: Config completa (dopo il merge di ``load_config``).

    Raises:
        TypeError: Se ``config`` non e' un dict.
        ValueError: Se una chiave richiesta manca oppure un valore e' incoerente.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")
    _assert_config_finite(config)
    data = config.get("data")
    general = config.get("general")
    if not isinstance(data, dict):
        raise ValueError("sezione 'data' mancante o non dict nella config")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict nella config")
    missing = [key for key in _REQUIRED_DATA_KEYS if key not in data]
    if missing:
        raise ValueError(f"chiavi mancanti in config['data']: {missing}")
    if "seed" not in general:
        raise ValueError("chiave mancante: general.seed")

    sequence_length = int(data["sequence_length"])
    if sequence_length <= 0:
        raise ValueError(f"sequence_length deve essere > 0, ricevuto: {sequence_length}")

    map_type = str(data["map_type"])
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type deve essere in {sorted(_MAP_TYPES)}, ricevuto: {map_type!r}")

    map_param = float(data["map_param"])
    if not math.isfinite(map_param):
        raise ValueError(f"map_param non finito: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= map_param <= _LOGISTIC_MU_MAX):
        #: la validazione di mu vale solo per la mappa logistica.
        raise ValueError(
            f"map_param (mu) fuori dal regime caotico "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )

    for key in ("fc_hz", "fs_hz", "max_doppler", "doppler_direct_max", "alpha_min", "alpha_max"):
        value = float(data[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"data.{key} deve essere finito e > 0, ricevuto: {value!r}")

    if float(data["max_doppler"]) >= 0.5:
        raise ValueError("data.max_doppler deve essere < 0.5 (guardia anti-aliasing)")
    if float(data["doppler_direct_max"]) > float(data["max_doppler"]):
        raise ValueError("data.doppler_direct_max deve essere <= data.max_doppler")

    if float(data["alpha_min"]) >= float(data["alpha_max"]):
        raise ValueError("data.alpha_min deve essere < data.alpha_max")

    rician_kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(rician_kappa_db) or rician_kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db deve essere >= 0, ricevuto: {rician_kappa_db!r}")

    snr_range = data["snr_range"]
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"data.snr_range deve essere [min, max], ricevuto: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min >= snr_max:
        raise ValueError(f"data.snr_range invalido: {snr_range!r}")

    snr_step = float(data["snr_step"])
    if not math.isfinite(snr_step) or snr_step <= 0.0:
        raise ValueError(f"data.snr_step deve essere > 0, ricevuto: {snr_step!r}")

    max_delay = int(data["max_delay"])
    if max_delay <= 0:
        raise ValueError(f"data.max_delay deve essere > 0, ricevuto: {max_delay}")
    if max_delay > sequence_length:
        # policy tau <= sequence_length (test_channel_extreme_delay_raises).
        raise ValueError(
            f"data.max_delay ({max_delay}) deve essere <= sequence_length ({sequence_length})"
        )

    echoes = data["echoes"]
    if not isinstance(echoes, (list, tuple)) or len(echoes) == 0:
        raise ValueError(f"data.echoes deve essere una lista non vuota, ricevuto: {echoes!r}")
    for k in echoes:
        if not isinstance(k, (int,)) or isinstance(k, bool) or k < 0:
            raise ValueError(f"data.echoes deve contenere int >= 0, ricevuto: {k!r}")
        if k > max_delay:
            raise ValueError(
                f"k={k} > max_delay={max_delay}: impossibile campionare {k} ritardi distinti"
            )

    for key in ("num_symbols_train", "num_symbols_val", "num_symbols_test"):
        value = data[key]
        if not isinstance(value, (int,)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"data.{key} deve essere un int > 0, ricevuto: {value!r}")

    for key in ("raw_dir", "processed_dir"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{key} deve essere una stringa non vuota, ricevuto: {value!r}")

    seed = general["seed"]
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"general.seed deve essere un int >= 0, ricevuto: {seed!r}")

    logger.debug(
        "config validata: seq_len=%d, map_type=%s, mu=%s, max_delay=%d, max_doppler=%s",
        sequence_length, map_type, map_param, max_delay, data["max_doppler"],
    )

def _iterate_map(map_type: str, map_param: float, x0: float, sequence_length: int) -> np.ndarray:
    """Itera la mappa caotica per ``sequence_length`` campioni da ``x0``.

    Args:
        map_type: Mappa da iterare (``"logistic"`` o ``"bernoulli"``).
        map_param: Parametro mu della mappa logistica (ignorato per Bernoulli).
        x0: Condizione iniziale (deve essere in (0, 1)).
        sequence_length: Numero di campioni da generare.

    Returns:
        Vettore float64 di shape ``(sequence_length,)`` con la traiettoria.
    """
    seq = np.empty(sequence_length, dtype=np.float64)
    if map_type == "logistic":
        x = x0
        for i in range(sequence_length):
            seq[i] = x
            if i + 1 == sequence_length:
                break
            x = map_param * x * (1.0 - x)  # Eq. (1) Sez. III-A
    else:
        # Eq. (2) Sez. III-A: mappa di Bernoulli (shift). In precisione finita
        # la forma binaria (2x mod 1) e' nilpotente (ogni x0 float e' dyadico ->
        # orbita a zero in <= 64 passi): si usa la forma moltiplicativa in base
        # a dispari x -> (a*x mod 1), permutazione dello stato (nessun collasso,
        # densita' invariante uniforme). Stato iniziale = pattern IEEE-754 di x0.
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
    """Genera una sequenza caotica deterministica dal seed (Eq. (1)/(2)).

    Args:
        map_type: ``"logistic"`` o ``"bernoulli"``.
        map_param: mu della mappa logistica (Eq. (1),).
        seed: Seed intero non negativo: stesso seed, stessa sequenza.
        sequence_length: Lunghezza della sequenza (N_seq, default 100).

    Returns:
        Vettore float64 di shape ``(sequence_length,)``, valori in (0, 1),
        energia positiva (guardia anti-degenerazione).

    Raises:
        ValueError: Se ``map_type`` non e' valido, oppure ``map_param`` fuori
            da [3.57, 4] per la logistica, oppure seed/sequence_length non
            validi.
        RuntimeError: Se la mappa degenera (NaN/Inf, energia nulla) dopo
            ``_MAP_RETRIES`` tentativi.
    """
    if not isinstance(map_type, str) or map_type not in _MAP_TYPES:
        raise ValueError(f"map_type deve essere in {sorted(_MAP_TYPES)}, ricevuto: {map_type!r}")
    if not isinstance(map_param, (int, float)) or not math.isfinite(float(map_param)):
        raise ValueError(f"map_param deve essere un float finito, ricevuto: {map_param!r}")
    if map_type == "logistic" and not (_LOGISTIC_MU_MIN <= float(map_param) <= _LOGISTIC_MU_MAX):
        raise ValueError(
            f"map_param (mu) fuori dal regime caotico "
            f"[{_LOGISTIC_MU_MIN}, {_LOGISTIC_MU_MAX}]: {map_param}"
        )
    if not isinstance(seed, (int,)) or isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed deve essere un int >= 0, ricevuto: {seed!r}")
    if not isinstance(sequence_length, (int,)) or int(sequence_length) <= 0:
        raise ValueError(f"sequence_length deve essere un int > 0, ricevuto: {sequence_length!r}")

    n = int(sequence_length)
    mu = float(map_param)
    # x0 deterministica dal seed, condivisa con la generazione vettorizzata
    # (``_x0_from_seed``): stesso seed -> stessa x0 -> stessa sequenza sia nel
    # generatore per-simbolo sia in ``generate_transmitted_batch`` (coerenza con
    # la rigenerazione del riferimento ISAC da bit+seed).
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
        # Guardia anti-collasso: con la forma moltiplicativa della Bernoulli
        # il collasso diadico e' eliminato, ma si esclude comunque ogni
        # sequenza pseudo-degenere (coda di zeri o energia nulla).
        and np.count_nonzero(seq) >= max(2, n // 10)
    )
    if valid:
        logger.debug(
            "mappa %s generata (x0=%.6f, energia=%.3e)",
            map_type, x0, energy,
        )
        return np.asarray(seq, dtype=np.float64)

    raise RuntimeError(
        f"mappa {map_type!r} degenerata: NaN/Inf o energia nulla"
    )

def sample_echo_parameters(
    k: int,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> List[EchoParams]:
    """Campiona i parametri di K echi (Eq. (3), Tab. I).

    Args:
        k: Numero di echi (K). Con K=0 restituisce lista vuota ().
        rng: Generatore NumPy (stream dedicato per riproducibilita').
        config: Config completa (chiavi data.max_delay, data.max_doppler,
            data.alpha_min, data.alpha_max).

    Returns:
        Lista di ``EchoParams`` con tau_k interi distinti in [1, max_delay],
        f_Dk in [0, max_doppler], alpha_k log-uniformi in [alpha_min, alpha_max].

    Raises:
        ValueError: Se ``k`` non e' valido, ``k > max_delay`` oppure le chiavi
            di config mancano o sono incoerenti.
    """
    if not isinstance(k, (int,)) or isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k deve essere un int >= 0, ricevuto: {k!r}")
    k = int(k)
    if k == 0:
        return []  #: nessun eco (smoke test, AWGN puro)
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
            f"k ({k}) > max_delay ({max_delay}): impossibile campionare "
            f"{k} ritardi interi distinti in [1, max_delay]"
        )

    taus = rng.choice(np.arange(1, max_delay + 1, dtype=np.int64), size=k, replace=False)
    dopplers = rng.uniform(0.0, max_doppler, size=k)
    if bool(data.get("alpha_tau_coupling", False)):
        # Law radar-like: alpha ~ alpha_max / tau (ampiezza ~ 1/range). Il floor
        # evita alpha->0 per tau grandi (altrimenti eco "invisibili" a ogni SNR).
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
        raise ValueError("parametri eco non finiti durante il campionamento")
    logger.debug(
        "campionati %d echi: tau=%s, fD=%s, alpha=%s",
        k, [e.tau for e in echoes], [e.f_doppler for e in echoes], [e.alpha for e in echoes],
    )
    return echoes

def sample_direct_path(rng: np.random.Generator, config: Dict[str, Any]) -> DirectPathParams:
    """Campiona il path diretto: h_c Rician a potenza unitaria e f_Dc.

    Args:
        rng: Generatore NumPy.
        config: Config completa (chiavi data.rician_kappa_db e
            data.doppler_direct_max).

    Returns:
        ``DirectPathParams`` con h_c complesso (E[abs(h_c)^2] = 1) e f_Dc in
        [0, doppler_direct_max].

    Raises:
        ValueError: Se ``rician_kappa_db`` e' negativo o non finito.
        RuntimeError: Se il campionamento produce valori non finiti.
    """
    data = config["data"]
    kappa_db = float(data["rician_kappa_db"])
    if not math.isfinite(kappa_db) or kappa_db < 0.0:
        raise ValueError(f"data.rician_kappa_db deve essere >= 0, ricevuto: {kappa_db!r}")
    doppler_direct_max = float(data["doppler_direct_max"])

    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    g = (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = float(rng.uniform(0.0, doppler_direct_max))  #
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag) and math.isfinite(f_dc)):
        raise RuntimeError("campionamento del path diretto non finito")
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
    """Applica il canale aereo multi-eco (Eq. (3)) con AWGN (Eq. (4)).

    Args:
        x: Sequenza trasmessa (reale, 1D, shape ``(N,)``).
        h_c: Coefficiente di fading Rician del path diretto (potenza unitaria).
        f_dc: Doppler normalizzato del path diretto.
        echoes: Lista di ``EchoParams`` (puo' essere vuota, K=0).
        snr_db: SNR per bit in dB (sigma_w^2 = P_s * 10^(-SNR/10)).
        rng: Generatore NumPy per l'AWGN complesso.

    Returns:
        ``(y, noise_var)``: segnale ricevuto complesso shape ``(N,)`` e
        varianza sigma_w^2 del rumore complesso.

    Raises:
        TypeError: Se ``echoes`` non e' una lista.
        ValueError: Se x, h_c, f_dc, snr_db non sono validi, oppure un ritardo
            supera la finestra di osservazione, oppure la potenza echi e' >= 1.
        RuntimeError: Se l'output contiene NaN/Inf o la potenza del segnale
            e' nulla (simbolo degenere).
    """
    if not isinstance(echoes, list):
        raise TypeError(f"echoes deve essere list[EchoParams], ricevuto: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x deve essere un vettore 1D non vuoto, ricevuto: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contiene NaN/Inf")
    if not (math.isfinite(f_dc) and 0.0 <= f_dc < 0.5):
        raise ValueError(f"f_dc deve essere in [0, 0.5), ricevuto: {f_dc!r}")
    if not (math.isfinite(h_c.real) and math.isfinite(h_c.imag)):
        raise ValueError(f"h_c non finito: {h_c!r}")
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db deve essere finito, ricevuto: {snr_db!r}")

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    echo_power = 0.0
    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"elemento di echoes non EchoParams: {echo!r}")
        if echo.tau > n_samples:
            # test_channel_extreme_delay_raises: ritardo fuori finestra.
            raise ValueError(
                f"eco con tau={echo.tau} oltre la finestra di osservazione (N={n_samples})"
            )
        echo_power += float(echo.alpha) ** 2
    if echo_power >= 1.0 - _POWER_EPS:
        raise ValueError(
            f"Somma alpha_k^2 = {echo_power:.6f} >= 1: normalizzazione Eq. (4) impossibile"
        )

    # Eq. (4): E[abs(h_c)^2] + Somma alpha_k^2 = 1, con E[abs(h_c)^2] = 1 in ingresso.
    h_c_eff = h_c * math.sqrt(1.0 - echo_power)

    # Eq. (3): path diretto con Doppler f_Dc.
    y_clean = h_c_eff * x_arr * np.exp(1j * 2.0 * math.pi * f_dc * n_idx)
    # Eq. (3): echi ritardati con zero-padding e Doppler f_Dk.
    for echo in echoes:
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo.alpha * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    #: potenza ricevuta misurata per simbolo -> SNR esatto per costruzione.
    signal_power = float(np.mean(np.abs(y_clean) ** 2))
    if not math.isfinite(signal_power) or signal_power <= 0.0:
        raise RuntimeError(f"potenza segnale non valida: {signal_power!r} (simbolo degenere)")

    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    # w ~ CN(0, noise_var): parte reale e immaginaria con varianza noise_var/2.
    w = math.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    y = y_clean + w

    if not np.all(np.isfinite(y)) or not math.isfinite(noise_var):
        raise RuntimeError("output del canale non finito (NaN/Inf): STOP  Sez. 1.2")

    realized_snr = 10.0 * math.log10(signal_power / noise_var) if noise_var > 0.0 else math.inf
    logger.debug(
        "canale: P_s=%.3e, noise_var=%.3e, SNR realizzato=%.2f dB (richiesto %.1f dB)",
        signal_power, noise_var, realized_snr, snr_db,
    )
    return y, noise_var

def apply_echo_only_channel(
    x: np.ndarray,
    echoes: List[EchoParams],
) -> np.ndarray:
    """Applica il canale deterministico echo-only (Eq. (3)) senza AWGN/fading.

    Variante dell'Esperimento 3 (``channel.echo_only_mode: true``): isola la
    degradazione dovuta a eco/Doppler eliminando AWGN e fading Rician/Rayleigh
    (``channel.add_awgn: false``, ``channel.echo_fading: "none"``). Il path
    diretto ha guadagno unitario e nessun Doppler; le K eco sono copie
    ritardate e scalate di ``x`` con rotazione di fase Doppler (Eq. (3)).

    Args:
        x: Sequenza trasmessa (reale, 1D, shape ``(N,)``).
        echoes: Lista di ``EchoParams`` (puo' essere vuota, K=0).

    Returns:
        Segnale ricevuto complesso shape ``(N,)``: path diretto + eco, senza
        rumore additivo.

    Raises:
        TypeError: Se ``x`` non e' ``np.ndarray`` oppure ``echoes`` non e' una
            lista.
        ValueError: Se ``x`` non e' un vettore 1D finito non vuoto, oppure un
            ritardo supera la finestra di osservazione.
        RuntimeError: Se l'output contiene NaN/Inf.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x deve essere np.ndarray, ricevuto: {type(x).__name__}")
    if not isinstance(echoes, list):
        raise TypeError(f"echoes deve essere list[EchoParams], ricevuto: {type(echoes).__name__}")
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError(f"x deve essere un vettore 1D non vuoto, ricevuto: shape={x_arr.shape}")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contiene NaN/Inf")

    n_samples = x_arr.size
    n_idx = np.arange(n_samples, dtype=np.float64)

    # Path diretto deterministico: guadagno unitario, nessun fading, nessun
    # Doppler (isola la degradazione dovuta alle sole eco).
    y_clean = x_arr.astype(np.complex128)

    # Eq. (3): echi ritardati con zero-padding, scaling alpha_k e Doppler f_Dk.
    for echo in echoes:
        if not isinstance(echo, EchoParams):
            raise TypeError(f"elemento di echoes non EchoParams: {echo!r}")
        if echo.tau > n_samples:
            # Policy tau > N_seq (test_channel_extreme_delay_raises).
            raise ValueError(
                f"eco con tau={echo.tau} oltre la finestra di osservazione (N={n_samples})"
            )
        x_delayed = np.zeros(n_samples, dtype=np.float64)
        x_delayed[echo.tau:] = x_arr[: n_samples - echo.tau]
        y_clean = y_clean + echo.alpha * x_delayed * np.exp(
            1j * 2.0 * math.pi * echo.f_doppler * n_idx
        )

    if not np.all(np.isfinite(y_clean)):
        raise RuntimeError(
            "output del canale echo-only non finito (NaN/Inf): STOP  Sez. 1.2"
        )

    logger.debug(
        "canale echo-only applicato: N=%d, K=%d echi, output finito",
        n_samples, len(echoes),
    )
    return y_clean

def build_snr_grid(snr_range: List[float], snr_step: float) -> List[float]:
    """Costruisce la griglia SNR aritmetica INCLUSIVA sugli estremi (T13).

    Args:
        snr_range: ``[min, max]`` dB.
        snr_step: Passo in dB (positivo).

    Returns:
        Lista di valori ``[min, min+step, ...]`` piu' ``max`` se non incluso.

    Raises:
        ValueError: Se ``snr_range`` non ha due valori finiti con min < max
            oppure ``snr_step`` non e' positivo.
    """
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"snr_range deve essere [min, max], ricevuto: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    step = float(snr_step)
    if not (math.isfinite(snr_min) and math.isfinite(snr_max) and math.isfinite(step)):
        raise ValueError(f"valori non finiti in snr_range/snr_step: {snr_range!r}, {snr_step!r}")
    if snr_min >= snr_max:
        raise ValueError(f"snr_min ({snr_min}) deve essere < snr_max ({snr_max})")
    if step <= 0.0:
        raise ValueError(f"snr_step deve essere > 0, ricevuto: {step}")

    n_points = int(math.ceil((snr_max - snr_min) / step - 1e-9))
    if n_points < 1:
        raise ValueError(f"griglia SNR vuota per {snr_range!r} con step {step}")
    points = [snr_min + i * step for i in range(n_points)]
    if abs(points[-1] - snr_max) > 1e-9:
        points.append(snr_max)  # estremo incluso (T13)
    logger.debug("griglia SNR inclusiva: %s", points)
    return points

def _map_type_for_bit(map_type: str, bit: int) -> str:
    """Restituisce la mappa associata al bit (CSK,).

    Convenzione: bit 0 -> ``map_type`` (default logistic); bit 1 -> mappa
    complementare (bernoulli se logistic, logistic se bernoulli).

    Raises:
        ValueError: Se ``map_type`` non e' valido oppure ``bit`` non e' 0/1.
    """
    if map_type not in _MAP_TYPES:
        raise ValueError(f"map_type deve essere in {sorted(_MAP_TYPES)}, ricevuto: {map_type!r}")
    if bit not in (0, 1):
        raise ValueError(f"bit deve essere 0 o 1, ricevuto: {bit!r}")
    if bit == 0:
        return map_type
    return "bernoulli" if map_type == "logistic" else "logistic"

def _dominant_echo_index(echoes: List[EchoParams]) -> int:
    """Indice dell'eco dominante = eco con alpha massimo (, Sez. IV-C).

    Args:
        echoes: Lista dei parametri degli echi.

    Returns:
        Indice dell'eco con alpha_k massimo ("strongest echo").

    Raises:
        ValueError: Se ``echoes`` e' vuota.
    """
    if not echoes:
        raise ValueError("echoes vuota: nessun eco dominante")
    return int(np.argmax([echo.alpha for echo in echoes]))

def resolve_n_per_combo(
    config: Dict[str, Any],
    split: str,
    num_combos: int,
) -> int:
    """Risolve il numero di simboli per combinazione (SNR, K) per uno split.

    Unica fonte di verita' del conteggio per-combo, condivisa da
    ``dataset_generator.generate_dataset`` e ``data_loader._expected_per_combo``
    (evita duplicazione e deriva tra generatore e loader). Applica il
    fail-fast costruttivo dei vincoli di bilanciamento:
      - split ``"train"``: ``num_symbols_train`` e' un TOTALE distribuito sulle
        ``num_combos`` combinazioni; deve essere divisibile per ``num_combos``
        e il quoziente deve essere pari (bit 0/1 bilanciati per ogni (SNR, K),) —.
      - split ``"val"``/``"test"``: ``num_symbols_val/test`` sono PER OGNI
        (SNR, K) e devono essere pari —.
    Il messaggio di errore riporta i valori reali, il vincolo violato e i
    valori validi da usare nel YAML (nessun reverse-engineering).

    Args:
        config: Config completa (chiavi ``data.num_symbols_train/val/test``).
        split: ``"train"``, ``"val"`` oppure ``"test"``.
        num_combos: Numero di combinazioni (SNR, K) della griglia (int > 0).

    Returns:
        Numero di simboli per ogni (SNR, K) nello split richiesto (int pari > 0).

    Raises:
        ValueError: Se ``split``/``num_combos`` non sono validi, se il conteggio
            per-combo non e' positivo oppure dispari, oppure se
            ``num_symbols_train`` non e' divisibile per ``num_combos``.
    """
    if split not in _SPLITS:
        raise ValueError(f"split deve essere in {sorted(_SPLITS)}, ricevuto: {split!r}")
    if not isinstance(num_combos, int) or isinstance(num_combos, bool) or num_combos <= 0:
        raise ValueError(f"num_combos deve essere un int > 0, ricevuto: {num_combos!r}")

    data = config["data"]
    if split == "train":
        total = int(data["num_symbols_train"])
        if total % num_combos != 0:
            #: bilanciamento per (SNR, K) impossibile con resto.
            nearest_down = num_combos * (total // num_combos)
            nearest_up = nearest_down + num_combos
            raise ValueError(
                f"num_symbols_train ({total}) non divisibile per num_combos ({num_combos}): "
                "bilanciamento per (SNR, K) impossibile ( Sez. 4.1). "
                f"Valori validi vicini: {nearest_down} oppure {nearest_up}."
            )
        n_per_combo = total // num_combos
    elif split == "val":
        n_per_combo = int(data["num_symbols_val"])  #: per (SNR, K)
    else:
        n_per_combo = int(data["num_symbols_test"])  #: per (SNR, K)

    if n_per_combo <= 0:
        raise ValueError(
            f"numero simboli per combinazione deve essere > 0, ricevuto: {n_per_combo}"
        )
    if n_per_combo % 2 != 0:
        if split == "train":
            unit = 2 * num_combos
            suggestion = unit * ((total + unit - 1) // unit)
            hint = (
                f"per split 'train' usare num_symbols_train multiplo di "
                f"2*num_combos={unit} (valore valido vicino: {suggestion})"
            )
        else:
            key = "num_symbols_val" if split == "val" else "num_symbols_test"
            hint = f"per split '{split}' usare {key} pari (es. {n_per_combo + 1})"
        raise ValueError(
            f"n_per_combo ({n_per_combo}) deve essere pari per bilanciare i bit 0/1 "
            f"per ogni (SNR, K) ( Sez. 4.1): {hint}"
        )
    return n_per_combo

def _center_normalize_symbol(x: np.ndarray) -> np.ndarray:
    """Centra (media zero) e normalizza (energia unitaria) un simbolo trasmesso.

    Le due mappe CSK (logistica a mu=3.9 e Bernoulli) hanno statistiche diverse
    (media/energia): senza normalizzazione le classi sarebbero separabili con un
    banale energy/DC-detector, non imparando la struttura caotica. Dopo questa
    trasformazione i simboli delle due classi hanno la stessa energia e media
    nulla: la decodifica richiede di discriminare la struttura delle mappe.

    Args:
        x: Sequenza caotica 1D in (0, 1).

    Returns:
        Sequenza centrata e normalizzata (somma = 0, somma quadrati = 1).

    Raises:
        RuntimeError: Se dopo il centraggio l'energia e' nulla (simbolo degenere).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_centered = x_arr - np.mean(x_arr)
    energy = float(np.sqrt(np.sum(x_centered ** 2)))
    if not math.isfinite(energy) or energy <= _ENERGY_EPS:
        raise RuntimeError(
            "simbolo degenere dopo centraggio/normalizzazione (energia nulla)"
        )
    return x_centered / energy


# ============================================================================
# Generazione vettorizzata (batch)
#
# Le funzioni seguenti producono gli stessi oggetti del generatore per-simbolo
# (``generate_chaotic_sequence`` / ``sample_echo_parameters`` /
# ``sample_direct_path`` / ``apply_aerial_channel``) ma operano su batch di
# simboli. Sono usate da:
#   - ``generate_dataset``  : speedup della pre-generazione su Colab (il loop
#     Python per-simbolo con 100 iterazioni di mappa era il collo di bottiglia).
#   - ``evaluate_model_online`` (src/evaluation/evaluator.py): valutazione BER
#     con simboli generati ON-THE-FLY fino a ``max_symbols_per_snr`` (rimozione
#     del floor di misura 1/(2*n_symbols) con n_symbols fisso a 20000).
# La coerenza con la rigenerazione del riferimento ISAC (``build_reference_matrix``,
# che chiama ``generate_chaotic_sequence(seed)``) e' garantita da ``_x0_from_seed``:
# stesso seed -> stessa x0 -> stessa sequenza, per-simbolo e per-batch.
# ============================================================================

def _x0_from_seed(seed: int, map_type: str, map_param: float) -> float:
    """Condizione iniziale x0 deterministica dal seed (== ``generate_chaotic_sequence``).

    Estratto il primo campione di ``np.random.default_rng(seed).uniform(
    _X0_MIN, _X0_MAX)`` con retry sui punti fissi della mappa (identica logica
    del generatore per-simbolo): stesso seed -> stessa x0 -> stessa sequenza.

    Args:
        seed: Seed intero non negativo.
        map_type: ``"logistic"`` o ``"bernoulli"``.
        map_param: ``mu`` della mappa logistica (per i punti fissi ``1 - 1/mu``).

    Returns:
        ``float`` in ``(X0_MIN, X0_MAX)`` lontano dai punti fissi.

    Raises:
        ValueError: Se ``map_type``/``seed`` non sono validi.
        RuntimeError: Se dopo ``_MAP_RETRIES`` tentativi nessuna x0 e' valida.
    """
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type deve essere in {sorted(_MAP_TYPES)}, ricevuto: {map_type!r}"
        )
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed deve essere un int >= 0, ricevuto: {seed!r}")
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
        f"mappa {map_type!r} degenerata dopo {_MAP_RETRIES} tentativi: x0 su punto fisso"
    )


def _iterate_map_batch(
    map_type: str,
    map_param: float,
    x0: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Itera la mappa caotica su un batch di condizioni iniziali (vettorizzato).

    Replica ``_iterate_map`` riga per riga: logistica ``x -> mu x (1-x)``
    (Eq. (1)) oppure Bernoulli moltiplicativa a base dispari (Eq. (2), forma
    non-nilpotente su ``Z/2^64``).

    Args:
        map_type: ``"logistic"`` o ``"bernoulli"``.
        map_param: ``mu`` della logistica (ignorato per Bernoulli).
        x0: Array ``(N,)`` di condizioni iniziali in ``(0, 1)``.
        sequence_length: Lunghezza di ciascuna sequenza (``N_seq``).

    Returns:
        Array ``(N, sequence_length)`` float64 con le traiettorie.
    """
    x0_arr = np.asarray(x0, dtype=np.float64)
    if x0_arr.ndim != 1:
        raise ValueError(f"x0 deve essere 1D, ricevuto: {x0_arr.shape}")
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
            x = map_param * x * (1.0 - x)  # Eq. (1) Sez. III-A
    else:
        state = np.asarray(x0_arr, dtype=np.float64).view(np.uint64) & _BERNOULLI_MASK
        for i in range(sequence_length):
            seq[:, i] = state / _BERNOULLI_SCALE
            state = (state * _BERNOULLI_MULTIPLIER) & _BERNOULLI_MASK
    return seq


def _map_types_for_bits(map_type: str, bits: np.ndarray) -> np.ndarray:
    """Mappe CSK associate ai bit di un batch (convenzione , Eq. (1)/(2)).

    bit 0 -> ``map_type`` (default logistic); bit 1 -> mappa complementare.

    Args:
        map_type: Mappa del bit 0.
        bits: Array ``(N,)`` di bit 0/1.

    Returns:
        Array ``(N,)`` di stringhe ``"logistic"``/``"bernoulli"``.

    Raises:
        ValueError: Se ``map_type`` non e' valido o i bit non sono 0/1.
    """
    if map_type not in _MAP_TYPES:
        raise ValueError(
            f"map_type deve essere in {sorted(_MAP_TYPES)}, ricevuto: {map_type!r}"
        )
    bits_arr = np.asarray(bits)
    if not np.all(np.isin(bits_arr, (0, 1))):
        raise ValueError(f"bits deve contenere solo 0/1, ricevuto: {bits_arr!r}")
    if map_type == "logistic":
        return np.where(bits_arr == 0, "logistic", "bernoulli")
    return np.where(bits_arr == 0, "bernoulli", "logistic")


def _center_normalize_batch(x: np.ndarray) -> np.ndarray:
    """Centra (media zero) e normalizza (energia unitaria) un batch di simboli.

    Replica vettorizzata di ``_center_normalize_symbol`` riga per riga.

    Args:
        x: Array ``(N, N_seq)`` di sequenze caotiche in ``(0, 1)``.

    Returns:
        Array ``(N, N_seq)`` con media 0 e norma 1 per riga.

    Raises:
        RuntimeError: Se qualche riga degenera (energia nulla / non finita).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x deve essere 2D (N, N_seq), ricevuto: {x_arr.shape}")
    x_centered = x_arr - np.mean(x_arr, axis=1, keepdims=True)
    energy = np.sqrt(np.sum(x_centered ** 2, axis=1, keepdims=True))
    if (not np.all(np.isfinite(x_centered))) or np.any(energy <= _ENERGY_EPS):
        raise RuntimeError(
            "batch degenere dopo centraggio/normalizzazione (energia nulla o NaN/Inf)"
        )
    return x_centered / energy


def generate_transmitted_batch(
    config: Dict[str, Any],
    bits: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Genera un batch di sequenze trasmesse (centrate + normalizzate).

    Vettorizzazione di ``generate_chaotic_sequence`` + ``_center_normalize_symbol``:
    stessa distribuzione e, per lo stesso ``(bit, seed)``, la STESSA sequenza del
    generatore per-simbolo (coerente con ``build_reference_matrix``, che rigenera
    il riferimento ISAC dal seed). Le due mappe del batch vengono iterate in
    parallelo su array ``(N,)`` (100 step vettorizzati invece di un loop Python
    per-simbolo): ~50-100x piu' veloce della generazione attuale.

    Args:
        config: Config completa (chiavi ``data.map_type``, ``data.map_param``,
            ``data.sequence_length``).
        bits: Array ``(N,)`` dei bit (0/1).
        seeds: Array ``(N,)`` dei seed interi (stessa lunghezza di ``bits``).

    Returns:
        Array ``(N, N_seq)`` float64 = sequenza trasmessa per ogni simbolo
        (media 0, energia 1).

    Raises:
        ValueError: Se shape/valori di ``bits``/``seeds`` non sono validi.
        RuntimeError: Se una mappa degenera dopo ``_MAP_RETRIES`` tentativi.
    """
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    if bits_arr.ndim != 1 or seeds_arr.ndim != 1 or bits_arr.shape != seeds_arr.shape:
        raise ValueError(
            f"bits/seeds devono essere 1D allineati, ricevuti: "
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
    """Batch di sequenze trasmesse con x0 campionata dal rng di batch (veloce).

    Variante di ``generate_transmitted_batch`` che NON deriva x0 dal seed
    (niente ``default_rng`` per-simbolo, il collo di bottiglia della versione
    seed-consistente): la distribuzione di x0 e' identica (uniforme in
    ``(X0_MIN, X0_MAX)`` con nudge sui punti fissi), ma ``(bit, seed)`` non
    riproduce piu' la stessa sequenza. Usata SOLO dalla valutazione online
    (``generate_test_batch``), dove il riferimento ``x_ref`` viene calcolato
    inline e mai rigenerato dai seed.

    Args:
        config: Config completa.
        bits: Array ``(N,)`` dei bit (0/1).
        rng: Generatore NumPy del batch (consumato per x0).

    Returns:
        Array ``(N, N_seq)`` float64 = sequenza trasmessa (media 0, energia 1).

    Raises:
        ValueError: Se ``bits`` non e' 1D di soli 0/1.
        RuntimeError: Se una mappa degenera (energia nulla / NaN).
    """
    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])

    bits_arr = np.asarray(bits, dtype=np.int64)
    if bits_arr.ndim != 1:
        raise ValueError(f"bits deve essere 1D, ricevuto: {bits_arr.shape}")
    n = int(bits_arr.shape[0])
    if n == 0:
        return np.empty((0, seq_len), dtype=np.float64)

    maps = _map_types_for_bits(map_type, bits_arr)
    out = np.empty((n, seq_len), dtype=np.float64)
    for mt in sorted(_MAP_TYPES):
        idx = np.where(maps == mt)[0]
        if idx.size == 0:
            continue
        # x0 uniforme in (X0_MIN, X0_MAX); i rarissimi punti vicini ai punti
        # fissi vengono spostati (probabilita' ~1e-12 per campione).
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
    """Applica il canale aereo multi-eco + AWGN a un batch (Eq. (3)/(4)).

    Vettorizzazione di ``sample_echo_parameters`` + ``sample_direct_path`` +
    ``apply_aerial_channel``: stessi campionamenti (ritardi interi distinti,
    Doppler uniformi, alpha log-uniformi, path diretto Rician con
    ``E[abs(h_c)^2] = 1``) ma su batch ``(N, ...)``.

    Args:
        x_norm: Array ``(N, N_seq)`` di sequenze trasmesse (media 0, energia 1).
        k: Numero di echi (int, fisso per batch).
        snr_db: SNR per bit in dB.
        config: Config completa (chiavi ``data.*`` del canale).
        rng: Generatore NumPy del batch.

    Returns:
        Tupla ``(y, tau_labels, f_d_labels)``:
            - ``y``: segnale ricevuto complesso ``(N, N_seq)``;
            - ``tau_labels``/``f_d_labels``: parametri dell'eco dominante
              (alpha massimo, Sez. IV-C); zeri se ``k == 0``.

    Raises:
        ValueError: Se ``x_norm`` non e' 2D, ``k`` non valido o potenza echi >= 1.
        RuntimeError: Se l'output contiene NaN/Inf o un simbolo e' degenere.
    """
    x_arr = np.asarray(x_norm, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x_norm deve essere 2D (N, N_seq), ricevuto: {x_arr.shape}")
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k deve essere un int >= 0, ricevuto: {k!r}")
    k = int(k)
    if not math.isfinite(float(snr_db)):
        raise ValueError(f"snr_db deve essere finito, ricevuto: {snr_db!r}")

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
            f"k ({k}) > max_delay ({max_delay}): impossibile campionare "
            f"{k} ritardi interi distinti in [1, max_delay]"
        )

    n = int(x_arr.shape[0])
    n_idx = np.arange(seq_len, dtype=np.float64)[None, :]  # (1, N_seq) per le fasi
    n_idx_i = np.arange(seq_len, dtype=np.int64)[None, :]  # (1, N_seq) per i ritardi

    # --- Echi (Eq. (3)): ritardi interi distinti per riga -------------------
    if k > 0:
        pool = np.tile(np.arange(1, max_delay + 1, dtype=np.int64)[None, :], (n, 1))
        taus = rng.permuted(pool, axis=1)[:, :k]  # (N, k): k ritardi distinti/riga
        dopplers = rng.uniform(0.0, max_doppler, size=(n, k))
        if bool(data.get("alpha_tau_coupling", False)):
            # Law radar-like (Tab. I): alpha ~ alpha_max/tau, ampiezza ~ 1/range.
            floor = float(data.get("alpha_floor", _ALPHA_COUPLING_FLOOR))
            alphas = np.clip(alpha_max / np.maximum(taus.astype(np.float64), 1.0),
                             floor, alpha_max)
        else:
            alphas = 10.0 ** rng.uniform(
                math.log10(alpha_min), math.log10(alpha_max), size=(n, k)
            )
        echo_power = np.sum(alphas ** 2, axis=1)
        if np.any(echo_power >= 1.0 - _POWER_EPS):
            raise ValueError("Somma alpha_k^2 >= 1: normalizzazione Eq. (4) impossibile")
    else:
        taus = np.empty((n, 0), dtype=np.int64)
        dopplers = np.empty((n, 0), dtype=np.float64)
        alphas = np.empty((n, 0), dtype=np.float64)
        echo_power = np.zeros(n, dtype=np.float64)

    # --- Path diretto (Rician, E[abs(h_c)^2] = 1) ---------------------------
    kappa_lin = 10.0 ** (kappa_db / 10.0)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2.0)
    h_c = (
        math.sqrt(kappa_lin / (kappa_lin + 1.0)) * np.exp(1j * theta)
        + math.sqrt(1.0 / (kappa_lin + 1.0)) * g
    )
    f_dc = rng.uniform(0.0, doppler_direct_max, size=n)

    # Eq. (4): E[abs(h_c)^2] + Somma alpha_k^2 = 1 (E[abs(h_c)^2] = 1).
    h_c_eff = h_c * np.sqrt(1.0 - echo_power)

    # Eq. (3): path diretto con Doppler f_Dc + echi ritardati con Doppler f_Dk.
    y = h_c_eff[:, None] * x_arr * np.exp(1j * 2.0 * math.pi * f_dc[:, None] * n_idx)
    for j in range(k):
        delay_idx = n_idx_i - taus[:, j, None]  # (N, N_seq): indice ritardato (int)
        valid = delay_idx >= 0
        clip_idx = np.clip(delay_idx, 0, seq_len - 1)
        x_delayed = np.where(
            valid, np.take_along_axis(x_arr, clip_idx, axis=1), 0.0
        )
        phase = np.exp(1j * 2.0 * math.pi * dopplers[:, j, None] * n_idx)
        y = y + alphas[:, j, None] * x_delayed * phase

    # AWGN con SNR esatto per costruzione: sigma_w^2 = P_s * 10^(-SNR/10).
    signal_power = np.mean(np.abs(y) ** 2, axis=1)
    if np.any(signal_power <= 0.0) or not np.all(np.isfinite(signal_power)):
        raise RuntimeError("potenza segnale non valida (simbolo degenere)")
    noise_var = signal_power * 10.0 ** (-float(snr_db) / 10.0)
    w = np.sqrt(noise_var[:, None] / 2.0) * (
        rng.standard_normal((n, seq_len)) + 1j * rng.standard_normal((n, seq_len))
    )
    y = y + w

    if not np.all(np.isfinite(y)):
        raise RuntimeError("output del canale non finito (NaN/Inf): STOP Sez. 1.2")

    # Etichette sensing: eco dominante (alpha massimo, Sez. IV-C).
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
    """Genera un batch di simboli di TEST on-the-fly (vettorizzato).

    Usata dalla valutazione online (``evaluate_model_online``) per rimuovere il
    floor di misura della valutazione statica: si generano batch finche' non si
    accumulano ``bit_error_threshold`` errori oppure ``max_symbols_per_snr``
    simboli (criterio del paper Sez. V-A, come in marco-siino).

    Stessa distribuzione di ``generate_dataset`` (mappe CSK, normalizzazione,
    canale multi-eco + AWGN). Restituisce anche ``x_ref`` (sequenza trasmessa),
    che alimenta il canale di riferimento ISAC della sensing head senza doverla
    rigenerare dai seed (``build_reference_matrix``).

    Args:
        config: Config completa.
        num_symbols: Numero di simboli del batch (int > 0).
        snr_db: SNR per bit in dB.
        k: Numero di echi (fisso per batch).
        rng: Generatore NumPy del batch (riproducibile per (SNR, batch)).

    Returns:
        Dict con chiavi ``x`` (complesso ``(N, N_seq)``), ``bit`` (int64),
        ``tau``/``f_d`` (float64), ``seed`` (int64) e ``x_ref`` (float32).

    Raises:
        ValueError: Se ``num_symbols``/``k``/``snr_db`` non sono validi.
    """
    if isinstance(num_symbols, bool) or int(num_symbols) <= 0:
        raise ValueError(
            f"num_symbols deve essere un int > 0, ricevuto: {num_symbols!r}"
        )
    n = int(num_symbols)
    if isinstance(k, bool) or int(k) < 0:
        raise ValueError(f"k deve essere un int >= 0, ricevuto: {k!r}")

    bits = rng.integers(0, 2, size=n, dtype=np.int64)
    seeds = rng.integers(0, _SEED_HIGH, size=n, dtype=np.int64)
    # Path veloce (x0 dal rng di batch): la valutazione online usa il riferimento
    # x_ref calcolato inline, senza mai rigenerarlo dai seed.
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
    """Genera e salva il dataset per uno split (bilanciato per (SNR, K)).

    Args:
        config: Config completa (dopo ``load_config`` e ``_validate_config``).
        split: ``"train"``, ``"val"`` oppure ``"test"``.
        output_dir: Directory di salvataggio dei file ``.npz``.

    Returns:
        Lista dei path dei file ``.npz`` salvati, uno per combinazione (SNR, K).

    Raises:
        ValueError: Se ``split`` non e' valido, la config e' invalida, il
            conteggio per combinazione non e' positivo o dispari, oppure
            ``num_symbols_train`` non e' divisibile per le combinazioni
            ().
        RuntimeError: Se i controlli di shape/range/finitezza falliscono.
    """
    if split not in _SPLITS:
        raise ValueError(f"split deve essere in {sorted(_SPLITS)}, ricevuto: {split!r}")
    _validate_config(config)
    data = config["data"]
    snr_grid = build_snr_grid(data["snr_range"], data["snr_step"])
    k_list = [int(k) for k in data["echoes"]]
    num_combos = len(snr_grid) * len(k_list)

    # Fonte unica di verita': divisibilita', parita' dei bit 0/1 e
    # conteggi per-combo sono delegati a resolve_n_per_combo
    # (condiviso con data_loader._expected_per_combo).
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

            # Bit bilanciati (meta' 0 e meta' 1) con shuffle deterministico.
            bits = np.array([0] * (n_per_combo // 2) + [1] * (n_per_combo // 2), dtype=np.uint8)
            rng.shuffle(bits)

            # Generazione VETTORIZZATA del blocco (sequenze trasmesse + canale
            # multi-eco + AWGN): stessa distribuzione del generatore per-simbolo
            # (``generate_transmitted_batch``/``apply_channel_batch``), ~50-100x
            # piu' veloce (il loop Python con 100 iterazioni di mappa per simbolo
            # era il collo di bottiglia della pre-generazione su Colab).
            seed_sym = rng.integers(0, _SEED_HIGH, size=n_per_combo, dtype=np.int64)
            x_ref = generate_transmitted_batch(config, bits, seed_sym)
            x_arr, tau, f_d = apply_channel_batch(x_ref, k, float(snr), config, rng)

            # Verifiche pre-salvataggio.
            if not np.all(np.isfinite(x_arr)):
                raise RuntimeError(f"array x non finito per {split} snr={snr} k={k}")
            if np.any(tau < 0.0) or np.any(tau > float(max_delay)):
                raise RuntimeError(f"label tau fuori range per {split} snr={snr} k={k}")
            if np.any(f_d < 0.0) or np.any(f_d > max_doppler):
                raise RuntimeError(f"label f_d fuori range per {split} snr={snr} k={k}")
            if not np.all(np.isin(bits, (0, 1))):
                raise RuntimeError(f"bit fuori da 0/1 per {split} snr={snr} k={k}")

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
                "salvato %s: %d simboli, shape x=%s, tau in [%.0f, %.0f], f_d in [%.3e, %.3e]",
                out_path, n_per_combo, x_arr.shape,
                float(np.min(tau)), float(np.max(tau)),
                float(np.min(f_d)), float(np.max(f_d)),
            )
            paths.append(out_path)

    logger.info(
        "split '%s': %d file generati, %d simboli per combinazione (combos=%d)",
        split, len(paths), n_per_combo, num_combos,
    )
    return paths

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI del generatore dataset: ``--config``, ``--splits``, ``--output-dir``.

    Args:
        argv: Argomenti da riga di comando (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: Se ``--config`` manca (argparse).
        ValueError: Se split non validi oppure config non valida.
    """
    parser = argparse.ArgumentParser(
        description="Generatore dataset Dual-Head Ultra-CAN (ISAC in IoD), Fase 1.1"
    )
    parser.add_argument("--config", required=True, help="path della config esperimento (YAML)")
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
        raise ValueError("--splits non contiene split validi")
    for split in splits:
        if split not in _SPLITS:
            raise ValueError(f"split non valido: {split!r} (attesi: {sorted(_SPLITS)})")

    # Default coerente con il runner: directory hashata data/raw/<hash> (stessa
    # fonte di pipeline.prepare_dataset). --output-dir resta l'override esplicito.
    from src.utils.dataset_utils import get_dataset_dir  # import lazy

    output_dir = Path(args.output_dir) if args.output_dir else get_dataset_dir(config)
    logger.info(
        "avvio generazione dataset: splits=%s, output_dir=%s, seed=%s",
        splits, output_dir, config["general"]["seed"],
    )
    for split in splits:
        generate_dataset(config, split, output_dir)
    logger.info("generazione dataset completata per %s", splits)

if __name__ == "__main__":
    main()

# ============================================================================
