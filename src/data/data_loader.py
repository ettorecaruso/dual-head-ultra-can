"""Data loader per l'architettura Dual-Head Ultra-CAN (ISAC in IoD).

  - Docstring iniziale con lo scopo del modulo                     [questo blocco]
  - Type hints su tutti i parametri e i ritorni
  - Logging strutturato (INFO eventi principali, DEBUG shape e valori)
  - Docstring Google-style su ogni funzione pubblica
  - Formule del paper commentate con riferimento Sezione e Equazione
  - Shape-check e range-check (nessun NaN/Inf)
  - Nessun parametro hard-coded: tutto da YAML via ``src.utils.config_loader``
    con verifica fail-fast all'avvio (``_validate_config``)

Responsabilita' (FLAG della pianificazione risolti in questo modulo):
  - l'.npz salva il segnale ricevuto complesso (complex128); la
             conversione real vs I/Q spetta a questo loader ed e' governata
             dalla chiave YAML ``data.feature_mode`` (nessun hard-coding).
  - [ALERT-2] con ``feature_mode="real"`` (``np.real(y)``) si scarta la
             componente di quadratura: energia dimezzata e circa 3 dB di
             SNR effettivo in meno rispetto alla rappresentazione I/Q.

Riferimenti paper (``latex.txt``):
  - Sez. IV-D: i target sensing tau e f_D sono Min-Max normalizzati in [0, 1]
               PRIMA della loss (tau/tau_max, f_D/f_Dmax; min = 0 per costruzione).
  - Sez. IV  : input del modello X_input in R^{100 x 1} -> feature_mode="real".

Dipendenze:
  - ``src.utils.config_loader`` (``load_config``, ``DEFAULT_BASE_CONFIG_PATH``)
  - ``src.utils.logger`` (``log_config_summary``, ``setup_logging``)
  - ``src.data.dataset_generator`` (``build_snr_grid``: griglia SNR condivisa,
    dipendenza unidirezionale)
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

# --- Bootstrap del path del repository (esecuzione come script) -------------
# Consente `python src/data/data_loader.py` da qualsiasi CWD: aggiunge la root
# del repo (paper/) a sys.path PRIMA degli import dei moduli interni.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset_generator import build_snr_grid, resolve_n_per_combo  # tipi condivisi (PLAN Sez. 5.2)
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
    save_config_snapshot,
)
from src.utils.logger import log_config_summary, setup_logging

# Logger di modulo (`` Sez. 5).
logger = logging.getLogger(__name__)

# --- Costanti (non parametri esperimento, quindi non nel YAML) --------------
_SPLITS = frozenset({"train", "val", "test"})
_FEATURE_MODES = frozenset({"real", "iq"})
# Normalizzazioni zero-parametro delle features (data.feature_norm):
#   "none"   -> nessuna (comportamento originale del paper)
#   "sign"   -> allineamento del segno della proiezione reale per frame
#   "energy" -> sign-alignment + normalizzazione L2 per frame (AGC)
_FEATURE_NORMS = frozenset({"none", "sign", "energy"})
_NPZ_KEYS = frozenset({"x", "bit", "tau", "f_d", "snr_db", "k", "seed"})
# Nomi generati da dataset_generator.py: <split>_snr<snr:g>_echo<k>.npz
_NPZ_NAME_RE = re.compile(
    r"^(train|val|test)_snr(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_echo(\d+)\.npz$"
)
_SNR_ROUND_DECIMALS = 6  # arrotondamento chiavi (SNR, K) per confronti float robusti
_RANGE_TOL = 1e-9        # tolleranza dei range-check ( Sez. 1.2)
_MIN_SEQUENCE_LENGTH = 1
_MIN_BATCH_SIZE = 1
_NUM_FEATURES_REAL = 1   # input (B, N_seq, 1) — paper Sez. IV
_NUM_FEATURES_IQ = 2     # input (B, N_seq, 2) — variante I/Q
# Canale di RIFERIMENTO (sequenza caotica trasmessa, nota al ricevitore ISAC):
# viene SEMPRE aggiunto come ultimo canale delle features (zero-parametro,
# rigenerato deterministicamente dal seed salvato nell'.npz). Con "iq" l'input
# del modello diventa (B, N_seq, 3) = [Re(y), Im(y), x_ref]; con "real"
# (B, N_seq, 2) = [Re(y), x_ref]. Il canale di riferimento alimenta SOLO la
# sensing head (matched-filter con cancellazione del path diretto); il backbone
# di comunicazione NON lo vede (nessun shortcut sulla modulazione CSK).
_NUM_REFERENCE_CHANNELS = 1

# Tipo condiviso del modulo.
DataDict = Dict[str, np.ndarray]


def model_input_channels(feature_mode: str) -> int:
    """Numero di canali dell'input del modello (base + canale di riferimento).

    Args:
        feature_mode: ``"real"`` oppure ``"iq"``.

    Returns:
        ``2`` per ``"real"`` (Re + x_ref), ``3`` per ``"iq"`` (Re + Im + x_ref).

    Raises:
        ValueError: Se ``feature_mode`` non e' valido.
    """
    if feature_mode == "iq":
        return _NUM_FEATURES_IQ + _NUM_REFERENCE_CHANNELS
    if feature_mode == "real":
        return _NUM_FEATURES_REAL + _NUM_REFERENCE_CHANNELS
    raise ValueError(
        f"feature_mode deve essere in {sorted(_FEATURE_MODES)}, ricevuto: {feature_mode!r}"
    )

def _parse_npz_name(path: Path) -> Tuple[str, float, int]:
    """Estrae ``(split, snr, k)`` dal nome di un file ``.npz`` del generatore.

    Args:
        path: Path del file ``.npz`` (es. ``train_snr5_echo1.npz``).

    Returns:
        Tupla ``(split, snr, k)`` con ``snr`` float e ``k`` intero.

    Raises:
        ValueError: Se il nome non rispetta il pattern del generatore
            (``<split>_snr<snr:g>_echo<k>.npz``).
    """
    match = _NPZ_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"nome file non riconosciuto: {path.name!r} "
            f"(atteso: <split>_snr<snr>_echo<k>.npz)"
        )
    split, snr_str, k_str = match.groups()
    return split, float(snr_str), int(k_str)

def _validate_npz_arrays(
    x: np.ndarray,
    bit: np.ndarray,
    tau: np.ndarray,
    f_d: np.ndarray,
    sequence_length: int,
    max_delay: float,
    max_doppler: float,
) -> None:
    """Validazione difensiva di un blocco di dati caricato da ``.npz``.

    Applica i check di shape, finitezza e range richiesti da 
    su array gia' materializzati (usata sia per singolo file sia
    per il risultato aggregato: difesa in profondita').

    Args:
        x: Segnale ricevuto complesso, shape (N, sequence_length).
        bit: Bit trasmessi, shape (N,).
        tau: Ritardi target, shape (N,).
        f_d: Doppler target, shape (N,).
        sequence_length: Lunghezza della sequenza (``data.sequence_length``).
        max_delay: Estremo superiore di tau (``data.max_delay``).
        max_doppler: Estremo superiore di f_d (``data.max_doppler``).

    Raises:
        ValueError: Se una shape, un range o la finitezza sono violati.
    """
    if x.ndim != 2 or x.shape[1] != sequence_length:
        raise ValueError(
            f"x deve avere shape (N, {sequence_length}), ricevuto: {x.shape}"
        )
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("blocco senza campioni (n=0)")
    if not (len(bit) == len(tau) == len(f_d) == n):
        raise ValueError(
            "lunghezze incoerenti: "
            f"x={n}, bit={len(bit)}, tau={len(tau)}, f_d={len(f_d)}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("x contiene NaN/Inf ( Sez. 1.2)")
    if not np.all(np.isin(bit, (0, 1))):
        raise ValueError("bit deve contenere solo i valori 0/1")
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0) or np.any(tau > max_delay + _RANGE_TOL):
        raise ValueError(f"tau fuori range [0, {max_delay}] o non finito")
    if not np.all(np.isfinite(f_d)) or np.any(f_d < 0.0) or np.any(f_d > max_doppler + _RANGE_TOL):
        raise ValueError(f"f_d fuori range [0, {max_doppler}] o non finito")

def load_npz_files(
    data_dir: Path,
    snr_values: List[float],
    echoes: List[int],
    split: str,
    config: Dict[str, Any],
) -> DataDict:
    """Carica e concatena i file ``.npz`` prodotti da ``dataset_generator.py``.

    I file ``<split>_snr<snr>_echo<k>.npz`` vengono caricati in ordine
    deterministico (stessa griglia del generatore: SNR esterno, K interno) e
    concatenati in un unico ``DataDict`` con ``snr_db`` e ``k`` per-campione.

    Args:
        data_dir: Directory contenente i file ``.npz`` (default ``data.raw_dir``).
        snr_values: Griglia SNR (dB) attesa, nello stesso ordine del generatore.
        echoes: Valori K attesi, nello stesso ordine del generatore.
        split: ``"train"``, ``"val"`` oppure ``"test"``.
        config: Config completa (chiavi ``data.sequence_length``,
            ``data.max_delay``, ``data.max_doppler``).

    Returns:
        ``DataDict`` con chiavi: ``x`` (complex128, (N, N_seq)), ``bit``
        (uint8, (N,)), ``tau``/``f_d`` (float64, (N,)), ``snr_db``
        (float64, (N,)) e ``k`` (int64, (N,)).

    Raises:
        ValueError: Se ``split``/``snr_values``/``echoes`` non sono validi,
            la griglia (SNR, K) e' incompleta, le chiavi del ``.npz`` mancano
            o i range di tau/f_d/bit sono violati.
        FileNotFoundError: Se ``data_dir`` non esiste oppure nessun file
            valido viene trovato per lo split richiesto.
    """
    if split not in _SPLITS:
        raise ValueError(f"split deve essere in {sorted(_SPLITS)}, ricevuto: {split!r}")
    if not snr_values:
        raise ValueError("snr_values non puo' essere vuoto")
    if not echoes:
        raise ValueError("echoes non puo' essere vuoto")
    if not all(math.isfinite(float(s)) for s in snr_values):
        raise ValueError(f"snr_values deve contenere solo valori finiti: {snr_values!r}")
    if not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in echoes):
        raise ValueError(f"echoes deve contenere int >= 0, ricevuto: {echoes!r}")

    data = config["data"]
    sequence_length = int(data["sequence_length"])
    max_delay = float(data["max_delay"])
    max_doppler = float(data["max_doppler"])

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir non esiste o non e' una directory: {data_dir}")

    # Griglia attesa nello stesso ordine del generatore: SNR esterno, K interno.
    combos: List[Tuple[float, int]] = [
        (float(snr), int(k)) for snr in snr_values for k in echoes
    ]

    # Glob robusto: il parse del nome evita discrepanze di formattazione float.
    files_by_combo: Dict[Tuple[float, int], Path] = {}
    for path in sorted(data_dir.glob(f"{split}_*.npz")):
        try:
            f_split, snr, k = _parse_npz_name(path)
        except ValueError as exc:
            logger.warning("file ignorato: %s", exc)
            continue
        if f_split != split:
            continue
        if (snr, k) in files_by_combo:
            logger.warning("combinazione (snr=%s, k=%d) duplicata: ignorato %s", snr, k, path.name)
            continue
        files_by_combo[(snr, k)] = path

    if not files_by_combo:
        raise FileNotFoundError(
            f"nessun file .npz valido per split={split!r} in {data_dir}"
        )

    missing = [combo for combo in combos if combo not in files_by_combo]
    if missing:
        raise ValueError(
            f"griglia (SNR, K) incompleta per split={split!r}: "
            f"file mancanti per {missing}"
        )

    x_parts: List[np.ndarray] = []
    bit_parts: List[np.ndarray] = []
    tau_parts: List[np.ndarray] = []
    f_d_parts: List[np.ndarray] = []
    snr_parts: List[np.ndarray] = []
    k_parts: List[np.ndarray] = []
    seed_parts: List[np.ndarray] = []

    for snr, k in combos:  # ordine deterministico = ordine del generatore
        path = files_by_combo[(snr, k)]
        with np.load(path, allow_pickle=False) as npz:
            missing_keys = _NPZ_KEYS - set(npz.files)
            if missing_keys:
                raise ValueError(
                    f"chiavi mancanti in {path.name}: {sorted(missing_keys)}"
                )
            x_file = np.asarray(npz["x"])
            bit_file = np.asarray(npz["bit"])
            tau_file = np.asarray(npz["tau"])
            f_d_file = np.asarray(npz["f_d"])
            seed_file = np.asarray(npz["seed"])
            snr_file = float(np.asarray(npz["snr_db"]))
            k_file = int(np.asarray(npz["k"]))

        # Coerenza nome-file vs metadati salvati nel .npz.
        if abs(snr_file - snr) > 1e-6 or k_file != k:
            raise ValueError(
                f"{path.name}: snr_db={snr_file} o k={k_file} incoerenti col nome file"
            )

        _validate_npz_arrays(
            x_file,
            bit_file,
            tau_file,
            f_d_file,
            sequence_length=sequence_length,
            max_delay=max_delay,
            max_doppler=max_doppler,
        )
        n_file = int(x_file.shape[0])
        if seed_file.shape != (n_file,):
            raise ValueError(
                f"{path.name}: seed shape {seed_file.shape} incoerente con n={n_file}"
            )
        x_parts.append(x_file)
        bit_parts.append(bit_file)
        tau_parts.append(tau_file)
        f_d_parts.append(f_d_file)
        snr_parts.append(np.full(n_file, snr_file, dtype=np.float64))
        k_parts.append(np.full(n_file, k_file, dtype=np.int64))
        seed_parts.append(seed_file)
        logger.debug(
            "caricato %s: %d campioni, snr=%.4g, k=%d", path.name, n_file, snr_file, k_file,
        )

    data_out: DataDict = {
        "x": np.concatenate(x_parts, axis=0),
        "bit": np.concatenate(bit_parts, axis=0),
        "tau": np.concatenate(tau_parts, axis=0),
        "f_d": np.concatenate(f_d_parts, axis=0),
        "snr_db": np.concatenate(snr_parts, axis=0),
        "k": np.concatenate(k_parts, axis=0),
        "seed": np.concatenate(seed_parts, axis=0),
    }

    # Validazione aggregata (Sez. 1.2, difesa in profondita').
    _validate_npz_arrays(
        data_out["x"],
        data_out["bit"],
        data_out["tau"],
        data_out["f_d"],
        sequence_length=sequence_length,
        max_delay=max_delay,
        max_doppler=max_doppler,
    )

    logger.info(
        "split '%s': %d campioni totali, x shape=%s, tau in [%.2f, %.2f], f_d in [%.3e, %.3e]",
        split,
        int(data_out["x"].shape[0]),
        data_out["x"].shape,
        float(np.min(data_out["tau"])),
        float(np.max(data_out["tau"])),
        float(np.min(data_out["f_d"])),
        float(np.max(data_out["f_d"])),
    )
    return data_out

def normalize_targets(
    tau: np.ndarray,
    f_d: np.ndarray,
    tau_max: float,
    fd_max: float,
) -> np.ndarray:
    """Min-Max normalizza i target sensing in [0, 1] (paper Sez. IV-D).

    Il minimo e' 0 per costruzione (tau >= 0, f_d >= 0), quindi la
    normalizzazione si riduce a ``tau / tau_max`` e ``f_d / fd_max``.

    Args:
        tau: Ritardi target (float64, shape (N,)).
        f_d: Doppler target (float64, shape (N,)).
        tau_max: Estremo superiore di tau (``data.max_delay``).
        fd_max: Estremo superiore di f_d (``data.max_doppler``).

    Returns:
        Array ``float32`` shape (N, 2): colonne ``[tau_norm, f_d_norm]``
        entrambe in [0, 1], pronto per la loss MSE.

    Raises:
        ValueError: Se gli input non sono validi (shape non allineate,
            NaN/Inf, ``tau_max``/``fd_max`` <= 0).
        RuntimeError: Se l'output esce da [0, 1] oltre la tolleranza.
    """
    tau_arr = np.asarray(tau, dtype=np.float64)
    f_d_arr = np.asarray(f_d, dtype=np.float64)
    if tau_arr.ndim != 1 or f_d_arr.ndim != 1:
        raise ValueError(
            f"tau e f_d devono essere 1D, ricevuti: {tau_arr.shape}, {f_d_arr.shape}"
        )
    if tau_arr.shape != f_d_arr.shape:
        raise ValueError(
            f"tau e f_d devono avere la stessa shape: {tau_arr.shape} vs {f_d_arr.shape}"
        )
    if not (math.isfinite(tau_max) and tau_max > 0.0):
        raise ValueError(f"tau_max deve essere finito e > 0, ricevuto: {tau_max!r}")
    if not (math.isfinite(fd_max) and fd_max > 0.0):
        raise ValueError(f"fd_max deve essere finito e > 0, ricevuto: {fd_max!r}")
    if not (np.all(np.isfinite(tau_arr)) and np.all(np.isfinite(f_d_arr))):
        raise ValueError("tau/f_d contengono NaN/Inf ( Sez. 1.2)")

    # Sez. IV-D: Min-Max con min = 0 per costruzione.
    tau_norm = tau_arr / float(tau_max)
    f_d_norm = f_d_arr / float(fd_max)

    if np.any(tau_norm < -_RANGE_TOL) or np.any(tau_norm > 1.0 + _RANGE_TOL):
        raise RuntimeError(
            f"tau_norm fuori da [0, 1]: range [{np.min(tau_norm)}, {np.max(tau_norm)}]"
        )
    if np.any(f_d_norm < -_RANGE_TOL) or np.any(f_d_norm > 1.0 + _RANGE_TOL):
        raise RuntimeError(
            f"f_d_norm fuori da [0, 1]: range [{np.min(f_d_norm)}, {np.max(f_d_norm)}]"
        )

    targets = np.stack([tau_norm, f_d_norm], axis=-1).astype(np.float32)
    logger.debug(
        "target normalizzati: shape=%s, tau_norm in [%.4f, %.4f], f_d_norm in [%.4f, %.4f]",
        targets.shape,
        float(np.min(tau_norm)),
        float(np.max(tau_norm)),
        float(np.min(f_d_norm)),
        float(np.max(f_d_norm)),
    )
    return targets

def regenerate_reference(bit: Any, seed: Any, config: Dict[str, Any]) -> np.ndarray:
    """Rigenera la sequenza caotica trasmessa (riferimento ISAC) da bit e seed.

    Il ricevitore ISAC conosce la propria forma d'onda trasmessa: la sequenza
    trasmessa ``x_ref`` e' rigenerabile deterministicamente dal seed salvato
    nell'.npz (stessa logica di ``dataset_generator.generate_dataset``:
    ``map_type_for_bit(bit)`` -> ``generate_chaotic_sequence`` ->
    ``_center_normalize_symbol``). La sequenza di riferimento alimenta la
    sensing head (matched-filter con cancellazione del path diretto).

    Args:
        bit: Bit trasmesso (0 o 1).
        seed: Seed della sequenza caotica (int >= 0).
        config: Config completa (chiavi ``data.map_type``, ``data.map_param``,
            ``data.sequence_length``).

    Returns:
        Array float64 shape ``(N_seq,)`` = sequenza trasmessa centrata e
        normalizzata (media 0, energia 1).

    Raises:
        ValueError: Se la rigenerazione fallisce (delegato al generatore).
    """
    from src.data.dataset_generator import (  # import locale (evita cicli)
        _center_normalize_symbol,
        _map_type_for_bit,
        generate_chaotic_sequence,
    )

    data = config["data"]
    seq_len = int(data["sequence_length"])
    map_type = str(data["map_type"])
    map_param = float(data["map_param"])
    x = generate_chaotic_sequence(
        _map_type_for_bit(map_type, int(bit)), map_param, int(seed), seq_len
    )
    return _center_normalize_symbol(x)


def build_reference_matrix(
    bit: np.ndarray, seed: np.ndarray, config: Dict[str, Any]
) -> np.ndarray:
    """Costruisce la matrice di riferimento (N, N_seq) per ogni campione.

    Args:
        bit: Array dei bit, shape (N,).
        seed: Array dei seed, shape (N,).
        config: Config completa (``data.sequence_length``).

    Returns:
        Array float32 shape (N, N_seq) con la sequenza trasmessa per campione.

    Raises:
        ValueError: Se ``bit``/``seed`` non sono 1D o non hanno la stessa
            lunghezza.
    """
    bit_arr = np.asarray(bit)
    seed_arr = np.asarray(seed)
    if bit_arr.ndim != 1 or seed_arr.ndim != 1 or bit_arr.shape != seed_arr.shape:
        raise ValueError(
            f"bit/seed devono essere 1D allineati, ricevuti: {bit_arr.shape}, {seed_arr.shape}"
        )
    n = int(bit_arr.shape[0])
    seq_len = int(config["data"]["sequence_length"])
    out = np.empty((n, seq_len), dtype=np.float32)
    for i in range(n):
        out[i] = regenerate_reference(bit_arr[i], seed_arr[i], config).astype(np.float32)
    logger.debug("matrice di riferimento costruita: shape=%s", out.shape)
    return out


def _build_feature_matrix(
    y_complex: np.ndarray,
    feature_mode: str,
    feature_norm: Optional[str] = None,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Converte il segnale ricevuto complesso in features ``float32``.

    Args:
        y_complex: Segnale ricevuto complesso, shape (N, N_seq).
        feature_mode: ``"real"`` -> features (N, N_seq, 1) con ``np.real(y)``;
            ``"iq"`` -> features (N, N_seq, 2) con ``[Re(y), Im(y)]``.
        feature_norm: Normalizzazione zero-parametro (``"none"``, ``"sign"``,
            ``"energy"``). ``None`` equivale a ``"none"`` (retro-compatibile).
        reference: Matrice di riferimento (N, N_seq) opzionale; se presente
            viene concatenata come ULTIMO canale (canale ISAC per la sensing
            head). Default ``None`` (nessun canale extra).

    Returns:
        Features ``float32`` shape (N, N_seq, num_features).

    Raises:
        ValueError: Se ``feature_mode``/``feature_norm`` non sono validi oppure
            ``y_complex`` non e' 2D o contiene NaN/Inf.
        RuntimeError: Se l'output non e' finito dopo la conversione float32.
    """
    y = np.asarray(y_complex)
    if y.ndim != 2:
        raise ValueError(f"y_complex deve essere 2D, ricevuto: {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_complex contiene NaN/Inf ( Sez. 1.2)")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"feature_mode deve essere in {sorted(_FEATURE_MODES)}, ricevuto: {feature_mode!r}"
        )
    if feature_norm is None:
        feature_norm = "none"
    if feature_norm not in _FEATURE_NORMS:
        raise ValueError(
            f"feature_norm deve essere in {sorted(_FEATURE_NORMS)}, ricevuto: {feature_norm!r}"
        )

    if feature_mode == "real":
        # [ALERT-2] np.real scarta la componente di quadratura (~3 dB).
        features = np.real(y)[..., np.newaxis]  # (N, N_seq, 1)
    else:
        features = np.stack([np.real(y), np.imag(y)], axis=-1)  # (N, N_seq, 2)

    # Normalizzazione zero-parametro per frame (solo mode "real": il segno della
    # proiezione reale Re(h_c) = |h_c| cos(theta) e' aleatorio e la sua ambiguita'
    # di fase penalizza i modelli senza capacita' di invarianza; l'allineamento
    # usa il fatto che entrambe le mappe CSK hanno media > 0).
    if feature_norm != "none":
        if feature_mode != "real":
            raise ValueError(
                "feature_norm != 'none' richiede feature_mode='real' "
                f"(ricevuto feature_mode={feature_mode!r})"
            )
        x = features[..., 0]
        if feature_norm in ("sign", "energy"):
            x = np.where(x.mean(axis=-1, keepdims=True) < 0.0, -x, x)
        if feature_norm == "energy":
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            norm = np.where(norm == 0.0, 1.0, norm)
            x = x / norm
        features = x[..., np.newaxis]

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.ndim != 2 or ref.shape[0] != features.shape[0]:
            raise ValueError(
                f"reference shape attesa (N, N_seq) con N={features.shape[0]}, "
                f"ricevuta: {ref.shape}"
            )
        if ref.shape[1] != features.shape[1]:
            raise ValueError(
                f"reference lunghezza {ref.shape[1]} incoerente con N_seq={features.shape[1]}"
            )
        features = np.concatenate([features, ref[..., np.newaxis]], axis=-1)

    features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("features non finite dopo la conversione float32")
    logger.debug(
        "features: mode=%s, norm=%s, shape=%s, dtype=%s",
        feature_mode, feature_norm, features.shape, features.dtype,
    )
    return features

def build_tf_dataset(
    data: DataDict,
    batch_size: int,
    config: Dict[str, Any],
    shuffle: bool = True,
    seed: Optional[int] = None,
    feature_mode: Optional[str] = None,
    feature_norm: Optional[str] = None,
) -> tf.data.Dataset:
    """Costruisce il ``tf.data.Dataset`` per training e valutazione.

    Features: ``(B, N_seq, 1)`` con ``feature_mode="real"`` oppure
    ``(B, N_seq, 2)`` con ``"iq"``. Labels: ``{"comm": bit,
    "sensing": [tau_norm, f_d_norm]}``. ``seed`` e ``feature_mode`` di
    default sono letti dalla config (nessun hard-coding, 
    Sez. 7); i parametri espliciti servono solo come override (es. test).

    Args:
        data: ``DataDict`` prodotto da ``load_npz_files``.
        batch_size: Dimensione del batch (int > 0).
        config: Config completa (``data.feature_mode``, ``general.seed``,
            ``data.max_delay``, ``data.max_doppler``, ``data.sequence_length``).
        shuffle: Se ``True`` applica shuffle con buffer pieno (uniforme e
            riproducibile via ``seed``).
        seed: Seed dello shuffle; default ``general.seed`` della config.
        feature_mode: Override di ``data.feature_mode`` (per i test).

    Returns:
        Dataset TF con ``element_spec``: features ``(None, N_seq, num_feat)``,
        labels ``comm (None,)`` e ``sensing (None, 2)``.

    Raises:
        ValueError: Se ``batch_size``/``seed`` non validi, ``feature_mode``
            non ammesso, ``data`` senza chiavi/campioni, shape non attese.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(
            f"batch_size deve essere un int >= {_MIN_BATCH_SIZE}, ricevuto: {batch_size!r}"
        )
    if seed is None:
        seed_val = config["general"].get("seed")
        if seed_val is None:
            raise ValueError("chiave mancante: general.seed")
        seed = int(seed_val)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed deve essere un int >= 0, ricevuto: {seed!r}")
    if feature_mode is None:
        feature_mode = config["data"].get("feature_mode")
        if not isinstance(feature_mode, str):
            raise ValueError("chiave mancante o non stringa: data.feature_mode")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"feature_mode deve essere in {sorted(_FEATURE_MODES)}, ricevuto: {feature_mode!r}"
        )
    if feature_norm is None:
        feature_norm = config["data"].get("feature_norm", "none")
        if not isinstance(feature_norm, str):
            raise ValueError("chiave mancante o non stringa: data.feature_norm")
    if feature_norm not in _FEATURE_NORMS:
        raise ValueError(
            f"feature_norm deve essere in {sorted(_FEATURE_NORMS)}, ricevuto: {feature_norm!r}"
        )
    if feature_norm != "none" and feature_mode != "real":
        raise ValueError(
            "feature_norm != 'none' richiede feature_mode='real' "
            f"(ricevuto feature_mode={feature_mode!r})"
        )

    missing = [key for key in ("x", "bit", "tau", "f_d", "seed") if key not in data]
    if missing:
        raise ValueError(f"data manca delle chiavi: {missing}")
    n = int(data["x"].shape[0])
    if n < 1:
        raise ValueError("data non contiene campioni (n < 1)")

    sequence_length = int(config["data"]["sequence_length"])
    max_delay = float(config["data"]["max_delay"])
    max_doppler = float(config["data"]["max_doppler"])

    # Features (B, N_seq, num_features) e target normalizzati in [0,1] (Sez. IV-D).
    # Il canale di riferimento (sequenza trasmessa nota al ricevitore ISAC)
    # viene rigenerato dal seed e concatenato come ultimo canale; se il DataDict
    # espone gia' ``x_ref`` (precalcolato una sola volta in pipeline.load_datasets)
    # lo si riusa (evita di rigenerarlo a ogni load).
    if "x_ref" in data and data["x_ref"] is not None:
        reference = data["x_ref"]
    else:
        reference = build_reference_matrix(data["bit"], data["seed"], config)
    features = _build_feature_matrix(data["x"], feature_mode, feature_norm, reference)
    targets = normalize_targets(
        data["tau"], data["f_d"], tau_max=max_delay, fd_max=max_doppler
    )
    bits = np.asarray(data["bit"], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, {"comm": bits, "sensing": targets})
    )
    if shuffle:
        # Buffer = intero dataset: shuffle uniforme e riproducibile col seed.
        dataset = dataset.shuffle(
            buffer_size=n, seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Shape-check esplicito su element_spec (canale di riferimento incluso).
    num_features = model_input_channels(feature_mode)
    features_spec = dataset.element_spec[0]
    comm_spec = dataset.element_spec[1]["comm"]
    sensing_spec = dataset.element_spec[1]["sensing"]
    if features_spec.shape.as_list() != [None, sequence_length, num_features]:
        raise ValueError(
            f"shape features attesa [None, {sequence_length}, {num_features}], "
            f"ricevuta: {features_spec.shape}"
        )
    if comm_spec.shape.as_list() != [None]:
        raise ValueError(f"shape comm attesa [None], ricevuta: {comm_spec.shape}")
    if sensing_spec.shape.as_list() != [None, 2]:
        raise ValueError(f"shape sensing attesa [None, 2], ricevuta: {sensing_spec.shape}")

    logger.info(
        "dataset costruito: n=%d, batch_size=%d, feature_mode=%s, shuffle=%s, seed=%d",
        n, batch_size, feature_mode, shuffle, seed,
    )
    logger.debug(
        "element_spec: features=%s, comm=%s, sensing=%s",
        features_spec, comm_spec, sensing_spec,
    )
    return dataset

def verify_snr_balance(
    data: DataDict,
    snr_values: List[float],
    echoes: List[int],
    expected_per_combo: Optional[int] = None,
) -> Dict[Tuple[float, int], int]:
    """Verifica il bilanciamento del dataset per (SNR, K).

    Ogni combinazione (SNR, K) deve avere lo stesso numero di campioni e
    tutti i punti della griglia devono essere presenti (nessuna combinazione
    estranea). Se ``expected_per_combo`` e' fornito, ogni conteggio deve
    anche coincidere col valore atteso.

    Args:
        data: ``DataDict`` con array per-campione ``snr_db`` e ``k``.
        snr_values: Griglia SNR attesa.
        echoes: Valori K attesi.
        expected_per_combo: Numero atteso di campioni per (SNR, K); se
            fornito, ogni conteggio deve coincidere.

    Returns:
        Dict ``{(snr, k): count}`` con chiavi arrotondate a 6 decimali.

    Raises:
        ValueError: Se gli array sono mancanti/incoerenti, la griglia non e'
            coperta esattamente, i conteggi non sono uniformi oppure
            differiscono da ``expected_per_combo``.
    """
    for key in ("snr_db", "k"):
        if key not in data:
            raise ValueError(f"data manca della chiave '{key}' per il bilanciamento")
    snr_arr = np.asarray(data["snr_db"], dtype=np.float64)
    k_arr = np.asarray(data["k"], dtype=np.int64)
    if snr_arr.ndim != 1 or k_arr.ndim != 1:
        raise ValueError("snr_db e k devono essere array 1D per-campione")
    if len(snr_arr) != len(k_arr):
        raise ValueError(
            f"snr_db ({len(snr_arr)}) e k ({len(k_arr)}) devono avere la stessa lunghezza"
        )
    if expected_per_combo is not None:
        if (
            not isinstance(expected_per_combo, int)
            or isinstance(expected_per_combo, bool)
            or expected_per_combo <= 0
        ):
            raise ValueError(
                f"expected_per_combo deve essere un int > 0, ricevuto: {expected_per_combo!r}"
            )

    # Arrotondamento a 6 decimali per confronti float robusti.
    counts: Dict[Tuple[float, int], int] = Counter(
        (round(float(snr_arr[i]), _SNR_ROUND_DECIMALS), int(k_arr[i]))
        for i in range(len(snr_arr))
    )

    expected_combos = {
        (round(float(snr), _SNR_ROUND_DECIMALS), int(k)) for snr in snr_values for k in echoes
    }
    got_combos = set(counts)
    missing = sorted(expected_combos - got_combos)
    extra = sorted(got_combos - expected_combos)
    if missing or extra:
        raise ValueError(
            f"griglia (SNR, K) sbilanciata: mancanti={missing}, estranei={extra}"
        )

    values = list(counts.values())
    if len(set(values)) != 1:
        raise ValueError(
            "dataset sbilanciato per (SNR, K): conteggi non uniformi -> "
            f"{dict(sorted(counts.items()))}"
        )
    if expected_per_combo is not None and values[0] != expected_per_combo:
        raise ValueError(
            f"dataset sbilanciato: conteggio per (SNR, K) = {values[0]}, "
            f"atteso = {expected_per_combo}"
        )

    logger.info(
        "bilanciamento verificato: %d campioni per ciascuna delle %d combinazioni (SNR, K)",
        values[0], len(expected_combos),
    )
    return dict(counts)

def _expected_per_combo(config: Dict[str, Any], split: str, num_combos: int) -> int:
    """Numero atteso di campioni per combinazione (SNR, K) in uno split.

    Delega a ``dataset_generator.resolve_n_per_combo``: unica fonte di verita'
    del conteggio per-combo (nessuna duplicazione della logica di e tra generatore e loader, incluso il controllo di parita' dei bit
    0/1 che prima mancava nel loader).

    Args:
        config: Config completa (chiavi ``data.num_symbols_train/val/test``).
        split: ``"train"``, ``"val"`` oppure ``"test"``.
        num_combos: Numero di combinazioni (SNR, K) della griglia.

    Returns:
        Campioni attesi per ogni (SNR, K) nello split richiesto.

    Raises:
        ValueError: Se ``split``/``num_combos`` non validi oppure
            ``num_symbols_train`` non e' divisibile per ``num_combos`` oppure
            il conteggio per-combo non e' pari (bit 0/1 bilanciati).
    """
    return resolve_n_per_combo(config, split, num_combos)

def _validate_config(config: Dict[str, Any]) -> None:
    """Validazione fail-fast delle chiavi usate dal loader.

    Args:
        config: Config completa (dopo ``load_config``).

    Raises:
        TypeError: Se ``config`` non e' un dict.
        ValueError: Se una chiave richiesta manca oppure un valore e'
            incoerente (fail-fast,).
    """
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")
    general = config.get("general")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(general, dict):
        raise ValueError("sezione 'general' mancante o non dict")
    if not isinstance(data, dict):
        raise ValueError("sezione 'data' mancante o non dict")
    if not isinstance(training, dict):
        raise ValueError("sezione 'training' mancante o non dict")

    seed = general.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"general.seed deve essere un int >= 0, ricevuto: {seed!r}")
    experiment_name = general.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("general.experiment_name deve essere una stringa non vuota")

    sequence_length = data.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < _MIN_SEQUENCE_LENGTH
    ):
        raise ValueError(
            f"data.sequence_length deve essere un int >= {_MIN_SEQUENCE_LENGTH}, "
            f"ricevuto: {sequence_length!r}"
        )

    feature_mode = data.get("feature_mode")
    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"data.feature_mode deve essere in {sorted(_FEATURE_MODES)}, ricevuto: {feature_mode!r}"
        )

    for key in ("max_delay", "max_doppler"):
        value = data.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"data.{key} deve essere finito e > 0, ricevuto: {value!r}")
    if float(data["max_doppler"]) >= 0.5:
        raise ValueError("data.max_doppler deve essere < 0.5 (guardia anti-aliasing)")

    snr_range = data.get("snr_range")
    if not isinstance(snr_range, (list, tuple)) or len(snr_range) != 2:
        raise ValueError(f"data.snr_range deve essere [min, max], ricevuto: {snr_range!r}")
    snr_min, snr_max = float(snr_range[0]), float(snr_range[1])
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min >= snr_max:
        raise ValueError(f"data.snr_range invalido: {snr_range!r}")

    snr_step = data.get("snr_step")
    if (
        not isinstance(snr_step, (int, float))
        or isinstance(snr_step, bool)
        or not math.isfinite(float(snr_step))
        or float(snr_step) <= 0.0
    ):
        raise ValueError(f"data.snr_step deve essere > 0, ricevuto: {snr_step!r}")

    echoes = data.get("echoes")
    if not isinstance(echoes, (list, tuple)) or len(echoes) == 0:
        raise ValueError(f"data.echoes deve essere una lista non vuota, ricevuto: {echoes!r}")
    for k in echoes:
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"data.echoes deve contenere int >= 0, ricevuto: {k!r}")

    for key in ("num_symbols_train", "num_symbols_val", "num_symbols_test"):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"data.{key} deve essere un int > 0, ricevuto: {value!r}")

    raw_dir = data.get("raw_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("data.raw_dir deve essere una stringa non vuota")

    batch_size = training.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < _MIN_BATCH_SIZE:
        raise ValueError(
            f"training.batch_size deve essere un int >= {_MIN_BATCH_SIZE}, ricevuto: {batch_size!r}"
        )

    logger.debug(
        "config validata: seq_len=%d, feature_mode=%s, max_delay=%d, max_doppler=%s, batch_size=%d",
        sequence_length,
        feature_mode,
        int(data["max_delay"]),
        data["max_doppler"],
        batch_size,
    )

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI di smoke test del loader: ``--config``, ``--splits``, ``--data-dir``.

    Carica la config (merge base <- esperimento), ricostruisce la griglia SNR
    con ``build_snr_grid`` (tipo condiviso dal generatore) e per ogni split
    esegue: ``load_npz_files`` -> ``verify_snr_balance`` ->
    ``normalize_targets`` -> ``build_tf_dataset``.

    Args:
        argv: Argomenti da riga di comando (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: Se ``--config`` manca (argparse).
        ValueError: Se la config non e' valida, gli split non sono validi
            oppure il dataset e' sbilanciato.
        FileNotFoundError: Se non esistono file ``.npz`` per uno split.
    """
    parser = argparse.ArgumentParser(
        description="Data loader Dual-Head Ultra-CAN (ISAC in IoD), Fase 1.2"
    )
    parser.add_argument("--config", required=True, help="path della config esperimento (YAML)")
    parser.add_argument(
        "--splits", default="train,val,test", help="split da validare, separati da virgola"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override della directory dei dati (default: data.raw_dir)",
    )
    args = parser.parse_args(argv)

    config = load_config(
        config_path=Path(args.config),
        base_config_path=DEFAULT_BASE_CONFIG_PATH,
    )
    _validate_config(config)

    experiment_name = str(config["general"]["experiment_name"])
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

    data_dir = Path(args.data_dir) if args.data_dir else Path(config["data"]["raw_dir"])
    snr_grid = build_snr_grid(config["data"]["snr_range"], config["data"]["snr_step"])
    echoes = [int(k) for k in config["data"]["echoes"]]
    num_combos = len(snr_grid) * len(echoes)
    batch_size = int(config["training"]["batch_size"])
    feature_mode = str(config["data"]["feature_mode"])

    logger.info(
        "avvio data loader: splits=%s, data_dir=%s, griglia SNR=%s, echoes=%s",
        splits, data_dir, snr_grid, echoes,
    )
    for split in splits:
        data = load_npz_files(data_dir, snr_grid, echoes, split, config)
        expected = _expected_per_combo(config, split, num_combos)
        verify_snr_balance(data, snr_grid, echoes, expected_per_combo=expected)
        targets = normalize_targets(
            data["tau"],
            data["f_d"],
            tau_max=float(config["data"]["max_delay"]),
            fd_max=float(config["data"]["max_doppler"]),
        )
        dataset = build_tf_dataset(data, batch_size, config)
        logger.info(
            "split '%s' pronto: %d campioni, targets shape=%s, feature_mode=%s",
            split, int(data["x"].shape[0]), targets.shape, feature_mode,
        )
        del dataset  # verifica di costruzione; il dataset sara' usato da trainer/evaluator
    logger.info("data loader completato")

if __name__ == "__main__":
    main()

# ============================================================================
