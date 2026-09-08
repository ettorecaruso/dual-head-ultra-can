"""Classical DCSK correlator and equations-only receivers."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    load_config,
)
from src.utils.logger import log_config_summary, setup_logging

logger = logging.getLogger(__name__)

_DETECTOR_NAMES: Tuple[str, ...] = ("dcsk", "matched_filter", "energy_detector")
_DCSK_REQUIRED_KEYS: Tuple[str, ...] = ("correlation_length", "threshold")
_ENERGY_EPS = 1e-12

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
            raise ValueError(f"valore non finito nella config: {path} = {value!r}")

def _as_positive_int(value: Any, label: str) -> int:
    
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} deve essere un intero positivo, ricevuto: {value!r}")
    as_int = int(value)
    if float(value) != float(as_int) or as_int <= 0:
        raise ValueError(f"{label} deve essere un intero positivo, ricevuto: {value!r}")
    return as_int

def _split_meta_symbol(y: np.ndarray, beta: int) -> Tuple[np.ndarray, np.ndarray]:
    """Divide un frame DCSK in meta' reference e meta' dati.

    Args:
        y: Array complesso/reale di shape ``(B, 2*beta)`` o ``(2*beta,)``.
        beta: Meta-lunghezza del meta-simbolo (``correlation_length``).

    Returns:
        Coppia ``(ref, data)``, entrambe di shape ``(B, beta)``.

    Raises:
        TypeError: Se ``y`` non e' un ``np.ndarray``.
        ValueError: Se ``y`` non ha ultima dimensione ``2*beta`` oppure contiene
            NaN/Inf.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y deve essere np.ndarray, ricevuto: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y deve avere shape (B, 2*beta) o (2*beta,), ricevuta: {y.shape}")
    if beta < 1:
        raise ValueError(f"beta deve essere >= 1, ricevuto: {beta!r}")
    if y.shape[1] != 2 * beta:
        raise ValueError(
            f"ultima dimensione di y deve essere 2*beta={2 * beta}, "
            f"ricevuta: {y.shape[1]} (shape {y.shape})"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene NaN/Inf ( Sez. 1.2)")
    ref = y[:, :beta]
    data = y[:, beta:]
    return ref, data

def dcsk_correlator_demodulate(
    y: np.ndarray,
    ref: Optional[np.ndarray] = None,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y deve essere np.ndarray, ricevuto: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y deve avere shape (B, 2*beta) o (2*beta,), ricevuta: {y.shape}")
    length = y.shape[1]
    if length < 2 or length % 2 != 0:
        raise ValueError(
            f"y deve avere lunghezza pari >= 2 (meta-simbolo [ref, data]), ricevuta: {length}"
        )
    beta = length // 2
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene NaN/Inf ( Sez. 1.2)")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold deve essere finito, ricevuto: {threshold!r}")

    ref_internal, data = _split_meta_symbol(y, beta)
    if ref is not None:
        if not isinstance(ref, np.ndarray):
            raise TypeError(f"ref deve essere np.ndarray, ricevuto: {type(ref).__name__}")
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if ref.ndim != 2 or ref.shape[1] != beta:
            raise ValueError(
                f"ref deve avere shape (beta,) o (1, beta) con beta={beta}, ricevuta: {ref.shape}"
            )
        if ref.shape[0] not in (1, y.shape[0]):
            raise ValueError(
                f"ref batch incompatibile: {ref.shape[0]} (attesi 1 o {y.shape[0]})"
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError("ref contiene NaN/Inf ( Sez. 1.2)")
        ref_internal = ref

    z = np.real(np.sum(np.conj(ref_internal) * data, axis=-1))
    if not np.all(np.isfinite(z)):
        raise RuntimeError(
            "statistica di correlazione DCSK non finita (NaN/Inf): "
            "STOP  Sez. 1.2"
        )
    bits = (z > threshold).astype(np.int8)
    logger.debug(
        "dcsk_correlator_demodulate: batch=%d, beta=%d, z in [%.4e, %.4e]",
        y.shape[0], beta, float(np.min(z)), float(np.max(z)),
    )
    return bits

def matched_filter_demodulate(
    y: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y deve essere np.ndarray, ricevuto: {type(y).__name__}")
    if not isinstance(template, np.ndarray):
        raise TypeError(
            f"template deve essere np.ndarray, ricevuto: {type(template).__name__}"
        )
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y deve avere shape (B, beta) o (beta,), ricevuta: {y.shape}")
    if template.ndim != 1:
        raise ValueError(f"template deve essere 1D (beta,), ricevuta: {template.shape}")
    if y.shape[1] != template.shape[0]:
        raise ValueError(
            f"template ({template.shape[0]} campioni) non allineato alla finestra "
            f"y ({y.shape[1]} campioni)"
        )
    if y.shape[1] == 0:
        raise ValueError("y vuoto (beta=0): nessun campione da correlare")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene NaN/Inf ( Sez. 1.2)")
    if not np.all(np.isfinite(template)):
        raise ValueError("template contiene NaN/Inf ( Sez. 1.2)")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold deve essere finito, ricevuto: {threshold!r}")

    template_energy = float(np.sum(np.abs(template) ** 2))
    if template_energy <= _ENERGY_EPS:
        logger.warning(
            "matched_filter_demodulate: template degenere (energia %.3e <= %.1e) — "
            "possibile bias tra le due mappe (logistica/Bernoulli)",
            template_energy, _ENERGY_EPS,
        )

    z = np.real(np.sum(np.conj(template) * y, axis=-1))
    if not np.all(np.isfinite(z)):
        raise RuntimeError(
            "statistica matched filter non finita (NaN/Inf): STOP  Sez. 1.2"
        )
    bits = (z > threshold).astype(np.int8)
    logger.debug(
        "matched_filter_demodulate: batch=%d, template_energy=%.4e, z in [%.4e, %.4e]",
        y.shape[0], template_energy, float(np.min(z)), float(np.max(z)),
    )
    return bits

def energy_detector_demodulate(
    y: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y deve essere np.ndarray, ricevuto: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y deve avere shape (B, N) o (N,), ricevuta: {y.shape}")
    if y.shape[1] == 0:
        raise ValueError("y vuoto (nessun campione)")
    if y.shape[0] == 0:
        raise ValueError("y vuoto (batch senza campioni)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contiene NaN/Inf ( Sez. 1.2)")
    if not math.isfinite(float(threshold)):
        raise ValueError(f"threshold deve essere finito, ricevuto: {threshold!r}")

    energy = np.sum(np.abs(y) ** 2, axis=-1)
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise RuntimeError(
            "energia non valida (NaN/Inf o negativa): STOP  Sez. 1.2"
        )
    bits = (energy > threshold).astype(np.int8)
    logger.debug(
        "energy_detector_demodulate: batch=%d, energia in [%.4e, %.4e]",
        y.shape[0], float(np.min(energy)), float(np.max(energy)),
    )
    return bits

def _ber(bits_hat: np.ndarray, bits_true: np.ndarray) -> float:
    
    bits_hat_arr = np.asarray(bits_hat)
    bits_true_arr = np.asarray(bits_true)
    if bits_hat_arr.ndim != 1 or bits_true_arr.ndim != 1:
        raise ValueError("bits_hat e bits_true devono essere 1D")
    if bits_hat_arr.shape != bits_true_arr.shape:
        raise ValueError(
            f"shape non allineate: bits_hat {bits_hat_arr.shape} != "
            f"bits_true {bits_true_arr.shape}"
        )
    n_bits = int(bits_hat_arr.shape[0])
    if n_bits == 0:
        raise ValueError("n_bits = 0: BER non definita")

    try:
        from src.evaluation.metrics import ber as _metrics_ber
    except (ImportError, AttributeError):
        _metrics_ber = None
    if _metrics_ber is not None:
        return float(_metrics_ber(bits_true_arr, bits_hat_arr))

    n_errors = int(np.count_nonzero(bits_hat_arr != bits_true_arr))
    return float(n_errors / n_bits)

def _validate_correlator_config(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, float]:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")
    baselines_cfg = config.get("baselines")
    if not isinstance(baselines_cfg, dict):
        raise ValueError("sezione 'baselines' mancante o non dict nella config")
    corr_cfg = baselines_cfg.get("dcsk_correlator")
    if not isinstance(corr_cfg, dict):
        raise ValueError("sezione 'baselines.dcsk_correlator' mancante o non dict")
    missing = [key for key in _DCSK_REQUIRED_KEYS if key not in corr_cfg]
    if missing:
        raise ValueError(f"chiavi mancanti in baselines.dcsk_correlator: {missing}")
    _assert_config_finite(corr_cfg)

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("sezione 'data' mancante o non dict nella config")
    if "sequence_length" not in data_cfg:
        raise ValueError("chiave 'data.sequence_length' mancante nella config")
    sequence_length = _as_positive_int(
        data_cfg["sequence_length"], "data.sequence_length"
    )

    beta = _as_positive_int(
        corr_cfg["correlation_length"],
        "baselines.dcsk_correlator.correlation_length",
    )
    if 2 * beta > sequence_length:
        raise ValueError(
            f"2*correlation_length ({2 * beta}) > sequence_length ({sequence_length}): "
            "il meta-simbolo DCSK non e' rappresentabile nel frame ricevuto"
        )
    threshold = float(corr_cfg["threshold"])
    if not math.isfinite(threshold):
        raise ValueError(
            "baselines.dcsk_correlator.threshold deve essere finito, "
            f"ricevuto: {threshold!r}"
        )
    logger.debug(
        "config correlatore valida: beta=%d, threshold=%g, 2*beta=%d <= N_seq=%d",
        beta, threshold, 2 * beta, sequence_length,
    )
    return corr_cfg, beta, threshold

def evaluate_classical(
    y: np.ndarray,
    bits_true: np.ndarray,
    detector: str,
    config: Dict[str, Any],
    ref: Optional[np.ndarray] = None,
    template: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    
    if detector not in _DETECTOR_NAMES:
        raise ValueError(
            f"detector non valido: {detector!r} (attesi: {list(_DETECTOR_NAMES)})"
        )
    _, beta, threshold = _validate_correlator_config(config)

    if not isinstance(y, np.ndarray):
        raise TypeError(f"y deve essere np.ndarray, ricevuto: {type(y).__name__}")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2:
        raise ValueError(f"y deve avere shape (B, N) o (N,), ricevuta: {y.shape}")

    bits_true_arr = np.asarray(bits_true)
    if bits_true_arr.ndim != 1:
        raise ValueError(f"bits_true deve essere 1D, ricevuta: {bits_true_arr.shape}")
    if bits_true_arr.shape[0] != y.shape[0]:
        raise ValueError(
            f"bits_true ({bits_true_arr.shape[0]}) non allineato al batch "
            f"y ({y.shape[0]})"
        )
    if not np.all(np.isin(bits_true_arr, (0, 1))):
        raise ValueError("bits_true deve contenere solo valori 0/1")

    if detector == "dcsk" and y.shape[1] != 2 * beta:
        raise ValueError(
            f"frame y ({y.shape[1]} campioni) incoerente con correlation_length "
            f"(2*beta={2 * beta}): attesi frame [ref, data] di 2*correlation_length"
        )
    if detector == "matched_filter" and y.shape[1] != beta:
        raise ValueError(
            f"frame y ({y.shape[1]} campioni) incoerente con correlation_length "
            f"(beta={beta}): attese finestre di correlation_length campioni"
        )

    if detector == "dcsk":
        bits_hat = dcsk_correlator_demodulate(y, ref=ref, threshold=threshold)
    elif detector == "matched_filter":
        if template is None:
            raise ValueError(
                "detector 'matched_filter' richiede 'template' "
                "(sequenza caotica di riferimento)"
            )
        bits_hat = matched_filter_demodulate(y, template, threshold=threshold)
    else:
        bits_hat = energy_detector_demodulate(y, threshold=threshold)

    n_bits = int(bits_true_arr.shape[0])
    n_errors = int(np.count_nonzero(bits_hat != bits_true_arr))
    ber = _ber(bits_hat, bits_true_arr)
    logger.info(
        "evaluate_classical: detector=%s, n_bits=%d, n_errors=%d, BER=%.6f",
        detector, n_bits, n_errors, ber,
    )
    return {"ber": ber, "n_errors": n_errors, "n_bits": n_bits}

def count_trainable_params(*_ignored: Any, **_ignored_kw: Any) -> int:
    
    return 0

def main(argv: Optional[Sequence[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test ricevitori classici "
            "(DCSK / matched filter / energy detector)"
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path della config esperimento (YAML) da cui leggere i parametri",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(
        config_path=config_path, base_config_path=DEFAULT_BASE_CONFIG_PATH
    )
    general_cfg = config.get("general")
    if not isinstance(general_cfg, dict):
        raise ValueError("sezione 'general' mancante o non dict nella config")
    experiment_name = str(general_cfg.get("experiment_name", "ultra_can_isac"))
    log_dir = _REPO_ROOT / "results" / experiment_name / "logs"
    setup_logging(
        log_dir=log_dir,
        level=str(general_cfg.get("log_level", "INFO")),
        experiment_name=experiment_name,
    )
    log_config_summary(config, logger)

    corr_cfg, beta, threshold = _validate_correlator_config(config)
    logger.info(
        "smoke DCSK: correlation_length=%d (beta), threshold=%g, 2*beta=%d",
        beta, threshold, 2 * beta,
    )

    from src.data.dataset_generator import generate_chaotic_sequence

    data_cfg = config["data"]
    seed = int(general_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    template_seed = int(rng.integers(1, 2 ** 31 - 1))
    ref_template = generate_chaotic_sequence(
        str(data_cfg["map_type"]),
        float(data_cfg["map_param"]),
        template_seed,
        beta,
    )
    if ref_template.shape != (beta,) or not np.all(np.isfinite(ref_template)):
        raise RuntimeError(
            f"reference template non valido: shape {ref_template.shape}, "
            "attesa (beta,) con valori finiti"
        )
    y0 = np.concatenate([ref_template, -ref_template])
    y1 = np.concatenate([ref_template, ref_template])
    y_smoke = np.stack([y0, y1])
    bits_true = np.array([0, 1], dtype=np.int8)

    bits_dcsk = dcsk_correlator_demodulate(y_smoke, threshold=threshold)
    ber_dcsk = _ber(bits_dcsk, bits_true)
    logger.info(
        "smoke DCSK: bit_hat=%s attesi=%s, BER=%.6f",
        bits_dcsk.tolist(), bits_true.tolist(), ber_dcsk,
    )
    if ber_dcsk > 0.0:
        logger.warning(
            "smoke DCSK: BER=%g > 0 su frame puliti — possibile errore di "
            "implementazione", ber_dcsk,
        )

    bits_mf = matched_filter_demodulate(
        y_smoke[:, beta:], ref_template, threshold=threshold
    )
    ber_mf = _ber(bits_mf, bits_true)
    logger.info(
        "smoke matched filter: bit_hat=%s attesi=%s, BER=%.6f",
        bits_mf.tolist(), bits_true.tolist(), ber_mf,
    )
    if ber_mf > 0.0:
        logger.warning(
            "smoke matched filter: BER=%g > 0 su frame puliti — possibile errore "
            "di implementazione", ber_mf,
        )

    bits_energy = energy_detector_demodulate(y_smoke, threshold=threshold)
    ber_energy = _ber(bits_energy, bits_true)
    logger.info(
        "smoke energy detector: bit_hat=%s attesi=%s, BER=%.6f "
        "(floor atteso ~0.5: 'i classici sbagliano')",
        bits_energy.tolist(), bits_true.tolist(), ber_energy,
    )

    params = count_trainable_params()
    if params != 0:
        raise RuntimeError(
            f"count_trainable_params deve essere 0 per un ricevitore classico, "
            f"ricevuto: {params}"
        )
    logger.info(
        "smoke completato: count_trainable_params=%d, config=%s", params, corr_cfg
    )

if __name__ == "__main__":
    main()

