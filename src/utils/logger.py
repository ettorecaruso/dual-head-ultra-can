"""Logging strutturato per il progetto Dual-Head Ultra-CAN (ISAC in IoD).

Modulo di utilità conforme alle linee guida del progetto:
  - Docstring iniziale con lo scopo del modulo                     [questo blocco]
  - Type hints su tutti i parametri e i ritorni
  - Logging strutturato con livelli INFO e DEBUG
  - Logger di modulo via ``logging.getLogger(__name__)``
  - Docstring Google-style su ogni funzione pubblica (scopo, argomenti,
    ritorni, eccezioni)

Contenuto:
  - ``get_logger``         : logger di modulo, da chiamare con ``__name__``.
  - ``setup_logging``      : handler su file (``<log_dir>/<experiment_name>.log``,
    utf-8, append) + handler su console (stderr); idempotente.
  - ``log_config_summary`` : dump INFO/DEBUG della config caricata + WARNING
    su valori float non finiti (NaN/Inf).

Note:
  - Dipendenze: nessuna (solo standard library)
  - Nessuna formula del paper (Eq. (1)-(15)) è implementata in questo modulo:
    utility pura, quindi non ci sono riferimenti matematici da commentare.
  - Il livello di logging NON è hard-coded: i punti di ingresso devono passare
    ``config["general"].get("log_level", "INFO")`` (chiave aggiunta a
    ``base_config.yaml`` nella Fase 0, `` Sez. 7).
"""

from __future__ import annotations

import logging
import math
import numbers
import sys
from pathlib import Path
from typing import Any, Iterator

_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_LOG_FORMAT: str = "%(asctime)s | %(name)-30s | %(levelname)-6s | %(message)s"
_LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_HANDLER_TAG: str = "_ultra_can_handler"

logger = logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"name deve essere una stringa non vuota, ricevuto: {name!r}")
    return logging.getLogger(name)

def _resolve_level(level: str) -> int:
    
    if not isinstance(level, str):
        raise ValueError(f"level deve essere una stringa, ricevuto: {type(level).__name__}")
    level_upper = level.strip().upper()
    if level_upper not in _LOG_LEVELS:
        raise ValueError(f"level non valido: {level!r}. Ammessi: {sorted(_LOG_LEVELS)}")
    return getattr(logging, level_upper)

def _find_file_handler(root_logger: logging.Logger, log_file: Path) -> logging.FileHandler | None:
    
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        try:
            if Path(handler.baseFilename).resolve() == log_file.resolve():
                return handler
        except (OSError, ValueError):
            continue
    return None

def _find_console_handler(root_logger: logging.Logger) -> logging.StreamHandler | None:
    
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and handler.stream in (sys.stdout, sys.stderr)
            and getattr(handler, _HANDLER_TAG, False)
        ):
            return handler
    return None

def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    experiment_name: str = "ultra_can_isac",
) -> Path:
    
    level_num = _resolve_level(level)

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError(
            f"experiment_name deve essere una stringa non vuota, ricevuto: {experiment_name!r}"
        )
    if "\x00" in experiment_name:
        raise ValueError(f"experiment_name non può contenere caratteri NUL: {experiment_name!r}")
    if "/" in experiment_name or "\\" in experiment_name or ".." in experiment_name:
        raise ValueError(
            f"experiment_name non può contenere separatori di percorso: {experiment_name!r}"
        )

    try:
        log_dir_path = Path(log_dir).resolve()
    except TypeError as exc:
        raise TypeError(
            f"log_dir deve essere Path o str, ricevuto: {type(log_dir).__name__}"
        ) from exc

    if log_dir_path.exists() and not log_dir_path.is_dir():
        raise ValueError(f"log_dir esiste ma non è una directory: {log_dir_path}")

    log_dir_path.mkdir(parents=True, exist_ok=True)
    experiment_name_clean = experiment_name.strip()
    log_file = log_dir_path / f"{experiment_name_clean}.log"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(min(level_num, logging.INFO))

    file_handler = _find_file_handler(root_logger, log_file)
    if file_handler is None:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        root_logger.addHandler(file_handler)
    setattr(file_handler, _HANDLER_TAG, True)
    file_handler.setLevel(min(level_num, logging.INFO))
    file_handler.setFormatter(formatter)

    console_handler = _find_console_handler(root_logger)
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stderr)
        root_logger.addHandler(console_handler)
    setattr(console_handler, _HANDLER_TAG, True)
    console_handler.setLevel(level_num)
    console_handler.setFormatter(formatter)

    root_logger.info(
        "Logging configurato: file=%s livello=%s",
        log_file,
        logging.getLevelName(level_num),
    )

    return log_file

def _iter_config_items(config: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _iter_config_items(value, dotted)
        else:
            yield dotted, value

def _iter_config_leaves(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_config_leaves(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_config_leaves(item, f"{path}[{index}]")
    else:
        yield path, value

def _log_config_value(logger: logging.Logger, path: str, value: Any) -> None:
    
    if value is None or isinstance(value, (str, int, float, bool)):
        logger.debug("config.%s = %s", path, value)
    elif isinstance(value, (list, tuple)):
        logger.debug("config.%s = %s (len=%d)", path, value, len(value))
    else:
        logger.debug("config.%s = %s (type=%s)", path, value, type(value).__name__)

def log_config_summary(config: dict, logger: logging.Logger) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere un dict, ricevuto: {type(config).__name__}")
    if not isinstance(logger, logging.Logger):
        raise TypeError(f"logger deve essere logging.Logger, ricevuto: {type(logger).__name__}")

    general = config.get("general") or {}
    data = config.get("data") or {}
    model = config.get("model") or {}
    training = config.get("training") or {}

    logger.info("experiment_name = %s", general.get("experiment_name", "N/D"))
    logger.info("seed = %s", general.get("seed", "N/D"))
    logger.info("lambda_mse = %s", training.get("lambda_mse", "N/D"))
    logger.info("sequence_length = %s", data.get("sequence_length", "N/D"))
    logger.info(
        "map_type = %s / map_param = %s",
        data.get("map_type", "N/D"),
        data.get("map_param", "N/D"),
    )
    logger.info("backbone_type = %s", model.get("backbone_type", "N/D"))
    logger.info("epochs = %s", training.get("epochs", "N/D"))

    for path, value in _iter_config_leaves(config, "config"):
        if isinstance(value, numbers.Real) and not math.isfinite(value):
            logger.warning("CONFIG non-finito: %s = %s", path, value)

    for path, value in _iter_config_items(config):
        _log_config_value(logger, path, value)

    logger.debug("Riepilogo config completato (%d sezioni)", len(config))

