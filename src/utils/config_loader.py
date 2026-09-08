"""Caricamento e validazione della configurazione YAML per il progetto
Dual-Head Ultra-CAN (ISAC in IoD).

  - Docstring iniziale con lo scopo del modulo                     [questo blocco]
  - Type hints su tutti i parametri e i ritorni
  - Logging strutturato con livelli INFO e DEBUG
  - Logger di modulo via ``logging.getLogger(__name__)``
  - Docstring Google-style su ogni funzione pubblica

Contenuto:
  - ``load_config``          : merge profondo
    base_config.yaml <- config esperimento <- override CLI (--set).
  - ``validate_config``      : fail-fast (``ValueError``) su chiavi mancanti.
  - ``parse_cli_overrides``  : parsing di ``--set path.to.key=value``.
  - ``save_config_snapshot`` : salva ``config_used.yaml`` nei metadati del run
    . Il valore di lambda_mse usato nel run DEVE
    comparire nei metadati salvati con i risultati).

Note:
  - Nessuna formula del paper (Eq. (1)-(15)) e' implementata in questo modulo:
    utility pura, quindi nessun riferimento matematico da commentare.
  - Il merge e' ricorsivo sui dizionari; le liste (es. ``snr_range``,
    ``echoes``) vengono SOSTITUITE, non concatenate.
  - Dipendenze: PyYAML (requirements.txt) e ``src.utils.logger``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_BASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base_config.yaml"

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _iter_config_items(config: Any, prefix: str = "config") -> Iterator[Tuple[str, Any]]:
    
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}"
            yield from _iter_config_items(value, path)
    elif isinstance(config, (list, tuple)):
        for index, value in enumerate(config):
            path = f"{prefix}[{index}]"
            yield from _iter_config_items(value, path)
    else:
        yield prefix, config

def load_config(
    config_path: Path,
    base_config_path: Path = DEFAULT_BASE_CONFIG_PATH,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    base_path = Path(base_config_path)
    exp_path = Path(config_path)
    for path, label in ((base_path, "base_config"), (exp_path, "config esperimento")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} non trovato: {path}")
    try:
        with open(base_path, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
        with open(exp_path, "r", encoding="utf-8") as fh:
            experiment = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"file YAML non valido: {exc}") from exc
    if not isinstance(base, dict):
        raise ValueError(f"{base_path} non contiene un dict in testa")
    if not isinstance(experiment, dict):
        raise ValueError(f"{exp_path} non contiene un dict in testa")

    merged = _deep_merge(base, experiment)
    if cli_overrides is not None:
        if not isinstance(cli_overrides, dict):
            raise TypeError(
                f"cli_overrides deve essere dict, ricevuto: {type(cli_overrides).__name__}"
            )
        merged = _deep_merge(merged, cli_overrides)

    logger.debug("config caricata: %s <- %s (override CLI: %s)", base_path, exp_path, bool(cli_overrides))
    return merged

def _key_exists(config: Dict[str, Any], dotted_path: str) -> bool:
    
    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True

def validate_config(config: Dict[str, Any], required_keys: Sequence[str]) -> None:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")
    missing = [key for key in required_keys if not _key_exists(config, key)]
    if missing:
        raise ValueError(f"chiavi richieste mancanti nella config: {missing}")
    logger.debug("validate_config OK: %d chiavi verificate", len(required_keys))

def _coerce_cli_value(raw_value: str) -> Any:
    
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def parse_cli_overrides(raw: Sequence[str]) -> Dict[str, Any]:
    
    overrides: Dict[str, Any] = {}
    for token in raw:
        item = token.strip()
        if not item or item == "--set":
            continue
        if item.startswith("--set="):
            item = item[len("--set="):]
        elif item.startswith("--set "):
            item = item[len("--set "):].strip()
        if "=" not in item:
            raise ValueError(f"override CLI malformato (atteso path.to.key=value): {token!r}")
        dotted_path, raw_value = item.split("=", 1)
        parts = [part for part in dotted_path.split(".") if part]
        if not parts:
            raise ValueError(f"path di override vuoto: {token!r}")
        value = _coerce_cli_value(raw_value)
        node: Dict[str, Any] = overrides
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise ValueError(f"collisione di tipo sul path {dotted_path!r}")
            node = child
        node[parts[-1]] = value
    logger.debug("override CLI parsati: %s", overrides)
    return overrides

def save_config_snapshot(config: Dict[str, Any], output_dir: Path) -> Path:
    
    if not isinstance(config, dict):
        raise TypeError(f"config deve essere dict, ricevuto: {type(config).__name__}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "config_used.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    logger.info("snapshot config salvato: %s", snapshot_path)
    return snapshot_path

