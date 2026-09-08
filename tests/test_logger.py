"""Test per il modulo ``src/utils/logger.py`` (Fase 0).

Casi coperti (equivalenza con i casi limite del template "dataset"):
  T1-T5  : creazione file di log, handler file+console, idempotenza, livelli
           e path invalidi (analoghi a SNR estremi / K=0 / ritardi fuori
           finestra del canale);
  T6-T8  : riepilogo config — lambda_mse sempre presente, dump DEBUG, WARNING
           su NaN/Inf (analoghi ai check di contenuto/energia del template);
  T9-T10 : validazione di ``get_logger`` e livello letto dalla chiave YAML
           ``general.log_level`` (niente hard-coding).
  T11-T18: correzioni dalla review — TypeError sugli input, sezioni None,
           chiavi mancanti, encoding UTF-8, caso positivo get_logger,
           stream stderr marcato, root-level minimo INFO, NaN in lista.

Comando di esecuzione (dal log):
    python -m pytest tests/test_logger.py -v
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

from src.utils.logger import (
    _HANDLER_TAG,
    get_logger,
    log_config_summary,
    setup_logging,
)

def _tiny_config() -> dict:
    
    return {
        "general": {
            "seed": 42,
            "experiment_name": "ultra_can_isac",
            "log_level": "INFO",
        },
        "data": {
            "sequence_length": 100,
            "map_type": "logistic",
            "map_param": 3.9,
            "snr_range": [-5, 20],
            "snr_step": 2,
            "max_delay": 33,
            "max_doppler": 0.00008,
        },
        "model": {"backbone_type": "conv1d"},
        "training": {"epochs": 5, "learning_rate": 0.001, "lambda_mse": 0.1},
    }

def _count_file_handlers(log_file: Path) -> int:
    
    return sum(
        1
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_file
    )

def test_setup_logging_creates_file(tmp_path: Path) -> None:
    
    log_dir = tmp_path / "logs"
    log_file = setup_logging(log_dir, level="INFO", experiment_name="ber_vs_snr")
    assert log_file == (log_dir.resolve() / "ber_vs_snr.log")
    assert log_file.exists()

    marker = "HELLO_LOGGER_T1"
    get_logger("test.creates_file").info(marker)

    content = log_file.read_text(encoding="utf-8")
    assert len(content) > 0
    assert marker in content

def test_setup_logging_console_and_file_handlers(tmp_path: Path) -> None:
    """T2: il root ha 1 FileHandler verso il file atteso e >=1 StreamHandler."""
    log_file = setup_logging(tmp_path / "logs", level="INFO", experiment_name="exp_handlers")
    root_logger = logging.getLogger()

    file_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_file
    ]
    console_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and handler.stream in (sys.stdout, sys.stderr)
    ]
    assert len(file_handlers) == 1
    assert len(console_handlers) >= 1

    marker = "HANDLER_FMT_CHECK"
    get_logger("test.handlers").info(marker)
    content = log_file.read_text(encoding="utf-8")
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| .*" + marker, content) is not None

def test_setup_logging_idempotent(tmp_path: Path) -> None:
    """T3: due chiamate non duplicano handler né righe di log."""
    log_dir = tmp_path / "logs"
    exp_name = "idem_exp"

    first = setup_logging(log_dir, level="INFO", experiment_name=exp_name)
    n_before = _count_file_handlers(first)
    second = setup_logging(log_dir, level="INFO", experiment_name=exp_name)
    n_after = _count_file_handlers(second)

    assert first == second
    assert n_before == n_after == 1

    marker = "IDEMPOTENCY_MARKER"
    log = get_logger("test.idempotent")
    log.info(marker)
    log.info(marker)
    content = first.read_text(encoding="utf-8")
    assert content.count(marker) == 2

def test_setup_logging_invalid_level_raises(tmp_path: Path) -> None:
    """T4: level non ammesso -> ValueError; level minuscolo accettato."""
    with pytest.raises(ValueError):
        setup_logging(tmp_path, level="TRACE", experiment_name="exp")
    with pytest.raises(ValueError):
        setup_logging(tmp_path, level=123, experiment_name="exp")  # type: ignore[arg-type]

    log_file = setup_logging(tmp_path / "logs_lower", level="debug", experiment_name="exp_lower")
    assert log_file.exists()

def test_setup_logging_empty_name_raises(tmp_path: Path) -> None:
    
    with pytest.raises(ValueError):
        setup_logging(tmp_path, experiment_name="")
    with pytest.raises(ValueError):
        setup_logging(tmp_path, experiment_name="   ")
    with pytest.raises(ValueError):
        setup_logging(tmp_path, experiment_name="a/b")
    with pytest.raises(ValueError):
        setup_logging(tmp_path, experiment_name="..")

def test_log_dir_is_file_raises(tmp_path: Path) -> None:
    """T5b: log_dir che esiste come file -> ValueError."""
    some_file = tmp_path / "not_a_dir"
    some_file.write_text("io sono un file", encoding="utf-8")
    with pytest.raises(ValueError):
        setup_logging(some_file, experiment_name="exp")

def test_log_config_summary_contains_lambda(caplog: pytest.LogCaptureFixture) -> None:
    """T6: il riepilogo INFO contiene lambda_mse ed experiment_name."""
    summary_logger = get_logger("test.config_summary_info")
    caplog.set_level(logging.INFO)
    log_config_summary(_tiny_config(), summary_logger)

    messages = [r.getMessage() for r in caplog.records if r.name == "test.config_summary_info"]
    assert any("lambda_mse = 0.1" in m for m in messages)
    assert any("experiment_name = ultra_can_isac" in m for m in messages)

def test_log_config_summary_debug_dump(caplog: pytest.LogCaptureFixture) -> None:
    """T7: a livello DEBUG ogni chiave compare come config.<sez>.<chiave>."""
    summary_logger = get_logger("test.config_summary_debug")
    caplog.set_level(logging.DEBUG)
    log_config_summary(_tiny_config(), summary_logger)

    messages = [r.getMessage() for r in caplog.records if r.name == "test.config_summary_debug"]
    assert any(m == "config.general.seed = 42" for m in messages)
    assert any(m.startswith("config.data.map_param = ") for m in messages)
    assert any("Riepilogo config completato (4 sezioni)" in m for m in messages)

def test_config_non_finite_warning(caplog: pytest.LogCaptureFixture) -> None:
    
    summary_logger = get_logger("test.config_non_finite")
    caplog.set_level(logging.WARNING)
    bad_config = {
        "general": {"experiment_name": "bad_run"},
        "training": {"lambda_mse": 0.1, "learning_rate": float("nan")},
        "data": {"map_param": float("inf")},
    }
    log_config_summary(bad_config, summary_logger)  # non deve sollevare

    records = [r for r in caplog.records if r.name == "test.config_non_finite"]
    assert any(
        r.levelno == logging.WARNING and "training.learning_rate" in r.getMessage()
        for r in records
    )
    assert any(
        r.levelno == logging.WARNING and "data.map_param" in r.getMessage()
        for r in records
    )

def test_get_logger_empty_name_raises() -> None:
    """T9: get_logger con nome vuoto/whitespace/non-str -> ValueError."""
    with pytest.raises(ValueError):
        get_logger("")
    with pytest.raises(ValueError):
        get_logger("   ")
    with pytest.raises(ValueError):
        get_logger(123)  # type: ignore[arg-type]

def test_level_from_config_key(tmp_path: Path) -> None:
    
    config = _tiny_config()
    config["general"]["log_level"] = "DEBUG"
    log_file = setup_logging(
        tmp_path / "logs",
        level=config["general"].get("log_level", "INFO"),
        experiment_name="cfg_level",
    )
    get_logger("test.level_from_config").debug("DEBUG_MARKER_T10")
    content = log_file.read_text(encoding="utf-8")
    assert "DEBUG_MARKER_T10" in content

def test_log_config_summary_type_raises() -> None:
    """T11: config non-dict o logger non-Logger -> TypeError (guardia G8)."""
    with pytest.raises(TypeError):
        log_config_summary(["not", "a", "dict"], get_logger("test.type_raises"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        log_config_summary({}, "not_a_logger")  # type: ignore[arg-type]

def test_config_section_none_no_crash(caplog: pytest.LogCaptureFixture) -> None:
    
    summary_logger = get_logger("test.config_section_none")
    caplog.set_level(logging.INFO)
    log_config_summary({"general": None, "training": None}, summary_logger)

    messages = [r.getMessage() for r in caplog.records if r.name == "test.config_section_none"]
    assert any("experiment_name = N/D" in m for m in messages)
    assert any("lambda_mse = N/D" in m for m in messages)

def test_log_config_summary_missing_keys_no_crash(caplog: pytest.LogCaptureFixture) -> None:
    
    summary_logger = get_logger("test.config_missing_keys")
    caplog.set_level(logging.INFO)
    log_config_summary({}, summary_logger)

    messages = [r.getMessage() for r in caplog.records if r.name == "test.config_missing_keys"]
    assert any("experiment_name = N/D" in m for m in messages)

def test_setup_logging_utf8_encoding(tmp_path: Path) -> None:
    """T14: record con caratteri non-ASCII -> nessun errore di encoding."""
    log_file = setup_logging(tmp_path / "logs", level="INFO", experiment_name="utf8_exp")
    marker = "τ fD λ_Π"
    get_logger("test.utf8").info(marker)

    content = log_file.read_text(encoding="utf-8")
    assert marker in content

def test_get_logger_valid_name() -> None:
    """T15: get_logger con nome valido -> Logger con il nome richiesto."""
    log = get_logger("test.valid_name")
    assert isinstance(log, logging.Logger)
    assert log.name == "test.valid_name"

def test_console_handler_stream_is_stderr(tmp_path: Path) -> None:
    """T16: lo StreamHandler di setup_logging punta a stderr ed è marcato."""
    setup_logging(tmp_path / "logs", level="INFO", experiment_name="stderr_exp")
    console_handlers = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.StreamHandler)
        and getattr(handler, _HANDLER_TAG, False)
        and handler.stream is sys.stderr
    ]
    assert len(console_handlers) >= 1

def test_root_level_min_info(tmp_path: Path) -> None:
    """T17: il root non scende mai sotto INFO (guardia G6)."""
    setup_logging(tmp_path / "logs_debug", level="DEBUG", experiment_name="root_debug")
    assert logging.getLogger().level == logging.DEBUG

    setup_logging(tmp_path / "logs_warn", level="WARNING", experiment_name="root_warn")
    assert logging.getLogger().level == logging.INFO

def test_config_nan_inside_list(caplog: pytest.LogCaptureFixture) -> None:
    
    summary_logger = get_logger("test.config_nan_in_list")
    caplog.set_level(logging.WARNING)
    log_config_summary({"data": {"snr_range": [0.0, float("nan")]}}, summary_logger)

    records = [r for r in caplog.records if r.name == "test.config_nan_in_list"]
    assert any(
        r.levelno == logging.WARNING and "data.snr_range[1]" in r.getMessage()
        for r in records
    )
