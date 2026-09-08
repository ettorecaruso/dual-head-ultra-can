"""Test per ``src/utils/config_loader.py``

Questi test verificano il caricatore di configurazione YAML, unica via di
accesso ai parametri (niente hard-coding). Coprono il
100% delle funzioni pubbliche del modulo:

  - ``load_config``           : merge profondo base <- esperimento <- CLI,
    T1-T4, T11, T14 (vedi commenti dei test in ``base_config.yaml`` e in
    ``config_loader.py``);
  - ``validate_config``       : fail-fast ``ValueError`` su chiavi mancanti
    (``test_missing_key_raises`` del PLAN);
  - ``parse_cli_overrides``   : parsing ``--set path.to.key=value`` con
    coercizione di tipo (``test_cli_override_parsing`` del PLAN);
  - ``save_config_snapshot``  : snapshot ``config_used.yaml`` round-trip
    identico (T10);
  - ``DEFAULT_BASE_CONFIG_PATH``: risoluzione del path canonico del base.

I test obbligatori (``tests/test_config_loader.py``) sono:
  test_load_base_config, test_merge_experiment_overrides,
  test_missing_key_raises, test_cli_override_parsing.

Scelta di pytest: coerente con la suite esistente (``tests/test_logger.py`` usa
pytest con fixtures, parametrize e ``tmp_path``) e con il comando dal log: ``python -m pytest tests/test_config_loader.py -v``.

TODO (test mancanti, segnalati esplicitamente —):
  - T5  (map_param fuori [3.57, 4.0] -> ValueError): validazione NON nel
        config_loader ma in ``dataset_generator._validate_config`` /
        ``generate_chaotic_sequence`` -> coperto da ``tests/test_input_validation.py``
        e ``tests/test_chaotic_maps.py``.
  - T6  (max_delay >= sequence_length -> ValueError): validazione in
        ``dataset_generator._validate_config`` -> coperto da
        ``tests/test_input_validation.py::test_generator_config_max_delay_policy_raises``.
  - T7  (SNR -20/+30 dB accettati, finiti): comportamento del canale
        ``apply_aerial_channel`` -> coperto da ``tests/test_channel.py``.
  - T8  (snr_range con nan / snr_step <= 0 -> ValueError): validazione in
        ``dataset_generator`` -> coperto da ``tests/test_input_validation.py``
        e ``tests/test_chaotic_maps.py::test_build_snr_grid_invalid_inputs``.
  - T12 (loss_weights.sensing != lambda_mse -> WARNING + autocorrezione):
        appartiene al ``Trainer`` (non ancora implementato) -> test dedicato
        in ``tests/test_training.py``.
  - T13 (griglia SNR inclusiva sugli estremi): appartiene a
        ``dataset_generator.build_snr_grid`` -> coperto da
        ``tests/test_chaotic_maps.py`` e ``tests/test_input_validation.py``.
  - Cascata ``load_config(base, [ber_vs_snr, exp2], cli)`` con LISTA di config
        esperimento: NON supportata dalla
        firma corrente di ``load_config`` (un solo ``config_path``). Testato
        qui il caso equivalente ``base=full_experiment`` (T4); il supporto
        multi-file va aggiunto al sorgente (ALERT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    _deep_merge,
    load_config,
    parse_cli_overrides,
    save_config_snapshot,
    validate_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _REPO_ROOT / "configs"
_BASE_CONFIG_PATH = _CONFIGS_DIR / "base_config.yaml"
_EXPERIMENTS_PATH = _CONFIGS_DIR / "experiments.yaml"

_MU_PAPER = 3.9
_LAMBDA_PAPER = 2.0
_FD_MAX = 8e-5
_N_SEQ = 100
_M_CSK = 2

def _write_yaml(tmp_path: Path, data: Dict[str, Any], name: str = "experiment.yaml") -> Path:
    
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return path

def _empty_experiment(tmp_path: Path) -> Path:
    
    return _write_yaml(tmp_path, {})

def _load_exp_config(exp_key: str, mode: str) -> Dict[str, Any]:
    
    base = load_config(_EXPERIMENTS_PATH, _BASE_CONFIG_PATH)
    with open(_EXPERIMENTS_PATH, "r", encoding="utf-8") as fh:
        experiments_full = yaml.safe_load(fh)
    section = experiments_full[exp_key][mode]
    return _deep_merge(base, section)

def test_default_base_config_path_resolves_to_configs_dir() -> None:
    """Caso normale: DEFAULT_BASE_CONFIG_PATH punta a configs/base_config.yaml."""
    assert DEFAULT_BASE_CONFIG_PATH == _BASE_CONFIG_PATH
    assert DEFAULT_BASE_CONFIG_PATH.is_file()

def test_load_base_config_essential_keys(tmp_path: Path) -> None:
    
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    for dotted in (
        "general.seed",
        "general.experiment_name",
        "general.log_level",
        "data.sequence_length",
        "data.map_type",
        "data.map_param",
        "data.max_delay",
        "data.max_doppler",
        "data.snr_range",
        "model.backbone_type",
        "training.lambda_mse",
        "training.loss_weights.comm",
        "training.loss_weights.sensing",
        "channel.model",
    ):
        node: Any = config
        for part in dotted.split("."):
            assert isinstance(node, dict) and part in node, f"chiave mancante: {dotted}"
            node = node[part]

def test_load_base_config_paper_values_T2_T14(tmp_path: Path) -> None:
    
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    data = config["data"]
    assert data["map_param"] == pytest.approx(_MU_PAPER)
    assert config["training"]["lambda_mse"] == pytest.approx(_LAMBDA_PAPER)
    assert data["max_doppler"] == pytest.approx(_FD_MAX)
    assert data["sequence_length"] == _N_SEQ
    assert config["model"]["communication_head"]["modulation_order"] == _M_CSK
    assert config["general"]["log_level"] == "INFO"

def test_merge_fast_test_overrides_T1(tmp_path: Path) -> None:
    
    config = _load_exp_config("ber_vs_snr", "fast")
    assert config["general"]["experiment_name"] == "ber_vs_snr_fast"
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == 32
    assert config["data"]["snr_range"] == [0, 10]
    assert config["data"]["snr_step"] == 5
    assert config["data"]["echoes"] == [0, 1]
    assert config["data"]["max_delay"] == 5
    assert config["data"]["num_symbols_train"] == 108
    assert config["data"]["num_symbols_val"] == 20
    assert config["data"]["num_symbols_test"] == 50
    assert config["data"]["snr_range"] == [0, 10]
    assert len(config["data"]["snr_range"]) == 2

def test_merge_preserves_base_values_T2(tmp_path: Path) -> None:
    
    config = _load_exp_config("ber_vs_snr", "fast")
    data = config["data"]
    assert data["map_param"] == pytest.approx(_MU_PAPER)
    assert config["training"]["lambda_mse"] == pytest.approx(_LAMBDA_PAPER)
    assert data["max_doppler"] == pytest.approx(_FD_MAX)
    assert data["sequence_length"] == _N_SEQ
    assert config["model"]["communication_head"]["modulation_order"] == _M_CSK
    assert config["model"]["backbone_type"] == "conv1d"

def test_merge_full_experiment_paper_values_T3(tmp_path: Path) -> None:
    """T3: ber_vs_snr.full allineato al paper Sez. V-A."""
    config = _load_exp_config("ber_vs_snr", "full")
    assert config["data"]["num_symbols_train"] == 105000
    assert config["data"]["num_symbols_val"] == 5000
    assert config["data"]["num_symbols_test"] == 20000
    assert config["data"]["echoes"] == [1, 3]
    assert config["data"]["snr_step"] == 2
    assert config["training"]["epochs"] == 50
    assert config["training"]["batch_size"] == 128
    assert config["training"]["early_stopping_patience"] == 10
    assert config["evaluation"]["online_generation"] is True

def test_merge_experiments_yaml_preserves_base_paths_T4(tmp_path: Path) -> None:
    """T4: experiments.yaml non altera i path di default (raw/processed)."""
    config = _load_exp_config("ber_vs_snr", "full")
    assert config["data"]["raw_dir"] == "data/raw"
    assert config["data"]["processed_dir"] == "data/processed"
    assert config["data"]["map_param"] == pytest.approx(_MU_PAPER)
    assert config["training"]["epochs"] == 50

def test_load_config_missing_base_file_raises(tmp_path: Path) -> None:
    """Input invalidi: base_config mancante -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="base_config"):
        load_config(_empty_experiment(tmp_path), base_config_path=tmp_path / "missing_base.yaml")

def test_load_config_missing_experiment_file_raises(tmp_path: Path) -> None:
    
    with pytest.raises(FileNotFoundError, match="esperimento"):
        load_config(tmp_path / "missing_exp.yaml", _BASE_CONFIG_PATH)

def test_load_config_invalid_yaml_raises(tmp_path: Path) -> None:
    """Input invalidi: YAML malformato -> ValueError (da yaml.YAMLError)."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: [1, 2\n  snr: {broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML non valido"):
        load_config(bad, _BASE_CONFIG_PATH)

def test_load_config_non_dict_top_level_raises(tmp_path: Path) -> None:
    """Input invalidi: YAML valido ma senza dict in testa -> ValueError."""
    list_yaml = _write_yaml(tmp_path, {"dummy": True}, "list.yaml")
    list_yaml.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dict in testa"):
        load_config(list_yaml, _BASE_CONFIG_PATH)

def test_load_config_cli_overrides_non_dict_raises(tmp_path: Path) -> None:
    """Input invalidi: cli_overrides non dict -> TypeError."""
    with pytest.raises(TypeError, match="cli_overrides"):
        load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH, cli_overrides="nope")

def test_load_config_cli_overrides_applied_last(tmp_path: Path) -> None:
    
    exp = _write_yaml(tmp_path, {"training": {"epochs": 3}})
    config = load_config(exp, _BASE_CONFIG_PATH, cli_overrides={"training": {"epochs": 7}})
    assert config["training"]["epochs"] == 7

def test_load_config_deep_merge_nested_dicts(tmp_path: Path) -> None:
    """Caso normale: dict annidati fusi ricorsivamente e sezioni nuove aggiunte."""
    exp = _write_yaml(
        tmp_path,
        {
            "data": {"max_delay": 10},
            "nuova_sezione": {"sottochiave": 1},
        },
    )
    config = load_config(exp, _BASE_CONFIG_PATH)
    assert config["data"]["max_delay"] == 10
    assert config["data"]["max_doppler"] == pytest.approx(_FD_MAX)
    assert config["nuova_sezione"]["sottochiave"] == 1

def test_load_config_lists_replaced_not_concatenated(tmp_path: Path) -> None:
    
    exp = _write_yaml(tmp_path, {"data": {"snr_range": [0.0, 10.0], "echoes": [1]}})
    config = load_config(exp, _BASE_CONFIG_PATH)
    assert config["data"]["snr_range"] == [0.0, 10.0]
    assert config["data"]["echoes"] == [1]

def test_load_config_empty_experiment_returns_base(tmp_path: Path) -> None:
    
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    raw_base = yaml.safe_load(_BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config == raw_base

def test_load_config_does_not_mutate_base_files(tmp_path: Path) -> None:
    
    before = _BASE_CONFIG_PATH.read_text(encoding="utf-8")
    load_config(
        _EXPERIMENTS_PATH,
        _BASE_CONFIG_PATH,
        cli_overrides={"training": {"epochs": 99}},
    )
    assert _BASE_CONFIG_PATH.read_text(encoding="utf-8") == before

def test_validate_config_ok(tmp_path: Path) -> None:
    
    config = _load_exp_config("ber_vs_snr", "fast")
    required: List[str] = [
        "general.experiment_name",
        "data.sequence_length",
        "data.map_param",
        "training.lambda_mse",
        "model.communication_head.modulation_order",
    ]
    validate_config(config, required)

def test_validate_config_missing_key_raises(tmp_path: Path) -> None:
    """ (test_missing_key_raises): chiave mancante -> ValueError."""
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    del config["training"]["lambda_mse"]
    with pytest.raises(ValueError, match="lambda_mse"):
        validate_config(config, ["training.lambda_mse"])

def test_validate_config_multiple_missing_keys(tmp_path: Path) -> None:
    
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    required = ["data.map_param", "data.max_delay", "training.lambda_mse"]
    for key in required:
        node: Any = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]
    with pytest.raises(ValueError) as excinfo:
        validate_config(config, required)
    assert "map_param" in str(excinfo.value)
    assert "max_delay" in str(excinfo.value)
    assert "lambda_mse" in str(excinfo.value)

def test_validate_config_nested_key_missing(tmp_path: Path) -> None:
    """Input limite: path annidato profondo non presente -> ValueError."""
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    with pytest.raises(ValueError, match="loss_weights.non_esiste"):
        validate_config(config, ["training.loss_weights.non_esiste"])

def test_validate_config_non_dict_type_error() -> None:
    """Input invalidi: config non dict -> TypeError."""
    with pytest.raises(TypeError, match="dict"):
        validate_config(["not", "a", "dict"], ["a.b"])

def test_validate_config_empty_required_keys_ok(tmp_path: Path) -> None:
    
    config = load_config(_empty_experiment(tmp_path), _BASE_CONFIG_PATH)
    validate_config(config, [])

def test_parse_cli_override_single_nested() -> None:
    """ (test_cli_override_parsing): dict annidato corretto."""
    overrides = parse_cli_overrides(("--set", "training.epochs=5"))
    assert overrides == {"training": {"epochs": 5}}

def test_parse_cli_value_coercion() -> None:
    """Caso normale: coercizione bool, int, float e stringa."""
    overrides = parse_cli_overrides(
        (
            "--set",
            "general.flag=true",
            "training.epochs=5",
            "data.map_param=3.9",
            "general.experiment_name=run_alpha",
        )
    )
    assert overrides["general"]["flag"] is True
    assert overrides["training"]["epochs"] == 5
    assert overrides["training"]["epochs"] is not True
    assert overrides["data"]["map_param"] == pytest.approx(3.9)
    assert overrides["general"]["experiment_name"] == "run_alpha"

def test_parse_cli_multiple_and_deep_path() -> None:
    
    overrides = parse_cli_overrides(
        (
            "training.loss_weights.comm=1.0",
            "training.loss_weights.sensing=0.1",
            "experiments.jamming.jsr=5",
        )
    )
    assert overrides["training"]["loss_weights"]["comm"] == pytest.approx(1.0)
    assert overrides["training"]["loss_weights"]["sensing"] == pytest.approx(0.1)
    assert overrides["experiments"]["jamming"]["jsr"] == 5

def test_parse_cli_numeric_path_creates_dict_key() -> None:
    
    overrides = parse_cli_overrides(("data.echoes.1=3",))
    assert overrides == {"data": {"echoes": {"1": 3}}}

def test_parse_cli_accepts_plain_tokens() -> None:
    """Caso limite: token senza prefisso ``--set`` accettati (docstring modulare)."""
    assert parse_cli_overrides(("training.epochs=5",)) == {"training": {"epochs": 5}}

def test_parse_cli_equals_prefixed_token() -> None:
    """Caso limite: token ``--set=path=value`` (prefisso con ``=``)."""
    overrides = parse_cli_overrides(("--set=data.max_delay=10",))
    assert overrides == {"data": {"max_delay": 10}}

def test_parse_cli_empty_tokens_skipped() -> None:
    """Caso limite: token vuoti ignorati -> override vuoto."""
    assert parse_cli_overrides(("", "  ", "--set")) == {}

def test_parse_cli_missing_equals_raises() -> None:
    """Input invalidi: token senza ``=`` -> ValueError con messaggio chiaro."""
    with pytest.raises(ValueError, match="malformato"):
        parse_cli_overrides(("training.epochs",))

def test_parse_cli_empty_path_raises() -> None:
    """Input invalidi: path di override vuoto -> ValueError."""
    with pytest.raises(ValueError, match="path di override vuoto"):
        parse_cli_overrides(("=5",))

def test_parse_cli_path_collision_raises() -> None:
    """Input invalidi: collisione di tipo sul path -> ValueError."""
    with pytest.raises(ValueError, match="collisione"):
        parse_cli_overrides(("training.epochs=5", "training.epochs.extra=1"))

def test_save_config_snapshot_creates_file(tmp_path: Path) -> None:
    
    config = _load_exp_config("ber_vs_snr", "fast")
    output_dir = tmp_path / "results" / "exp1_fast" / "logs"
    snapshot = save_config_snapshot(config, output_dir)
    assert snapshot == (output_dir / "config_used.yaml")
    assert snapshot.is_file()

def test_save_config_snapshot_round_trip_T10(tmp_path: Path) -> None:
    
    config = _load_exp_config("ber_vs_snr", "full")
    snapshot = save_config_snapshot(config, tmp_path / "logs")
    reloaded = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert reloaded == config

def test_save_config_snapshot_utf8_round_trip(tmp_path: Path) -> None:
    
    config: Dict[str, Any] = {"training": {"lambda_mse": 0.1}, "data": {"tau_max": "tau"}}
    snapshot = save_config_snapshot(config, tmp_path / "logs")
    reloaded = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert reloaded == config

def test_save_config_snapshot_non_dict_raises(tmp_path: Path) -> None:
    """Input invalidi: config non dict -> TypeError."""
    with pytest.raises(TypeError, match="dict"):
        save_config_snapshot("not_a_dict", tmp_path)

def test_deep_merge_replaces_lists() -> None:
    
    merged = _deep_merge({"data": {"snr_range": [-5.0, 20.0]}}, {"data": {"snr_range": [0.0, 10.0]}})
    assert merged == {"data": {"snr_range": [0.0, 10.0]}}

def test_deep_merge_does_not_mutate_base() -> None:
    
    base: Dict[str, Any] = {"training": {"epochs": 5, "lambda_mse": 0.1}}
    _deep_merge(base, {"training": {"epochs": 50}})
    assert base == {"training": {"epochs": 5, "lambda_mse": 0.1}}

def test_deep_merge_nested_dicts_merged() -> None:
    
    base = {"training": {"epochs": 5, "batch_size": 64}, "data": {"map_param": 3.9}}
    override = {"training": {"epochs": 50}, "general": {"log_level": "DEBUG"}}
    merged = _deep_merge(base, override)
    assert merged["training"] == {"epochs": 50, "batch_size": 64}
    assert merged["data"]["map_param"] == pytest.approx(3.9)
    assert merged["general"]["log_level"] == "DEBUG"

def test_integration_load_validate_snapshot_round_trip(tmp_path: Path) -> None:
    """INT: load -> validate -> snapshot -> reload identico (pipeline reale)."""
    config = _load_exp_config("ber_vs_snr", "fast")
    validate_config(
        config,
        ["general.experiment_name", "data.sequence_length", "training.lambda_mse"],
    )
    snapshot = save_config_snapshot(config, tmp_path / "logs")
    reloaded = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert reloaded == config
    assert reloaded["training"]["lambda_mse"] == pytest.approx(_LAMBDA_PAPER)

def test_integration_cli_override_end_to_end(tmp_path: Path) -> None:
    """INT: ``--set`` CLI parsato e applicato sul merge completo."""
    cli = parse_cli_overrides(("--set", "training.epochs=1", "data.max_delay=10"))
    config = load_config(_EXPERIMENTS_PATH, _BASE_CONFIG_PATH, cli_overrides=cli)
    assert config["training"]["epochs"] == 1
    assert config["data"]["max_delay"] == 10
    assert config["general"]["experiment_name"] == "ultra_can_isac"

def test_tiny_config_fixture_contract(tiny_config: Dict[str, Any]) -> None:
    """conftest: tiny_config rispetta il contratto del ."""
    assert tiny_config["data"]["map_param"] == pytest.approx(_MU_PAPER)
    assert tiny_config["training"]["lambda_mse"] == pytest.approx(_LAMBDA_PAPER)
    assert tiny_config["data"]["max_delay"] == 10
    assert tiny_config["data"]["max_doppler"] == pytest.approx(_FD_MAX)
    assert tiny_config["data"]["sequence_length"] == _N_SEQ
    assert tiny_config["general"]["experiment_name"] == "tiny_test"

def test_tiny_dataset_fixture_contract(tiny_dataset: Dict[str, Any]) -> None:
    """conftest: tiny_dataset con shape, range e finitezza corretti (128, K=1, SNR=0)."""
    x = tiny_dataset["x"]
    assert x.shape == (128, _N_SEQ, 2)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
    assert np.all(x[..., 0] > 0.0) and np.all(x[..., 0] < 1.0)
    assert np.all(x[..., 1] >= 0.0)

    comm = tiny_dataset["comm_labels"]
    assert comm.shape == (128,)
    assert set(np.unique(comm)).issubset({0, 1})

    sensing = tiny_dataset["sensing_labels"]
    assert sensing.shape == (128, 2)
    assert np.all(sensing[:, 0] >= 0.0) and np.all(sensing[:, 0] <= 10.0)
    assert np.all(sensing[:, 1] >= 0.0) and np.all(sensing[:, 1] <= _FD_MAX)
    assert np.all(np.isfinite(sensing))

    assert np.all(tiny_dataset["snr_db"] == 0.0)
    assert np.all(tiny_dataset["k"] == 1)

def test_tiny_model_fixture_contract(tiny_model: Any) -> None:
    
    assert tiny_model is not None
