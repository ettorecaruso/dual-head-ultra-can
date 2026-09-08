"""Test per ``src.experiments.pipeline.prepare_all_datasets``.

Verifica la pre-generazione "una tantum" di TUTTI i dataset degli esperimenti
leggendo ``configs/experiments.yaml``:

  - scopre i blocchi TOP-LEVEL ``ber_vs_snr..final_report`` (regressione: la versione
    precedente iterava ``experiments.*`` e trovava ZERO configurazioni);
  - per ``ber_vs_snr``/``exp2`` espande gli scenari (``echoes``/``max_doppler``) con
    lo stesso hash che il runner (``run_exp1``/``run_exp2``) richiede a runtime;
  - ``jamming`` usa il blocco ``data`` piano; ``classical_receivers``/``final_report`` non producono
    dataset (nessun caricamento da disco);
  - deduplica per hash (ber_vs_snr/exp2 condividono le directory degli scenari);
  - il ramo ``--split`` chiama ``generate_dataset(cfg, split, output_dir)``.

I test MONKEYPATCHANO ``prepare_dataset`` / ``generate_dataset`` per registrare
le config senza generare file (``get_dataset_dir`` è una funzione pura).

Esecuzione (comando dal log):
  python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import pipeline
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config
from src.utils.dataset_utils import get_dataset_dir

EXPERIMENTS_PATH = _REPO_ROOT / "configs" / "experiments.yaml"
BASE_CONFIG_PATH = DEFAULT_BASE_CONFIG_PATH

def _load_sources() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    base = load_config(EXPERIMENTS_PATH, BASE_CONFIG_PATH)
    raw = yaml.safe_load(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    return base, raw

def _merged(
    base: Dict[str, Any],
    raw: Dict[str, Any],
    exp_name: str,
    mode: str,
    scenario: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Replica il merge del runner: base data <- expN.mode.data <- scenario.

    Args:
        base: Config base mergiata (da ``_load_sources``).
        raw: Raw experiments.yaml.
        exp_name: Nome del blocco top-level (es. ``"ber_vs_snr"``).
        mode: ``"full"`` o ``"fast"``.
        scenario: Dict scenario con ``echoes``/``max_doppler`` (opzionale).

    Returns:
        Config completa con ``data`` mergiata (e scenario applicato se fornito).
    """
    data_params = raw[exp_name][mode].get("data") or {}
    merged = base.copy()
    merged["data"] = {**base["data"], **data_params}
    if scenario is not None:
        merged["data"]["echoes"] = list(scenario["echoes"])
        merged["data"]["max_doppler"] = float(scenario["max_doppler"])
        scenario_data = scenario.get("data")
        if scenario_data is not None:
            merged["data"].update(scenario_data)
    return merged

def _expected_dirs(base: Dict[str, Any], raw: Dict[str, Any]) -> Set[str]:
    
    dirs: Set[str] = set()
    default_scenarios = (
        (base.get("experiments") or {})
        .get("ber_vs_snr", {})
        .get("scenarios") or []
    )
    for mode in ("full", "fast"):
        for scenario in raw["ber_vs_snr"][mode]["experiments"]["ber_vs_snr"]["scenarios"]:
            dirs.add(str(get_dataset_dir(_merged(base, raw, "ber_vs_snr", mode, scenario))))
        for sname in raw["exp2"][mode]["experiments"]["exp2_reduced_params"]["scenarios"]:
            scenario = next(s for s in default_scenarios if s.get("name") == sname)
            dirs.add(str(get_dataset_dir(_merged(base, raw, "exp2", mode, scenario))))
        if "data" in raw["jamming"][mode]:
            dirs.add(str(get_dataset_dir(_merged(base, raw, "jamming", mode))))
    return dirs

def test_prepare_all_datasets_discovers_scenario_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scopre i blocchi top-level ber_vs_snr..final_report e espande gli scenari di ber_vs_snr/exp2."""
    recorded: List[Dict[str, Any]] = []

    def fake_prepare_dataset(cfg: Dict[str, Any], no_regen: bool = False) -> Path:
        
        recorded.append(cfg)
        return get_dataset_dir(cfg)

    monkeypatch.setattr(pipeline, "prepare_dataset", fake_prepare_dataset)

    pipeline.prepare_all_datasets(EXPERIMENTS_PATH)

    assert recorded, "prepare_dataset deve essere chiamata per ogni config unica"

    recorded_dirs = {str(get_dataset_dir(cfg)) for cfg in recorded}
    base, raw = _load_sources()
    expected_dirs = _expected_dirs(base, raw)

    assert recorded_dirs == expected_dirs, (
        "le directory generate non coincidono con quelle richieste dal runner "
        "(scenari ber_vs_snr/exp2 + jamming piano)"
    )
    assert len(recorded) == len(expected_dirs), (
        "dedup per hash fallito: attese %d directory uniche, registrate %d"
        % (len(expected_dirs), len(recorded))
    )

def test_prepare_all_datasets_scenario_overrides_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    
    recorded: List[Dict[str, Any]] = []

    def fake_prepare_dataset(cfg: Dict[str, Any], no_regen: bool = False) -> Path:
        recorded.append(cfg)
        return get_dataset_dir(cfg)

    monkeypatch.setattr(pipeline, "prepare_dataset", fake_prepare_dataset)
    pipeline.prepare_all_datasets(EXPERIMENTS_PATH)

    base, raw = _load_sources()
    by_dir = {str(get_dataset_dir(cfg)): cfg["data"] for cfg in recorded}

    sc_limited = next(
        s for s in raw["ber_vs_snr"]["full"]["experiments"]["ber_vs_snr"]["scenarios"]
        if s["name"] == "k3_doppler_limited"
    )
    dir_limited = str(get_dataset_dir(_merged(base, raw, "ber_vs_snr", "full", sc_limited)))
    assert dir_limited in by_dir
    assert by_dir[dir_limited]["echoes"] == [3]
    assert by_dir[dir_limited]["max_doppler"] == 4e-5

    sc_k1 = next(
        s for s in raw["ber_vs_snr"]["full"]["experiments"]["ber_vs_snr"]["scenarios"]
        if s["name"] == "k1_doppler_full"
    )
    dir_k1 = str(get_dataset_dir(_merged(base, raw, "ber_vs_snr", "full", sc_k1)))
    assert dir_k1 in by_dir
    assert by_dir[dir_k1]["feature_mode"] == "iq"
    assert by_dir[dir_k1]["echoes"] == [1]
    assert by_dir[dir_k1]["max_doppler"] == 8e-5

    dir_exp4_full = str(get_dataset_dir(_merged(base, raw, "jamming", "full")))
    assert dir_exp4_full in by_dir
    assert by_dir[dir_exp4_full]["echoes"] == [1, 3]

def test_apply_scenario_config_iq_override() -> None:
    """`_apply_scenario_config` applica gli override data.* opzionali (feature_mode)."""
    base, raw = _load_sources()
    merged = pipeline._merge_data_config(base, raw["ber_vs_snr"]["full"].get("data") or {})
    assert merged["data"]["feature_mode"] == "iq"

    sc_iq = {"name": "k1_doppler_full_iq", "echoes": [1], "max_doppler": 0.00008,
             "data": {"feature_mode": "iq"}}
    out = pipeline._apply_scenario_config(merged, sc_iq)
    assert out["data"]["feature_mode"] == "iq"
    assert out["data"]["echoes"] == [1]
    assert out["data"]["max_doppler"] == 8e-5
    assert out["data"]["sequence_length"] == merged["data"]["sequence_length"]
    sc_real = {"name": "k1_real_ref", "echoes": [1], "max_doppler": 0.00008,
               "data": {"feature_mode": "real"}}
    out_real = pipeline._apply_scenario_config(merged, sc_real)
    assert out_real["data"]["feature_mode"] == "real"
    assert merged["data"]["feature_mode"] == "iq"

def test_apply_scenario_config_rejects_non_dict_data() -> None:
    
    base, raw = _load_sources()
    merged = pipeline._merge_data_config(base, raw["ber_vs_snr"]["full"].get("data") or {})
    sc_bad = {"name": "bad", "echoes": [1], "max_doppler": 0.00008,
              "data": "iq"}
    with pytest.raises(ValueError, match="'data' deve essere un dict"):
        pipeline._apply_scenario_config(merged, sc_bad)

def test_prepare_all_datasets_split_calls_generate_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: List[Tuple[str, str, List[int], float]] = []

    def fake_get_dataset_dir(cfg: Dict[str, Any]) -> Path:
        real = get_dataset_dir(cfg)
        return tmp_path / Path(real).name

    def fake_generate_dataset(
        cfg: Dict[str, Any], split: str, output_dir: Path
    ) -> List[Path]:
        
        calls.append(
            (
                split,
                str(output_dir),
                list(cfg["data"]["echoes"]),
                float(cfg["data"]["max_doppler"]),
            )
        )
        return []

    monkeypatch.setattr(pipeline, "get_dataset_dir", fake_get_dataset_dir)
    monkeypatch.setattr(pipeline, "generate_dataset", fake_generate_dataset)

    pipeline.prepare_all_datasets(EXPERIMENTS_PATH, split="train")

    assert calls, "generate_dataset deve essere chiamata per ogni config unica"
    assert all(s == "train" for s, _, _, _ in calls), "split deve essere 'train'"

    base, raw = _load_sources()
    expected_tmp = {str(tmp_path / Path(d).name) for d in _expected_dirs(base, raw)}
    actual_out = {out for _, out, _, _ in calls}
    assert actual_out == expected_tmp, "le directory del ramo --split non coincidono"

    sc_limited = next(
        s for s in raw["ber_vs_snr"]["full"]["experiments"]["ber_vs_snr"]["scenarios"]
        if s["name"] == "k3_doppler_limited"
    )
    dir_limited = str(
        tmp_path
        / Path(str(get_dataset_dir(_merged(base, raw, "ber_vs_snr", "full", sc_limited)))).name
    )
    by_out = {out: (ech, dop) for _, out, ech, dop in calls}
    assert by_out[dir_limited] == ([3], 4e-5)

def test_prepare_all_datasets_forwards_no_regen(monkeypatch: pytest.MonkeyPatch) -> None:
    
    seen: List[bool] = []

    def fake_prepare_dataset(cfg: Dict[str, Any], no_regen: bool = False) -> Path:
        seen.append(no_regen)
        return get_dataset_dir(cfg)

    monkeypatch.setattr(pipeline, "prepare_dataset", fake_prepare_dataset)

    pipeline.prepare_all_datasets(EXPERIMENTS_PATH, no_regen=True)

    assert seen, "prepare_dataset deve essere chiamata"
    assert all(v is True for v in seen), "no_regen=True deve essere propagato"

def test_prepare_all_datasets_invalid_split_raises() -> None:
    """Split non ammesso -> ValueError (fail-fast,)."""
    with pytest.raises(ValueError, match="split"):
        pipeline.prepare_all_datasets(EXPERIMENTS_PATH, split="bogus")

def test_prepare_all_datasets_missing_config_raises(tmp_path: Path) -> None:
    """config_path inesistente -> FileNotFoundError."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        pipeline.prepare_all_datasets(missing)
