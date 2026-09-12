#!/usr/bin/env python3

"""Single entry point for all paper experiments."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import pipeline
from src.models.dcsk_correlator import evaluate_classical, dcsk_correlator_demodulate
from src.utils.config_loader import (
    DEFAULT_BASE_CONFIG_PATH,
    _deep_merge,
    load_config,
    save_config_snapshot,
    validate_config,
)
from src.utils.dataset_utils import get_dataset_dir
from src.utils.logger import get_logger, log_config_summary, setup_logging
from src.utils.model_io import load_model
from src.utils.model_names import canonical_model_name

logger = get_logger(__name__)

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False


def _log_rss(tag: str) -> None:
    
    if _HAS_PSUTIL:
        rss_gb = _psutil.Process().memory_info().rss / 1e9
    else:
        import resource as _res
        rss_gb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6
    logger.info("RSS [%s] = %.2f GB", tag, rss_gb)

_EXPERIMENT_ORDER = ("ber_vs_snr", "classical_receivers", "jamming",
                     "jamming_interpretability", "final_report")

_LEGACY_NUMERIC_IDS = {
    "1": "ber_vs_snr",
    "2": "classical_receivers",
    "3": "jamming",
    "4": "jamming_interpretability",
    "5": "final_report",
}
_VALID_MODELS = ("conv1d", "qkv", "lstm", "mc_dlsk")
_DEFAULT_MODEL = None
_DEFAULT_MODE = "fast"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results"


def _validate_common_config(config: Dict[str, Any]) -> None:
    
    required_keys = (
        "general.experiment_name",
        "general.seed",
        "general.log_level",
        "data.sequence_length",
        "data.feature_mode",
        "data.snr_range",
        "data.snr_step",
        "data.echoes",
        "data.max_delay",
        "data.max_doppler",
        "training.batch_size",
        "training.lambda_mse",
        "training.loss_weights.comm",
        "training.loss_weights.sensing",
    )
    validate_config(config, required_keys)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Single entry point for the DH-ISAC experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/experiments/runner.py --experiments 1 --mode fast
  python src/experiments/runner.py --experiments 1,2,3 --mode full --model qkv
  python src/experiments/runner.py --experiments all --mode full --no-regen
        """,
    )
    parser.add_argument(
        "--experiments",
        required=True,
        help="Experiment identifiers to run, comma separated (e.g. 1,2,3) or 'all'",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default=_DEFAULT_MODE,
        help=f"Execution mode (default: {_DEFAULT_MODE})",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            "Architecture to run (single value, e.g. conv1d, or a comma "
            "separated list for jamming_interpretability). "
            f"(choices: {', '.join(_VALID_MODELS)}). When omitted, all "
            "architectures from the experiment config are used."
        ),
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="If set, do not regenerate the dataset (use existing files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "experiments.yaml",
        help="Path to the experiments.yaml file (default: configs/experiments.yaml)",
    )
    return parser.parse_args(argv)

def parse_experiment_list(experiments_arg: str) -> List[str]:
    """Map a --experiments CLI value to an ordered list of experiment names.

    Accepts experiment names (e.g. "jamming"), legacy numeric ids (e.g. "3")
    or "all".

    Args:
        experiments_arg: Comma-separated experiment identifiers or "all".

    Returns:
        Ordered list of experiment names.

    Raises:
        ValueError: If an identifier is not recognised.
    """
    if experiments_arg.strip().lower() == "all":
        return list(_EXPERIMENT_ORDER)

    parts = [p.strip() for p in experiments_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("--experiments cannot be empty")

    names: List[str] = []
    for token in parts:
        if token in _EXPERIMENT_ORDER:
            names.append(token)
        elif token in _LEGACY_NUMERIC_IDS:
            names.append(_LEGACY_NUMERIC_IDS[token])
        else:
            valid = ", ".join(list(_EXPERIMENT_ORDER) + list(_LEGACY_NUMERIC_IDS))
            raise ValueError(f"Invalid experiment identifier {token!r}. Valid: {valid}")
    return list(dict.fromkeys(names))

def load_experiment_config(
    experiment_name: str,
    mode: str,
    experiments_yaml_path: Path,
    base_config_path: Path = DEFAULT_BASE_CONFIG_PATH,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    base = load_config(
        config_path=experiments_yaml_path,
        base_config_path=base_config_path,
        cli_overrides=None,
    )

    with open(experiments_yaml_path, "r", encoding="utf-8") as f:
        experiments_full = yaml.safe_load(f)

    exp_key = experiment_name
    if exp_key not in experiments_full:
        raise ValueError(f"Section '{exp_key}' missing in {experiments_yaml_path}")

    exp_section = experiments_full[exp_key]
    if mode not in exp_section:
        raise ValueError(
            f"Mode '{mode}' missing for {exp_key} in {experiments_yaml_path}"
        )

    exp_config = exp_section[mode]

    merged = _deep_merge(base, exp_config)

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    _validate_common_config(merged)

    logger.debug("Config loaded for %s (%s): %d keys", experiment_name, mode, len(merged))
    return merged


def _evaluate_model(
    model: tf.keras.Model,
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.evaluation.evaluator import (
        compute_ber_curve,
        evaluate_model as eval_model,
        evaluate_model_online,
        plot_ber_vs_snr,
    )

    if config.get("evaluation", {}).get("online_generation", False):
        results = evaluate_model_online(model, config)
    else:
        results = eval_model(model, test_data, config)
    df_ber = compute_ber_curve(results)

    if "corr_tau" in df_ber.columns and len(df_ber) > 0:
        corr_tau_max = float(df_ber["corr_tau"].max())
        mse_tau_high = float(df_ber["mse_tau"].iloc[-1])
        if corr_tau_max < 0.15:
            logger.warning(
                "SENSING COLLAPSED? max corr(tau) = %.3f (threshold 0.15), "
                "MSE_tau high-SNR = %.2f (~variance %s): check "
                "use_reference_profile/feature_mode in model %s",
                corr_tau_max, mse_tau_high, "90.8", model.name,
            )
        else:
            logger.info(
                "Sensing OK: corr(tau) max = %.3f, MSE_tau high-SNR = %.2f",
                corr_tau_max, mse_tau_high,
            )
        if "corr_fd" in df_ber.columns:
            corr_fd_max = float(df_ber["corr_fd"].max())
            if corr_fd_max < 0.1:
                logger.warning(
                    "fD is not resolvable from the current parameters (max corr(fD) = %.3f): "
                    "the Doppler phase accumulated over N_seq=%d samples with "
                    "max_doppler=%g is ~0.05 rad (Cramer-Rao bound). "
                    "Estimating fD requires longer windows or a larger max_doppler.",
                    corr_fd_max,
                    int(config.get("data", {}).get("sequence_length", 100)),
                    config.get("data", {}).get("max_doppler", 8e-5),
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    df_ber.to_csv(csv_path, index=False)

    plot_format = config["visualization"].get("plot_format", "pdf")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_ber_vs_snr(df_ber, plot_dir, plot_format, model.name)

    return {"metrics": df_ber, "plots": {"ber_vs_snr": plot_path}}

def _evaluate_dcsk_on_test_data(
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    snr_test_range = config["evaluation"]["snr_test_range"]
    batch_size = int(config["training"]["batch_size"])
    max_symbols_per_snr = config["evaluation"]["max_symbols_per_snr"]
    bit_error_threshold = config["evaluation"]["bit_error_threshold"]

    results: List[Dict[str, Any]] = []

    template = generate_chaotic_sequence(
        map_type=config["data"]["map_type"],
        map_param=config["data"]["map_param"],
        seed=int(config["general"]["seed"]),
        sequence_length=beta,
    )

    for snr_target in snr_test_range:
        mask = np.isclose(test_data["snr_db"], snr_target, rtol=0, atol=1e-6)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        y_snr = test_data["x"][idx]
        bit_snr = test_data["bit"][idx]
        n_total = len(bit_snr)

        accum_errors = 0
        accum_symbols = 0

        start = 0
        while (
            start < n_total
            and accum_symbols < max_symbols_per_snr
            and accum_errors < bit_error_threshold
        ):
            end = min(start + batch_size, n_total)
            batch_x = y_snr[start:end]
            batch_bit = bit_snr[start:end]

            bits_pred = dcsk_correlator_demodulate(batch_x, threshold=threshold)

            errors = np.sum(bits_pred != batch_bit)
            accum_errors += int(errors)
            accum_symbols += len(batch_bit)
            start = end

        if accum_symbols > 0:
            ber_val = accum_errors / accum_symbols
            results.append(
                {
                    "snr_db": float(snr_target),
                    "ber": ber_val,
                    "n_errors": accum_errors,
                    "n_symbols": accum_symbols,
                    "mse_tau": 0.0,
                    "mse_fd": 0.0,
                }
            )

    df = pd.DataFrame(results)
    df = df.sort_values("snr_db").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    return df

def _find_scenario_by_name(scenarios: List[Dict], name: str) -> Optional[Dict]:
    
    for s in scenarios:
        if s.get("name") == name:
            return s
    return None

def _compute_winner_summary(
    curves: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, float]:
    
    arch_bers: Dict[str, List[float]] = {}
    for scenario_curves in curves.values():
        for arch, df in scenario_curves.items():
            if "ber" in df.columns and not df.empty:
                arch_bers.setdefault(arch, []).append(float(df["ber"].mean()))

    summary = {arch: float(np.mean(bers)) for arch, bers in arch_bers.items()}
    return summary

def _generate_pdf_from_tex(tex_path: Path) -> Optional[Path]:
    """Convert a .tex file to PDF using pdflatex (if available)."""
    pdf_path = tex_path.with_suffix(".pdf")

    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("pdflatex unavailable, skipping PDF generation")
            return None

        cwd = tex_path.parent
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        logger.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd)

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            logger.error("pdflatex failed (code %d): %s", result.returncode, result.stderr)
            return None

        if pdf_path.exists():
            logger.info("PDF generated: %s", pdf_path)
            return pdf_path
        else:
            logger.warning("PDF not generated although pdflatex ran without errors")
            return None

    except FileNotFoundError:
        logger.warning("pdflatex not found in PATH, skipping PDF generation")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timeout after 30 seconds")
        return None
    except Exception as e:
        logger.warning("Error while generating the PDF: %s", e)
        return None


def run_ber_vs_snr(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("ber_vs_snr experiment: BER vs SNR (5 architectures, 3 scenarios)")
    logger.info("=" * 60)

    exp_cfg = config["experiments"]["ber_vs_snr"]
    architectures = list(exp_cfg["architectures"])
    if model_type is not None:
        architectures = [a for a in architectures if a == model_type]
        logger.info(
            "ber_vs_snr: --model=%s filters architectures to: %s", model_type, architectures
        )
    scenarios = exp_cfg["scenarios"]
    plot_format = config["visualization"].get("plot_format", "pdf")

    results: Dict[str, Any] = {
        "scenarios": {},
        "curves": {},
        "plots": {},
    }

    for scenario in scenarios:
        scenario_name = scenario["name"]
        logger.info("--- Scenario: %s ---", scenario_name)

        scenario_config = pipeline._apply_scenario_config(config, scenario)

        data_dir = pipeline.prepare_dataset(scenario_config, no_regen)
        online_eval = bool(
            (scenario_config.get("evaluation") or {}).get("online_generation", False)
        )
        # The static test split is only needed for the offline evaluation and for
        # the model-free receivers (blind_stat/dcsk). Full runs use online
        # evaluation, so the test split (the largest one) is not loaded for the
        # DL-only runs: this keeps the Colab RAM footprint well below the limit.
        needs_test = (not online_eval) or any(
            a in ("blind_stat", "dcsk") for a in architectures
        )
        train_data, train_ds, val_ds, test_data = pipeline.load_datasets(
            scenario_config, data_dir, include_test=needs_test
        )

        scenario_curves: Dict[str, pd.DataFrame] = {}
        scenario_results = {}

        for arch in architectures:
            logger.info("Architecture: %s", arch)
            if arch in ("conv1d", "qkv", "lstm", "mc_dlsk"):
                arch_output_dir = output_dir / scenario_name / arch
                model = pipeline.build_model(scenario_config, arch)
                scenario_config.setdefault("general", {})["run_output_dir"] = str(
                    arch_output_dir
                )
                res = pipeline.train_and_evaluate(
                    config=scenario_config,
                    model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_data=test_data,
                    output_dir=arch_output_dir,
                )
                logs_dir = arch_output_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                save_config_snapshot(scenario_config, logs_dir)
                df_ber = res["metrics"]
                scenario_curves[arch] = df_ber
                scenario_results[arch] = res
                res.pop("model", None)
                del model
                gc.collect()
                tf.keras.backend.clear_session()
                _log_rss(f"ber_vs_snr {scenario_name}/{arch}")
            elif arch == "dcsk":
                arch_output_dir = output_dir / scenario_name / "dcsk_correlator"
                df_dcsk = _evaluate_dcsk_on_test_data(
                    test_data, scenario_config, arch_output_dir
                )
                scenario_curves["dcsk"] = df_dcsk
                scenario_results["dcsk"] = {"metrics": df_dcsk}
            elif arch == "blind_stat":
                from src.models.blind_stat import evaluate_on_test_data as _eval_blind

                arch_output_dir = output_dir / scenario_name / "blind_stat"
                df_blind = _eval_blind(test_data, scenario_config, arch_output_dir)
                scenario_curves["blind_stat"] = df_blind
                scenario_results["blind_stat"] = {"metrics": df_blind}
            else:
                logger.warning("Unsupported architecture: %s, skipping", arch)
                continue

        plot_dir = output_dir / "plots"
        plot_path = pipeline.collect_and_plot_overlay(
            curves=scenario_curves,
            scenario=scenario_name,
            output_dir=plot_dir,
            plot_format=plot_format,
        )
        results["plots"][scenario_name] = plot_path
        results["curves"][scenario_name] = scenario_curves
        results["scenarios"][scenario_name] = scenario_results
        del train_data, train_ds, val_ds, test_data
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"ber_vs_snr {scenario_name} fine scenario")

    logger.info("ber_vs_snr experiment completed.")
    return results

def _find_scenario_trained_model(
    mode: str,
    scenario_name: str,
    arch: str,
    run_root: Optional[Path] = None,
) -> Optional[Path]:
    
    candidates: List[Path] = []
    if run_root is not None:
        candidates.append(
            run_root / "ber_vs_snr" / scenario_name / arch / "best_model.keras"
        )
    candidates.extend(
        [
            _REPO_ROOT / "results" / mode / "ber_vs_snr" / scenario_name / arch / "best_model.keras",
            _REPO_ROOT / "results" / "experiments" / "ber_vs_snr" / scenario_name / arch / "best_model.keras",
        ]
    )
    for ckpt in candidates:
        if ckpt.is_file():
            return ckpt
    return None


def run_classical_receivers(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("Experiment classical_receivers: equations only (no DL) - classical receivers")
    logger.info("=" * 60)

    echo_config = copy.deepcopy(config)
    echo_config["channel"] = {
        "add_awgn": False,
        "echo_fading": "none",
        "echo_only_mode": True,
    }

    sweep_k_cfg = config.get("sweep_k", {})
    k_min = sweep_k_cfg.get("min", 1)
    k_max = sweep_k_cfg.get("max", 10)
    k_step = sweep_k_cfg.get("step", 1)
    k_range = range(k_min, k_max + 1, k_step)

    sweep_doppler_cfg = config.get("sweep_doppler", {})
    doppler_min = sweep_doppler_cfg.get("min", 0.0)
    doppler_max = sweep_doppler_cfg.get("max", 8e-5)
    doppler_num = sweep_doppler_cfg.get("num_points", 10)
    doppler_range = np.linspace(doppler_min, doppler_max, doppler_num)

    echo_cfg = config.get("echo_only", {})
    doppler_fixed = echo_cfg.get("doppler_fixed", 4e-5)
    k_fixed = echo_cfg.get("k_fixed", 3)

    num_symbols = config["data"].get("num_symbols_test", 2000)

    results_k = _sweep_k_echo_only(
        echo_config, k_range, doppler_fixed, num_symbols, output_dir, echo_cfg
    )

    results_doppler = _sweep_doppler_echo_only(
        echo_config, k_fixed, doppler_range, num_symbols, output_dir, echo_cfg
    )

    rows_k = [{"k": k, **ber_dict} for k, ber_dict in sorted(results_k.items())]
    pd.DataFrame(rows_k).to_csv(output_dir / "ber_vs_k.csv", index=False)
    rows_d = [{"doppler_max": d, **ber_dict} for d, ber_dict in sorted(results_doppler.items())]
    pd.DataFrame(rows_d).to_csv(output_dir / "ber_vs_doppler.csv", index=False)
    logger.info("classical_receivers results saved to %s/{ber_vs_k,ber_vs_doppler}.csv", output_dir)

    logger.info("classical_receivers experiment completed.")
    return {
        "results_k": results_k,
        "results_doppler": results_doppler,
    }

def _find_clean_ber_curve(
    mode: str,
    arch: str,
    run_root: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    
    bases: List[Path] = []
    if run_root is not None:
        bases.append(run_root / "ber_vs_snr")
    bases.append(_REPO_ROOT / "results" / mode / "ber_vs_snr")

    for base in bases:
        for scenario in ("k1_doppler_full", "k3_doppler_full", "k3_doppler_limited"):
            path = base / scenario / arch / "metrics.csv"
            if not path.is_file():
                continue
            try:
                df = pd.read_csv(path)
                if {"snr_db", "ber"}.issubset(df.columns) and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("ber_vs_snr metrics.csv not readable (%s): %s", path, exc)
    logger.debug("clean ber_vs_snr curve not found for %s (mode=%s)", arch, mode)
    return None


def _plot_jamming_vs_clean_ber(
    model_name: str,
    jamming_dfs: Dict[str, pd.DataFrame],
    clean_curve: Optional[pd.DataFrame],
    output_dir: Path,
    plot_format: str,
) -> Optional[Path]:
    
    if not jamming_dfs:
        logger.warning("No jammed curve for %s: skipping the comparison", model_name)
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_clean, ax_jam) = plt.subplots(1, 2, figsize=(14, 6))

    clean_mean: Optional[float] = None
    if clean_curve is not None and not clean_curve.empty:
        ax_clean.semilogy(
            clean_curve["snr_db"], clean_curve["ber"],
            marker="o", linestyle="-", color="#1f77b4", linewidth=2,
            label="Clean (ber_vs_snr)",
        )
        ax_clean.set_xlabel("SNR (dB)")
        ax_clean.set_ylabel("Bit Error Rate (BER)")
        ax_clean.set_title(f"{model_name} - clean (ber_vs_snr)")
        ax_clean.grid(True, which="both", linestyle="--", alpha=0.6)
        ax_clean.legend()
        clean_mean = float(np.mean(clean_curve["ber"]))
    else:
        ax_clean.text(
            0.5, 0.5, "clean ber_vs_snr curve not available",
            ha="center", va="center", transform=ax_clean.transAxes,
        )
        ax_clean.set_title(f"{model_name} - clean (ber_vs_snr)")

    _JAM_COLORS = {"cw": "blue", "barrage": "red", "partial_band": "green"}
    _JAM_MARKERS = {"cw": "o", "barrage": "s", "partial_band": "D"}
    for jammer_type, df in jamming_dfs.items():
        if df is None or df.empty or "jsr_db" not in df or "ber" not in df:
            continue
        ax_jam.semilogy(
            df["jsr_db"], df["ber"],
            marker=_JAM_MARKERS.get(jammer_type, "o"),
            linestyle="-",
            color=_JAM_COLORS.get(jammer_type),
            linewidth=2,
            label=jammer_type.capitalize().replace("_", " "),
        )
    if clean_mean is not None:
        ax_jam.axhline(
            clean_mean, linestyle="--", color="gray", linewidth=1.5,
            label=f"Clean mean (ber_vs_snr) = {clean_mean:.4f}",
        )
    ax_jam.set_xlabel("JSR (dB)")
    ax_jam.set_ylabel("Bit Error Rate (BER)")
    ax_jam.set_title(f"{model_name} - jammed")
    ax_jam.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_jam.legend()

    fig.suptitle(f"{model_name}: Jammed BER vs clean BER (ber_vs_snr)", y=1.02)
    out_path = output_dir / f"ber_vs_jsr_{model_name}_vs_clean.{plot_format}"
    plt.savefig(out_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Jamming vs clean ber_vs_snr comparison saved to %s", out_path)
    return out_path


def run_jamming(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("Experiment jamming: jamming robustness + explainability")
    logger.info("=" * 60)

    jamming_cfg = config.get("experiments", {}).get("jamming", {}) or {}
    models_to_test = list(jamming_cfg.get("models") or ["conv1d", "qkv"])
    if model_type is not None:
        models_to_test = [m for m in models_to_test if m == model_type]
        logger.info(
            "jamming: --model=%s -> models filtered to: %s", model_type, models_to_test
        )
    logger.info("Models to test: %s", models_to_test)

    data_dir = pipeline.prepare_dataset(config, no_regen)
    # The static test split is needed only for the jamming evaluation, while the
    # training (run_single_experiment below) loads train/val only when the
    # evaluation is online. Loading the test split once here (instead of through
    # load_datasets with train+val) avoids keeping two full copies of the largest
    # split in RAM during training.
    test_data = pipeline.load_test_data(config, data_dir)

    results: Dict[str, Any] = {"models": {}}
    visualize_layers = bool(jamming_cfg.get("visualize_layers", False))

    for model_name in models_to_test:
        logger.info("=" * 60)
        logger.info("Model: %s", model_name)
        logger.info("=" * 60)

        model_output_dir = output_dir / model_name
        res = pipeline.run_single_experiment(
            config=config,
            model_type=model_name,
            output_dir=model_output_dir,
            no_regen=no_regen,
        )
        model = res.get("model")
        if model is None:
            raise RuntimeError("run_single_experiment did not return a model")

        baseline_results = _evaluate_model(
            model, test_data, config, model_output_dir / "jamming"
        )

        jamming_dir = model_output_dir / "jamming"
        jamming_results = pipeline.evaluate_with_jamming(
            config=config,
            model=model,
            test_data=test_data,
            output_dir=jamming_dir,
            model_name=model_name,
        )

        if visualize_layers:
            layer_dir = model_output_dir / "layer_activity"
            pipeline.plot_layer_activity(
                config=config,
                model=model,
                test_data=test_data,
                output_dir=layer_dir,
            )

        results["models"][model_name] = {
            "baseline": baseline_results,
            "jamming": jamming_results,
            "model_path": model_output_dir / "best_model.keras",
        }

        plot_format = config["visualization"].get("plot_format", "pdf")
        _plot_jamming_vs_clean_ber(
            model_name=model_name,
            jamming_dfs=(jamming_results or {}).get("results", {}),
            clean_curve=_find_clean_ber_curve(
                mode, model_name, run_root=output_dir.parent
            ),
            output_dir=jamming_dir,
            plot_format=plot_format,
        )

        del model, res
        gc.collect()
        tf.keras.backend.clear_session()
        _log_rss(f"jamming {model_name}")

    _plot_jamming_multi_model(results, output_dir, plot_format)

    logger.info("jamming experiment completed.")
    return results

def _plot_jamming_multi_model(
    results: Dict[str, Any],
    output_dir: Path,
    plot_format: str,
) -> Optional[Path]:
    
    models = results.get("models", {})
    if not models:
        logger.warning("No models evaluated: skipping the multi-model plot")
        return None

    first = next(iter(models.values()))
    all_dfs = (first.get("jamming") or {}).get("results", {})
    if not all_dfs:
        logger.warning("No jamming results for the multi-model plot")
        return None

    jammer_types = list(all_dfs.keys())
    _JAM_PALETTE = {
        "conv1d": "#1f77b4", "qkv": "#ff7f0e", "lstm": "#2ca02c", "mc_dlsk": "#d62728",
    }

    fig, axes = plt.subplots(1, len(jammer_types), figsize=(6 * len(jammer_types), 5), squeeze=False)
    for ax, jt in zip(axes[0], jammer_types):
        for name, mod in models.items():
            dfs = (mod.get("jamming") or {}).get("results", {})
            df = dfs.get(jt)
            if df is None or df.empty:
                logger.warning("Model %s without jamming data for %s", name, jt)
                continue
            ax.semilogy(
                df["jsr_db"], df["ber"],
                marker="o", linestyle="-", linewidth=2, markersize=6,
                color=_JAM_PALETTE.get(name, "gray"),
                label=name,
            )
        ax.set_xlabel("JSR (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(jt.capitalize().replace("_", " "))
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.set_ylim([1e-6, 1.0])
        ax.legend()

    fig.suptitle("Jammed BER comparison - all models", y=1.02)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"ber_vs_jsr_all_models.{plot_format}"
    plt.savefig(out_path, format=plot_format, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Multi-model plot saved to %s", out_path)
    return out_path

def run_jamming_interpretability(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    from src.data.data_loader import build_reference_matrix, load_npz_files
    from src.experiments.jamming_interpretability import run_jamming_interpretability_probe

    jamming_interpretability_cfg = (config.get("experiments") or {}).get("jamming_interpretability") or {}
    models_to_test = list(jamming_interpretability_cfg.get("models") or ["conv1d", "qkv"])
    if model_type is not None:
        wanted = {model_type} if isinstance(model_type, str) else set(model_type)
        models_to_test = [m for m in models_to_test if m in wanted]
    jsr_values = [float(v) for v in (jamming_interpretability_cfg.get("jsr_values") or [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0])]
    jammer_types = list(jamming_interpretability_cfg.get("jamming_types") or ["cw", "barrage", "partial_band"])
    snr_eval = [float(v) for v in (jamming_interpretability_cfg.get("snr_eval") or [-1.0, 3.0, 7.0, 11.0, 15.0, 20.0])]
    ret_subset = int(jamming_interpretability_cfg.get("ret_subset", 3000))

    logger.info("Experiment jamming_interpretability (models=%s)", models_to_test)
    logger.info("  snr_eval=%s jsr=%s jammer=%s ret_subset=%d",
                snr_eval, jsr_values, jammer_types, ret_subset)

    data_dir = pipeline.prepare_dataset(config, no_regen)
    echoes = [int(k) for k in config["data"]["echoes"]]
    test_data = load_npz_files(data_dir, snr_eval, echoes, "test", config)
    test_data["x_ref"] = build_reference_matrix(test_data["bit"], test_data["seed"], config)
    logger.info("jamming_interpretability: test set of %d samples (SNR %s)", test_data["x"].shape[0], snr_eval)

    results: Dict[str, Any] = {"models": {}}
    for arch in models_to_test:
        logger.info("=" * 60)
        logger.info("jamming_interpretability architecture: %s", arch)

        arch_out = output_dir / arch
        candidates = [
            output_dir.parent / "jamming" / arch / "best_model.keras",
            _REPO_ROOT / "results" / mode / "jamming" / arch / "best_model.keras",
            _REPO_ROOT / "results" / "full" / "jamming" / arch / "best_model.keras",
        ]
        ckpt_path = next((c for c in candidates if c.is_file()), None)
        if ckpt_path is not None:
            logger.info("jamming_interpretability %s: jamming model loaded from %s", arch, ckpt_path)
            model = load_model(ckpt_path)
        else:
            logger.warning("jamming_interpretability %s: jamming checkpoint not found, training a fallback", arch)
            jamming_dir = output_dir.parent / "jamming" / arch
            jamming_dir.mkdir(parents=True, exist_ok=True)
            res_train = pipeline.run_single_experiment(
                config=config, model_type=arch, output_dir=jamming_dir, no_regen=no_regen,
            )
            model = res_train["model"]

        arch_res = run_jamming_interpretability_probe(
            model=model,
            arch=arch,
            test_data=test_data,
            config=config,
            out_dir=arch_out,
            jsr_values=jsr_values,
            jammer_types=jammer_types,
            ret_subset=ret_subset,
        )
        results["models"][arch] = arch_res

    logger.info("jamming_interpretability experiment completed. Output in %s", output_dir)
    return results

def _find_trained_model(
    config: Dict[str, Any],
    arch: str,
    output_dir: Path,
) -> Optional[Path]:
    
    candidates: List[Path] = []

    ber_vs_snr_cfg = config.get("experiments", {}).get("ber_vs_snr", {})
    scenarios = ber_vs_snr_cfg.get("scenarios", []) if isinstance(ber_vs_snr_cfg, dict) else []

    run_root_candidates = [
        output_dir.parent,
        output_dir.parents[1],
    ]
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("name"):
            for run_root in run_root_candidates:
                candidates.append(
                    run_root / "ber_vs_snr" / str(scenario["name"]) / arch / "best_model.keras"
                )
    candidates.append(_REPO_ROOT / "results" / "experiments" / "ber_vs_snr" / arch / "best_model.keras")

    experiment_name = config["general"]["experiment_name"]
    model_dir = _REPO_ROOT / "results" / experiment_name / arch / "models"
    candidates.extend([
        model_dir / "best_model.keras",
        model_dir / "best_model.h5",
    ])

    for ckpt_path in candidates:
        if ckpt_path.is_file():
            logger.debug("Checkpoint found for '%s': %s", arch, ckpt_path)
            return ckpt_path
    logger.debug("No checkpoint found for '%s' among: %s", arch, candidates)
    return None

def run_final_report(
    config: Dict[str, Any],
    mode: str,
    model_type: Optional[str],
    no_regen: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    
    logger.info("=" * 60)
    logger.info("final_report experiment: final weight table (SWaP-C)")
    logger.info("=" * 60)

    models: Dict[str, Optional[tf.keras.Model]] = {}
    dl_archs = ["conv1d", "qkv", "lstm", "mc_dlsk"]

    for arch in dl_archs:
        ckpt_path = _find_trained_model(config, arch, output_dir)
        if ckpt_path is not None:
            logger.info("Loading model %s from %s", arch, ckpt_path)
            try:
                model = load_model(ckpt_path)
                models[arch] = model
            except Exception as e:
                logger.warning(
                    "Failed to load %s (%s): building from scratch",
                    arch, e,
                )
                try:
                    models[arch] = pipeline.build_model(config, arch)
                except Exception as build_exc:
                    logger.error("Building failed for %s: %s", arch, build_exc)
                    models[arch] = None
        else:
            logger.info("Building model %s from scratch (no checkpoint found)", arch)
            try:
                model = pipeline.build_model(config, arch)
                models[arch] = model
            except Exception as e:
                logger.error("Building failed for %s: %s", arch, e)
                models[arch] = None

    models["classical"] = None

    final_report_cfg = config["experiments"]["final_report"]
    output_rel = final_report_cfg.get("output", "weight_table.tex")
    output_path = Path(output_rel) if Path(output_rel).is_absolute() else output_dir / output_rel
    pipeline.generate_weight_table(
        config=config,
        models=models,
        output_path=output_path,
    )

    pdf_path = _generate_pdf_from_tex(output_path)

    report_path = None
    try:
        from src.experiments.final_report import generate_final_report
        report_path = generate_final_report(output_dir.parent, output_dir / "final_report.md", models)
        logger.info("Final report written to %s", report_path)
    except Exception as exc:
        logger.warning("Could not generate final report: %s", exc)

    logger.info("final_report experiment completed.")
    return {
        "tex_path": output_path,
        "pdf_path": pdf_path,
        "report_path": report_path,
    }


def _sweep_k_echo_only(
    config: Dict[str, Any],
    k_range: range,
    doppler_fixed: float,
    num_symbols: int,
    output_dir: Path,
    echo_cfg: Dict[str, Any],
) -> Dict[int, Dict[str, float]]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    map_type = config["data"]["map_type"]
    map_param = config["data"]["map_param"]
    seed = int(config["general"]["seed"])

    template = generate_chaotic_sequence(map_type, map_param, seed, beta)
    doppler_direct = float(echo_cfg.get("doppler_direct_fixed", 1e-5))

    results: Dict[int, Dict[str, float]] = {}

    for k in k_range:
        logger.debug("K=%d, doppler_direct=%.2e", k, doppler_direct)
        y, bits_true = _build_echo_only_dataset(
            config, k, doppler_direct, num_symbols, echo_cfg
        )

        ber_dict: Dict[str, float] = {}
        for detector in ("dcsk", "matched_filter", "energy_detector"):
            if detector == "matched_filter":
                res = evaluate_classical(
                    y[:, beta:], bits_true, detector, config,
                    ref=None, template=template
                )
            else:
                res = evaluate_classical(
                    y, bits_true, detector, config,
                    ref=None, template=None
                )
            ber = float(res["ber"])
            if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
                raise RuntimeError(
                    f"BER not finite or out of range for K={k}, detector={detector}: {ber}"
                )
            ber_dict[detector] = ber

        results[k] = ber_dict

    return results

def _sweep_doppler_echo_only(
    config: Dict[str, Any],
    k_fixed: int,
    doppler_range: np.ndarray,
    num_symbols: int,
    output_dir: Path,
    echo_cfg: Dict[str, Any],
) -> Dict[float, Dict[str, float]]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    threshold = float(config["baselines"]["dcsk_correlator"]["threshold"])
    map_type = config["data"]["map_type"]
    map_param = config["data"]["map_param"]
    seed = int(config["general"]["seed"])

    template = generate_chaotic_sequence(map_type, map_param, seed, beta)

    results: Dict[float, Dict[str, float]] = {}

    for doppler in doppler_range:
        logger.debug("doppler=%.2e, k_fixed=%d", doppler, k_fixed)
        y, bits_true = _build_echo_only_dataset(
            config, k_fixed, doppler, num_symbols, echo_cfg
        )

        ber_dict: Dict[str, float] = {}
        for detector in ("dcsk", "matched_filter", "energy_detector"):
            if detector == "matched_filter":
                res = evaluate_classical(
                    y[:, beta:], bits_true, detector, config,
                    ref=None, template=template
                )
            else:
                res = evaluate_classical(
                    y, bits_true, detector, config,
                    ref=None, template=None
                )
            ber = float(res["ber"])
            if not np.isfinite(ber) or ber < 0.0 or ber > 1.0:
                raise RuntimeError(
                    f"BER not finite or out of range for doppler={doppler}, detector={detector}: {ber}"
                )
            ber_dict[detector] = ber

        results[doppler] = ber_dict

    return results

def _build_echo_only_dataset(
    config: Dict[str, Any],
    k: int,
    doppler_direct: float,
    num_symbols: int,
    echo_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    
    from src.data.dataset_generator import generate_chaotic_sequence

    data_cfg = config["data"]
    sequence_length = int(data_cfg["sequence_length"])
    map_type = str(data_cfg["map_type"])
    map_param = float(data_cfg["map_param"])
    max_delay = int(data_cfg["max_delay"])
    beta = int(config["baselines"]["dcsk_correlator"]["correlation_length"])
    if 2 * beta != sequence_length:
        raise ValueError(
            f"classical_receivers requires 2*correlation_length ({2 * beta}) == sequence_length "
            f"({sequence_length}): the DCSK [ref|data] frame is not representable"
        )

    offset = int(echo_cfg.get("tau_offset", 7))
    alpha_base = float(echo_cfg.get("alpha_base", 0.2))
    alpha_variation = float(echo_cfg.get("alpha_variation", 0.5))
    echo_doppler_max = float(echo_cfg.get("doppler_fixed", 4e-5))
    snr_db = echo_cfg.get("snr_db")

    seed = int(config["general"]["seed"])
    rng = np.random.default_rng(seed + 42)

    bits_true = np.array([0] * (num_symbols // 2) + [1] * (num_symbols // 2), dtype=np.uint8)
    rng.shuffle(bits_true)

    frames = np.zeros((num_symbols, 2 * beta), dtype=np.float64)
    for i in range(num_symbols):
        ref = generate_chaotic_sequence(map_type, map_param, int(rng.integers(1, 2**31 - 1)), beta)
        data = ref if int(bits_true[i]) == 1 else -ref
        frames[i, :beta] = ref
        frames[i, beta:] = data

    guard = int(max_delay)
    stream_len = num_symbols * 2 * beta + guard
    stream = np.zeros(stream_len, dtype=np.float64)
    for i in range(num_symbols):
        stream[i * 2 * beta:(i + 1) * 2 * beta] = frames[i]

    n_idx = np.arange(stream_len, dtype=np.float64)

    alphas = [alpha_base * (1.0 + alpha_variation * (j / max(1, k - 1))) for j in range(k)]
    g = math.sqrt(max(1e-6, 1.0 - sum(a * a for a in alphas)))

    y = (g * stream).astype(np.complex128)

    for j in range(k):
        tau_j = (j * offset + 1) % max_delay + 1
        fd_j = (j / max(1, k - 1)) * echo_doppler_max
        x_delayed = np.zeros(stream_len, dtype=np.float64)
        x_delayed[tau_j:] = stream[: stream_len - tau_j]
        y = y + alphas[j] * x_delayed * np.exp(1j * 2.0 * math.pi * fd_j * n_idx)

    if snr_db is not None:
        snr_db = float(snr_db)
        ps = float(np.mean(np.abs(y) ** 2))
        noise_var = ps * 10.0 ** (-snr_db / 10.0)
        y = y + rng.normal(0.0, math.sqrt(noise_var / 2.0), stream_len) \
              + 1j * rng.normal(0.0, math.sqrt(noise_var / 2.0), stream_len)

    if not np.all(np.isfinite(y)):
        raise RuntimeError("echo-only signal not finite (NaN/Inf)")

    local_n = np.arange(2 * beta, dtype=np.float64)
    direct_phase = np.exp(1j * 2.0 * math.pi * float(doppler_direct) * local_n)
    y_sym = np.stack([
        y[i * 2 * beta:(i + 1) * 2 * beta] * direct_phase
        for i in range(num_symbols)
    ])
    return y_sym, bits_true



def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the requested experiments in sequence."""
    args = parse_args(argv)

    if isinstance(args.model, str) and args.model.strip():
        args.model = ",".join(
            canonical_model_name(part)
            for part in args.model.split(",")
            if part.strip()
        )

    exp_names = parse_experiment_list(args.experiments)
    logger.info("Experiments to run: %s", exp_names)

    if isinstance(args.model, str) and "," in args.model:
        parts = [m.strip() for m in args.model.split(",") if m.strip()]
        invalid = [m for m in parts if m not in _VALID_MODELS]
        if invalid:
            raise ValueError(f"Invalid models in --model: {invalid}")
        if set(exp_names) - {"jamming_interpretability"}:
            raise ValueError(
                "--model with multiple architectures is allowed only with "
                "--experiments jamming_interpretability"
            )
        args.model = parts
        logger.info("Requested models (multi): %s", parts)

    run_root = args.output_dir / args.mode
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=log_dir, level="INFO", experiment_name="runner")
    logger.info("Mode: %s", args.mode)
    logger.info("Models: %s", args.model if args.model is not None else "ALL (config)")
    logger.info("No dataset regeneration: %s", args.no_regen)

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    runners: Dict[str, Any] = {
        "ber_vs_snr": run_ber_vs_snr,
        "classical_receivers": run_classical_receivers,
        "jamming": run_jamming,
        "jamming_interpretability": run_jamming_interpretability,
        "final_report": run_final_report,
    }

    all_results: Dict[str, Any] = {}
    for exp_name in exp_names:
        logger.info("=" * 80)
        logger.info("EXPERIMENT %s", exp_name)
        logger.info("=" * 80)
        try:
            cli_overrides: Dict[str, Any] = {}
            if isinstance(args.model, str):
                cli_overrides = {"model": {"backbone_type": args.model}}
            config = load_experiment_config(
                experiment_name=exp_name,
                mode=args.mode,
                experiments_yaml_path=args.config,
                base_config_path=DEFAULT_BASE_CONFIG_PATH,
                cli_overrides=cli_overrides,
            )

            log_config_summary(config, logger)

            exp_output_dir = run_root / exp_name
            exp_log_dir = exp_output_dir / "logs"
            exp_log_dir.mkdir(parents=True, exist_ok=True)
            save_config_snapshot(config, exp_log_dir)
            config.setdefault("general", {})["run_output_dir"] = str(exp_output_dir)

            runner_fn = runners[exp_name]
            result = runner_fn(
                config=config,
                mode=args.mode,
                model_type=args.model,
                no_regen=args.no_regen,
                output_dir=exp_output_dir,
            )
            all_results[exp_name] = result
            logger.info("Experiment %s completed successfully.", exp_name)
        except Exception as exc:
            logger.error("Experiment %s failed: %s", exp_name, exc, exc_info=True)
            continue

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for exp_name, result in all_results.items():
        logger.info("%s: %s", exp_name, "OK" if result else "FAILED")
    logger.info("All requested experiments finished. Output in: %s", args.output_dir)

if __name__ == "__main__":
    main()
