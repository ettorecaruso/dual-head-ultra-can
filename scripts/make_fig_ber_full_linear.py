#!/usr/bin/env python3
"""Blind receiver evaluation and full-range BER figure (logarithmic scale)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset_generator import (
    _x0_from_seed,
    _iterate_map_batch,
    _center_normalize_batch,
    apply_channel_batch,
)
from src.models.blind_stat import blind_features, fit_lda, blind_decide

SEQLEN = 100
MU = 3.9
SNR_GRID = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
SCENARIOS = {
    "k1_doppler_full": (1, 8e-5),
    "k3_doppler_full": (3, 8e-5),
    "k3_doppler_limited": (3, 4e-5),
}
MODELS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LEGEND = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
    "blind_stat": "Blind statistical",
}
# Visual language shared with the other figure scripts: colourblind-safe
# palette, filled markers with a white edge, grey dashed blind reference.
STYLES = {
    "conv1d": dict(color="#1f77b4", marker="o"),
    "qkv": dict(color="#d62728", marker="s"),
    "lstm": dict(color="#2ca02c", marker="^"),
    "mc_dlsk": dict(color="#9467bd", marker="D"),
}
BLIND_STYLE = dict(color="#4d4d4d", ls=(0, (5, 2.2)), lw=1.3)
LINE = dict(lw=1.0, ms=3.8, mec="white", mew=0.6)
SCENARIO_TITLES = {
    "k1_doppler_full": "Single echo (K = 1)",
    "k3_doppler_full": "Multi-echo (K = 3)",
    "k3_doppler_limited": "Multi-echo, low Doppler (K = 3)",
}
SCENARIO_SLUGS = {
    "k1_doppler_full": "k1_single_echo",
    "k3_doppler_full": "k3_multi_echo",
    "k3_doppler_limited": "k3_low_doppler",
}


def _results_root() -> Path:
    """Locate the results tree, tolerating the ``results_old`` archive."""
    for candidate in (REPO / "results" / "full", REPO / "results_old" / "full"):
        if candidate.is_dir():
            return candidate
    return REPO / "results" / "full"


SRC_FULL = _results_root() / "ber_vs_snr"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#dcdcdc",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#c8c8c8",
            "lines.solid_capstyle": "round",
        }
    )


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_curves(out_dir, scenario: str) -> dict:
    curves = {}
    blind = Path(out_dir) / scenario / "blind_stat" / "metrics.csv"
    if blind.exists():
        curves["blind_stat"] = pd.read_csv(blind).sort_values("snr_db")
    for model in MODELS:
        path = Path(out_dir) / scenario / model / "metrics.csv"
        if path.exists():
            curves[model] = pd.read_csv(path).sort_values("snr_db")
    return curves


def _log_floor(curves: dict) -> float:
    mins = []
    for df in curves.values():
        ber = df["ber"].to_numpy(dtype=float)
        ber = ber[np.isfinite(ber) & (ber > 0)]
        if ber.size:
            mins.append(float(ber.min()))
    return (min(mins) / 2.5) if mins else 1e-5


def make_cfg(max_doppler: float) -> dict:
    return {
        "data": {
            "sequence_length": SEQLEN,
            "max_delay": 33,
            "max_doppler": max_doppler,
            "doppler_direct_max": 1e-5,
            "rician_kappa_db": 10.0,
            "alpha_min": 0.05,
            "alpha_max": 0.3,
            "map_type": "logistic",
            "map_param": MU,
            "alpha_tau_coupling": False,
            "alpha_floor": 0.05,
        }
    }


def gen(rng, n, k, snr_db, cfg):
    bits = rng.integers(0, 2, size=n).astype(np.int64)
    seeds = rng.integers(0, 2 ** 31 - 1, size=n)
    x0l = np.array([_x0_from_seed(int(s), "logistic", MU) for s in seeds])
    x0b = np.array([_x0_from_seed(int(s), "bernoulli", MU) for s in seeds])
    seq_l = _center_normalize_batch(_iterate_map_batch("logistic", MU, x0l, SEQLEN))
    seq_b = _center_normalize_batch(_iterate_map_batch("bernoulli", MU, x0b, SEQLEN))
    tx = np.where((bits == 0)[:, None], seq_l, seq_b)
    y, _, _ = apply_channel_batch(tx, k, snr_db, cfg, rng)
    return bits, y


def blind_ber_at_snr(cfg, k, snr_db, n, seed):
    rng = np.random.default_rng(seed)
    bits, y = gen(rng, n, k, snr_db, cfg)
    half = n // 2
    w, thr = fit_lda(blind_features(y[:half]), bits[:half])
    pred = blind_decide(y[half:], w, thr)
    errs = int(np.sum(pred != bits[half:]))
    return errs, half


def evaluate_blind(scenario, max_doppler, n, out_dir):
    k, _ = SCENARIOS[scenario]
    cfg = make_cfg(max_doppler)
    rows = []
    for i, snr in enumerate(SNR_GRID):
        errs, tot = blind_ber_at_snr(cfg, k, snr, n, seed=44000 + i * 17)
        rows.append(
            {
                "snr_db": float(snr),
                "ber": errs / tot,
                "n_errors": errs,
                "n_symbols": tot,
            }
        )
    df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)
    dest = out_dir / scenario / "blind_stat"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "metrics.csv", index=False)
    return df


def copy_stored_metrics(scenario, out_dir):
    for model in MODELS:
        src = SRC_FULL / scenario / model / "metrics.csv"
        if not src.exists():
            print(f"  [warn] missing metrics: {src}")
            continue
        dest = out_dir / scenario / model
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / "metrics.csv")


def _draw_panel(ax, curves: dict, title: str, floor: float) -> None:
    """Draw one BER panel on a logarithmic axis, with its own legend."""
    ax.set_yscale("log")
    ax.axhline(0.5, color="#c4c4c4", lw=0.9, ls=":", zorder=0)
    for model in MODELS:
        if model not in curves:
            continue
        df = curves[model]
        ax.plot(df["snr_db"], df["ber"], label=LEGEND[model],
                **STYLES[model], **LINE, zorder=3)
    if "blind_stat" in curves:
        df = curves["blind_stat"]
        ax.plot(df["snr_db"], df["ber"], label=LEGEND["blind_stat"],
                marker="x", ms=4.0, zorder=2, **BLIND_STYLE)
    ax.set_title(title, pad=6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylim(floor, 0.72)
    ax.margins(x=0.03)
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", alpha=0.45)
    ax.legend(loc="lower left", handlelength=2.6, borderpad=0.5,
              labelspacing=0.35)
    _despine(ax)


def _save(fig, out_dir, stem: str) -> None:
    fig_dir = REPO / "figures"
    for target in (fig_dir / f"{stem}.pdf", Path(out_dir) / "plots" / f"{stem}.pdf"):
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight", pad_inches=0.03)
    print("saved", fig_dir / f"{stem}.pdf")


def plot_full_range(out_dir) -> None:
    """Log-scale full-range BER: combined 3-panel figure + one figure per K."""
    _apply_style()
    out_dir = Path(out_dir)
    curves = {s: _load_curves(out_dir, s) for s in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no BER curves found in", out_dir)
        return

    available = [c for c in curves.values() if c]
    floor = min(_log_floor(c) for c in available)

    # Combined three-panel figure, legend repeated on every panel.
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        _draw_panel(ax, curves.get(scenario, {}),
                    SCENARIO_TITLES[scenario], floor)
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    _save(fig, out_dir, "ber_full_linear")
    plt.close(fig)

    # Standalone figure per scenario (one per K), easier to place in the paper.
    for scenario in SCENARIOS:
        if not curves.get(scenario):
            continue
        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        _draw_panel(ax, curves[scenario], SCENARIO_TITLES[scenario],
                    _log_floor(curves[scenario]))
        ax.set_ylabel("BER")
        fig.tight_layout()
        _save(fig, out_dir, f"ber_full_{SCENARIO_SLUGS[scenario]}")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "newRes")
    ap.add_argument("--symbols", type=int, default=20000)
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--only-figure", action="store_true",
                    help="only regenerate the figure from the results already present in output-dir")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_figure:
        plot_full_range(out_dir)
        print("output dir:", out_dir)
        return

    for scenario, (k, max_doppler) in SCENARIOS.items():
        print(f"[scenario] {scenario} (K={k}, fD_max={max_doppler:.0e})")
        copy_stored_metrics(scenario, out_dir)
        df = evaluate_blind(scenario, max_doppler, args.symbols, out_dir)
        print(df.to_string(index=False))

    if not args.skip_figure:
        plot_full_range(out_dir)
    print("output dir:", out_dir)


if __name__ == "__main__":
    main()

