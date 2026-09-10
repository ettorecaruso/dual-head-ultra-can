#!/usr/bin/env python3
"""Blind receiver evaluation and full-range BER figure."""
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
    "conv1d": "Ultra-CAN",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
    "blind_stat": "Blind statistical",
}
SRC_FULL = REPO / "results" / "full" / "ber_vs_snr"


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


def plot_full_range(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
    styles = {
        "conv1d": dict(color="#1f77b4", marker="o"),
        "qkv": dict(color="#ff7f0e", marker="s"),
        "lstm": dict(color="#2ca02c", marker="^"),
        "mc_dlsk": dict(color="#d62728", marker="D"),
    }
    for ax_idx, scenario in enumerate(SCENARIOS):
        ax = axes[ax_idx]
        blind = pd.read_csv(out_dir / scenario / "blind_stat" / "metrics.csv")
        ax.plot(
            blind["snr_db"],
            blind["ber"],
            color="0.35",
            lw=2.2,
            ls="--",
            label="Blind statistical",
        )
        for model in MODELS:
            path = out_dir / scenario / model / "metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            ax.plot(
                df["snr_db"],
                df["ber"],
                lw=1.4,
                label=LEGEND[model],
                **styles[model],
            )
        ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(-0.02, 0.62)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("BER")
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig_path = REPO / "figures" / "ber_full_linear.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "ber_full_linear.pdf")
    plt.close(fig)
    print("saved", fig_path)


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

