#!/usr/bin/env python3
"""Delay-estimation figure (single-echo scenario, publication style).

Two panels are produced: delay correlation and delay RMSE in samples, one
curve per architecture.  Inputs are read from ``results/full/ber_vs_snr`` and
fall back to ``results_old/full/ber_vs_snr``.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCENARIO = "k1_doppler_full"
ARCHS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
STYLES = {
    "conv1d": dict(color="#1f77b4", marker="o"),
    "qkv": dict(color="#d62728", marker="s"),
    "lstm": dict(color="#2ca02c", marker="^"),
    "mc_dlsk": dict(color="#9467bd", marker="D"),
}
LINE = dict(lw=1.0, ms=3.8, mec="white", mew=0.6)


def _results_root() -> Path:
    """Locate the results tree, tolerating the ``results_old`` archive."""
    for candidate in (REPO / "results" / "full", REPO / "results_old" / "full"):
        if candidate.is_dir():
            return candidate
    return REPO / "results" / "full"


RESULTS = _results_root() / "ber_vs_snr" / SCENARIO


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


def _load() -> dict:
    curves = {}
    for arch in ARCHS:
        path = RESULTS / arch / "metrics.csv"
        if path.exists():
            curves[arch] = pd.read_csv(path).sort_values("snr_db")
    return curves


def main() -> None:
    _apply_style()
    curves = _load()
    if not curves:
        print("[warn] no metrics found under", RESULTS)
        return

    fig, (ax_corr, ax_rmse) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    corr_all, rmse_all = [], []
    for arch, df in curves.items():
        ax_corr.plot(df["snr_db"], df["corr_tau"], label=LABELS[arch],
                     **STYLES[arch], **LINE)
        corr_all.extend(df["corr_tau"].to_numpy(dtype=float))
        if "mse_tau" in df.columns:
            rmse = np.sqrt(np.clip(df["mse_tau"].to_numpy(dtype=float), 0.0, None))
            ax_rmse.plot(df["snr_db"], rmse, label=LABELS[arch],
                         **STYLES[arch], **LINE)
            rmse_all.extend(rmse.tolist())

    ax_corr.set_title("Delay correlation", pad=6)
    ax_corr.set_xlabel("SNR (dB)")
    ax_corr.set_ylabel(r"corr($\hat{\tau}$, $\tau$)")
    ax_corr.set_ylim((min(corr_all) - 0.05) if corr_all else 0.0, 1.0)

    ax_rmse.set_title("Delay RMSE", pad=6)
    ax_rmse.set_xlabel("SNR (dB)")
    ax_rmse.set_ylabel(r"RMSE of $\hat{\tau}$ (samples)")
    if rmse_all:
        ax_rmse.set_ylim(0.0, max(rmse_all) * 1.10)

    for ax in (ax_corr, ax_rmse):
        ax.grid(True, which="major", axis="both")
        ax.grid(True, which="minor", axis="both", alpha=0.4)
        ax.margins(x=0.03)
        _despine(ax)

    ax_corr.legend(loc="lower right", handlelength=2.6, borderpad=0.5,
                   labelspacing=0.35)
    ax_rmse.legend(loc="upper right", handlelength=2.6, borderpad=0.5,
                   labelspacing=0.35)

    fig.tight_layout()
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sensing_delay_single.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
