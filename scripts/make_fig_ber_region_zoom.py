#!/usr/bin/env python3
"""Operating-region BER figure (SNR >= 5 dB, log scale, publication style).

Produces a combined three-panel figure, one panel per K scenario with the
legend repeated in every panel, plus one standalone figure per scenario.
Inputs are read from ``results/full/ber_vs_snr`` and fall back to
``results_old/full/ber_vs_snr``.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
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
SCENARIOS = [
    ("k1_doppler_full", "k1_single_echo", "Single echo (K = 1)"),
    ("k3_doppler_full", "k3_multi_echo", "Multi-echo (K = 3)"),
    ("k3_doppler_limited", "k3_low_doppler", "Multi-echo, low Doppler (K = 3)"),
]
MIN_DB = 5.0
TARGET = 1e-4
Y_BOTTOM = 3.5e-5
Y_TOP = 1.3e-3


def _results_root() -> Path:
    """Locate the results tree, tolerating the ``results_old`` archive."""
    for candidate in (REPO / "results" / "full", REPO / "results_old" / "full"):
        if candidate.is_dir():
            return candidate
    return REPO / "results" / "full"


RESULTS = _results_root() / "ber_vs_snr"


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


def _curve(scenario: str) -> dict:
    curves = {}
    for arch in ARCHS:
        path = RESULTS / scenario / arch / "metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["snr_db"] >= MIN_DB].sort_values("snr_db")
        if len(df):
            curves[arch] = df
    return curves


def _draw_panel(ax, curves: dict, title: str) -> None:
    ax.set_yscale("log")
    ax.axhspan(Y_BOTTOM, TARGET, color="#e8f4ec", zorder=0)
    ax.axhline(TARGET, color="#4c9a63", lw=1.0, ls="--", zorder=1,
               label=r"target BER $= 10^{-4}$")
    for arch in ARCHS:
        if arch not in curves:
            continue
        df = curves[arch]
        ax.plot(df["snr_db"], df["ber"], label=LABELS[arch],
                **STYLES[arch], **LINE, zorder=3)
    ax.set_title(title, pad=6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.margins(x=0.03)
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="both", alpha=0.4)
    ax.legend(loc="upper right", handlelength=2.6, borderpad=0.5,
              labelspacing=0.35)
    _despine(ax)


def _save(fig, stem: str) -> None:
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print("saved", out)


def main() -> None:
    _apply_style()
    curves = {scenario: _curve(scenario) for scenario, _, _ in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no metrics found under", RESULTS)
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, (scenario, _, title) in zip(axes, SCENARIOS):
        _draw_panel(ax, curves.get(scenario, {}), title)
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    _save(fig, "ber_region_zoom")
    plt.close(fig)

    for scenario, slug, title in SCENARIOS:
        if not curves.get(scenario):
            continue
        fig, ax = plt.subplots(figsize=(5.4, 4.1))
        _draw_panel(ax, curves[scenario], title)
        ax.set_ylabel("BER")
        fig.tight_layout()
        _save(fig, f"ber_region_zoom_{slug}")
        plt.close(fig)


if __name__ == "__main__":
    main()
