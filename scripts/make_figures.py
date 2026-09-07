import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CURVES = ROOT / "results" / "curves"
FIGURES = ROOT / "results" / "figures"

MODEL_COLORS = {
    "conv1d": "#4c72b0",
    "qkv": "#dd8452",
    "lstm": "#55a868",
    "mc_dlsk": "#c44e52",
}
MODEL_LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
SCENARIO_LABELS = {
    "k1_doppler_full": "Single-echo (K=1, full Doppler)",
    "k3_doppler_full": "Dense swarm (K=3, full Doppler)",
    "k3_doppler_limited": "Dense swarm (K=3, reduced Doppler)",
}


def figure_operating_region(outdir=None):
    outdir = Path(outdir or FIGURES)
    df = pd.read_csv(CURVES / "ber_curves.csv")
    df = df[df.snr_db >= 5]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.1), sharey=True)
    for ax, scenario in enumerate(["k1_doppler_full", "k3_doppler_full", "k3_doppler_limited"]):
        ax = axs[ax]
        for model in MODEL_COLORS:
            sub = df[(df.scenario == scenario) & (df.model == model)]
            ax.plot(sub.snr_db, sub.ber, color=MODEL_COLORS[model], lw=1.4,
                    label=MODEL_LABELS[model], marker="o", ms=3)
        ax.axhline(1e-4, color="black", ls="--", lw=0.9)
        ax.set_yscale("log")
        ax.set_xlim(5, 20)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=7)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("BER", fontsize=8)
    axs[0].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "operating_region_ber.pdf")
    plt.close(fig)


def figure_sensing_single(outdir=None):
    outdir = Path(outdir or FIGURES)
    df = pd.read_csv(CURVES / "sensing_curves.csv")
    sub = df[df.scenario == "k1_doppler_full"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for model in MODEL_COLORS:
        rows = sub[sub.model == model]
        ax.plot(rows.snr_db, rows.delay_corr, color=MODEL_COLORS[model],
                lw=1.4, label=MODEL_LABELS[model], marker="o", ms=3)
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("corr", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "sensing_delay_single.pdf")
    plt.close(fig)


def figure_activation_retention(outdir=None):
    outdir = Path(outdir or FIGURES)
    df = pd.read_csv(CURVES / "retention.csv")
    panels = {
        "conv1d": ["conv1", "conv2", "attention_pooling"],
        "qkv": ["conv1", "conv2", "qkv_attention"],
        "lstm": ["lstm_state", "projection"],
        "mc_dlsk": ["bilstm", "projection"],
    }
    names = {
        "conv1": "conv1", "conv2": "conv2", "attention_pooling": "attn pooling",
        "qkv_attention": "QKV attention", "lstm_state": "LSTM state",
        "bilstm": "BiLSTM", "projection": "projection",
    }
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.15), sharey=True)
    for i, arch in enumerate(panels):
        ax = axs[i]
        arch_df = df[df.arch == arch]
        stages = panels[arch]
        bars = [arch_df[arch_df.stage == s].cosine.iloc[0] for s in stages]
        ax.bar(range(len(stages)), bars, color="#4c72b0", width=0.6)
        for j, v in enumerate(bars):
            ax.text(j, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5)
        sensor = arch_df[arch_df.stage == "sensing_position"].cosine.iloc[0]
        ax.axhline(sensor, color="#55a868", ls="--", lw=1.2)
        ax.text(len(stages) - 1, sensor + 0.03, f"{sensor:.2f}", color="#55a868",
                ha="right", fontsize=6.5)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([names[s] for s in stages], fontsize=6.5)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(MODEL_LABELS[arch], fontsize=7.5)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("cosine", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "activation_retention.pdf")
    plt.close(fig)


def figure_jamaware(outdir=None):
    outdir = Path(outdir or FIGURES)
    df = pd.read_csv(CURVES / "jamaware_ber.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.1), sharey=True)
    for ax, jammer in zip(fig.axes, ["cw", "barrage", "partial_band"]):
        sub = df[df.jammer == jammer]
        ax.plot(sub.jsr_db, sub.clean_trained, color="#c44e52", lw=1.4,
                label="clean-trained", marker="o", ms=3)
        ax.plot(sub.jsr_db, sub.jam_aware, color="#4c72b0", lw=1.4,
                label="jamming-aware", marker="s", ms=3)
        ax.set_yscale("log")
        ax.set_xlabel("JSR (dB)", fontsize=8)
        ax.set_title(jammer, fontsize=8)
        ax.grid(alpha=0.3)
    axs0 = fig.axes[0]
    axs0.set_ylabel("BER", fontsize=8)
    axs0.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "jamming_aware_training.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(FIGURES))
    parser.add_argument("--figures", nargs="+", default=["all"])
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if "all" in args.figures or "operating_region" in args.figures:
        figure_operating_region(outdir)
    if "all" in args.figures or "sensing" in args.figures:
        figure_sensing_single(outdir)
    if "all" in args.figures or "retention" in args.figures:
        figure_activation_retention(outdir)
    if "all" in args.figures or "jamaware" in args.figures:
        figure_jamaware(outdir)
    print("figures written to", outdir)


if __name__ == "__main__":
    main()
