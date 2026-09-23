#!/usr/bin/env python3
"""Frequency-agility figure: BER vs JSR (fixed carrier vs hopping) and BER vs hop rate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "full" / "frequency_agility"
OUT = REPO / "figures" / "frequency_agility.pdf"
MIRROR = REPO / "results" / "figures" / "frequency_agility.pdf"

ARCH = "qkv"
IN_CHANNEL = "cw"
JAMMERS = ["barrage", "fixed_partial", "sweep", "follower"]
TITLES = {
    "barrage": "Barrage (control)",
    "fixed_partial": "Fixed partial coverage",
    "sweep": "Sweeping jammer",
    "follower": "Reactive follower",
}
COLORS = {"barrage": "#8c8c8c", "fixed_partial": "#c44e52",
          "sweep": "#4c72b0", "follower": "#55a868"}


def _load(name: str) -> pd.DataFrame:
    path = RESULTS / ARCH / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def main() -> None:
    jsr_df = _load("frequency_agility_vs_jsr.csv")
    jsr_df = jsr_df[(jsr_df.in_channel == IN_CHANNEL) & (jsr_df.arch == ARCH)]
    dwell_df = _load("frequency_agility_vs_dwell.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax = axes[0]
    clean_value = float(jsr_df["ber_clean"].iloc[0])
    ax.axhline(clean_value, color="black", ls=":", lw=1.2,
               label=f"No jammer (BER={clean_value:.1e})")
    for jammer in JAMMERS:
        sub = jsr_df[jsr_df.jammer_model == jammer]
        if sub.empty:
            continue
        for modality, style in (("fh_off", "--"), ("fh_on", "-")):
            row = sub[sub.modality == modality].sort_values("jsr_db")
            if row.empty:
                continue
            label = f"{TITLES[jammer]}, {'fixed carrier' if modality == 'fh_off' else 'hopping'}"
            ax.semilogy(row["jsr_db"], row["ber"], linestyle=style,
                        color=COLORS[jammer], marker="o", ms=2.6, lw=1.3, label=label)
            lower = np.clip(row["ber"] - row["ber_std"], 1e-9, 1.0)
            upper = np.clip(row["ber"] + row["ber_std"], 1e-9, 1.0)
            ax.fill_between(row["jsr_db"], lower, upper, color=COLORS[jammer],
                            alpha=0.12, linewidth=0)
    ax.set_xlabel("JSR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"{ARCH.upper()}, {IN_CHANNEL} jammer: fixed carrier vs frequency hopping")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=6.4, ncol=1, loc="lower right")

    ax = axes[1]
    for jammer in ["sweep", "follower"]:
        sub = dwell_df[dwell_df.jammer_model == jammer].sort_values("hop_rate_hz")
        if sub.empty:
            continue
        ax.semilogy(sub["hop_rate_hz"], sub["ber"], marker="s", ms=3, lw=1.4,
                    color=COLORS[jammer], label=TITLES[jammer])
        lower = np.clip(sub["ber"] - sub["ber_std"], 1e-9, 1.0)
        upper = np.clip(sub["ber"] + sub["ber_std"], 1e-9, 1.0)
        ax.fill_between(sub["hop_rate_hz"], lower, upper, color=COLORS[jammer],
                        alpha=0.14, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Hop rate (hops/s)")
    ax.set_ylabel("BER")
    ax.set_title(f"{ARCH.upper()}, JSR = {float(dwell_df['jsr_db'].iloc[0]):.0f} dB")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=7)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)
    if MIRROR.parent.is_dir():
        fig.savefig(MIRROR, bbox_inches="tight")
        print("saved", MIRROR)
    plt.close(fig)

    pivot = jsr_df.pivot_table(
        index=["jammer_model", "in_channel", "jsr_db"], columns="modality", values="ber"
    ).reset_index()
    if {"fh_off", "fh_on"}.issubset(pivot.columns):
        pivot["gain_db"] = 10.0 * np.log10(
            pivot["fh_off"].clip(lower=1e-9) / pivot["fh_on"].clip(lower=1e-9)
        )
        gain_path = RESULTS / ARCH / "frequency_agility_gain.csv"
        pivot.to_csv(gain_path, index=False)
        print("saved", gain_path)


if __name__ == "__main__":
    main()
