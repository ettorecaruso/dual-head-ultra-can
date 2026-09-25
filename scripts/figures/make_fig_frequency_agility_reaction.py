#!/usr/bin/env python3
"""BER against the reaction latency of a tracking jammer.

    python scripts/figures/make_fig_frequency_agility_reaction.py [run_dir]

One panel per architecture listed in the CSV, one curve per jammer archetype,
with the confidence interval of every point as a light band. The message is the
one the hop-rate sweep cannot give on its own: the protection of frequency agility
comes from hopping faster than the jammer can re-tune, so the BER is flat until
the latency exceeds the slot duration and then climbs.

``run_dir`` defaults to ``results/full/frequency_agility``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

DEFAULT_RUN_DIR = REPO / "results" / "full" / "frequency_agility"
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
ARCH_TITLES = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
MODEL_LABELS = {
    "follower": "Reactive follower",
    "sweep": "Sweeping jammer",
    "fixed_partial": "Fixed partial (25%)",
    "barrage": "Barrage (control)",
}
Y_LO, Y_HI = 1e-4, 1.0
_Z95 = 1.959963984540054


def _load(run_dir: Path, arch: str) -> Optional[pd.DataFrame]:
    path = Path(run_dir) / arch / "frequency_agility_vs_reaction.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame[np.isfinite(frame["ber"])].sort_values("reaction_bursts")


def _band(frame: pd.DataFrame) -> Optional[tuple]:
    y = frame["ber"].to_numpy(dtype=float)
    if "ber_ci_lo" in frame and "ber_ci_hi" in frame:
        return (
            np.clip(frame["ber_ci_lo"].to_numpy(dtype=float), Y_LO, Y_HI),
            np.clip(frame["ber_ci_hi"].to_numpy(dtype=float), Y_LO, Y_HI),
        )
    if "ber_std" in frame and "n_realizations" in frame:
        n = np.maximum(frame["n_realizations"].to_numpy(dtype=float), 1.0)
        half = _Z95 * frame["ber_std"].to_numpy(dtype=float) / np.sqrt(n)
        return (np.clip(y - half, Y_LO, Y_HI), np.clip(y + half, Y_LO, Y_HI))
    return None


def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    frames: Dict[str, pd.DataFrame] = {}
    for arch in ARCHS:
        frame = _load(run_dir, arch)
        if frame is not None and not frame.empty:
            frames[arch] = frame
    if not frames:
        raise SystemExit(
            f"no frequency_agility_vs_reaction.csv under {run_dir}: run the "
            "frequency agility experiment with frequency_hopping.reaction_sweep set"
        )

    archs = [arch for arch in ARCHS if arch in frames]
    fs.apply_style()
    fig, axes = plt.subplots(1, len(archs), figsize=(4.4 * len(archs), 4.2), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        fs.log_axis(ax, Y_LO, Y_HI, x_step=2.0)
        frame = frames[arch]
        for model in dict.fromkeys(frame["jammer_model"]):
            subset = frame[frame["jammer_model"] == model]
            kw = fs.series_kwargs(model)
            x = subset["latency_us"].to_numpy(dtype=float)
            y = subset["ber"].to_numpy(dtype=float)
            band = _band(subset)
            if band is not None:
                ax.fill_between(x, band[0], band[1], color=kw["color"], alpha=0.15,
                                linewidth=0, zorder=2)
            ax.plot(x, y, label=MODEL_LABELS.get(model, model), zorder=3, **kw)
        ax.set_title(ARCH_TITLES.get(arch, arch), pad=8)
        ax.set_xlabel("Jammer reaction latency ($\\mu$s)")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("BER")
    fig.suptitle("Frequency agility against a tracking jammer", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fs.legend_below(axes[len(axes) // 2], ncol=2)
    fs.outer_frame(fig, axes)
    fs.save(fig, "frequency_agility_reaction")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
