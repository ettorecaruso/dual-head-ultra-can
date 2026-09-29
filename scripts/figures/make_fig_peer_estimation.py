#!/usr/bin/env python3
"""Peer-aware sensing: where the reported range lands.

    python scripts/make_fig_peer_estimation.py

Two panels, both stacked composition bars -- no curve families, no line-style
vocabulary to decode:

* left -- the reported range against the distance to the nearest cooperative
  peer (``offset = tau_obstacle - tau_peer``, clipped at +-4 samples).  The
  estimate lands on the obstacle, on the peer (the mirror failure the fail-safe
  exists for), or somewhere else;
* right -- the same composition against the amplitude ratio between the obstacle
  echo and the expected peer echo, from a stronger peer to an obstacle twice as
  strong.

Reading.  Exact identification (within the half-sample tolerance the metric uses)
is 8% overall and never exceeds 12.5%, and it goes to zero as soon as the peer
arrives *earlier* than the obstacle (negative offset) or is the stronger echo: in
those conditions the receiver locks onto the peer.  The two receivers are drawn
together because they agree on these rates to the fourth decimal -- the decision
is a deterministic test on the estimated delay, so the rates are a property of the
geometry, not of the architecture.  Accuracy and the price of the fail-safe are in
``peer_estimation_price.pdf``.

Written dataset: ``results/full/peer_estimation/<arch>/peer_estimation.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

SRC = REPO / "results" / "full" / "peer_estimation"
ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
RATIO_ORDER = ("peer_stronger", "comparable", "target_gt1.25", "target_gt2")
RATIO_LABELS = {
    "peer_stronger": "peer\nstronger",
    "comparable": "comparable",
    "target_gt1.25": "target\nx1.25",
    "target_gt2": "target\nx2",
}
REFERENCE_GUARD = 1.5
#: Colours of the three outcomes of the range report.
OUTCOME = {
    "obstacle": ("#00CC96", "reports the obstacle"),
    "peer": ("#EF553B", "reports the peer (mirror failure)"),
    "elsewhere": ("#B0B0B0", "neither"),
}


def load() -> pd.DataFrame:
    frames = []
    for arch in ARCHS:
        path = SRC / arch / "peer_estimation.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run the peer_estimation experiment")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["key_bin"] = frame["key_bin"].astype(str)
    return frame


def _rates(frame: pd.DataFrame, summary: str, keys: List[str],
           guard: float = REFERENCE_GUARD) -> Dict[str, np.ndarray]:
    """Obstacle / peer / neither rates per key, averaged over the two receivers."""
    picked = frame[(frame["summary"] == summary)
                   & frame["key_bin"].isin(keys)
                   & np.isclose(frame["guard_factor"], float(guard))]
    obstacle, peer, counts = [], [], []
    for key in keys:
        sub = picked[picked["key_bin"] == key]
        obstacle.append(float(sub["target_id_rate"].mean()))
        peer.append(float(sub["peer_false_rate"].mean()))
        counts.append(int(sub["n"].mean()))
    obstacle = np.array(obstacle)
    peer = np.array(peer)
    return {
        "obstacle": obstacle,
        "peer": peer,
        "elsewhere": np.clip(1.0 - obstacle - peer, 0.0, None),
        "n": np.array(counts),
    }


def _rates_panel(ax, keys: List[str], labels: List[str], rates: Dict[str, np.ndarray],
                 xlabel: str, title: str) -> None:
    """Grouped bars for the two informative outcomes of one range report.

    The third outcome ("neither", i.e. the estimate lands neither on the obstacle
    nor on a peer) is 80-95% of every bin: stacking it would shrink the two rates
    the figure is about to slivers, so it is quoted as a number in each bin
    instead of drawn.
    """
    x = np.arange(len(keys))
    width = 0.38
    peak = float(np.nanmax(np.concatenate([rates["obstacle"], rates["peer"]])))
    top = max(0.05, peak) * 1.45
    for name, offset in (("obstacle", -width / 2), ("peer", width / 2)):
        color, label = OUTCOME[name]
        values = rates[name]
        ax.bar(x + offset, values, width=width, color=color, label=label, zorder=3)
        for xi, value in enumerate(values):
            ax.annotate(f"{100 * value:.0f}", (xi + offset, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color=color)
    for xi, (count, elsewhere) in enumerate(zip(rates["n"], rates["elsewhere"])):
        ax.annotate(f"n={count}\n{100 * elsewhere:.0f}% elsewhere", (xi, top),
                    ha="center", va="top", fontsize=8.5, color="#777777")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of samples")
    ax.set_ylim(0.0, top)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    fs.apply_style()
    frame = load()

    offsets = sorted(int(value) for value in
                     frame.loc[frame["summary"] == "offset_samples", "key_bin"].unique())
    offset_labels = [("<=-4" if value == min(offsets) else
                      ">=+4" if value == max(offsets) else f"{value:+d}")
                     for value in offsets]
    offset_rates = _rates(frame, "offset_samples", [str(v) for v in offsets])
    ratio_rates = _rates(frame, "amplitude_ratio", list(RATIO_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    _rates_panel(axes[0], offsets, offset_labels, offset_rates,
                 "obstacle delay - nearest peer (samples)",
                 "A peer arriving earlier takes the report over")
    _rates_panel(axes[1], list(RATIO_ORDER), [RATIO_LABELS[k] for k in RATIO_ORDER],
                 ratio_rates, "obstacle / peer echo amplitude",
                 "A stronger obstacle is identified more often")

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "peer_estimation")
    plt.close(fig)

    overall = frame[(frame["summary"] == "all") & (frame["key_bin"] == "all")
                    & np.isclose(frame["guard_factor"], REFERENCE_GUARD)]
    print("overall identification per receiver:",
          {row["arch"]: round(float(row["target_id_rate"]), 4)
           for _, row in overall.iterrows()})
    print("worst receiver disagreement on the rates: %.2e"
          % float((frame[frame["summary"] == "all"].groupby("key_bin")
                   ["target_id_rate"].max()
                   - frame[frame["summary"] == "all"].groupby("key_bin")
                   ["target_id_rate"].min()).abs().max()))


if __name__ == "__main__":
    main()


