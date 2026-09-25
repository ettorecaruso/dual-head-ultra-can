#!/usr/bin/env python3
"""Why the sensing head stops working off the nominal channel.

    python scripts/figures/make_fig_sensing_observability.py [run_dir]

One panel, two measured curves, both in ``[0, 1]`` so they share an axis:

* ``P_oracle`` -- the probability that the sensing **label** (the strongest
  *geometric* echo, which is what the dataset writes) is also the strongest tap
  in the received waveform. This is an oracle: it needs no model and no
  training, it is measured on the tap set the generator applies by
  ``diagnostics/channel_label_observability``.
* ``corr_tau`` -- what the trained head extracts, the top over SNR of the delay
  correlation of ``channel_generalization/<arch>/summary.csv``.

They move together, which is the whole point: the collapse of the sensing metric
is not the receiver failing, it is the label leaving the observability of the
waveform. The light 5%-clutter variant is the dose-response control that proves
it (1.00 -> 0.62 -> 0.00).

The retraining control confirms the reading from the other side: it recovers the
BER floor to the in-domain value and leaves ``corr_tau`` at the noise floor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

DEFAULT_RUN_DIR = REPO / "results" / "full" / "channel_generalization"
ORACLE = (REPO / "results" / "full" / "diagnostics"
          / "channel_label_observability" / "oracle.csv")
ARCH = "qkv"
TICKS = {"nominal": "nominal", "b_tdl_d_light": "TDL-D\n5% clutter",
         "b_tdl_d": "TDL-D\n25% clutter", "c_two_ray_jakes": "two-ray\n+Jakes",
         "b_tdl_a": "TDL-A\nNLOS"}


def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    if not ORACLE.is_file():
        raise SystemExit(
            f"missing {ORACLE}: run scripts/diagnostics/channel_label_observability.py"
        )
    oracle = pd.read_csv(ORACLE).set_index("channel_variant")
    summary = pd.read_csv(Path(run_dir) / ARCH / "summary.csv").set_index(
        "channel_variant")

    variants: List[str] = [v for v in oracle.index if v in summary.index]
    variants.sort(key=lambda v: -float(oracle.loc[v, "p_label_is_strongest"]))
    x = np.arange(len(variants), dtype=float)
    p_oracle = [float(oracle.loc[v, "p_label_is_strongest"]) for v in variants]
    corr = [float(summary.loc[v, "corr_tau_top"]) for v in variants]

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x, p_oracle, width=0.55, color="#636EFA", alpha=0.35,
           edgecolor="#636EFA", linewidth=1.2, zorder=2,
           label=r"$P_{oracle}$: the label is the strongest tap")
    kw = fs.series_kwargs(ARCH)
    ax.plot(x, corr, zorder=4, label=f"trained head, {ARCH} (corr)", **kw)
    for xi, cv in zip(x, corr):
        ax.annotate(f"{cv:.3f}", (xi, cv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#EF553B")
    # The bars at P = 0 carry no label on purpose: a number there would collide
    # with the x tick labels below it (and with the corr value just above it), so
    # the absence of the bar is left to speak and is stated once, to the right.
    for xi, pv in zip(x, p_oracle):
        if pv < 0.05:
            continue
        ax.annotate(f"{pv:.2f}", (xi, pv), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color="#3A56C4")
    if min(p_oracle) < 0.05:
        ax.annotate(r"$P_{oracle}=0.00$ where the bar is absent",
                    (0.52, 0.55), xycoords="axes fraction", ha="left",
                    va="center", fontsize=9.5, color="#3A56C4")

    ax.set_xticks(x)
    ax.set_xticklabels([TICKS.get(v, v) for v in variants], fontsize=9.5)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="major", axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_ylabel("probability / correlation")
    ax.set_title("Sensing collapses because the label leaves the observable "
                 "waveform", pad=8)
    fig.tight_layout()
    fs.legend_below(ax, ncol=2, y=-0.20)
    fs.save(fig, "sensing_observability")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
