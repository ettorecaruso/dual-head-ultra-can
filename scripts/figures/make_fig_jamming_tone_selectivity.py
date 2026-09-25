#!/usr/bin/env python3
"""CW jamming is tone-selective: one tone in fifteen is nearly harmless.

    python scripts/figures/make_fig_jamming_tone_selectivity.py

This is the figure behind the anomaly of the CW curve of the jamming-aware
control panel. The legacy evaluation drew **one random CW tone per JSR point**,
so the shape of the old curve was partly the luck of the draw: at JSR -2 dB the
15-tone Monte Carlo has a mean BER of 0.276 with a range of 0.0005 to 0.4301. The
reason is visible here: the damage depends on where the tone falls relative to
the receiver's processing structure. One tone (``f_cw = 0.25``) stays below 1e-3
up to JSR +2 dB, while the fourteen others saturate around 0.5.

That is why the paper re-ran the CW curve over the tones and reports a mean
instead of a single realization, and why the tone selectivity is a limitation
worth stating: a jammer that knows the receiver can pick the damaging tones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

SRC = (REPO / "results/full/diagnostics/jamming_artifact"
       / "ber_vs_jsr_cw_realizations.csv")
JSR_PLOT = (-6.0, -2.0, 2.0, 6.0, 10.0)
SPECIAL = 0.25
Y_LO, Y_HI = 1e-4, 1.0


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}: run "
                         "scripts/diagnostics/diagnose_jamming_mc.py --snapshots 15")
    df = pd.read_csv(SRC)
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fs.log_axis(ax, Y_LO, Y_HI)

    colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for jsr, colour in zip(JSR_PLOT, colours):
        sub = df[np.isclose(df["jsr_db"], jsr)].sort_values("f_cw")
        ax.plot(sub["f_cw"], np.clip(sub["ber"], Y_LO, Y_HI), marker="o",
                ms=4.5, color=colour, lw=1.3, label=f"JSR = {jsr:+.0f} dB",
                zorder=3)

    ax.axvline(SPECIAL, color="0.35", ls=(0, (4, 2)), lw=1.3, zorder=1)
    ax.annotate(r"$f_{cw}=0.25$" + ":\nnearly immune", (SPECIAL, Y_LO * 1.4),
                textcoords="offset points", xytext=(6, 0), fontsize=9.5,
                color="#333333", va="bottom")
    ax.set_xlabel(r"CW tone frequency $f_{cw}$ (normalised)")
    ax.set_ylabel("BER")
    ax.set_title("One tone in fifteen is nearly harmless: the CW curve needs a "
                 "Monte Carlo", pad=8)
    ax.set_xlim(-0.02, 0.52)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fs.legend_below(ax, ncol=5, y=-0.15)
    fs.save(fig, "jamming_tone_selectivity")
    plt.close(fig)


if __name__ == "__main__":
    main()
