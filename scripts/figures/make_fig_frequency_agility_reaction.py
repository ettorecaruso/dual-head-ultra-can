#!/usr/bin/env python3
"""Reaction latency of a tracking jammer: the four receivers collapse on one law.

    python scripts/figures/make_fig_frequency_agility_reaction.py [run_dir]

**One panel, not four.** The sweep contains a single jammer archetype (the
reactive follower) on the CW in-channel at one dwell, so a panel per receiver
would repeat the same curve four times. The interesting statement is the opposite
one: the four receivers are indistinguishable here, because what happens is
geometry, not signal processing.

The protected fraction of the time is fixed by the reaction latency ``L`` and the
dwell (in bursts) as ``f = max(0, 1 - L/dwell)``, so the measured BER must follow

    BER(L) = f(L) * BER_jammed + (1 - f(L)) * BER_clean

with ``BER_jammed`` the value at ``L = 0`` (where ``f = 1``) and ``BER_clean``
the no-jammer floor. The analytic curve is drawn on top of the four measured
ones and the script **verifies** the agreement, so the collapse cannot silently
break: hopping is defeated exactly when the jammer re-tunes faster than the slot,
and the *height* of the penalty is still whatever the jammer achieves while
aligned.

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
ARCH_LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
Y_LO, Y_HI = 1e-5, 1.0
#: The mixing law must reproduce the sweep to this relative tolerance.
LAW_TOLERANCE = 0.05


def _load(run_dir: Path, arch: str) -> Optional[pd.DataFrame]:
    path = Path(run_dir) / arch / "frequency_agility_vs_reaction.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return frame[np.isfinite(frame["ber"])].sort_values("latency_us")


def _endpoints(frame: pd.DataFrame) -> tuple:
    """The two endpoints of the sweep: f = 1 (jammer aligned) and f = 0 (free).

    Both are taken from the same sweep, so the law is tested against the
    condition it describes instead of against a clean floor measured elsewhere:
    at f = 0 the measured BER *is* the no-jammer floor of this condition.
    """
    frac = frame["jammed_fraction"].to_numpy(dtype=float)
    if frac.max() < 1.0 - 1e-9 or frac.min() > 1e-9:
        raise AssertionError(
            "the reaction sweep must contain both f = 1 (latency 0) and f = 0 "
            f"(latency >= dwell), found f in [{frac.min():.3f}, {frac.max():.3f}]"
        )
    jammed = float(frame.loc[np.isclose(frac, 1.0), "ber"].iloc[0])
    clean = float(frame.loc[np.isclose(frac, 0.0), "ber"].iloc[0])
    return jammed, clean


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

    # The analytic law is cross-checked against every receiver before drawing.
    worst = 0.0
    for arch in archs:
        frame = frames[arch]
        jammed, clean = _endpoints(frame)
        predicted = frame["jammed_fraction"] * jammed + (1 - frame["jammed_fraction"]) * clean
        interior = (frame["jammed_fraction"] > 1e-9) & (frame["jammed_fraction"] < 1.0 - 1e-9)
        measured = frame.loc[interior, "ber"].to_numpy(dtype=float)
        rel = np.abs(measured - predicted[interior].to_numpy(dtype=float)) / np.maximum(
            predicted[interior].to_numpy(dtype=float), 1e-12
        )
        if rel.size:
            worst = max(worst, float(np.max(rel)))
    if worst > LAW_TOLERANCE:
        raise AssertionError(
            f"the mixing law does not describe the reaction sweep: worst relative "
            f"error {worst:.3f} > {LAW_TOLERANCE:.2f} on the interior points. The "
            "jump in jammed_fraction and the BER jump must come from the same "
            "geometry."
        )
    print(f"mixing-law check (all receivers, interior points): worst {worst:.4f}")

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    fs.log_axis(ax, Y_LO, Y_HI)

    for arch in archs:
        frame = frames[arch]
        kw = fs.series_kwargs(arch)
        ax.plot(frame["latency_us"], frame["ber"], label=ARCH_LABELS[arch],
                zorder=3, **kw)

    reference = frames[archs[0]]
    jsr = float(reference["jsr_db"].iloc[0])
    jammed, clean = _endpoints(reference)
    lat = reference["latency_us"].to_numpy(dtype=float)
    frac = reference["jammed_fraction"].to_numpy(dtype=float)
    ax.plot(lat, frac * jammed + (1 - frac) * clean,
            color="black", lw=2.2, ls=(0, (1, 1.4)), zorder=5,
            label=r"law $\,f\,$BER$_{jam}+(1-f)\,$BER$_{clean}$")

    for x, f in zip(lat, frac):
        ax.annotate(rf"$f={f:.3f}$", (x, Y_LO * 1.6), ha="center", va="bottom",
                    fontsize=9, color="#444444")

    ax.set_xticks(sorted(float(v) for v in lat))
    ax.set_xlim(float(lat.min()) - 8.0, float(lat.max()) + 8.0)
    ax.set_xlabel(r"Jammer reaction latency $L$ ($\mu$s), dwell = "
                  f"{int(reference['dwell_bursts'].iloc[0])} bursts")
    ax.set_ylabel("BER")
    half = reference[np.isclose(frac, 0.5)]
    if not half.empty:
        ax.annotate("half the bursts jammed, half the BER:\n"
                    r"the mixture is linear in $f$, there is no soft regime",
                    (float(half["latency_us"].iloc[0]),
                     float(half["ber"].iloc[0]) * 1.7),
                    ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_title("Hopping wins only while the slot is shorter than the jammer's "
                 f"reaction (dwell = {int(reference['dwell_bursts'].iloc[0])} "
                 f"bursts, JSR = +{jsr:.0f} dB)", pad=8, fontsize=11)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fs.legend_below(ax, ncol=2, y=-0.20)
    fs.save(fig, "frequency_agility_reaction")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
