#!/usr/bin/env python3
"""What frequency hopping buys, and why it is geometry rather than signal processing.

    python scripts/make_fig_frequency_agility.py [run_dir]

Two panels from the ``frequency_agility`` run (8 channels, 10 us bursts, one
burst per hop unless the dwell sweep says otherwise, SNR = 21 dB, channel held
per hop):

* left -- **coverage**: the fraction of bursts the jammer reaches.  This is the
  only quantity hopping changes.  A full-band jammer covers everything whatever
  the transmitter does (it cannot be out-hopped); a band-agnostic jammer covers a
  share of the band it must guess; a hopping link leaves the jammer whatever its
  model can follow.  The annotations give the ideal (geometric) gain of each
  hopping point, ``-10 log10(f)``.
* right -- **consequence**: BER against JSR for the hopping link, with the
  fixed-carrier reference (omniscient jammer) as the grey band.  The vertical gap
  to that band is what the transmitter buys, and the ordering of the curves is
  the one of the left panel: nobody escapes the barrage control, while against
  partial-band and sweeping jammers the hopping link buys a fraction of the
  geometric value -- the rest is eaten by the receiver's own no-jamming floor.

The two panels are tied together by one relation, checked here before drawing:
the hopping link jams the same bursts at the same JSR, so its BER is the mixture

    BER_hop = f * BER_jam + (1 - f) * BER_clean

with ``BER_jam`` the omniscient fixed-carrier value, ``BER_clean`` the no-jammer
floor of the same condition and ``f`` the coverage.  The check runs over every
architecture, jammer archetype, in-channel type and JSR point; the script also
keeps the matched-control gate (under barrage, hopping on/off must agree to a
small fraction of the confidence half-width, because the exposure is identical).

``run_dir`` defaults to ``results/full/frequency_agility``.
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

DEFAULT_RUN_DIR = REPO / "results" / "full" / "frequency_agility"
ARCH = "qkv"
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
MODELS = [("barrage", "Barrage"),
          ("fixed_partial", "Fixed partial (25%)"),
          ("sweep", "Sweeping"),
          ("follower", "Follower (20 us)")]
IN_CHANNEL = "cw"
MIN_REALIZATIONS = 5
#: The aggregate fh_off/fh_on difference under barrage must stay inside this
#: fraction of the confidence half-width (see ``_check_matched_control``).
MATCHED_CONTROL_FRACTION = 0.05
#: The mixture must reproduce the hopping curve to this relative tolerance.
LAW_TOLERANCE = 0.05
Y_HI = 1.0
GAIN_JSR = 10.0
#: Marker style of the three modalities of the coverage panel.
MODALITY_STYLE = {
    "fh_off": ("Fixed carrier, omniscient jammer", "#7F7F7F", "o"),
    "fh_off_blind": ("Fixed carrier, band-agnostic jammer", "#FFA15A", "s"),
    "fh_on": ("Frequency hopping (100 k hop/s)", "#636EFA", "^"),
}


def _read(run_dir: Path, arch: str, name: str) -> pd.DataFrame:
    path = Path(run_dir) / arch / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {arch}/{name} (run the frequency_agility experiment)")
    df = pd.read_csv(path)
    if "n_realizations" in df:
        n = df["n_realizations"].dropna()
        if len(n) and int(n.max()) < MIN_REALIZATIONS:
            raise RuntimeError(f"{arch}/{name}: expected >= {MIN_REALIZATIONS} realizations, "
                               f"found {int(n.max())}")
    return df


def _jsr(run_dir: Path, arch: str) -> pd.DataFrame:
    df = _read(run_dir, arch, "frequency_agility_vs_jsr.csv")
    return df[(df.in_channel == IN_CHANNEL) & (df.arch == arch)]


def _curve(df: pd.DataFrame, model: str, modality: str) -> pd.DataFrame:
    out = df[(df.jammer_model == model) & (df.modality == modality)]
    return out.dropna(subset=["jsr_db"]).sort_values("jsr_db")


def _check_matched_control(run_dir: Path, arch: str) -> Tuple[float, float]:
    """Barrage is the matched control: hopping must not change the exposure.

    The jammer used as the control covers the whole band, so switching hopping on
    cannot change how much of the transmission it reaches: the two BER curves
    must agree to a small fraction of their confidence interval.  Both numbers
    are returned and printed, so the run log carries the evidence.
    """
    rea = _read(run_dir, arch, "frequency_agility_realizations.csv")
    rea = rea[rea.in_channel == IN_CHANNEL]
    piv = rea.pivot_table(index=["realization", "jammer_model", "jsr_db"],
                          columns="modality", values="ber").dropna()
    bar = piv.xs("barrage", level="jammer_model")
    per_realization = float((bar["fh_off"] - bar["fh_on"]).abs().max())

    jsr = _read(run_dir, arch, "frequency_agility_vs_jsr.csv")
    jsr = jsr[(jsr.in_channel == IN_CHANNEL) & (jsr.jammer_model == "barrage")]
    half = float(((jsr["ber_ci_hi"] - jsr["ber_ci_lo"]) / 2.0).max())
    means = jsr.pivot_table(index="jsr_db", columns="modality", values="ber")
    aggregate = float((means["fh_off"] - means["fh_on"]).abs().max())

    if not aggregate < MATCHED_CONTROL_FRACTION * half:
        raise AssertionError(
            f"{arch}: barrage is not a matched control, the aggregate "
            f"max|off-on| = {aggregate:.3e} is not inside "
            f"{MATCHED_CONTROL_FRACTION:.2f} of the confidence half-width "
            f"({half:.3e}). The two modalities must share the channel process and "
            "the transmission (see src/experiments/frequency_agility.py)."
        )
    return per_realization, aggregate


def _check_mixing_law(run_dir: Path) -> float:
    """BER_hop = f * BER_jam + (1 - f) * BER_clean, over every grid point."""
    worst = 0.0
    for arch in ARCHS:
        df = _read(run_dir, arch, "frequency_agility_vs_jsr.csv")
        for in_channel in sorted(df.in_channel.unique()):
            sub = df[(df.arch == arch) & (df.in_channel == in_channel)]
            for model, _label in MODELS:
                off = _curve(sub, model, "fh_off")
                on = _curve(sub, model, "fh_on")
                if off.empty or on.empty:
                    continue
                merged = on.merge(off[["jsr_db", "ber"]], on="jsr_db", suffixes=("", "_jam"))
                f = merged["jammed_fraction"].to_numpy(dtype=float)
                predicted = (f * merged["ber_jam"].to_numpy(dtype=float)
                             + (1.0 - f) * merged["ber_clean"].to_numpy(dtype=float))
                measured = merged["ber"].to_numpy(dtype=float)
                rel = np.abs(measured - predicted) / np.maximum(predicted, 1e-12)
                worst = max(worst, float(np.max(rel)))
    if worst > LAW_TOLERANCE:
        raise AssertionError(
            f"the mixture law does not describe the hopping curve: worst relative "
            f"error {worst:.3f} > {LAW_TOLERANCE:.2f}. Hopping must only change the "
            "coverage, not how bad an exposed burst is."
        )
    return worst


def _coverage(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """Coverage ``f`` per (archetype, modality), averaged over archs and JSR."""
    frames = [_read(run_dir, arch, "frequency_agility_vs_jsr.csv") for arch in ARCHS]
    everything = pd.concat(frames, ignore_index=True)
    everything = everything[everything.in_channel == IN_CHANNEL]
    out: Dict[str, Dict[str, float]] = {}
    for model, _label in MODELS:
        out[model] = {}
        for modality in MODALITY_STYLE:
            sub = everything[(everything.jammer_model == model)
                             & (everything.modality == modality)]
            out[model][modality] = float(sub["jammed_fraction"].mean())
    return out


def _panel_coverage(ax, coverage: Dict[str, Dict[str, float]]) -> None:
    """Left panel: the share of bursts the jammer reaches, per modality.

    Grouped bars instead of scattered markers: the three modalities are then
    read left to right inside each archetype, with no overlapping sticks and no
    text inside the plotting area.  The hopping value and the ideal gain it
    implies (``-10 log10 f``) are written under the archetype name, where they
    belong to the group as a whole.
    """
    positions = np.arange(len(MODELS))
    width = 0.26
    for index, (modality, (label, color, _marker)) in enumerate(MODALITY_STYLE.items()):
        values = [coverage[model][modality] for model, _label in MODELS]
        bars = ax.bar(positions + (index - 1) * width, values, width=width * 0.92,
                      color=color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{100 * value:.0f}%",
                        (bar.get_x() + bar.get_width() / 2.0, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color=color)

    labels = []
    for model, label in MODELS:
        f = coverage[model]["fh_on"]
        if f > 0.0:
            labels.append(f"{label}\n{f:.3f} exposed, ideal {-10.0 * np.log10(f):.1f} dB")
        else:
            labels.append(f"{label}\n0 exposed, nothing to buy")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Fraction of bursts the jammer reaches")
    ax.set_title("Hopping moves the jammer off most bursts", fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)


def _panel_ber(ax, jsr: pd.DataFrame) -> None:
    """Right panel: the consequence on BER, hopping against the control."""
    control = pd.concat([_curve(jsr, model, "fh_off") for model, _ in MODELS],
                        ignore_index=True)
    grouped = control.groupby("jsr_db")["ber"]
    x = np.array(sorted(grouped.groups.keys()), dtype=float)
    mean = grouped.mean().reindex(x).to_numpy(dtype=float)
    low = grouped.min().reindex(x).to_numpy(dtype=float)
    high = grouped.max().reindex(x).to_numpy(dtype=float)
    ax.fill_between(x, low, high, color="#7F7F7F", alpha=0.22, lw=0, zorder=2)
    ax.plot(x, mean, color="#7F7F7F", ls=(0, (4, 2)), lw=1.6, zorder=3,
            label="Fixed carrier, omniscient (any archetype)")

    for model, label in MODELS:
        c = _curve(jsr, model, "fh_on")
        if c.empty:
            continue
        kw = fs.series_kwargs(model)
        ax.plot(c["jsr_db"], c["ber"], label=label, zorder=4, **kw)

    follower = _curve(jsr, "follower", "fh_on")
    if not follower.empty:
        last = follower.iloc[-1]
        ax.annotate("follower: never engages at this dwell,\n"
                    "so this is the no-jamming floor, not a gain",
                    (float(last["jsr_db"]), float(last["ber"])),
                    textcoords="offset points", xytext=(-6, 14), ha="right",
                    va="bottom", fontsize=8.5, color=fs.series_kwargs("follower")["color"])

    ax.set_xlabel("JSR (dB)")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xticks([float(v) for v in x])
    ax.set_title("What it buys: the gap to the control", fontsize=11, pad=8)


def main(argv: List[str] | None = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    deltas = {a: _check_matched_control(run_dir, a) for a in ARCHS}
    print("matched control (barrage, per realization / aggregate):",
          {a: f"{d[0]:.1e} / {d[1]:.1e}" for a, d in deltas.items()})
    worst = _check_mixing_law(run_dir)
    print("mixture-law check (all archs, jammers, in-channels, JSR points): "
          f"worst {worst:.4f}")

    coverage = _coverage(run_dir)
    print(f"coverage f and the gain it implies at JSR = +{GAIN_JSR:.0f} dB:")
    for model, label in MODELS:
        f = coverage[model]["fh_on"]
        geo = -10.0 * np.log10(f) if f > 0 else float("nan")
        gains = []
        for arch in ARCHS:
            d = _jsr(run_dir, arch)
            off = _curve(d, model, "fh_off")
            on = _curve(d, model, "fh_on")
            at = np.isclose(off["jsr_db"], GAIN_JSR) & np.isclose(on["jsr_db"], GAIN_JSR)
            if at.any():
                ratio = float(off.loc[at, "ber"].iloc[0]) / max(float(on.loc[at, "ber"].iloc[0]), 1e-12)
                gains.append(f"{arch} {10 * np.log10(ratio):.2f}")
        print(f"  {label:<20} f={f:.3f}  geometric {geo:5.2f} dB | measured "
              + ", ".join(gains))

    jsr = _jsr(run_dir, ARCH)
    drawn = [_curve(jsr, model, modality)["ber"].to_numpy(dtype=float)
             for model, _ in MODELS for modality in ("fh_on", "fh_off")
             if not _curve(jsr, model, modality).empty]
    y_lo = fs.ber_floor(drawn, Y_HI)
    print(f"BER axis: {y_lo:g} .. {Y_HI:g}")

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    _panel_coverage(axes[0], coverage)
    fs.log_axis(axes[1], y_lo, Y_HI, x_step=4.0)
    _panel_ber(axes[1], jsr)
    axes[1].set_ylabel("BER")

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=4)
    fs.save(fig, "frequency_agility")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])

