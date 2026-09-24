#!/usr/bin/env python3
"""Frequency-agility figures, in the same publication look as the BER figures.

Figure ``frequency_agility`` (three panels):
  A. fixed carrier (no hopping): BER vs JSR. Without agility every jammer model
     jams every burst (``jammed_fraction = 1``), so the four curves collapse onto
     each other;
  B. frequency hopping at 100 k hop/s (1-burst dwell): the curves separate by how
     much of the band/time their jammer really covers (barrage 100% > fixed
     partial 25% > sweeping 1/8 of the channels), while the follower never
     engages (``jammed_fraction = 0``) and therefore stays flat at its
     no-jamming BER: that flat line is the result, not a missing measurement
     (see ``results/RESULTS_GENERATION.md``, note on the follower);
  C. BER vs hop rate (dwell sweep, JSR = +6 dB) for the sweeping jammer and for
     the reactive follower. Only these two models are swept (config key
     ``frequency_hopping.dwell_sweep_models``): barrage and fixed partial cover a
     *fixed* share of the channels (100% and 25%) at every dwell, so their curves
     would just be flat duplicates of the vs-JSR panels. The sweeper is
     insensitive to the hop rate as well (its blind hit probability stays 1/8),
     whereas the follower is defeated by short dwells: at 50-100 k hop/s the dwell
     is shorter than its reaction time (2 bursts) and it stops jamming.

Figure ``frequency_agility_gain``: two panels at JSR = +10 dB.
  A. the absolute hopping gain per jammer archetype, one marker per architecture
     (dot plot, no bars). The gain is *not* a receiver property: it is the share
     of bursts that the hopping moves away from the jammer, so the four
     receivers sit on the same value and the barrage control is exactly 0 dB;
     the dashed line is the geometric value -10 log10(jammed fraction);
  B. the residual of every architecture with respect to that geometric value, in
     mdB. The architecture-to-architecture spread is a few *hundredths* of a dB,
     so only this zoomed residual scale makes it visible.
The follower is excluded because at a 1-burst dwell it never engages (its "gain"
is not a BER gain, it is the absence of jamming).

Every curve is the mean over the ``n_realizations`` jammer realizations stored in
``frequency_agility_vs_jsr.csv`` and ``frequency_agility_vs_dwell.csv``; the
per-realization spread is *not* drawn (no +-1 sigma envelope on the curves). The
script asserts the matched-control invariant: with barrage both modalities jam
the same bursts on the same channel, so their per-realization BER must coincide.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402

RESULTS = REPO / "results" / "full" / "frequency_agility"
ARCH = "qkv"
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN (QKV)",
               "lstm": "LSTM-OFDM-DCSK", "mc_dlsk": "MC-DLCSK"}
#: Short architecture names for the x axis of the gain panels.
ARCH_TICKS = {"conv1d": "Conv1D", "qkv": "QKV", "lstm": "LSTM\nOFDM-DCSK",
              "mc_dlsk": "MC-DLCSK"}
MODELS = [("barrage", "Barrage"),
          ("fixed_partial", "Fixed partial (25%)"),
          ("sweep", "Sweeping"),
          ("follower", "Follower (20 \u00b5s)")]
IN_CHANNEL = "cw"
MIN_REALIZATIONS = 5
Y_LO, Y_HI = 3e-5, 1.0
DWELL_JSR = 6.0
GAIN_JSR = 10.0
GAIN_MODELS = ("barrage", "fixed_partial", "sweep")
#: Jammer archetypes drawn in the gain figure. ``barrage`` is the matched control
#: (gain exactly 0 dB for every receiver): it is stated as a value in the panel
#: instead of being drawn as an invisible zero-height bar.
GAIN_PLOT = ("fixed_partial", "sweep")


def _read(arch: str, name: str) -> pd.DataFrame:
    path = RESULTS / arch / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {arch}/{name} (run the frequency_agility experiment)")
    df = pd.read_csv(path)
    if "n_realizations" in df:
        n = df["n_realizations"].dropna()
        if len(n) and int(n.max()) < MIN_REALIZATIONS:
            raise RuntimeError(f"{arch}/{name}: expected >= {MIN_REALIZATIONS} realizations, "
                               f"found {int(n.max())}")
    return df


def _jsr(arch: str) -> pd.DataFrame:
    df = _read(arch, "frequency_agility_vs_jsr.csv")
    return df[(df.in_channel == IN_CHANNEL) & (df.arch == arch)]


def _dwell(arch: str) -> pd.DataFrame:
    df = _read(arch, "frequency_agility_vs_dwell.csv")
    return df[(df.in_channel == IN_CHANNEL) & (df.arch == arch)]


def _check_matched_control(arch: str) -> float:
    """Barrage: the two modalities must give the same BER per realization."""
    rea = _read(arch, "frequency_agility_realizations.csv")
    rea = rea[rea.in_channel == IN_CHANNEL]
    piv = rea.pivot_table(index=["realization", "jammer_model", "jsr_db"],
                          columns="modality", values="ber").dropna()
    bar = piv.xs("barrage", level="jammer_model")
    delta = float((bar["fh_off"] - bar["fh_on"]).abs().max())
    if delta > 1e-9:
        raise AssertionError(
            f"{arch}: barrage is not a matched control, max|off-on| = {delta:.3e}. "
            "The two modalities must share the per-slot channel process and the "
            "transmission (see src/experiments/frequency_agility.py)."
        )
    return delta


def _curve(df: pd.DataFrame, model: str, modality: str = "fh_on") -> pd.DataFrame:
    out = df[(df.jammer_model == model) & (df.modality == modality)]
    return out.dropna(subset=["jsr_db"]).sort_values("jsr_db")


def _kfmt(value: float) -> str:
    return f"{value / 1000:.4g}k" if value >= 1000 else f"{value:.4g}"


def _panel_jsr(ax, jsr: pd.DataFrame, modality: str, title: str) -> None:
    """One BER-vs-JSR panel: the four jammer models for a given modality.

    No no-jammer reference line is drawn: it is flat by definition and it only
    adds another flat entry to the legend.
    """
    fs.log_axis(ax, Y_LO, Y_HI, x_step=4.0)
    xs = None
    for model, label in MODELS:
        c = _curve(jsr, model, modality)
        if c.empty:
            continue
        kw = fs.series_kwargs(model)
        x = c["jsr_db"].to_numpy(dtype=float)
        y = c["ber"].to_numpy(dtype=float)
        xs = x
        ax.plot(x, y, label=label, zorder=3, **kw)
    ax.set_title(title, pad=8)
    ax.set_xlabel("JSR (dB)")
    if xs is not None:
        ax.set_xlim(float(xs.min()), float(xs.max()))
        ax.set_xticks(sorted(float(v) for v in xs))


def _panel_dwell(ax, dwell: pd.DataFrame, title: str) -> None:
    """BER vs hop rate at a fixed JSR, for the sweeping and the follower jammer."""
    fs.log_axis(ax, Y_LO, Y_HI)
    ax.set_xscale("log")
    rates = None
    for model in ("sweep", "follower"):
        c = dwell[(dwell.jammer_model == model) & np.isclose(dwell.jsr_db, DWELL_JSR)]
        c = c.sort_values("hop_rate_hz")
        if c.empty:
            continue
        kw = fs.series_kwargs(model)
        x = c["hop_rate_hz"].to_numpy(dtype=float)
        y = c["ber"].to_numpy(dtype=float)
        rates = x
        ax.plot(x, y, label=dict(MODELS)[model], zorder=3, **kw)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Hop rate (hops/s)")
    if rates is not None:
        ticks = sorted(set(float(v) for v in rates))
        ax.set_xlim(ticks[0] * 0.78, ticks[-1] * 1.3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_kfmt(v) for v in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())


def _geometric_gain(fractions: dict, model: str) -> float:
    """``-10 log10(jammed fraction)`` of a jammer archetype (mean over archs).

    This is the gain the hopping modality buys when the protected bursts are
    error-free: it depends on the jammer coverage only, never on the receiver.
    """
    frac = float(np.nanmean([float(fractions[a][model]) for a in ARCHS]))
    if not np.isfinite(frac) or frac <= 0.0:
        return float("nan")
    return -10.0 * np.log10(frac)


def _gain_figure(gains: dict, fractions: dict, title: str) -> None:
    """Two panels: the absolute hopping gain and its residual on the geometry.

    Left panel: the gain of the jammer archetypes, one marker per architecture.
    No bars are used (with a dot plot a zoomed axis is legitimate) and the
    barrage control is drawn as an ordinary marker, not as a zero-height bar:
    its gain is exactly 0 dB for every architecture, so a bar would be invisible.

    Right panel: measured gain minus the geometric value, in mdB, i.e. the only
    zoom on which the few *hundredths* of a dB that separate the receivers are
    visible at all.
    """
    geo = {model: _geometric_gain(fractions, model) for model in GAIN_PLOT}
    bar = float(np.nanmean([float(gains[a]["barrage"]) for a in ARCHS]))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax0, ax1 = axes

    # ---- left: absolute gain, one marker per architecture ------------------
    x = np.arange(1 + len(GAIN_PLOT))
    means = [bar] + [float(np.nanmean([gains[a][m] for a in ARCHS]))
                     for m in GAIN_PLOT]
    for i, model in enumerate(GAIN_PLOT, start=1):
        if np.isfinite(geo[model]):
            colour = fs.series_kwargs(model)["color"]
            ax0.axhline(geo[model], color=colour, lw=1.1, ls="--", alpha=0.55,
                        zorder=1)
    for dx, arch in zip(np.linspace(-0.09, 0.09, len(ARCHS)), ARCHS):
        vals = [bar] + [float(gains[arch][m]) for m in GAIN_PLOT]
        kw = fs.series_kwargs(arch)
        ax0.plot(x + dx, vals, ls="", marker=kw["marker"], ms=8.5,
                 color=kw["color"], label=ARCH_LABELS[arch], zorder=4)
    for xi, value in zip(x, means):
        ax0.annotate(f"{value:+.2f} dB", (xi, value), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9.5, zorder=5)
    finite = [v for v in means + list(geo.values()) if np.isfinite(v)]
    ax0.set_ylim(min(finite) - 0.5, max(finite) + 0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(["Barrage"] +
                        [dict(MODELS)[m].replace(" (", "\n(") for m in GAIN_PLOT],
                        fontsize=9)
    ax0.set_ylabel("Gain over fixed carrier (dB)")
    ax0.set_title("Absolute gain per jammer archetype (dashed: geometric "
                  r"$-10\log_{10}f$)", fontsize=11, pad=8)

    # ---- right: residual on the geometric value, in mdB --------------------
    xr = np.arange(len(ARCHS))
    resid = []
    for model in GAIN_PLOT:
        vals = [1000.0 * (float(gains[a][model]) - geo[model]) for a in ARCHS]
        resid += [v for v in vals if np.isfinite(v)]
        kw = fs.series_kwargs(model)
        ax1.plot(xr, vals, color=kw["color"], lw=1.0, alpha=0.7, zorder=2)
        ax1.plot(xr, vals, ls="", marker=kw["marker"], ms=8.5, color=kw["color"],
                 label=dict(MODELS)[model], zorder=3)
        for xi, value in zip(xr, vals):
            if np.isfinite(value):
                ax1.annotate(f"{value:+.0f}", (xi, value),
                             textcoords="offset points", xytext=(0, 10),
                             ha="center", fontsize=9)
    ax1.axhline(0.0, color="#555555", lw=1.1, ls="--", alpha=0.8, zorder=1)
    if resid:
        ax1.set_ylim(min(resid) - 25.0, max(resid) + 35.0)
    ax1.set_xticks(xr)
    ax1.set_xticklabels([ARCH_TICKS[a] for a in ARCHS], fontsize=9)
    ax1.set_ylabel("gain − geometric (mdB)")
    ax1.set_title("Residual: architecture dependence", fontsize=11, pad=8)

    for ax in (ax0, ax1):
        ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=1.0)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.0)
    fs.legend_below(ax0, ncol=2, y=-0.24)
    fs.legend_below(ax1, ncol=2, y=-0.24)
    fs.outer_frame(fig, axes)
    fs.save(fig, "frequency_agility_gain")
    plt.close(fig)


def main() -> None:
    deltas = {a: _check_matched_control(a) for a in ARCHS}
    print("matched control (barrage, per realization):",
          {a: f"max|off-on|={d:.1e}" for a, d in deltas.items()})

    jsr = _jsr(ARCH)
    dwell = _dwell(ARCH)

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    _panel_jsr(axes[0], jsr, "fh_off", "Fixed carrier (no hopping)")
    _panel_jsr(axes[1], jsr, "fh_on", "Frequency hopping (100 k hop/s)")
    _panel_dwell(axes[2], dwell, f"BER vs hop rate (JSR = +{DWELL_JSR:.0f} dB)")
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    fs.legend_below(axes[1], ncol=2)
    fs.outer_frame(fig, axes)
    fs.save(fig, "frequency_agility")
    plt.close(fig)

    gains = {}
    fractions = {}
    for arch in ARCHS:
        d = _jsr(arch)
        gains[arch] = {}
        fractions[arch] = {}
        for model in GAIN_MODELS:
            off = _curve(d, model, "fh_off")
            on = _curve(d, model, "fh_on")
            off = off[np.isclose(off["jsr_db"], GAIN_JSR)]
            on = on[np.isclose(on["jsr_db"], GAIN_JSR)]
            if off.empty or on.empty:
                gains[arch][model] = float("nan")
                fractions[arch][model] = float("nan")
                continue
            ratio = max(float(off["ber"].iloc[0]), 1e-12) / max(float(on["ber"].iloc[0]), 1e-12)
            gains[arch][model] = float(10.0 * np.log10(ratio))
            fractions[arch][model] = (float(on["jammed_fraction"].iloc[0])
                                      if "jammed_fraction" in on else float("nan"))
    print(f"gain at JSR = +{GAIN_JSR:.0f} dB:",
          {a: {k: (round(v, 2) if np.isfinite(v) else None) for k, v in g.items()}
           for a, g in gains.items()})
    print(f"jammed fraction at JSR = +{GAIN_JSR:.0f} dB:",
          {a: {k: (round(v, 4) if np.isfinite(v) else None) for k, v in f.items()}
           for a, f in fractions.items()})
    for model in GAIN_MODELS:
        print(f"geometric -10log10(f) for {model}: "
              f"{_geometric_gain(fractions, model):.3f} dB")
    _gain_figure(gains, fractions,
                 f"Hopping gain over the fixed carrier (JSR = +{GAIN_JSR:.0f} dB)")


if __name__ == "__main__":
    main()

