#!/usr/bin/env python3
"""Frequency-agility figures, in the same publication look as the BER figures.

Figure ``frequency_agility`` (three panels):
  A. fixed carrier (no hopping): BER vs JSR for the four jammer models;
  B. frequency hopping at 100 k hop/s: the same four curves + the no-jammer line;
  C. BER vs hop rate (dwell sweep, JSR = +6 dB) for the sweeping jammer and for
     the reactive follower, with the fixed-carrier level as reference.

Figure ``frequency_agility_gain``: gain of the hopping link over the fixed carrier
at JSR = +10 dB, per architecture (barrage = control, fixed partial coverage,
sweeping jammer). It shows that the gain is architecture-independent; the
follower is excluded because at a 1-burst dwell it never engages (its "gain" is
not a BER gain, it is the absence of jamming).

Every curve is the mean over the ``n_realizations`` jammer realizations stored in
``frequency_agility_vs_jsr.csv`` and ``frequency_agility_vs_dwell.csv``. The script
asserts the matched-control invariant: with barrage both modalities jam the same
bursts on the same channel, so their per-realization BER must coincide.
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
MODELS = [("barrage", "Barrage (control)"),
          ("fixed_partial", "Fixed partial (25%)"),
          ("sweep", "Sweeping"),
          ("follower", "Follower (20 \u00b5s)")]
IN_CHANNEL = "cw"
MIN_REALIZATIONS = 5
Y_LO, Y_HI = 3e-5, 1.0
DWELL_JSR = 6.0
GAIN_JSR = 10.0
GAIN_MODELS = ("barrage", "fixed_partial", "sweep")


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


def _panel_jsr(ax, jsr: pd.DataFrame, modality: str, title: str,
               clean: float | None) -> None:
    """One BER-vs-JSR panel: the four jammer models for a given modality."""
    fs.log_axis(ax, Y_LO, Y_HI, x_step=4.0)
    if clean is not None:
        ax.axhline(clean, color="#7F7F7F", lw=1.1, ls=":", zorder=1, label="no jammer")
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


def _panel_dwell(ax, dwell: pd.DataFrame, jsr: pd.DataFrame, title: str) -> None:
    """BER vs hop rate at a fixed JSR, for the sweeping and the follower jammer."""
    fs.log_axis(ax, Y_LO, Y_HI)
    ax.set_xscale("log")
    ref = _curve(jsr, "sweep", "fh_off")
    ref = ref[np.isclose(ref["jsr_db"], DWELL_JSR)]
    if len(ref):
        ax.axhline(float(ref["ber"].iloc[0]), color=fs.series_kwargs("fixed_partial")["color"],
                   lw=1.1, ls=":", zorder=1, label="fixed carrier (sweeping jammer)")
    rates = None
    for model in ("sweep", "follower"):
        c = dwell[(dwell.jammer_model == model) & np.isclose(dwell.jsr_db, DWELL_JSR)]
        c = c.sort_values("hop_rate_hz")
        if c.empty:
            continue
        kw = fs.series_kwargs(model)
        x = c["hop_rate_hz"].to_numpy(dtype=float)
        y = c["ber"].to_numpy(dtype=float)
        s = np.nan_to_num(c["ber_std"].to_numpy(dtype=float), nan=0.0)
        rates = x
        ax.fill_between(x, np.clip(y - s, Y_LO, Y_HI), np.clip(y + s, Y_LO, Y_HI),
                        color=kw["color"], alpha=0.15, lw=0, zorder=2)
        ax.plot(x, y, label=dict(MODELS)[model], zorder=3, **kw)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Hop rate (hops/s)")
    if rates is not None:
        ticks = sorted(set(float(v) for v in rates))
        ax.set_xlim(ticks[0] * 0.78, ticks[-1] * 1.3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_kfmt(v) for v in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())


def _gain_figure(gains: dict, title: str) -> None:
    """Grouped bars: hopping gain over the fixed carrier, per architecture."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(ARCHS))
    width = 0.26
    allv = []
    for i, model in enumerate(GAIN_MODELS):
        vals = [float(gains[a][model]) for a in ARCHS]
        allv += [v for v in vals if np.isfinite(v)]
        kw = fs.series_kwargs(model)
        ax.bar(x + (i - 1) * width, vals, width, label=dict(MODELS)[model],
               color=kw["color"], edgecolor="black", linewidth=0.6, zorder=3)
        for xi, v in zip(x + (i - 1) * width, vals):
            if not np.isfinite(v):
                continue
            ax.text(xi, v + (0.25 if v >= 0 else -0.25), f"{v:+.1f}", ha="center",
                    fontsize=9, va="bottom" if v >= 0 else "top")
    lo = min(allv + [0.0]) - 1.2
    hi = max(allv + [0.0]) + 1.2
    ax.set_ylim(lo, hi)
    ax.axhline(0.0, color="black", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=9.5)
    ax.set_ylabel("Gain over fixed carrier (dB)")
    ax.set_title(title, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fs.legend_below(ax, ncol=3, y=-0.14)
    fs.save(fig, "frequency_agility_gain")
    plt.close(fig)


def main() -> None:
    deltas = {a: _check_matched_control(a) for a in ARCHS}
    print("matched control (barrage, per realization):",
          {a: f"max|off-on|={d:.1e}" for a, d in deltas.items()})

    jsr = _jsr(ARCH)
    dwell = _dwell(ARCH)
    clean_rows = jsr[(jsr.modality == "fh_on") & (jsr.jammer_model == "barrage")]
    clean = float(clean_rows["ber_clean"].mean()) if len(clean_rows) else None

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    _panel_jsr(axes[0], jsr, "fh_off", "Fixed carrier (no hopping)", clean)
    _panel_jsr(axes[1], jsr, "fh_on", "Frequency hopping (100 k hop/s)", clean)
    _panel_dwell(axes[2], dwell, jsr, f"BER vs hop rate (JSR = +{DWELL_JSR:.0f} dB)")
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    fs.legend_below(axes[1], ncol=3)
    fs.outer_frame(fig, axes)
    fs.save(fig, "frequency_agility")
    plt.close(fig)

    gains = {}
    for arch in ARCHS:
        d = _jsr(arch)
        gains[arch] = {}
        for model in GAIN_MODELS:
            off = _curve(d, model, "fh_off")
            on = _curve(d, model, "fh_on")
            off = off[np.isclose(off["jsr_db"], GAIN_JSR)]
            on = on[np.isclose(on["jsr_db"], GAIN_JSR)]
            if off.empty or on.empty:
                gains[arch][model] = float("nan")
                continue
            ratio = max(float(off["ber"].iloc[0]), 1e-12) / max(float(on["ber"].iloc[0]), 1e-12)
            gains[arch][model] = float(10.0 * np.log10(ratio))
    print(f"gain at JSR = +{GAIN_JSR:.0f} dB:",
          {a: {k: (round(v, 2) if np.isfinite(v) else None) for k, v in g.items()}
           for a, g in gains.items()})
    _gain_figure(gains, f"Hopping gain over the fixed carrier (JSR = +{GAIN_JSR:.0f} dB)")


if __name__ == "__main__":
    main()

