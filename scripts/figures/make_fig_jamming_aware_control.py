#!/usr/bin/env python3
"""Jamming-aware training control figure.

Panels: CW, barrage (control) and partial-band. In every panel the clean-trained
QKV receiver and the jamming-aware one are plotted as the **mean BER over the
jammer realizations** stored in ``conditions.csv`` (``n_realizations``).

Only the mean-over-realizations curves are drawn: the +-1 sigma envelope is *not*
plotted (the spread is tabulated in ``conditions_realizations.csv``), and the
per-realization table and single-run curves are deliberately *not* plotted
either, because a single CW realization is the artefact that made the curve
non-monotone (see ``results/full/diagnostics/jamming_artifact``). The two
no-jammer reference lines are not drawn either: each regime has its own flat
clean BER and the low-JSR tail of its curve already *is* that no-jamming floor,
so the lines only add two flat entries to the legend. The script asserts that
the clean-trained CW and partial-band curves are monotone in JSR, so that
artefact can not silently come back.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

RESULTS = REPO / "results" / "full" / "jamming_interpretability"
CLEAN = RESULTS / "qkv"
AWARE = RESULTS / "jamming_aware_training" / "qkv"
JAMMERS = [("cw", "CW"), ("barrage", "Barrage"),
           ("partial_band", "Partial band")]
MONOTONE = ("cw", "partial_band")
MIN_REALIZATIONS = 5
Y_LO, Y_HI = 1e-3, 1.0
_Z95 = 1.959963984540054


def _load(path: Path, who: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing {who} conditions: {path}")
    df = pd.read_csv(path)
    n = df["n_realizations"].dropna().unique() if "n_realizations" in df else []
    if len(n) != 1 or int(n[0]) < MIN_REALIZATIONS:
        raise RuntimeError(
            f"{who}: expected the mean over >= {MIN_REALIZATIONS} realizations, "
            f"found n_realizations={list(n)}"
        )
    return df


def _curve(df: pd.DataFrame, jammer: str) -> pd.DataFrame:
    out = df[df.jammer == jammer].copy()
    return out.dropna(subset=["jsr_db"]).sort_values("jsr_db")


def _check_monotone(df: pd.DataFrame, who: str) -> None:
    """The mean-over-realizations curves must grow monotonically with JSR."""
    for jammer in MONOTONE:
        c = _curve(df, jammer)
        if len(c) < 2:
            continue
        d = np.diff(c["ber"].to_numpy(dtype=float))
        if np.any(d < -1e-9):
            raise AssertionError(
                f"{who}/{jammer}: BER is not monotone in JSR (worst step {float(d.min()):.3e}). "
                "A single-realization artefact is back: check n_realizations and the "
                "jammer geometry (results/full/diagnostics/jamming_artifact)."
            )


def _band(frame: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Interval of a mean curve: the stored CI, or the SEM of the realizations."""
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


def _draw_panel(ax, clean: pd.DataFrame, aware: pd.DataFrame,
                jammer: str, title: str) -> None:
    """One JSR panel: mean curves of the two training regimes, no envelope."""
    fs.log_axis(ax, Y_LO, Y_HI, x_step=4.0)
    xs = None
    for df, name, label in ((clean, "clean_trained", "Clean-trained"),
                            (aware, "jamming_aware", "Jamming-aware")):
        c = _curve(df, jammer)
        if c.empty:
            continue
        kw = fs.series_kwargs(name)
        x = c["jsr_db"].to_numpy(dtype=float)
        y = c["ber"].to_numpy(dtype=float)
        xs = x
        ax.plot(x, y, label=label, zorder=3, **kw)
    ax.set_title(title, pad=8)
    ax.set_xlabel("JSR (dB)")
    if xs is not None:
        ax.set_xlim(float(xs.min()), float(xs.max()))
        ax.set_xticks(sorted(float(v) for v in xs))
    ax.set_axisbelow(True)


def main() -> None:
    clean = _load(CLEAN / "conditions.csv", "clean-trained")
    aware = _load(AWARE / "conditions.csv", "jamming-aware")
    _check_monotone(clean, "clean-trained")
    _check_monotone(aware, "jamming-aware")
    print(f"n_realizations: clean-trained={int(clean['n_realizations'].dropna().iloc[0])}, "
          f"jamming-aware={int(aware['n_realizations'].dropna().iloc[0])}")

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, (jammer, title) in zip(axes, JAMMERS):
        _draw_panel(ax, clean, aware, jammer, title)
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    fs.legend_below(axes[1], ncol=2)
    fs.save(fig, "jamming_aware_control")
    plt.close(fig)


if __name__ == "__main__":
    main()

