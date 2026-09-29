#!/usr/bin/env python3
"""Peer-aware sensing: how accurate is the range, and what does the fail-safe cost.

    python scripts/make_fig_peer_estimation_price.py

Two panels that answer the questions the composition figure leaves open:

* left -- the delay error in samples, per amplitude-ratio bin, as the mean (bar)
  with the 90th percentile (whisker) and the half-sample hit tolerance as a
  reference line.  The error is quoted against the 33-sample delay window the
  receiver works in, so the number can be read as a share of the observable span;
* right -- the fraction of samples where the conservative fallback trips, as a
  function of the guard factor, split between peers sharing the obstacle's range
  (the fail-safe doing its job) and separated peers (its price).

Both panels read the per-sample table (``peer_samples.csv``), so the two
receivers can be compared on the same samples; where they agree the script says
so in the run log instead of drawing a second indistinguishable curve.

Written datasets: ``results/full/peer_estimation/<arch>/peer_samples.csv`` and
``peer_estimation.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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
#: Same bins as the experiment (``src/experiments/peer_estimation.py``).
RATIO_BINS = [0.0, 0.75, 1.25, 2.0, np.inf]
RATIO_ORDER = ("peer_stronger", "comparable", "target_gt1.25", "target_gt2")
RATIO_LABELS = {
    "peer_stronger": "peer\nstronger",
    "comparable": "comparable",
    "target_gt1.25": "target\nx1.25",
    "target_gt2": "target\nx2",
}
GUARD = 1.5
#: The metric calls an estimate correct when it is within this many samples.
TOLERANCE = 0.5
#: Delay window of the receiver (``data.max_delay``): the error scale.
WINDOW = 33.0


def load_samples() -> Dict[str, pd.DataFrame]:
    frames = {}
    for arch in ARCHS:
        path = SRC / arch / "peer_samples.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run the peer_estimation experiment")
        frame = pd.read_csv(path)
        frames[arch] = frame[np.isclose(frame["guard_factor"], GUARD)]
    return frames


def load_summary() -> pd.DataFrame:
    frames = []
    for arch in ARCHS:
        frames.append(pd.read_csv(SRC / arch / "peer_estimation.csv"))
    frame = pd.concat(frames, ignore_index=True)
    frame["key_bin"] = frame["key_bin"].astype(str)
    return frame


def _binned(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["ratio"], bins=RATIO_BINS, labels=list(RATIO_ORDER),
                  right=False).astype(str)


def _accuracy(samples: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for arch, frame in samples.items():
        binned = _binned(frame)
        mean, p90 = [], []
        for key in RATIO_ORDER:
            err = frame.loc[binned == key, "abs_err"].to_numpy(dtype=float)
            mean.append(float(np.mean(err)) if err.size else np.nan)
            p90.append(float(np.percentile(err, 90)) if err.size else np.nan)
        out[arch] = {"mean": np.array(mean), "p90": np.array(p90)}
    return out


def _panel_accuracy(ax, accuracy: Dict[str, Dict[str, np.ndarray]]) -> None:
    x = np.arange(len(RATIO_ORDER))
    width = 0.34
    for index, arch in enumerate(ARCHS):
        kw = fs.series_kwargs(arch)
        offset = (index - 0.5) * width
        values = accuracy[arch]["mean"]
        lower = np.zeros_like(values)
        upper = np.clip(accuracy[arch]["p90"] - values, 0.0, None)
        ax.bar(x + offset, values, width=width, color=kw["color"],
               label=ARCH_LABELS[arch], zorder=3)
        ax.errorbar(x + offset, values, yerr=np.vstack([lower, upper]), fmt="none",
                    ecolor="#555555", elinewidth=1.1, capsize=3.5, zorder=4)
    ax.axhline(TOLERANCE, color="#EF553B", ls="--", lw=1.4, zorder=5)
    ax.annotate(f"hit tolerance = {TOLERANCE:g} sample", (len(RATIO_ORDER) - 0.5,
                TOLERANCE), textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=9, color="#EF553B")
    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABELS[k] for k in RATIO_ORDER], fontsize=9.5)
    ax.set_xlabel("obstacle / peer echo amplitude")
    ax.set_ylabel("delay error (samples; bar = mean, whisker = P90)")
    ax.set_ylim(0.0, max(4.0, float(np.nanmax([accuracy[a]["p90"].max() for a in ARCHS]))) * 1.1)
    ax.set_title("Off by several samples, while the metric asks for half",
                 fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _panel_fallback(ax, summary: pd.DataFrame) -> float:
    guards = sorted(float(v) for v in
                    summary.loc[summary["summary"] == "co_range", "guard_factor"].unique())
    worst = 0.0
    for key, dashes, label in (("co_range", "-", "peer shares the obstacle range"),
                               ("separated", (0, (4, 2)), "peer well separated")):
        values = []
        for guard in guards:
            sub = summary[(summary["summary"] == "co_range")
                          & (summary["key_bin"] == key)
                          & np.isclose(summary["guard_factor"], guard)]
            values.append(float(sub["fallback_rate"].mean()))
            worst = max(worst, float(sub["fallback_rate"].max() - sub["fallback_rate"].min()))
        ax.plot(guards, values, ls=dashes, marker="o", ms=6, lw=1.8,
                color="#636EFA" if key == "co_range" else "#EF553B", label=label,
                zorder=3)
    ax.set_xticks(guards)
    ax.set_xlabel("guard factor")
    ax.set_ylabel("fraction of samples with the fallback tripped")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("The fail-safe works, and its price is a third to three quarters",
                 fontsize=11, pad=8)
    ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return worst


def main() -> None:
    fs.apply_style()
    samples = load_samples()
    summary = load_summary()
    accuracy = _accuracy(samples)

    qkv = samples[ARCHS[-1]]["abs_err"]
    print("delay error (samples, window %.0f): mean %.2f, median %.2f, P90 %.2f"
          % (WINDOW, qkv.mean(), qkv.median(), np.percentile(qkv, 90)))
    print("fraction within 0.5 / 1 / 2 / 5 samples: "
          + " / ".join(f"{100 * float((qkv <= t).mean()):.1f}%" for t in (0.5, 1, 2, 5)))
    for arch in ARCHS:
        print(f"  {ARCH_LABELS[arch]:<18} mean {samples[arch]['abs_err'].mean():.2f} | "
              f"P90 {np.percentile(samples[arch]['abs_err'], 90):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    _panel_accuracy(axes[0], accuracy)
    worst = _panel_fallback(axes[1], summary)
    print("worst receiver disagreement on the fallback rate: %.2e" % worst)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=2)
    fs.save(fig, "peer_estimation_price")
    plt.close(fig)


if __name__ == "__main__":
    main()

