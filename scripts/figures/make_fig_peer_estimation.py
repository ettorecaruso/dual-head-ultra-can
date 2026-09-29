#!/usr/bin/env python3
"""Peer-aware sensing: which echo the receiver reports, and the co-range fail-safe.

    python scripts/figures/make_fig_peer_estimation.py

Three panels at the top of the SNR grid (21 dB), with the peer-trained receivers
of the ``iod_peers`` scenario. Left: the fraction of samples whose reported
range is the obstacle's, against the delay difference between the obstacle and
the nearest peer (zero is the co-range case); the dashed curve is the mirror
failure, a peer echo reported as an obstacle. Centre: the same rate conditioned
on the amplitude ratio between the obstacle and the nearest peer echo. Right:
the rate at which the conservative fallback trips, as a function of the guard
factor, split between the co-range and the separated cases (solid and dashed):
that fail-safe is what keeps an obstacle sharing a peer's range inside the
obstacle list, and the separated curve is its price.

Written dataset: ``results/full/peer_estimation/<arch>/peer_estimation.csv``.

Axes. The two identification panels are framed on the rates they draw instead of
a fixed 0-1 axis: the reported-obstacle rate runs from 0 to 0.12 and a 0-1 axis
turned every curve into a flat line lying on the bottom spine. The fallback panel
keeps the full 0-1 range, because there the rates do span 0.3-1.0, so the three
panels no longer share one y scale.  The line styles carry information of their
own (solid: the obstacle is reported / the co-range fail-safe; dashed: the peer is
reported / the price paid on separated peers), so they are named in the legend
next to the two architectures.
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
#: Colour of the line-style proxies that name the solid/dashed series of a panel.
STYLE_PROXY = dict(color="#444444", lw=1.6, marker="")


def _scaled_to_data(ax, headroom: float = 1.3) -> None:
    """Frame a rate panel on the values it draws (see the module docstring)."""
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float).ravel()
                             for line in ax.get_lines()])
    values = values[np.isfinite(values)]
    top = float(values.max()) if values.size else 0.1
    ax.set_ylim(0.0, max(top * headroom, 1e-3))


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


def _rate(
    frame: pd.DataFrame,
    summary: str,
    key: str,
    column: str,
    guard: float | None = None,
) -> dict:
    sub = frame[(frame["summary"] == summary) & (frame["key_bin"] == key)]
    if guard is not None:
        sub = sub[np.isclose(sub["guard_factor"], float(guard))]
    return {str(row["arch"]): float(row[column]) for _, row in sub.iterrows()}


def _panel_offsets(ax, frame, offsets) -> None:
    for arch in ARCHS:
        hit = [
            _rate(frame, "offset_samples", str(o), "target_id_rate", REFERENCE_GUARD).get(arch, np.nan)
            for o in offsets
        ]
        false = [
            _rate(frame, "offset_samples", str(o), "peer_false_rate", REFERENCE_GUARD).get(arch, np.nan)
            for o in offsets
        ]
        kw = fs.series_kwargs(arch)
        ax.plot(offsets, hit, label=ARCH_LABELS[arch], **kw)
        ax.plot(offsets, false, color=kw["color"], ls=(0, (4, 2)), lw=1.2, alpha=0.65)
    # The two curves of each architecture differ by line style only, so the style
    # needs its own legend entries: without them the dashed curves are unnamed.
    ax.plot([], [], ls="-", label="target identification (solid)", **STYLE_PROXY)
    ax.plot([], [], ls=(0, (4, 2)), label="mirror failure (dashed)", **STYLE_PROXY)
    ax.set_xlabel("obstacle delay - nearest peer (samples)")
    ax.set_ylabel("fraction of samples")
    ax.set_title("Target identification vs peer distance")
    ax.set_xticks(offsets)


def _panel_ratio(ax, frame) -> None:
    positions = np.arange(len(RATIO_ORDER))
    for arch in ARCHS:
        values = [
            _rate(frame, "amplitude_ratio", key, "target_id_rate", REFERENCE_GUARD).get(arch, np.nan)
            for key in RATIO_ORDER
        ]
        kw = fs.series_kwargs(arch)
        ax.plot(positions, values, label=ARCH_LABELS[arch], **kw)
    ax.set_xticks(positions)
    ax.set_xticklabels([RATIO_LABELS[key] for key in RATIO_ORDER])
    ax.set_xlabel("obstacle / peer echo amplitude")
    ax.set_title("Target identification vs amplitude ratio")


def _panel_fallback(ax, frame) -> None:
    guards = sorted(
        float(value)
        for value in frame.loc[
            frame["summary"] == "co_range", "guard_factor"
        ].unique()
    )
    for arch in ARCHS:
        kw = fs.series_kwargs(arch)
        for key, dashes, alpha in (
            ("co_range", "-", 1.0),
            ("separated", (0, (4, 2)), 0.55),
        ):
            values = [
                _rate(frame, "co_range", key, "fallback_rate", guard).get(arch, np.nan)
                for guard in guards
            ]
            ax.plot(
                guards,
                values,
                color=kw["color"],
                ls=dashes,
                lw=1.4,
                alpha=alpha,
                marker=kw.get("marker", "o"),
                ms=fs.MARKER_SIZE * 0.7,
            )
    ax.set_xticks(guards)
    # Same reason as in the offset panel: solid and dashed are two different
    # quantities (the fail-safe on co-range peers and its price elsewhere).
    ax.plot([], [], ls="-", label="co-range: fallback trips (solid)", **STYLE_PROXY)
    ax.plot([], [], ls=(0, (4, 2)), label="separated: fallback price (dashed)", **STYLE_PROXY)
    ax.set_xlabel("guard factor")
    ax.set_title("Fallback: fail-safe vs false alarm")


def main() -> None:
    fs.apply_style()
    frame = load()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=False)

    offsets = sorted(
        int(value)
        for value in frame.loc[
            frame["summary"] == "offset_samples", "key_bin"
        ].unique()
    )
    _panel_offsets(axes[0], frame, offsets)
    _panel_ratio(axes[1], frame)
    _panel_fallback(axes[2], frame)

    # The identification panels are framed on the rates they draw; the fallback
    # panel keeps the full 0-1 range (its rates span 0.3-1.0). Sharing one y axis
    # across the three would either clip the fallback or flatten the other two.
    _scaled_to_data(axes[0])
    _scaled_to_data(axes[1])
    axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.grid(True, axis="y", color=fs.GRID_COLOR, lw=0.6, alpha=0.7)

    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=3)
    fs.save(fig, "peer_estimation")
    plt.close(fig)
    print(
        "overall target identification:",
        {
            arch: round(
                _rate(frame, "all", "all", "target_id_rate").get(arch, float("nan")), 4
            )
            for arch in ARCHS
        },
    )


if __name__ == "__main__":
    main()
