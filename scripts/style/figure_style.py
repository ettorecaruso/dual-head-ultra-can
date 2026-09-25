"""Shared publication style for the figures of the Dual-Head Ultra-CAN paper.

The look reproduces the ``plotly_white`` palette used in the notebooks: white
background, black frame, light major grid, dotted sub-decade grid, Unicode log
ticks and a single shared legend *below* the panels (so it never sits on a curve).
Each panel keeps its own frame: the panels are deliberately **not** joined into
one box, so a row of panels reads as separate measurements.

Typical use in a figure script::

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from style import figure_style as fs

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, ... in ...:
        fs.log_axis(ax, y_lo, y_hi, x_step=2.0)
        ax.plot(x, y, label=..., **fs.series_kwargs("qkv"))
        ax.set_title(...); ax.set_xlabel(...)
    fs.legend_below(axes[1], ncol=2)      # or legend_below_fig for a centred one
    fs.save(fig, "my_figure")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]

LINE_W = 1.5
MARKER_SIZE = 6.5
GRID_COLOR = "#D3D3D3"
TARGET_COLOR = "#00CC96"
BASELINE_FILL = "#F0FBF5"

#: ``plotly_white`` palette of the paper notebooks, per architecture.
ARCH_COLORS = {"conv1d": "#636EFA", "qkv": "#EF553B",
               "lstm": "#00CC96", "mc_dlsk": "#AB63FA"}
ARCH_MARKERS = {"conv1d": "o", "qkv": "s", "lstm": "D", "mc_dlsk": "+"}
ARCH_DASHES = {
    "conv1d": "-",
    "qkv": (0, (6, 2)),
    "lstm": (0, (1, 1.6)),
    "mc_dlsk": (0, (4, 1.2, 1, 1.2)),
}

#: Same palette family, for the series of the jamming / frequency-agility
#: figures. ``barrage`` and ``clean`` are the grey controls.
SERIES = {
    "clean": ("#7F7F7F", "o", (0, (1, 1.6))),
    "clean_trained": ("#EF553B", "o", "-"),
    "jamming_aware": ("#636EFA", "s", (0, (6, 2))),
    "barrage": ("#7F7F7F", "o", "-"),
    "fixed_partial": ("#EF553B", "s", (0, (6, 2))),
    "sweep": ("#636EFA", "D", (0, (4, 1.2, 1, 1.2))),
    "follower": ("#00CC96", "^", (0, (1, 1.6))),
    "fixed carrier": ("#EF553B", "o", "-"),
    "hopping": ("#636EFA", "s", (0, (6, 2))),
}

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sup_exp(exp: int) -> str:
    """Unicode log label, e.g. -4 -> '10⁻⁴' (as in the plotly notebooks)."""
    return "10" + str(exp).translate(_SUPERSCRIPT)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "gray",
            "legend.facecolor": "white",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": LINE_W,
            "lines.markersize": MARKER_SIZE,
        }
    )


def series_kwargs(name: str) -> dict:
    """Line/marker style of a named series (see :data:`SERIES`)."""
    if name in SERIES:
        color, marker, ls = SERIES[name]
    elif name in ARCH_COLORS:
        color, marker, ls = ARCH_COLORS[name], ARCH_MARKERS[name], ARCH_DASHES[name]
    else:
        raise KeyError(f"unknown series {name!r}")
    return dict(color=color, marker=marker, ls=ls, lw=LINE_W, ms=MARKER_SIZE)


def interior_ticks(values):
    """Label the interior points only (the extremes collide with the frame)."""
    values = sorted(values)
    return values[1:-1] if len(values) > 2 else values


def log_axis(ax, y_lo: float, y_hi: float, x_step: float | None = None) -> None:
    """``plotly_white`` log axis: black frame, light grid, dotted sub-decades."""
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, lw=1.0)
    ax.grid(True, which="minor", axis="y", color=GRID_COLOR, lw=1.0, ls=":")
    e0 = int(np.floor(np.log10(y_lo)))
    e1 = int(np.ceil(np.log10(y_hi)))
    exps = [e for e in range(e0, e1 + 1) if y_lo * 0.999 <= 10.0 ** e <= y_hi * 1.001]
    ax.set_yticks([10.0 ** e for e in exps])
    ax.set_yticklabels([sup_exp(e) for e in exps])
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100)
    )
    if x_step:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_step))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", color="black")


def legend_below(ax, ncol: int, y: float = -0.17) -> None:
    """Single legend under the panel, outside the axes (never on a curve)."""
    handles, labels, seen = [], [], set()
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in seen:
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, framealpha=0.8, edgecolor="gray", facecolor="white",
              borderpad=0.6, labelspacing=0.4, handlelength=2.6)


def legend_below_fig(fig, axes, ncol: int, gap: float = 0.055) -> None:
    """One legend centred under the **whole row** of panels, in its own band.

    ``legend_below`` anchors to a single axes, so on a two- or four-panel figure it
    ends up under one of them and looks off-centre. This variant collects the
    handles of every panel and anchors to the figure instead.

    The vertical position is **measured, not guessed**: the legend hangs just
    below ``min(axes tight bottom) - gap``, where the tight bottom includes the
    tick labels *and* the x labels, so the legend never touches them whatever the
    figure height is. Call it *after* ``fig.tight_layout()``, which is what fixes
    the panel positions being read here.
    """
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    handles, labels, seen = [], [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    bottom = min(
        fig.transFigure.inverted().transform(
            ax.get_tightbbox(fig.canvas.get_renderer()).p0
        )[1]
        for ax in axes
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, bottom - gap),
               ncol=ncol, framealpha=0.8, edgecolor="gray", facecolor="white",
               borderpad=0.6, labelspacing=0.4, handlelength=2.6)


def save(fig, stem: str) -> None:
    """Write ``figures/<stem>.pdf`` and mirror it next to the results.

    No PNG raster copy is written: the deliverables are the vector PDFs only.
    """
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    print("saved", out)
    # Final deliverables live next to the results when that tree exists.
    mirror = REPO / "results" / "figures" / f"{stem}.pdf"
    if mirror.parent.is_dir():
        fig.savefig(mirror, bbox_inches="tight", pad_inches=0.12)
        print("saved", mirror)

