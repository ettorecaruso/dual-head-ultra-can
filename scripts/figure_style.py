"""Shared publication style for the figures of the Dual-Head Ultra-CAN paper.

The look is the one of ``make_fig_ber_region_zoom.py`` (matplotlib engine), which
reproduces the ``plotly_white`` palette used in the notebooks: white background,
black frame, light major grid, dotted sub-decade grid, Unicode log ticks and a
single shared legend *below* the panels (so it never sits on a curve).

Typical use in a figure script::

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figure_style as fs

    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, ... in ...:
        fs.log_axis(ax, y_lo, y_hi, x_step=2.0)
        ax.plot(x, y, label=..., **fs.series_kwargs("qkv"))
        ax.set_title(...); ax.set_xlabel(...)
    fs.legend_below(axes[1], ncol=2)
    fs.outer_frame(fig, axes)
    fs.save(fig, "my_figure")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[1]

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


def outer_frame(fig, axes) -> None:
    """Close the outer border across the gaps between panels."""
    from matplotlib.patches import Rectangle

    pos = [ax.get_position() for ax in axes]
    x0 = min(p.x0 for p in pos)
    x1 = max(p.x1 for p in pos)
    y0 = min(p.y0 for p in pos)
    y1 = max(p.y1 for p in pos)
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0,
                             transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=1.0,
                             clip_on=False, zorder=10))


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

