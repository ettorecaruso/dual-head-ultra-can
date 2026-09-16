#!/usr/bin/env python3
"""Delay-estimation figure (single-echo scenario, publication style).

One panel is produced: the delay correlation, one curve per architecture.  The
delay-RMSE panel is not used by the paper (the caption and the text discuss the
correlation only), so it is not drawn.  Inputs are read from
``results/full/ber_vs_snr``.
"""
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
SCENARIO = "k1_doppler_full"
ARCHS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
# "plotly_white" curve language (M. Siino's notebooks): plotly palette,
# big markers (plotly size=8), width=1.5 lines, dash+symbol cycled, no
# white marker edge.
PLOTLY_COLORS = {
    "conv1d": "#636EFA",
    "qkv": "#EF553B",
    "lstm": "#00CC96",
    "mc_dlsk": "#AB63FA",
}
PLOTLY_MARKERS = {"conv1d": "o", "qkv": "s", "lstm": "D", "mc_dlsk": "+"}
PLOTLY_DASHES = {
    "conv1d": "-",
    "qkv": (0, (6, 2)),
    "lstm": (0, (1, 1.6)),
    "mc_dlsk": (0, (4, 1.2, 1, 1.2)),
}
LINE_W = 1.5
MARKER_SIZE = 6.5


def _model_kwargs(arch: str) -> dict:
    return dict(color=PLOTLY_COLORS[arch], marker=PLOTLY_MARKERS[arch],
                ls=PLOTLY_DASHES[arch], lw=LINE_W, ms=MARKER_SIZE)


def _results_root() -> Path:
    """Locate the results tree produced by the runner (``results/full``)."""
    return REPO / "results" / "full"


RESULTS = _results_root() / "ber_vs_snr" / SCENARIO


def _apply_style() -> None:
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
            "grid.color": "#D3D3D3",
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


def _x_ticks(values):
    """Label the SNR axis with the interior points only.

    The first and last SNR values sit on the axis corners (their labels collide
    with the frame), and the last two points (19, 20 dB) are only 1 dB apart.
    Dropping the extremes leaves evenly spaced ticks.
    """
    values = sorted(values)
    return values[1:-1] if len(values) > 2 else values


def _outer_frame(fig, axes) -> None:
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


def _siino_frame(ax, x_step: float = 2.0) -> None:
    """plotly_white frame: black box, light major grid, dotted minor grid."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color="#D3D3D3", lw=1.0)
    ax.grid(True, which="minor", axis="both", color="#D3D3D3", lw=1.0, ls=":")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_step))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", color="black")


def _load() -> dict:
    curves = {}
    for arch in ARCHS:
        path = RESULTS / arch / "metrics.csv"
        if path.exists():
            curves[arch] = pd.read_csv(path).sort_values("snr_db")
    return curves


def plot_sensing_plotly() -> None:
    """Same delay figures rendered with Plotly (paper-notebook look)."""
    from plotly.subplots import make_subplots

    import plotly_style as ps

    curves = _load()
    if not curves:
        print("[warn] no metrics found under", RESULTS)
        return

    xs = sorted({float(v) for df in curves.values() for v in df["snr_db"]})
    ticks = ps.interior_ticks(xs)

    fig = make_subplots(rows=1, cols=1, subplot_titles=["Delay correlation"])
    for arch, df in curves.items():
        fig.add_trace(ps.scatter(df["snr_db"], df["corr_tau"], LABELS[arch], arch),
                      row=1, col=1)
    fig.update_layout(template="plotly_white", width=610, height=520,
                      font=dict(size=13), legend=ps.legend("lower right"),
                      margin=dict(l=70, r=20, t=50, b=60),
                      shapes=[ps.frame_shape()])
    fig.update_yaxes(ps.linear_yaxis("corr(τ̂, τ)"), row=1, col=1)
    fig.update_xaxes(ps.xaxis(ticks), row=1, col=1)
    ps.write(fig, REPO / "figures" / "plotly" / "sensing_delay_single.pdf",
             REPO / "figures" / "plotly" / "sensing_delay_single.html")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["matplotlib", "plotly"],
                    default="matplotlib",
                    help="plotly reproduces the paper-notebook look (PDF + HTML)")
    args = ap.parse_args(argv)
    if args.engine == "plotly":
        plot_sensing_plotly()
        return

    _apply_style()
    curves = _load()
    if not curves:
        print("[warn] no metrics found under", RESULTS)
        return

    fig, ax_corr = plt.subplots(figsize=(5.8, 4.6))
    corr_all = []
    for arch, df in curves.items():
        ax_corr.plot(df["snr_db"], df["corr_tau"], label=LABELS[arch],
                     **_model_kwargs(arch))
        corr_all.extend(df["corr_tau"].to_numpy(dtype=float))

    ax_corr.set_title("Delay correlation", pad=8)
    ax_corr.set_xlabel("SNR (dB)")
    ax_corr.set_ylabel(r"corr($\hat{\tau}$, $\tau$)")
    ax_corr.set_ylim((min(corr_all) - 0.05) if corr_all else 0.0, 1.0)

    _siino_frame(ax_corr)
    ax_corr.margins(x=0.03)

    xs = sorted({float(v) for df in curves.values() for v in df["snr_db"]})
    if xs:
        ax_corr.set_xlim(xs[0], xs[-1])
        ax_corr.set_xticks(_x_ticks(xs))

    ax_corr.legend(loc="lower right", title="Models", framealpha=0.8,
                   edgecolor="gray", facecolor="white", borderpad=0.6,
                   labelspacing=0.4, handlelength=2.6)

    fig.tight_layout()
    _outer_frame(fig, (ax_corr,))
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sensing_delay_single.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    print("saved", out)
    # Final deliverables live next to the results when that tree exists.
    mirror = REPO / "results" / "figures" / out.name
    if mirror.parent.is_dir():
        fig.savefig(mirror, bbox_inches="tight", pad_inches=0.12)
        print("saved", mirror)
    plt.close(fig)


if __name__ == "__main__":
    main()
