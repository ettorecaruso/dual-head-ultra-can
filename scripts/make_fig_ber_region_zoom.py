#!/usr/bin/env python3
"""Operating-region BER figure (SNR >= 5 dB, log scale, publication style).

Produces a combined three-panel figure, one panel per K scenario with the
legend repeated in every panel, plus one standalone figure per scenario.
Inputs are read from ``results/full/ber_vs_snr`` and fall back to
``results_old/full/ber_vs_snr``.
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
SCENARIOS = [
    ("k1_doppler_full", "k1_single_echo", "Single echo (K = 1)"),
    ("k3_doppler_full", "k3_multi_echo", "Multi-echo (K = 3)"),
    ("k3_doppler_limited", "k3_low_doppler", "Multi-echo, low Doppler (K = 3)"),
]
MIN_DB = 5.0
TARGET = 1e-4
Y_BOTTOM = 4e-5
Y_TOP = 1.2e-3


def _results_root() -> Path:
    """Locate the results tree, tolerating the ``results_old`` archive."""
    for candidate in (REPO / "results" / "full", REPO / "results_old" / "full"):
        if candidate.is_dir():
            return candidate
    return REPO / "results" / "full"


RESULTS = _results_root() / "ber_vs_snr"


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup_exp(exp: int) -> str:
    """Unicode log label, e.g. -4 -> '10⁻⁴' (as in the plotly notebooks)."""
    return "10" + str(exp).translate(_SUPERSCRIPT)


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


def _siino_axis(ax, y_lo: float, y_hi: float, x_step: float = 2.0) -> None:
    """plotly_white log axis: black frame, light grid, dotted sub-decades."""
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which="major", axis="both", color="#D3D3D3", lw=1.0)
    ax.grid(True, which="minor", axis="y", color="#D3D3D3", lw=1.0, ls=":")
    e0 = int(np.floor(np.log10(y_lo)))
    e1 = int(np.ceil(np.log10(y_hi)))
    exps = [e for e in range(e0, e1 + 1)
            if y_lo * 0.999 <= 10.0 ** e <= y_hi * 1.001]
    ax.set_yticks([10.0 ** e for e in exps])
    ax.set_yticklabels([_sup_exp(e) for e in exps])
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100)
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_step))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", color="black")


def _curve(scenario: str) -> dict:
    curves = {}
    for arch in ARCHS:
        path = RESULTS / scenario / arch / "metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["snr_db"] >= MIN_DB].sort_values("snr_db")
        if len(df):
            curves[arch] = df
    return curves


def _draw_panel(ax, curves: dict, title: str, legend_title=None) -> None:
    """One operating-region panel with the plotly_white look."""
    _siino_axis(ax, Y_BOTTOM, Y_TOP)
    ax.axhspan(Y_BOTTOM, TARGET, color="#F0FBF5", zorder=0)
    ax.axhline(TARGET, color="#00CC96", lw=1.2, ls="--", zorder=1,
               label="target BER = 10⁻⁴")
    for arch in ARCHS:
        if arch not in curves:
            continue
        df = curves[arch]
        ax.plot(df["snr_db"], df["ber"], label=LABELS[arch],
                zorder=3, **_model_kwargs(arch))
    ax.set_title(title, pad=8)
    ax.set_xlabel("SNR (dB)")
    xs = sorted({float(v) for df in curves.values() for v in df["snr_db"]})
    if xs:
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xticks(_x_ticks(xs))
    ax.legend(loc="upper right", title=legend_title, framealpha=0.8,
              edgecolor="gray", facecolor="white", borderpad=0.6,
              labelspacing=0.4, handlelength=2.6)


def _save(fig, stem: str) -> None:
    fig_dir = REPO / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    print("saved", out)


def plot_region_zoom_plotly() -> None:
    """Same operating-region figures rendered with Plotly (notebook look)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import plotly_style as ps

    curves = {scenario: _curve(scenario) for scenario, _, _ in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no metrics found under", RESULTS)
        return

    scenarios = [(s, sl, t) for s, sl, t in SCENARIOS if curves.get(s)]
    xs = sorted({float(v) for c in curves.values() for df in c.values()
                 for v in df["snr_db"]})
    ticks = ps.interior_ticks(xs)
    ydict = ps.log_yaxis(Y_BOTTOM, Y_TOP)
    ydict_plain = {k: v for k, v in ydict.items() if k != "title"}

    def _add(fig, scenario, x_ticks, first, **kw):
        fig.add_hrect(y0=Y_BOTTOM, y1=TARGET, fillcolor="#F0FBF5",
                      line_width=0, layer="below", **kw)
        fig.add_trace(ps.scatter([min(x_ticks), max(x_ticks)], [TARGET, TARGET],
                                 "target BER = 10⁻⁴", "qkv", dash="dash",
                                 color="#00CC96", width=1.2,
                                 showlegend=first), **kw)
        for arch in ARCHS:
            if arch in curves[scenario]:
                df = curves[scenario][arch]
                fig.add_trace(ps.scatter(df["snr_db"], df["ber"], LABELS[arch],
                                         arch, showlegend=first), **kw)

    # ---- combined three-panel figure ------------------------------------
    fig = make_subplots(rows=1, cols=len(scenarios), shared_yaxes=True,
                        horizontal_spacing=0.04,
                        subplot_titles=[t for _, _, t in scenarios])
    for col, (scenario, _, _) in enumerate(scenarios, start=1):
        _add(fig, scenario, ticks, first=(col == 1), row=1, col=col)
    fig.update_layout(template="plotly_white", width=1240, height=460,
                      font=dict(size=13), legend=ps.legend("upper right"),
                      margin=dict(l=70, r=20, t=50, b=60),
                      shapes=(ps.minor_grid_shapes(Y_BOTTOM, Y_TOP)
                              + [ps.frame_shape()]))
    for col in range(1, len(scenarios) + 1):
        fig.update_yaxes(ydict if col == 1 else ydict_plain, row=1, col=col)
        fig.update_xaxes(ps.xaxis(ticks), row=1, col=col)
    ps.write(fig, REPO / "figures" / "plotly" / "ber_region_zoom.pdf",
             REPO / "figures" / "plotly" / "ber_region_zoom.html")

    # ---- one square figure per K ----------------------------------------
    for scenario, slug, _ in scenarios:
        fig = go.Figure()
        _add(fig, scenario, ticks, first=True)
        fig.update_layout(template="plotly_white", width=610, height=610,
                          font=dict(size=14), legend=ps.legend("upper right"),
                          margin=dict(l=70, r=20, t=40, b=60),
                          shapes=(ps.minor_grid_shapes(Y_BOTTOM, Y_TOP)
                                  + [ps.frame_shape()]))
        fig.update_yaxes(ydict)
        fig.update_xaxes(ps.xaxis(ticks))
        stem = f"ber_region_zoom_{slug}"
        ps.write(fig, REPO / "figures" / "plotly" / f"{stem}.pdf",
                 REPO / "figures" / "plotly" / f"{stem}.html")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["matplotlib", "plotly"],
                    default="matplotlib",
                    help="plotly reproduces the paper-notebook look (PDF + HTML)")
    args = ap.parse_args(argv)
    if args.engine == "plotly":
        plot_region_zoom_plotly()
        return

    _apply_style()
    curves = {scenario: _curve(scenario) for scenario, _, _ in SCENARIOS}
    if not any(curves.values()):
        print("[warn] no metrics found under", RESULTS)
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, (scenario, _, title) in zip(axes, SCENARIOS):
        _draw_panel(ax, curves.get(scenario, {}), title)
    axes[0].set_ylabel("BER")
    fig.tight_layout()
    _outer_frame(fig, axes)
    _save(fig, "ber_region_zoom")
    plt.close(fig)

    for scenario, slug, title in SCENARIOS:
        if not curves.get(scenario):
            continue
        fig, ax = plt.subplots(figsize=(6.1, 6.1))
        _draw_panel(ax, curves[scenario], title, legend_title="Models")
        ax.set_ylabel("BER")
        fig.tight_layout()
        _save(fig, f"ber_region_zoom_{slug}")
        plt.close(fig)


if __name__ == "__main__":
    main()
