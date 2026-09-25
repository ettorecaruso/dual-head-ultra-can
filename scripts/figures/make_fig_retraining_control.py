#!/usr/bin/env python3
"""Does retraining on the new channel bring the receiver back? (the control)

    python scripts/figures/make_fig_retraining_control.py [run_root]

``channel_generalization`` leaves the frozen receivers on a BER floor of about
0.15 on TDL-D: no SNR recovers them, because a channel mismatch is a bias and not
extra noise. This figure is the control that closes that argument -- same channel,
same evaluation protocol, but with the heads retrained in-domain on TDL-D
(``ber_vs_snr/k3_doppler_full_b_tdl_d``, the run seeded from the ``k3_doppler_full``
config with the channel replaced by the ``b_tdl_d`` variant of
``configs/channels.yaml``).

Three arms per panel, all read from the CSVs:

* nominal channel, frozen head -- the in-domain reference of the paper;
* TDL-D, frozen head -- the mismatched arm of ``channel_generalization``;
* TDL-D, retrained head -- the control.

The retraining does what the mismatch should require: it removes the floor and
lands on the in-domain reference (three orders of magnitude lower). That is also
what makes the companion ``sensing_observability`` readable -- recovering the
communication head does *not* recover the delay correlation, whose collapse is a
property of the label (see ``diagnostics/channel_label_observability``) and not of
the receiver.

Both claims are checked against the CSVs before drawing, so the control cannot
silently stop being a control.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import figure_style as fs  # noqa: E402

DEFAULT_RUN_ROOT = REPO / "results" / "full"
#: Only the two architectures the retraining was run for.
ARCHS = ("conv1d", "qkv")
ARCH_LABELS = {"conv1d": "Ultra-CAN (Conv1D)", "qkv": "Ultra-CAN-QKV"}
VARIANT = "b_tdl_d"
#: (sub-tree, variant sub-directory). The cross-channel tree puts the variant
#: between the architecture and the file; the ``ber_vs_snr`` tree has no variant.
IN_DOMAIN = (("ber_vs_snr", "k3_doppler_full"), None)
RETRAINED = (("ber_vs_snr", "k3_doppler_full_b_tdl_d"), None)
FROZEN = (("channel_generalization",), VARIANT)
Y_LO, Y_HI = 5e-5, 1.0
#: 14 samples at 2 dB would collide in a half-width panel.
X_STEP = 4.0
#: The control must pull the floor at least this far down, and land this close to
#: the in-domain floor, or it stops being the control of the section.
MIN_FLOOR_RATIO = 100.0
MAX_GAP_TO_IN_DOMAIN = 3.0
#: Red is the frozen receiver and blue the one retrained on the offending channel,
#: the same pair as in ``jamming_aware_control``; grey is the in-domain reference.
ARMS = (
    ("in_domain", "nominal channel, frozen head",
     {"color": "#7F7F7F", "marker": "o", "ls": (0, (1, 1.6))}),
    ("frozen", "TDL-D, frozen head",
     {"color": "#EF553B", "marker": "o", "ls": "-"}),
    ("retrained", "TDL-D, retrained head",
     {"color": "#636EFA", "marker": "s", "ls": (0, (6, 2))}),
)


def _tex(value: float) -> str:
    """Two significant digits in mathtext, e.g. 1.457e-04 -> $1.5\\times10^{-4}$."""
    exp = int(np.floor(np.log10(value)))
    return rf"${value / 10.0 ** exp:.1f}\times10^{{{exp}}}$"


def _curve(run_root: Path, arch: str, tree: tuple,
           variant: Optional[str]) -> Optional[pd.DataFrame]:
    path = Path(run_root).joinpath(*tree) / arch
    if variant:
        path = path / variant
    path = path / "metrics.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if variant and "channel_variant" in frame.columns:
        frame = frame[frame["channel_variant"] == variant]
    return frame[np.isfinite(frame["ber"])].sort_values("snr_db")


def _floors(curves: Dict[str, Dict[str, pd.DataFrame]], arch: str) -> Dict[str, float]:
    """The BER floor of each arm: the best BER it reaches anywhere in the sweep."""
    return {key: float(curves[key][arch]["ber"].min()) for key, _, _ in ARMS}


def main(argv: Optional[List[str]] = None) -> None:
    run_root = Path(argv[0]).resolve() if argv else DEFAULT_RUN_ROOT
    curves: Dict[str, Dict[str, pd.DataFrame]] = {key: {} for key, _, _ in ARMS}
    for arch in ARCHS:
        for key, spec in (("in_domain", IN_DOMAIN), ("frozen", FROZEN),
                          ("retrained", RETRAINED)):
            frame = _curve(run_root, arch, *spec)
            if frame is not None and not frame.empty:
                curves[key][arch] = frame

    empty = [key for key, _, _ in ARMS if not curves[key]]
    if empty:
        raise SystemExit(
            f"missing the arm(s) {empty} under {run_root}: the control needs "
            "ber_vs_snr/k3_doppler_full, ber_vs_snr/k3_doppler_full_b_tdl_d and "
            f"channel_generalization/<arch>/{VARIANT}"
        )

    floors = {arch: _floors(curves, arch) for arch in ARCHS}
    for arch in ARCHS:
        floor = floors[arch]
        ratio = floor["frozen"] / floor["retrained"]
        if ratio < MIN_FLOOR_RATIO:
            raise AssertionError(
                f"{arch}: retraining on TDL-D lowers the BER floor only "
                f"{ratio:.1f}x ({floor['frozen']:.4f} -> {floor['retrained']:.4f}), "
                f"expected at least {MIN_FLOOR_RATIO:.0f}x"
            )
        gap = floor["retrained"] / floor["in_domain"]
        if gap > MAX_GAP_TO_IN_DOMAIN:
            raise AssertionError(
                f"{arch}: the retrained floor {floor['retrained']:.3e} is {gap:.1f}x "
                f"the in-domain floor {floor['in_domain']:.3e}: the retraining did "
                "not land on the reference"
            )
        print(f"{arch}: floor {floor['frozen']:.4f} -> {floor['retrained']:.4f} "
              f"({ratio:.0f}x lower), in-domain {floor['in_domain']:.4f} "
              f"({gap:.2f}x above it)")

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        fs.log_axis(ax, Y_LO, Y_HI, x_step=X_STEP)
        for key, label, style in ARMS:
            frame = curves[key].get(arch)
            if frame is None:
                continue
            ax.plot(frame["snr_db"], frame["ber"], label=label, zorder=3, **style)
        ax.set_title(ARCH_LABELS[arch], pad=8)
        ax.set_xlabel("SNR (dB)")
        floor = floors[arch]
        # In the band between the flat frozen tail (BER ~ 0.15) and the top frame,
        # which both panels leave empty.
        ax.annotate("frozen on TDL-D: floor " + _tex(floor["frozen"]) + "\n"
                    "retrained: " + _tex(floor["retrained"]) + " ("
                    + _tex(floor["frozen"] / floor["retrained"]) + " lower)",
                    (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#333333")
    axes[0].set_ylabel("BER")
    fig.suptitle("Retraining the heads on the new channel removes the BER floor "
                 "of the frozen receivers (TDL-D, 25% clutter, K = 3)", y=1.02)
    fig.tight_layout()
    fs.legend_below_fig(fig, axes, ncol=3)
    fs.save(fig, "retraining_control")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
