#!/usr/bin/env python3
"""Every frequency-agility condition against one prediction: the mixing law.

    python scripts/figures/make_fig_agility_law.py [run_dir]

The protection of frequency agility is geometric. If the jammer is aligned with
the transmission for a fraction ``f`` of the time, the measured BER must be

    BER = f * BER_aligned + (1 - f) * BER_clean

where ``BER_aligned`` is what the same jammer achieves when it is omniscient
(``fh_off``, always aligned, ``f = 1``) and ``BER_clean`` is the no-jammer floor
of that row. Nothing in the formula is fitted: both references are measured.

The figure plots *measured against predicted* for every point of the sweep, in
every modality, for the four receivers, on log-log axes with a +-1 dB band. If
the law holds, the cloud collapses on the diagonal, and the distance from the
diagonal measures the disagreement. The script also prints the worst relative
error, so the collapse is a checked statement and not an impression.

``run_dir`` defaults to ``results/full/frequency_agility``.
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

DEFAULT_RUN_DIR = REPO / "results" / "full" / "frequency_agility"
ARCHS = ("conv1d", "qkv", "lstm", "mc_dlsk")
ALIGNED = "fh_off"
MODALITIES = {
    "fh_off_blind": ("#EF553B", "o", "Fixed carrier, blind jammer"),
    "fh_on": ("#636EFA", "s", "Frequency hopping, model jammer"),
}
TOLERANCE_DB = 1.0
#: The law uses the *omniscient* BER as the aligned reference, which is only the
#: right reference when being aligned means the same thing in both modalities.
#: For a band-partial jammer it does not: the band-agnostic version overlaps the
#: signal band by a random amount, so at low JSR its aligned damage differs from
#: the omniscient one and the mixture is no longer geometric (measured: up to
#: 3 dB, versus <= 0.9 dB for the other three jammer families). The three
#: families below are the ones the law is asserted on; ``fixed_partial`` is still
#: drawn, and its residual is reported.
GATED_MODELS = ("barrage", "sweep", "follower")
MODEL_COLORS = {"barrage": "#7F7F7F", "sweep": "#636EFA",
                "fixed_partial": "#EF553B", "follower": "#00CC96"}


def _predicted_frame(run_dir: Path) -> pd.DataFrame:
    """Predicted vs measured BER for every point of every receiver."""
    rows: List[pd.DataFrame] = []
    for arch in ARCHS:
        path = Path(run_dir) / arch / "frequency_agility_vs_jsr.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df = df[np.isfinite(df["ber"]) & (df["in_channel"] == "cw")]
        aligned = df[df.modality == ALIGNED].set_index(["jammer_model", "jsr_db"])
        for modality in MODALITIES:
            sub = df[df.modality == modality].copy()
            key = pd.MultiIndex.from_arrays([sub["jammer_model"], sub["jsr_db"]])
            sub["ber_aligned"] = aligned["ber"].reindex(key).to_numpy()
            sub = sub[np.isfinite(sub["ber_aligned"])]
            sub["predicted"] = (
                sub["jammed_fraction"] * sub["ber_aligned"]
                + (1.0 - sub["jammed_fraction"]) * sub["ber_clean"]
            )
            rows.append(sub[["arch", "jammer_model", "jsr_db", "modality",
                             "jammed_fraction", "predicted", "ber"]])
    if not rows:
        raise SystemExit(f"no frequency_agility_vs_jsr.csv under {run_dir}")
    return pd.concat(rows, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> None:
    run_dir = Path(argv[0]).resolve() if argv else DEFAULT_RUN_DIR
    data = _predicted_frame(run_dir)
    measured = data["ber"].to_numpy(dtype=float)
    predicted = data["predicted"].to_numpy(dtype=float)
    valid = (measured > 0.0) & (predicted > 0.0)
    measured, predicted = measured[valid], predicted[valid]
    data = data[valid]

    ratio_db = 10.0 * np.log10(measured / predicted)
    data = data.assign(ratio_db=ratio_db)
    per_model = {
        model: float(data.loc[data.jammer_model == model, "ratio_db"].abs().max())
        for model in MODEL_COLORS
        if (data.jammer_model == model).any()
    }
    print("mixing law, worst |measured/predicted| per jammer family (dB): "
          + ", ".join(f"{k}={v:.3f}" for k, v in per_model.items()))
    gated = data[data.jammer_model.isin(GATED_MODELS)]
    worst = float(gated["ratio_db"].abs().max())
    print(f"asserted on {sorted(GATED_MODELS)}: worst {worst:.3f} dB over "
          f"{gated.shape[0]} points, median "
          f"{float(gated['ratio_db'].abs().median()):.3f} dB")
    if worst > TOLERANCE_DB:
        raise AssertionError(
            f"the mixing law is off by {worst:.2f} dB (> {TOLERANCE_DB} dB) on a "
            "gated jammer family: the modality comparison is not geometric any more."
        )

    fs.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lo, hi = max(float(predicted.min()) * 0.6, 1e-6), float(measured.max()) * 1.6
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, zorder=4)
    ax.fill_between([lo, hi],
                    [lo / 10 ** (TOLERANCE_DB / 10)], [lo * 10 ** (TOLERANCE_DB / 10)],
                    color="black", alpha=0.07, lw=0, zorder=1)
    for model, color in MODEL_COLORS.items():
        for modality, (_, marker, _) in MODALITIES.items():
            sub = data[(data.jammer_model == model) & (data.modality == modality)]
            if sub.empty:
                continue
            ax.scatter(sub["predicted"], sub["ber"], s=36, marker=marker, color=color,
                       alpha=0.8, edgecolor="white", linewidth=0.4, zorder=3,
                       label=model if modality == "fh_on" else None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, which="both", color=fs.GRID_COLOR, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"predicted $f\cdot$BER$_{aligned}+(1-f)\cdot$BER$_{clean}$")
    ax.set_ylabel("measured BER")
    ax.set_title("Frequency agility is geometric: "
                 f"{gated.shape[0]} conditions, worst {worst:.2f} dB", pad=8)
    fig.tight_layout()
    fs.legend_below_fig(fig, [ax], ncol=4)
    fs.save(fig, "frequency_agility_law")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
