#!/usr/bin/env python3
"""Cross-channel generalization table: pooled BER and minimum SNR at the target BER."""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "full" / "channel_generalization"
BASELINE = REPO / "results" / "full" / "ber_vs_snr"
OUT = REPO / "results" / "full" / "channel_generalization" / "generalization_table.tex"

ARCHS = ["conv1d", "qkv", "lstm", "mc_dlsk"]
LABELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN-QKV",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}
BASELINE_SCENARIO = "k3_doppler_full"
OPERATING_SNR_DB = 5.0
TARGET_BER = 1e-4


def _pooled(metrics: pd.DataFrame) -> float:
    sub = metrics[metrics["snr_db"] >= OPERATING_SNR_DB - 1e-9]
    errors = float(sub["n_errors"].sum())
    symbols = float(sub["n_symbols"].sum())
    return errors / symbols if symbols > 0.0 else float("nan")


def _min_snr(metrics: pd.DataFrame) -> float:
    for _, row in metrics.sort_values("snr_db").iterrows():
        if float(row["ber"]) <= TARGET_BER:
            return float(row["snr_db"])
    return float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    rows = []
    for arch in ARCHS:
        pair = {
            "arch": arch,
            "variant": "nominal (train = test)",
            "pooled_ber": float("nan"),
            "min_snr_at_1e-4": float("nan"),
        }
        baseline_metrics = BASELINE / BASELINE_SCENARIO / arch / "metrics.csv"
        if baseline_metrics.is_file():
            metrics = pd.read_csv(baseline_metrics)
            pair["pooled_ber"] = _pooled(metrics)
            pair["min_snr_at_1e-4"] = _min_snr(metrics)
        rows.append(pair)

        summary_path = RESULTS / arch / "summary.csv"
        if not summary_path.is_file():
            continue
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            rows.append({
                "arch": arch,
                "variant": str(row["variant"]),
                "pooled_ber": float(row["pooled_ber"]),
                "min_snr_at_1e-4": float(row["min_snr_at_1e-4"]),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "generalization_table.csv", index=False)

    variants = [v for v in frame["variant"].dropna().unique() if v != "nominal (train = test)"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cross-channel generalization of the frozen receivers (no retraining): "
        "pooled BER over the operating region and minimum SNR at BER $\\le 10^{-4}$.}",
        "\\label{tab:generalization}",
        "\\begin{tabular}{l" + "cc" * len(ARCHS) + "}",
        "\\toprule",
        " & " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{LABELS[a]}}}" for a in ARCHS
        ) + " \\\\",
        " & " + " & ".join("Pooled BER & min SNR" for _ in ARCHS) + " \\\\",
        "\\midrule",
    ]

    def _line(name: str) -> str:
        cells = []
        for arch in ARCHS:
            match = frame[(frame["arch"] == arch) & (frame["variant"] == name)]
            if match.empty:
                cells.extend(["n/a", "n/a"])
            else:
                row = match.iloc[0]
                cells.append(f"${_fmt(float(row['pooled_ber']), 6)}$")
                min_snr = float(row["min_snr_at_1e-4"])
                cells.append(f"${min_snr:.0f}$" if np.isfinite(min_snr) else "n/a")
        return name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\"

    lines.append(_line("nominal (train = test)"))
    lines.append("\\midrule")
    for variant in variants:
        lines.append(_line(variant))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("saved", OUT)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
