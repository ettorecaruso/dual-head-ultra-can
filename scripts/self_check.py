import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import dataset, models, swapc

EXPECTED_COUNTS = {"conv1d": 26597, "qkv": 43172, "lstm": 45988, "mc_dlsk": 44756}
REPORT = []


def check(name, condition, detail=""):
    REPORT.append(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    return condition


def pooled(group):
    mask = group.snr_db >= 5
    return (group.ber[mask] * group.n_symbols[mask]).sum() / group.n_symbols[mask].sum()


def main():
    all_ok = True
    for key, expected in EXPECTED_COUNTS.items():
        model = models.build_model(key)
        all_ok &= check(f"count {key}", model.count_params() == expected,
                        f"(got {model.count_params()})")

    curves = pd.read_csv(ROOT / "results" / "curves" / "ber_curves.csv")
    table = pd.read_csv(ROOT / "results" / "tables" / "table_pooled_ber.csv")
    all_ok &= check("ber_curves loaded", len(curves) > 100, f"rows={len(curves)}")
    scenario = "k1_doppler_full"
    sub = curves[curves.scenario == scenario]
    for model in ["conv1d", "qkv"]:
        value = pooled(sub[sub.model == model])
        ref = float(table.loc[table.scenario == scenario, model].iloc[0])
        all_ok &= check(f"pooled {model}", abs(value - ref) < 1e-6,
                        f"(got {value:.3e}, table {ref:.3e})")

    retention = pd.read_csv(ROOT / "results" / "curves" / "retention.csv")
    all_ok &= check("retention loaded", retention.arch.nunique() == 4)

    jam = pd.read_csv(ROOT / "results" / "curves" / "jamaware_ber.csv")
    row = jam[(jam.jammer == "cw") & (jam.jsr_db == -2)].iloc[0]
    all_ok &= check("jam-aware CW at -2 dB",
                    abs(row.clean_trained - 0.36) < 0.02 and abs(row.jam_aware - 0.007) < 0.002,
                    f"(clean {row.clean_trained:.4f} -> {row.jam_aware:.4f})")

    seed = dataset.ARCH_SEEDS["conv1d"]
    train = dataset.generate_split(600, 1, seed)
    val = dataset.generate_split(100, 1, seed + 100000)
    from src import training

    model = models.build_model("conv1d")
    training.compile_model(model)
    history = training.fit_model(model, train, val, epochs=1, batch_size=64)
    last = float(history.history["loss"][-1])
    all_ok &= check("smoke training conv1d", np.isfinite(last), f"(loss {last:.3f})")

    print("\n".join(REPORT))
    print("SELF_CHECK", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
