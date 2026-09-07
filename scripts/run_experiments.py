import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import dataset, models, swapc, training, utils

MODELS = ["conv1d", "qkv", "lstm", "mc_dlsk"]


def build_and_report():
    rows = []
    for key in MODELS:
        model = models.build_model(key)
        rows.append({"model": key, "params": model.count_params(),
                     "memory_kb": utils.float32_kb(model.count_params())})
    print(pd.DataFrame(rows).to_string(index=False))


def export_swapc():
    frame = swapc.load_weight_table()
    out_csv = ROOT / "results" / "tables" / "table_swapc.csv"
    out_tex = ROOT / "results" / "tables" / "table_swapc.tex"
    utils.save_csv(out_csv, frame)
    tex = frame.to_latex(index=False)
    out_tex.write_text(tex, encoding="utf-8")
    print("swapc written to", out_csv)


def smoke_train(model_type="conv1d", n_train=2000, n_val=400, epochs=1):
    seed = dataset.ARCH_SEEDS[model_type]
    model = training.train_model(model_type, n_train=n_train, n_val=n_val,
                                 n_echoes=1, epochs=epochs, seed=seed)
    print(model_type, "params", model.count_params(), "smoke done")


def main():
    parser = argparse.ArgumentParser(description="Dual-Head Ultra-CAN experiments")
    parser.add_argument("command", choices=["build", "swapc", "smoke"])
    parser.add_argument("--model", default="conv1d")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "build":
        build_and_report()
    elif args.command == "swapc":
        export_swapc()
    elif args.command == "smoke":
        smoke_train(args.model, n_train=args.samples, epochs=args.epochs)


if __name__ == "__main__":
    main()
