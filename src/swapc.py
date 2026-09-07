import pandas as pd

from .models import build_model
from .utils import float32_kb

MODELS = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
}


def weight_table():
    rows = []
    for key, name in MODELS.items():
        model = build_model(key)
        params = model.count_params()
        rows.append({"Receiver": name, "Parameters": params,
                     "Memory (kB)": float32_kb(params)})
    return pd.DataFrame(rows)


def load_weight_table():
    frame = weight_table()
    classical = pd.DataFrame([{"Receiver": "Classical (DCSK)", "Parameters": 0,
                               "Memory (kB)": 0.0}])
    return pd.concat([frame, classical], ignore_index=True)
