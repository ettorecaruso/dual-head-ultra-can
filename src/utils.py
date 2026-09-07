import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npy(path, array) -> None:
    ensure_dir(Path(path).parent)
    np.save(path, array)


def load_npy(path):
    return np.load(path)


def save_csv(path, frame) -> None:
    ensure_dir(Path(path).parent)
    frame.to_csv(path, index=False)


def load_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def float32_kb(params: int) -> float:
    return params * 4.0 / 1024.0
