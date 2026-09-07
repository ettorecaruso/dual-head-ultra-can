import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import dataset
from .models import build_model


def _labels(bits, delay):
    sensing = np.stack([delay, np.zeros_like(delay)], axis=-1).astype(np.float32)
    return {"comm_logits": bits, "sensing_out": sensing}


def compile_model(model, learning_rate=1e-3, lambda_sensing=2.0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "comm_logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "sensing_out": keras.losses.MeanSquaredError(),
        },
        loss_weights={"comm_logits": 1.0, "sensing_out": lambda_sensing},
    )
    return model


def fit_model(model, train, val, epochs=30, batch_size=128, patience=10, callbacks=None,
              validation_batch=1.0):
    x_tr, b_tr, d_tr = train
    x_va, b_va, d_va = val
    n_val = max(1, int(len(b_va) * validation_batch))
    if n_val < len(b_va):
        idx = np.random.default_rng(0).choice(len(b_va), size=n_val, replace=False)
        x_va, b_va, d_va = x_va[idx], b_va[idx], d_va[idx]
    history = model.fit(
        x_tr,
        _labels(b_tr, d_tr),
        validation_data=(x_va, _labels(b_va, d_va)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True)
        ],
        verbose=2,
    )
    return history


def train_model(model_type, n_train=105000, n_val=5000, n_echoes=1, epochs=30,
                batch_size=128, seed=None):
    seed = seed if seed is not None else dataset.ARCH_SEEDS[model_type]
    train = dataset.generate_split(n_train, n_echoes, seed)
    val = dataset.generate_split(n_val, n_echoes, seed + 100000)
    model = build_model(model_type)
    compile_model(model)
    fit_model(model, train, val, epochs=epochs, batch_size=batch_size)
    return model
