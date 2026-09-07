import numpy as np
import tensorflow as tf
import pandas as pd

from . import channel, dataset, training
from .evaluation import ber_from_logits, predict


def jammed_batch(model, jammer, jsr_db, snr_db, n_echoes, base_seed, num_symbols=8192):
    x, bits, delay = dataset.generate_batch(num_symbols, snr_db, n_echoes, base_seed)
    x_j = channel.apply_jamming(x, jammer, jsr_db, base_seed + 1)
    logits, _ = predict(model, x_j)
    return ber_from_logits(logits, bits), x, x_j


def ber_vs_jsr(model, jammer, jsrs, snr_db=11.0, n_echoes=1, base_seed=42, num_symbols=8192):
    rows = []
    for jsr in jsrs:
        ber, _, _ = jammed_batch(model, jammer, jsr, snr_db, n_echoes, base_seed, num_symbols)
        rows.append({"jammer": jammer, "jsr_db": jsr, "ber": ber})
    return pd.DataFrame(rows)


def layer_outputs(model, x, names):
    layers = [model.get_layer(n) for n in names]
    extractor = tf.keras.Model(inputs=model.input, outputs=[l.output for l in layers])
    return extractor(x, training=False)


def retention_cosine(model, clean_x, jammed_x, names):
    clean = layer_outputs(model, clean_x, names)
    jammed = layer_outputs(model, jammed_x, names)
    rows = []
    for name, a, b in zip(names, clean, jammed):
        a = tf.reshape(a, (tf.shape(a)[0], -1))
        b = tf.reshape(b, (tf.shape(b)[0], -1))
        sim = tf.reduce_sum(a * b, axis=1) / (
            tf.norm(a, axis=1) * tf.norm(b, axis=1) + 1e-8
        )
        rows.append({name: float(tf.reduce_mean(sim).numpy())})
    return pd.DataFrame(rows)


def train_jamming_aware(model_type, seed, jammer_pool=("cw", "barrage", "partial_band"),
                        jsr_pool=(-6, -2, 2), epochs=10, n_train=20000, n_val=3000,
                        n_echoes=1, snr_db=11.0):
    train = dataset.generate_split(n_train, n_echoes, seed)
    val = dataset.generate_split(n_val, n_echoes, seed + 100000)
    x_tr, b_tr, d_tr = train
    half = len(b_tr) // 2
    rng = np.random.default_rng(seed)
    for i in range(half):
        jammer = jammer_pool[int(rng.integers(0, len(jammer_pool)))]
        jsr = float(jsr_pool[int(rng.integers(0, len(jsr_pool)))])
        x_tr[i] = channel.apply_jamming(x_tr[i][None], jammer, jsr, int(rng.integers(0, 2**31)))[0]
    model = training.build_model(model_type)
    training.compile_model(model)
    training.fit_model(model, (x_tr, b_tr, d_tr), val, epochs=epochs)
    return model
