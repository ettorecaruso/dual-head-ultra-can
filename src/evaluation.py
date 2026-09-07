import numpy as np
import pandas as pd

from . import channel, dataset
from .utils import float32_kb


def predict(model, x):
    out = model.predict(x, verbose=0)
    if isinstance(out, list):
        return out[0], out[1]
    if isinstance(out, dict):
        names = list(model.output_names)
        return out[names[0]], out[names[1]]
    return out, None


def ber_from_logits(logits, bits):
    pred = np.argmax(logits, axis=-1).astype(np.int32)
    return float(np.mean(pred != bits))


def evaluate_at_snr(model, snr_db, n_echoes, base_seed, error_threshold=100,
                    max_symbols=2_000_000, chunk=4096, correlator=None):
    rng = np.random.default_rng(base_seed)
    errors = 0
    total = 0
    seed_counter = base_seed
    while errors < error_threshold and total < max_symbols:
        size = min(chunk, max_symbols - total)
        x, bits, delay = dataset.generate_batch(size, snr_db, n_echoes, seed_counter, bit_rng=rng)
        logits, _ = predict(model, x)
        pred = np.argmax(logits, axis=-1).astype(np.int32)
        errors += int(np.sum(pred != bits))
        total += size
        seed_counter += size * 7919 + int(snr_db) * 104729
    ber = errors / float(total) if total else 1.0
    return ber, total


def ber_curve(model, snrs=None, n_echoes=1, base_seed=42, correlator=None):
    snrs = snrs or dataset.SNR_GRID
    rows = []
    for snr in snrs:
        ber, total = evaluate_at_snr(model, snr, n_echoes, base_seed, correlator=correlator)
        rows.append({"snr_db": snr, "ber": ber, "symbols": total})
    return pd.DataFrame(rows)


def pooled_ber(df, min_snr=5):
    mask = df.snr_db >= min_snr
    errs = (df.ber[mask] * df.symbols[mask]).sum()
    syms = df.symbols[mask].sum()
    return float(errs / syms)


def minimum_snr_at_floor(df, floor=1e-4):
    below = df[df.ber <= floor]
    return float(below.snr_db.min()) if len(below) else float("nan")


def delay_correlation(model, snr_db, n_echoes, base_seed, num_symbols=4096):
    x, bits, delay = dataset.generate_batch(num_symbols, snr_db, n_echoes, base_seed)
    _, sensing = predict(model, x)
    predicted = sensing[:, 0]
    return float(np.corrcoef(predicted, delay)[0, 1])


def sensing_curve(model, snrs=None, n_echoes=1, base_seed=42):
    snrs = snrs or dataset.SNR_GRID
    rows = []
    for snr in snrs:
        corr = delay_correlation(model, snr, n_echoes, base_seed)
        rows.append({"snr_db": snr, "delay_corr": corr})
    return pd.DataFrame(rows)


def parameter_count(model):
    return model.count_params()


def footprint_kb(model):
    return float32_kb(model.count_params())
