import numpy as np

from . import channel

ARCH_SEEDS = {"conv1d": 5, "qkv": 67, "lstm": 63, "mc_dlsk": 80}
SNR_GRID = list(range(-5, 21, 2))


def snr_points(low=-5, high=20, step=2):
    return list(range(low, high + 1, step))


def generate_batch(size, snr_db, n_echoes, base_seed, bit_rng=None):
    if bit_rng is None:
        bit_rng = np.random.default_rng(base_seed)
    bits = bit_rng.integers(0, 2, size=size).astype(np.int32)
    x = np.empty((size, 100, 3), dtype=np.float32)
    delay = np.empty((size,), dtype=np.float32)
    for i, b in enumerate(bits):
        x[i], delay[i] = channel.generate_symbol(
            int(b), seed=int(base_seed + i * 7919 + snr_db * 104729),
            snr_db=float(snr_db), n_echoes=n_echoes)
    return x, bits, delay


def generate_split(num_symbols, n_echoes, base_seed, snrs=None):
    snrs = snrs or snr_points()
    per_snr = max(1, num_symbols // len(snrs))
    x_parts, b_parts, d_parts = [], [], []
    rng = np.random.default_rng(base_seed)
    for snr in snrs:
        x, b, d = generate_batch(per_snr, snr, n_echoes, base_seed + snr, bit_rng=rng)
        x_parts.append(x)
        b_parts.append(b)
        d_parts.append(d)
    return (np.concatenate(x_parts), np.concatenate(b_parts), np.concatenate(d_parts))


def train_val_sets(n_train, n_val, n_echoes, base_seed, snrs=None):
    train = generate_split(n_train, n_echoes, base_seed, snrs)
    val = generate_split(n_val, n_echoes, base_seed + 100000, snrs)
    return train, val
