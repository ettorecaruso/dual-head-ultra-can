import numpy as np


def logistic_step(x: np.ndarray, mu: float) -> np.ndarray:
    return mu * x * (1.0 - x)


def bernoulli_step(x: np.ndarray) -> np.ndarray:
    y = 2.0 * x
    return np.where(y >= 1.0, y - 1.0, y)


def iterate_map(x0: float, length: int, kind: str, mu: float = 3.9) -> np.ndarray:
    out = np.empty(length, dtype=np.float64)
    x = float(x0)
    for n in range(length):
        x = logistic_step(x, mu) if kind == "logistic" else bernoulli_step(x)
        out[n] = x
    return out


def x0_from_seed(seed: int, mu: float = 3.9) -> float:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.001, 0.999)
    return float(x0)


def reference_for_bit(bit: int, length: int, seed: int, mu: float = 3.9) -> np.ndarray:
    x0 = x0_from_seed(seed, mu)
    kind = "logistic" if bit == 0 else "bernoulli"
    return iterate_map(x0, length, kind, mu)


def reference_matrix(bits: np.ndarray, length: int, seeds: np.ndarray, mu: float = 3.9) -> np.ndarray:
    out = np.empty((bits.shape[0], length), dtype=np.float32)
    for i, (b, s) in enumerate(zip(bits, seeds)):
        out[i] = reference_for_bit(int(b), length, int(s), mu)
    return out
