import numpy as np

from .chaos import reference_for_bit


def rician_gain(rng: np.random.Generator, kappa_db: float = 10.0) -> complex:
    kappa = 10.0 ** (kappa_db / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * (kappa + 1.0)))
    los = np.sqrt(kappa / (kappa + 1.0))
    ray = sigma * (rng.standard_normal() + 1j * rng.standard_normal())
    return complex(los + ray)


def rayleigh_gain(rng: np.random.Generator) -> complex:
    return complex(rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)


def sample_direct_path(rng: np.random.Generator, max_doppler: float, kappa_db: float = 10.0):
    gain = rician_gain(rng, kappa_db)
    doppler = rng.uniform(-max_doppler, max_doppler)
    return gain, doppler


def sample_echoes(rng: np.random.Generator, n_echoes: int, max_delay: int, max_doppler: float, alpha_min: float, alpha_max: float):
    delays = rng.choice(np.arange(1, max_delay + 1), size=n_echoes, replace=False)
    echoes = []
    for tau in delays:
        alpha = rng.uniform(alpha_min, alpha_max)
        gain = rayleigh_gain(rng)
        doppler = rng.uniform(-max_doppler, max_doppler)
        echoes.append((tau, alpha * gain, doppler))
    return echoes


def sample_link(rng, n_echoes, max_delay, max_doppler, doppler_direct_max, kappa_db, alpha_min, alpha_max):
    direct, fdc = sample_direct_path(rng, doppler_direct_max, kappa_db)
    echoes = sample_echoes(rng, n_echoes, max_delay, max_doppler, alpha_min, alpha_max)
    return direct, fdc, echoes


def apply_link(x_ref, direct, fdc, echoes, snr_db, rng):
    n = x_ref.shape[0]
    x = x_ref.astype(np.float64)
    t = np.arange(n)
    y = direct * x * np.exp(2j * np.pi * fdc * t)
    for tau, alpha_h, fd in echoes:
        y[tau:] += alpha_h * x[: n - tau] * np.exp(2j * np.pi * fd * t[tau:])
    noise_power = np.mean(np.abs(y) ** 2) / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (y + noise).astype(np.complex64)


def symbol_to_feature(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.stack([y.real, y.imag, reference], axis=-1)


def generate_symbol(bit: int, seed: int, snr_db: float, n_echoes: int, length: int = 100,
                    mu: float = 3.9, max_delay: int = 33, max_doppler: float = 8e-5,
                    doppler_direct_max: float = 1e-5, kappa_db: float = 10.0,
                    alpha_min: float = 0.05, alpha_max: float = 0.3):
    rng = np.random.default_rng(seed)
    ref = reference_for_bit(int(bit), length, seed, mu)
    direct, fdc, echoes = sample_link(rng, n_echoes, max_delay, max_doppler,
                                      doppler_direct_max, kappa_db, alpha_min, alpha_max)
    y = apply_link(ref, direct, fdc, echoes, snr_db, rng)
    delays = [tau for tau, alpha_h, fd in echoes] if echoes else [0]
    gains = [abs(alpha_h) for tau, alpha_h, fd in echoes] if echoes else [0.0]
    dominant_tau = delays[int(np.argmax(gains))]
    feature = symbol_to_feature(y, ref.astype(np.float32))
    delay_target = dominant_tau / float(max_delay)
    return feature, delay_target



def apply_jamming(symbols: np.ndarray, jammer: str, jsr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = symbols.astype(np.float64).copy()
    b, n, c = out.shape
    power = np.mean(out[..., :2] ** 2)
    target = power * (10.0 ** (jsr_db / 10.0))
    if jammer == "cw":
        freq = rng.uniform(0.0, 0.5)
        tone = np.sqrt(2.0 * target) * np.cos(2 * np.pi * freq * np.arange(n))
        out[..., 0] += tone
    elif jammer == "barrage":
        noise = rng.standard_normal((b, n)) * np.sqrt(target)
        out[..., 0] += noise
    elif jammer == "partial_band":
        frac = 0.5
        n_bins = 64
        spec = np.fft.rfft(out[..., 0] + 1j * out[..., 1], axis=-1)
        bins = spec.shape[-1]
        jam_bins = rng.choice(np.arange(bins), size=max(1, int(frac * bins)), replace=False)
        noise_spec = np.fft.rfft(rng.standard_normal((b, n)) * np.sqrt(2.0 * target / frac), axis=-1)
        spec[:, jam_bins] += noise_spec[:, jam_bins]
        rec = np.fft.irfft(spec, n=n, axis=-1)
        out[..., 0] = rec.real
        out[..., 1] = rec.imag
    return out.astype(np.float32)
