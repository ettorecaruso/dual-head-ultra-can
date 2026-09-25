#!/usr/bin/env python3
"""Spectral detectability of the emitted waveform (low-probability-of-intercept).

    python scripts/diagnostics/lpi_detectability.py [--out-root results/full]

The question this answers is the one a reviewer asks about a spread-spectrum
link: *is the emitted waveform easier to intercept than a conventional one?*
It is answered with measurements, not with a claim.

Conventions, stated so that the comparison is fair:

* every waveform carries the **same energy in the detection window** and the
  x-axis is the **total SNR in that window**, ``gamma = E_s / N0``; the noise
  variance is therefore set as ``sigma^2 = 1/gamma`` and the detector thresholds
  do not depend on the waveform;
* the **peer** of the proposed chaos-shift-keying waveform is DSSS-BPSK (a PN
  sequence at the same symbol rate), i.e. the classic LPI baseline. OOK and BPSK
  with a rectangular pulse are the *narrowband* standard modulations the
  objection refers to and are reported for reference;
* the waveforms are built by the repository's own transmitter
  (``generate_transmitted_batch_fast``) and energy-normalized per symbol, with
  no channel and no receiver: this is a property of the emitted signal.

Two detectors are reported at ``Pfa = 1e-2``, calibrated by Monte Carlo:

* a **plain radiometer** (total energy) -- blind to the waveform shape by
  construction, so its curve is the same for every equal-energy waveform and
  serves as the control;
* a **channelized radiometer** (maximum energy over 64 sub-bands) -- this is the
  detector that can exploit a non-flat spectrum, so the gap between the two
  curves is what a shaped waveform gives away.

Outputs (CSV always; the PDF needs a working matplotlib):

    results/full/diagnostics/lpi_detectability/lpi_descriptors.csv
    results/full/diagnostics/lpi_detectability/lpi_roc.csv
    results/full/diagnostics/lpi_detectability/lpi_detectability.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.dataset_generator import generate_transmitted_batch_fast
from src.utils.config_loader import DEFAULT_BASE_CONFIG_PATH, load_config

SYMBOLS = 4096
PFA = 1.0e-2
N_DET = 1024
TRIALS = 2000
N_BINS = 64
SNR_GRID = np.arange(0.0, 26.0, 2.0)
SEED_WAVEFORM = 20260925
SEED_THRESHOLD = 7
SEED_DETECTION = 11
WAVEFORMS = ("proposed (CSK)", "OOK", "BPSK", "DSSS-BPSK")
WAVEFORMS_ABBR = {
    "proposed (CSK)": "CSK",
    "OOK": "OOK",
    "BPSK": "BPSK",
    "DSSS-BPSK": "DSSS",
}


def norm_symbols(x: np.ndarray, seq_len: int) -> np.ndarray:
    """Energy-normalize each symbol; no centring (that would kill OOK/BPSK)."""
    block = x.reshape(-1, seq_len).copy()
    norm = np.linalg.norm(block, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (block / norm).ravel()


def build_waveforms(cfg, rng):
    seq_len = int(cfg["data"]["sequence_length"])
    bits = rng.integers(0, 2, size=SYMBOLS)
    bipolar = 2.0 * bits - 1.0
    chaotic = generate_transmitted_batch_fast(cfg, bits, rng).ravel()
    pn = 2.0 * rng.integers(0, 2, size=SYMBOLS * seq_len) - 1.0
    return {
        "proposed (CSK)": norm_symbols(chaotic, seq_len),
        "OOK": norm_symbols(np.repeat(bits.astype(float), seq_len), seq_len),
        "BPSK": norm_symbols(np.repeat(bipolar, seq_len), seq_len),
        "DSSS-BPSK": norm_symbols(np.repeat(bipolar, seq_len) * pn, seq_len),
    }


def descriptors(x: np.ndarray, fs: float) -> dict:
    """Spectral flatness, peak-to-floor, PAPR and the 99% occupied bandwidth."""
    freq, psd = welch(x, fs=fs, nperseg=2048, return_onesided=True)
    psd, freq = psd[1:], freq[1:]
    cumulative = np.cumsum(psd) / np.sum(psd)
    edge = max(2, int(np.searchsorted(cumulative, 0.99)) + 1)
    band = psd[:edge]
    flatness = float(
        np.exp(np.mean(np.log(np.maximum(band, 1e-300)))) / np.mean(band)
    )
    return {
        "spectral_flatness": flatness,
        "peak_to_floor_db": float(10.0 * np.log10(band.max() / np.median(band))),
        "papr_db": float(10.0 * np.log10((x ** 2).max() / np.mean(x ** 2))),
        "occupied_bw_khz": float(freq[edge - 1]) / 1e3,
    }


def _subband_energy(power: np.ndarray) -> np.ndarray:
    edges = np.linspace(0, power.shape[1], N_BINS + 1).astype(int)
    return np.max(np.stack(
        [power[:, edges[b]:edges[b + 1]].sum(axis=1) for b in range(N_BINS)],
        axis=1,
    ), axis=1)


def unit_thresholds(rng) -> tuple:
    """Pfa thresholds for unit noise: plain energy and max sub-band energy."""
    plain, channelized = [], []
    for _ in range(60):
        noise = rng.standard_normal((1000, N_DET))
        power = np.abs(np.fft.rfft(noise, axis=1)) ** 2
        plain.append(np.sum(power, axis=1))
        channelized.append(_subband_energy(power))
    return (float(np.quantile(np.concatenate(plain), 1.0 - PFA)),
            float(np.quantile(np.concatenate(channelized), 1.0 - PFA)))


def roc(x: np.ndarray, thresholds: tuple, rng) -> list:
    """Pd of both detectors versus the total SNR in the window."""
    window = x[:N_DET]
    window = window / np.linalg.norm(window)
    rows = []
    for snr_db in SNR_GRID:
        sigma = np.sqrt(1.0 / 10.0 ** (snr_db / 10.0))
        thr_plain = thresholds[0] * sigma ** 2
        thr_chan = thresholds[1] * sigma ** 2
        hits_plain = hits_chan = 0
        done = 0
        while done < TRIALS:
            m = min(1000, TRIALS - done)
            received = window[None, :] + sigma * rng.standard_normal((m, N_DET))
            power = np.abs(np.fft.rfft(received, axis=1)) ** 2
            hits_plain += int(np.sum(np.sum(power, axis=1) > thr_plain))
            hits_chan += int(np.sum(_subband_energy(power) > thr_chan))
            done += m
        rows.append({
            "snr_db": float(snr_db),
            "pd_plain": hits_plain / TRIALS,
            "pd_channelized": hits_chan / TRIALS,
        })
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_REPO / "results" / "full",
        help="root the outputs are written to (default: results/full)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(DEFAULT_BASE_CONFIG_PATH)
    fs = float(cfg["data"]["fs_hz"])
    waves = build_waveforms(cfg, np.random.default_rng(SEED_WAVEFORM))

    out = args.out_root / "diagnostics" / "lpi_detectability"
    out.mkdir(parents=True, exist_ok=True)

    desc = {name: descriptors(x, fs) for name, x in waves.items()}
    with open(out / "lpi_descriptors.csv", "w", encoding="utf-8") as handle:
        handle.write("waveform,spectral_flatness,peak_to_floor_db,papr_db,"
                     "occupied_bw_khz\n")
        for name in WAVEFORMS:
            d = desc[name]
            handle.write(f"{name},{d['spectral_flatness']:.6f},"
                         f"{d['peak_to_floor_db']:.4f},{d['papr_db']:.4f},"
                         f"{d['occupied_bw_khz']:.3f}\n")

    thresholds = unit_thresholds(np.random.default_rng(SEED_THRESHOLD))
    curves = {
        name: roc(x, thresholds, np.random.default_rng(SEED_DETECTION))
        for name, x in waves.items()
    }
    with open(out / "lpi_roc.csv", "w", encoding="utf-8") as handle:
        handle.write("waveform,snr_db,pd_plain,pd_channelized\n")
        for name in WAVEFORMS:
            for row in curves[name]:
                handle.write(f"{name},{row['snr_db']:.1f},{row['pd_plain']:.4f},"
                             f"{row['pd_channelized']:.4f}\n")

    print(f"[lpi] fs={fs:.0e} Hz  symbols={SYMBOLS}  N_det={N_DET}  "
          f"Pfa={PFA:.0e}  trials={TRIALS}  bins={N_BINS}")
    print(f"{'waveform':16s} {'flatness':>9s} {'pk/flr dB':>10s} "
          f"{'PAPR dB':>8s} {'BW99 kHz':>9s}")
    for name in WAVEFORMS:
        d = desc[name]
        print(f"{name:16s} {d['spectral_flatness']:9.3f} "
              f"{d['peak_to_floor_db']:10.2f} {d['papr_db']:8.2f} "
              f"{d['occupied_bw_khz']:9.1f}")
    print(f"[lpi] Pfa thresholds (unit noise): plain={thresholds[0]:.1f} "
          f"channelized={thresholds[1]:.1f}")
    print("[lpi] gamma dB : Pd plain / Pd channelized")
    for index, snr_db in enumerate(SNR_GRID):
        cells = "  ".join(
            f"{WAVEFORMS_ABBR[name]} {curves[name][index]['pd_plain']:.2f}/"
            f"{curves[name][index]['pd_channelized']:.2f}"
            for name in WAVEFORMS
        )
        print(f"[lpi] {snr_db:8.1f} : {cells}")
    print(f"[lpi] saved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[lpi] matplotlib unavailable ({exc}): CSVs written, PDF skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name in WAVEFORMS:
        rows = curves[name]
        x = [r["snr_db"] for r in rows]
        axes[0].semilogy(x, [max(r["pd_plain"], 1e-3) for r in rows],
                         marker="o", label=WAVEFORMS_ABBR[name])
        axes[1].semilogy(x, [max(r["pd_channelized"], 1e-3) for r in rows],
                         marker="s", label=WAVEFORMS_ABBR[name])
    for ax, title in zip(axes, ("Plain radiometer (blind to shape)",
                                f"Channelized radiometer, {N_BINS} bins")):
        ax.axhline(PFA, color="0.4", ls=":", lw=1.0)
        ax.set_xlabel(r"total SNR in the window $\gamma$ (dB)")
        ax.set_title(title, pad=8)
        ax.grid(True, which="both", alpha=0.4)
    axes[0].set_ylabel(r"$P_d$")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Detectability of the emitted waveform (Pfa={PFA:.0e}, "
                 f"N={N_DET} samples)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "lpi_detectability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[lpi] saved {out / 'lpi_detectability.pdf'}")


if __name__ == "__main__":
    main()
