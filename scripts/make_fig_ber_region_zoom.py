#!/usr/bin/env python3
"""Operating-region BER figure (SNR >= 5 dB)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'ber_region_zoom.pdf'
ARCHS = ['conv1d', 'qkv', 'lstm', 'mc_dlsk']
LABELS = {'conv1d': 'Ultra-CAN (Conv1D)', 'qkv': 'Ultra-CAN (QKV)',
          'lstm': 'LSTM-OFDM-DCSK', 'mc_dlsk': 'MC-DLCSK'}
COLORS = {'conv1d': 'tab:blue', 'qkv': 'tab:red', 'lstm': 'tab:green',
          'mc_dlsk': 'tab:purple'}
SCENARIOS = [('single_echo', 'k1_doppler_full'),
             ('multi_echo', 'k3_doppler_full'),
             ('multi_echo_low_doppler', 'k3_doppler_limited')]
MIN_DB = 5.0

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
for ax, (title, scenario) in zip(axes, SCENARIOS):
    for arch in ARCHS:
        path = RESULTS / 'ber_vs_snr' / scenario / arch / 'metrics.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df['snr_db'] >= MIN_DB]
        ax.semilogy(df['snr_db'], df['ber'], marker='o', ms=3, lw=1.3,
                    color=COLORS[arch], label=LABELS[arch])
    ax.set_title(title.replace('_', ' '), fontsize=7)
    ax.set_xlabel('SNR (dB)', fontsize=8)
    ax.grid(alpha=0.35, which='both')
    ax.set_ylim(4e-5, 1e-3)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel('BER', fontsize=8)
axes[0].legend(fontsize=5, loc='upper right')
axes[0].axhline(1e-4, color='k', ls=':', lw=1)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print('saved', OUT)
