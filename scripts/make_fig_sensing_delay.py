#!/usr/bin/env python3
"""Delay-estimation figure (single echo)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'sensing_delay_single.pdf'
SCENARIO = 'k1_doppler_full'
ARCHS = ['conv1d', 'qkv', 'lstm', 'mc_dlsk']
LABELS = {'conv1d': 'Ultra-CAN (Conv1D)', 'qkv': 'Ultra-CAN (QKV)',
          'lstm': 'LSTM-OFDM-DCSK', 'mc_dlsk': 'MC-DLCSK'}
STYLES = {'conv1d': dict(color='#1f77b4', marker='o'),
          'qkv': dict(color='#ff7f0e', marker='s'),
          'lstm': dict(color='#2ca02c', marker='^'),
          'mc_dlsk': dict(color='#d62728', marker='D')}

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for arch in ARCHS:
    path = RESULTS / 'ber_vs_snr' / SCENARIO / arch / 'metrics.csv'
    if not path.exists():
        continue
    df = pd.read_csv(path)
    ax.plot(df['snr_db'], df['corr_tau'], lw=1.5, label=LABELS[arch], **STYLES[arch])
ax.set_xlabel('SNR (dB)')
ax.set_ylabel(r'corr($\hat{\tau},\tau$)')
ax.set_title('Delay estimation (single-echo scenario)')
ax.grid(alpha=0.3)
ax.legend(fontsize=7)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print('saved', OUT)
