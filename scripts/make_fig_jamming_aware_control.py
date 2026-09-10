#!/usr/bin/env python3
"""Jamming-aware training control figure."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'jamming_aware_control.pdf'
CLEAN = RESULTS / 'jamming_interpretability' / 'qkv'
AWARE = RESULTS / 'jamming_interpretability' / 'jamming_aware_training' / 'qkv'
JAMMERS = ['cw', 'barrage', 'partial_band']
TITLES = {'cw': 'CW', 'barrage': 'Barrage', 'partial_band': 'Partial-band'}

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
clean = pd.read_csv(CLEAN / 'conditions.csv')
aware = pd.read_csv(AWARE / 'conditions.csv')
for ax, jammer in zip(axes, JAMMERS):
    c = clean[clean.jammer == jammer].sort_values('jsr_db')
    a = aware[aware.jammer == jammer].sort_values('jsr_db')
    ax.plot(c['jsr_db'], c['ber'], color='#c44e52', ls='-', marker='o', ms=3, lw=1.4,
            label='Clean-trained')
    ax.plot(a['jsr_db'], a['ber'], color='#4c72b0', ls='--', marker='s', ms=3, lw=1.4,
            label='Jamming-aware')
    ax.set_title(TITLES[jammer], fontsize=8)
    ax.set_xlabel('JSR (dB)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('BER', fontsize=8)
axes[0].legend(fontsize=6)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print('saved', OUT)
