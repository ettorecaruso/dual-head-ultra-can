#!/usr/bin/env python3
"""Jamming-aware training control figure."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'jamming_aware_control.pdf'
MIRROR = REPO / 'results' / 'figures' / 'jamming_aware_control.pdf'
CLEAN = RESULTS / 'jamming_interpretability' / 'qkv'
AWARE = RESULTS / 'jamming_interpretability' / 'jamming_aware_training' / 'qkv'
JAMMERS = ['cw', 'barrage', 'partial_band']
TITLES = {'cw': 'CW', 'barrage': 'Barrage', 'partial_band': 'Partial-band'}


def _legend_below(ax, ncol: int, y: float = -0.20) -> None:
    """Single legend under the panel, outside the axes (never on a curve)."""
    handles, labels, seen = [], [], set()
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in seen:
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y),
              ncol=ncol, fontsize=7, framealpha=0.8, edgecolor='gray',
              facecolor='white', borderpad=0.5, labelspacing=0.3,
              handlelength=2.2)

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
clean = pd.read_csv(CLEAN / 'conditions.csv')
aware = pd.read_csv(AWARE / 'conditions.csv')


def _band(df, sign):
    if 'ber_std' not in df:
        return None
    std = np.nan_to_num(df['ber_std'].to_numpy(dtype=float), nan=0.0)
    ber = df['ber'].to_numpy(dtype=float)
    return np.clip(ber + sign * std, 1e-9, 1.0)


def _draw_band(ax, df, color):
    lower = _band(df, -1.0)
    upper = _band(df, +1.0)
    if lower is None or upper is None:
        return
    ax.fill_between(df['jsr_db'].to_numpy(dtype=float), lower, upper,
                    color=color, alpha=0.18, linewidth=0)


bers = []
for df in (clean, aware):
    lower = _band(df, -1.0)
    source = df['ber'].to_numpy(dtype=float) if lower is None else lower
    bers.extend(float(v) for v in source if v > 0)
y_lo = 10.0 ** np.floor(np.log10(min(bers)))
y_hi = 1.0

for ax, jammer in zip(axes, JAMMERS):
    c = clean[clean.jammer == jammer].sort_values('jsr_db')
    a = aware[aware.jammer == jammer].sort_values('jsr_db')
    ax.plot(c['jsr_db'], c['ber'], color='#c44e52', ls='-', marker='o', ms=3, lw=1.4,
            label='Clean-trained')
    _draw_band(ax, c, '#c44e52')
    ax.plot(a['jsr_db'], a['ber'], color='#4c72b0', ls='--', marker='s', ms=3, lw=1.4,
            label='Jamming-aware')
    _draw_band(ax, a, '#4c72b0')
    ax.set_yscale('log')
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(TITLES[jammer], fontsize=8)
    ax.set_xlabel('JSR (dB)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which='both')
axes[0].set_ylabel('BER', fontsize=8)
fig.tight_layout()
_legend_below(axes[1], ncol=2)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.12)
print('saved', OUT)
if MIRROR.parent.is_dir():
    fig.savefig(MIRROR, bbox_inches='tight', pad_inches=0.12)
    print('saved', MIRROR)
plt.close(fig)
