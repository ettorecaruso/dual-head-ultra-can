#!/usr/bin/env python3
"""Layer retention figure under jamming."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results' / 'full'
OUT = REPO / 'figures' / 'jamming_layer_retention.pdf'
PANELS = [
    ('conv1d', 'Ultra-CAN (Conv1D)',
     [('conv1', 'conv1_cos'), ('conv2', 'conv2_cos'),
      ('attn pooling', 'attn_pool_cos'), ('sensing\nposition', 'sensing_position_cos')]),
    ('qkv', 'Ultra-CAN (QKV)',
     [('conv1', 'conv1_cos'), ('conv2', 'conv2_cos'),
      ('QKV attention', 'qkv_attention_cos'), ('sensing\nposition', 'sensing_position_cos')]),
    ('lstm', 'LSTM-OFDM-DCSK',
     [('LSTM state', 'lstm_1_cos'), ('projection', 'lstm_proj_cos'),
      ('sensing\nposition', 'sensing_position_cos')]),
    ('mc_dlsk', 'MC-DLCSK',
     [('BiLSTM', 'mc_bilstm_1_cos'), ('proj', 'mc_proj_cos'),
      ('sensing\nposition', 'sensing_position_cos')]),
]

fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15), sharey=True)
for ax, (key, title, bars) in zip(axes, PANELS):
    path = RESULTS / 'jamming_interpretability' / key / 'conditions.csv'
    df = pd.read_csv(path)
    row = df[(df.jammer == 'barrage') & (df.jsr_db == 10.0)].iloc[0]
    names = [b[0] for b in bars]
    vals = [float(row[b[1]]) for b in bars]
    cols = ['#4c72b0'] * (len(vals) - 1) + ['#55a868']
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=6.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel('cosine', fontsize=8)
fig.suptitle('Layer-wise activation retention, barrage jamming at JSR +10 dB', fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.93])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print('saved', OUT)
