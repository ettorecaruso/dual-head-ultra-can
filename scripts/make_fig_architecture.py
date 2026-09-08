#!/usr/bin/env python3
"""Render the architecture diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'figures' / 'architecture.pdf'
XMAX, YMAX = 17.0, 8.0
FS = 8.6
MY, TY, BY = 4.05, 5.9, 2.1
EDGE = '#2f6db3'
SHARED = dict(fc='#dcebf7', ec=EDGE)
AGG = dict(fc='#e2efda', ec='#3a7d44')
VBOX = dict(fc='#fdf6e3', ec='#b8860b')
HEAD = dict(fc='#fde9d9', ec='#c55a11')
INP = dict(fc='#efefef', ec='#666666')

fig = plt.figure(figsize=(7.3, 3.65))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.axis('off')


def box(cx, cy, w, h, style):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.12',
                                lw=1.0, **style))


def label(cx, cy, text, size=FS, color='black'):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size,
            color=color, linespacing=1.4)


def note(cx, cy, text, size, color):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size,
            color=color, style='italic')


def arrow(x1, y1, x2, y2, lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


box(1.7, MY, 2.7, 1.9, INP)
box(4.2, MY, 2.1, 1.3, SHARED)
box(6.45, MY, 2.1, 1.3, SHARED)
box(9.35, TY, 3.3, 1.8, AGG)
box(9.35, BY, 3.3, 1.8, AGG)
box(11.9, MY, 1.6, 0.9, VBOX)
box(14.2, TY, 2.7, 1.9, HEAD)
box(14.2, BY, 2.7, 1.9, HEAD)
label(1.7, MY + 0.35, r'$\mathbf{X}\in\mathbb{R}^{100\times3}$', size=FS - 0.2)
label(1.7, MY - 0.55, 'received I/Q\n+ chaotic reference', size=FS - 0.8)
label(4.2, MY, 'Conv1D\n' + r'$F_1{=}32$, ReLU', size=FS - 0.4)
label(6.45, MY, 'Conv1D\n' + r'$F_2{=}64$, ReLU', size=FS - 0.4)
label(9.35, TY, 'Attention pooling\n(temporal softmax)\n(Ultra-CAN)', size=FS - 0.8)
label(9.35, BY, 'QKV multi-head attention\n' + r'($H{=}8$, $d_k{=}8$)'
      + '\n(Ultra-CAN-QKV)', size=FS - 0.9)
label(11.9, MY, r'$\mathbf{v}\in\mathbb{R}^{64}$', size=FS - 0.4)
label(14.2, TY, 'Communication head\nDense(128), ReLU\n'
      + r'$\rightarrow$ logits (softmax)', size=FS - 1.1)
label(14.2, BY, 'Sensing head\nDense(32), ReLU\n'
      + r'$\mathbf{v}$ + delay profile', size=FS - 1.1)
note(5.55, 5.30, 'shared feature extractor', FS - 1.6, '#1a4f7a')
note(9.35, 4.05, 'aggregation variant', FS - 1.6, '#3a7d44')
arrow(3.05, MY, 3.18, MY)
arrow(5.25, MY, 5.42, MY)
arrow(7.50, MY, 7.72, 5.45)
arrow(7.50, MY, 7.72, 2.65)
arrow(11.00, TY, 11.14, 4.50)
arrow(11.00, BY, 11.14, 3.60)
arrow(12.70, 4.45, 12.90, 5.00)
arrow(12.70, 3.65, 12.90, 3.10)
arrow(15.55, TY, 15.85, TY)
arrow(15.55, BY, 15.85, BY)
label(16.00, TY, r'$\hat{b}$', size=FS + 1.6)
label(16.00, BY, r'$\hat{\tau}$', size=FS + 1.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, format='pdf')
print('saved', OUT)
