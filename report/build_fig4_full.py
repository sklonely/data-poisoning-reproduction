"""Build the full Fig 4 reproduction figure — paper-style 6-curve panel.

Pulls latest aggregation (cell_id → ASR) from local victim metrics, plots whatever's
ready as solid lines; missing budgets shown as dashed extrapolation skipped.
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXISTING_PA = ROOT / 'hpc-results' / 'phase-a-victims'  # A01-A04
STAGE = ROOT / 'hpc-results' / 'staging-temp'
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

cells_meta = {
    'A01': ('ConvNetBN', 'dog-bird', 5, 5),     'A02': ('ConvNetBN', 'dog-bird', 50, 5),
    'A03': ('ConvNetBN', 'dog-bird', 500, 5),   'A04': ('ConvNetBN', 'dog-bird', 2500, 5),
    'A05': ('ConvNetBN', 'dog-bird', 5000, 5),
    'A06': ('ConvNetBN', 'frog-plane', 5, 6),   'A07': ('ConvNetBN', 'frog-plane', 50, 6),
    'A08': ('ConvNetBN', 'frog-plane', 500, 6), 'A09': ('ConvNetBN', 'frog-plane', 2500, 6),
    'A10': ('ConvNetBN', 'frog-plane', 5000, 6),
    'A11': ('VGG13BN', 'dog-bird', 5, 5),       'A12': ('VGG13BN', 'dog-bird', 50, 5),
    'A13': ('VGG13BN', 'dog-bird', 500, 5),     'A14': ('VGG13BN', 'dog-bird', 2500, 5),
    'A15': ('VGG13BN', 'dog-bird', 5000, 5),
    'A16': ('VGG13BN', 'frog-plane', 5, 6),     'A17': ('VGG13BN', 'frog-plane', 50, 6),
    'A18': ('VGG13BN', 'frog-plane', 500, 6),   'A19': ('VGG13BN', 'frog-plane', 2500, 6),
    'A20': ('VGG13BN', 'frog-plane', 5000, 6),
    'A21': ('ResNet', 'dog-bird', 5, 5),        'A22': ('ResNet', 'dog-bird', 50, 5),
    'A23': ('ResNet', 'dog-bird', 500, 5),      'A24': ('ResNet', 'dog-bird', 2500, 5),
    'A25': ('ResNet', 'dog-bird', 5000, 5),
    'A26': ('ResNet', 'frog-plane', 5, 6),      'A27': ('ResNet', 'frog-plane', 50, 6),
    'A28': ('ResNet', 'frog-plane', 500, 6),    'A29': ('ResNet', 'frog-plane', 2500, 6),
    'A30': ('ResNet', 'frog-plane', 5000, 6),
}


def victim_target_pred(victim_dir: Path):
    mp = victim_dir / 'metrics.jsonl'
    if not mp.exists(): return None
    last = None
    for line in mp.open():
        r = json.loads(line)
        mm = r.get('metrics', {})
        if any(k.startswith('class-0-') for k in mm):
            classes = {int(k.split('-')[-1]): v for k, v in mm.items() if k.startswith('class-0-')}
            last = min(classes, key=classes.get)
    return last


def cell_results(cell, yadv):
    for cd in [EXISTING_PA / f'victim-{cell}', STAGE / 'outputs-phase-a' / 'local' / f'victim-{cell}']:
        if cd.exists():
            preds = []
            for d in sorted(cd.iterdir()):
                if not d.is_dir(): continue
                p = victim_target_pred(d)
                if p is not None: preds.append(p)
            if preds:
                return preds, sum(1 for x in preds if x == yadv) / len(preds)
    return [], None


# Aggregate per (arch, pair) curve
curves = defaultdict(list)  # (arch, pair) -> list of (budget_pct, asr_pct, n)
for cell, (arch, pair, npoison, yadv) in cells_meta.items():
    preds, asr = cell_results(cell, yadv)
    if asr is not None:
        curves[(arch, pair)].append((npoison / 50000 * 100, asr * 100, len(preds)))

for k in curves:
    curves[k].sort()


# Paper visual readouts (Fig 4 right)
paper = {
    ('ConvNetBN', 'dog-bird'):   ([0.001, 0.01, 0.1, 1.0, 10.0], [2,  5,  18, 60, 90]),
    ('ConvNetBN', 'frog-plane'): ([0.001, 0.01, 0.1, 1.0, 10.0], [3,  6,  20, 55, 88]),
    ('VGG13BN',   'dog-bird'):   ([0.001, 0.01, 0.1, 1.0, 10.0], [1,  3,  12, 40, 80]),
    ('VGG13BN',   'frog-plane'): ([0.001, 0.01, 0.1, 1.0, 10.0], [2,  4,  14, 42, 78]),
    ('ResNet',    'dog-bird'):   ([0.001, 0.01, 0.1, 1.0, 10.0], [4,  8,  25, 72, 92]),
    ('ResNet',    'frog-plane'): ([0.001, 0.01, 0.1, 1.0, 10.0], [3,  7,  22, 60, 88]),
}

color_map = {'ConvNetBN': 'C0', 'VGG13BN': 'C2', 'ResNet': 'C3'}
marker_map = {'dog-bird': 'o', 'frog-plane': 's'}

# === Side-by-side: ours (left) vs paper (right) ===
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for (arch, pair), pts in sorted(curves.items()):
    if not pts: continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ns = [p[2] for p in pts]
    color = color_map[arch]
    marker = marker_map[pair]
    label = f'{arch} {pair} ({len(pts)} budgets, n=6)'
    ax_l.plot(xs, ys, '-' + marker, color=color, markersize=10, linewidth=2.2, label=label)
    for x, y, n in zip(xs, ys, ns):
        ax_l.annotate(f'{y:.0f}%', xy=(x, y), xytext=(6, -10),
                      textcoords='offset points', fontsize=8, color=color)

ax_l.set_xscale('log')
ax_l.set_xlabel('Poison budget (%)', fontsize=11)
ax_l.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax_l.set_title(f'Our reproduction (Phase A, n=6 single-target)\n'
               f'{sum(len(v) for v in curves.values())} datapoints across '
               f'{len([k for k,v in curves.items() if v])} of 6 curves',
               fontsize=11, fontweight='bold')
ax_l.set_ylim(-5, 110)
ax_l.set_xlim(0.005, 20)
ax_l.grid(alpha=0.3, which='both')
ax_l.legend(loc='lower right', fontsize=8)

for (arch, pair), (px, py) in paper.items():
    color = color_map[arch]
    marker = marker_map[pair]
    label = f'{arch} {pair}'
    ax_r.plot(px, py, '--' + marker, color=color, markersize=8, linewidth=1.8, alpha=0.8, label=label)

ax_r.set_xscale('log')
ax_r.set_xlabel('Poison budget (%)', fontsize=11)
ax_r.set_title('Paper Fig 4 right (Huang et al. 2020)\nVisual estimate of all 6 curves',
               fontsize=11, fontweight='bold')
ax_r.set_ylim(-5, 110)
ax_r.set_xlim(0.0005, 20)
ax_r.grid(alpha=0.3, which='both')
ax_r.legend(loc='lower right', fontsize=8)

fig.suptitle('Fig 4 reproduction (in-progress): our results vs paper Fig 4',
             fontsize=13, y=1.01)
fig.tight_layout()
out = OUT / 'fig4_full_progress.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
plt.close(fig)

# Print summary table
print()
print(f'{"arch":<10} {"pair":<11} {"datapoints":<12}')
print('-' * 35)
for (arch, pair), pts in sorted(curves.items()):
    print(f'{arch:<10} {pair:<11} {len(pts):<12}')
