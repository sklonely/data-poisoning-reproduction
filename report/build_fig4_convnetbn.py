"""ConvNetBN-focused Fig 4 comparison: our TF reproduction (left) vs paper Fig 4 cropped image (right)."""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).parent.parent
EXISTING_PA = ROOT / 'hpc-results' / 'phase-a-victims'
STAGE = ROOT / 'hpc-results' / 'staging-temp' / 'outputs-phase-a' / 'local'
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
PAPER_FIG4_RIGHT = OUT / 'paper_fig4_right_asr.png'

cells = {
    'A01': ('dog-bird', 5, 5),     'A02': ('dog-bird', 50, 5),
    'A03': ('dog-bird', 500, 5),   'A04': ('dog-bird', 2500, 5),
    'A05': ('dog-bird', 5000, 5),
    'A06': ('frog-plane', 5, 6),   'A07': ('frog-plane', 50, 6),
    'A08': ('frog-plane', 500, 6), 'A09': ('frog-plane', 2500, 6),
    'A10': ('frog-plane', 5000, 6),
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
    preds = []
    for cd in [EXISTING_PA / f'victim-{cell}', STAGE / f'victim-{cell}']:
        if cd.exists():
            for d in sorted(cd.iterdir()):
                if not d.is_dir(): continue
                p = victim_target_pred(d)
                if p is not None: preds.append(p)
    for ti in range(1, 5):
        for cd in [STAGE / f'victim-{cell}_t{ti}']:
            if cd.exists():
                for d in sorted(cd.iterdir()):
                    if not d.is_dir(): continue
                    p = victim_target_pred(d)
                    if p is not None: preds.append(p)
    return preds


curves = defaultdict(list)
for cell, (pair, npoison, yadv) in cells.items():
    preds = cell_results(cell, yadv)
    if preds:
        n = len(preds)
        k = sum(1 for p in preds if p == yadv)
        asr = k / n
        curves[pair].append((npoison/50000*100, asr*100, n))
for k in curves: curves[k].sort()


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.2]})

color_map = {'dog-bird': 'C0', 'frog-plane': 'C1'}
marker_map = {'dog-bird': 'o', 'frog-plane': 's'}

# Left: ours (no CI)
for pair, pts in sorted(curves.items()):
    if not pts: continue
    xs, ys, ns = zip(*pts)
    color = color_map[pair]
    marker = marker_map[pair]
    ax_l.plot(xs, ys, '-' + marker, color=color, markersize=11, linewidth=2.5,
              label=f'ConvNetBN {pair} (n̄={int(np.mean(ns))})')
    for x, y, n in zip(xs, ys, ns):
        ax_l.annotate(f'{y:.0f}%', xy=(x, y), xytext=(8, -10),
                      textcoords='offset points', fontsize=10, color=color)

ax_l.set_xscale('log')
ax_l.set_xlabel('Poison budget (%)', fontsize=11)
ax_l.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax_l.set_title('Our TF reproduction — ConvNetBN', fontsize=12, fontweight='bold')
ax_l.set_ylim(-5, 110)
ax_l.set_xlim(0.005, 20)
ax_l.grid(alpha=0.3, which='both')
ax_l.legend(loc='lower right', fontsize=10)

# Right: actual cropped paper image
if PAPER_FIG4_RIGHT.exists():
    paper_img = mpimg.imread(str(PAPER_FIG4_RIGHT))
    ax_r.imshow(paper_img)
    ax_r.axis('off')
    ax_r.set_title('Paper Fig 4 right (Huang et al. 2020) — full 6 curves\n'
                   '(image cropped from arXiv 2004.00225v2 page 7)',
                   fontsize=11, fontweight='bold')
else:
    ax_r.text(0.5, 0.5, 'paper_fig4_right_asr.png missing\nrun crop_paper_fig4.py',
              ha='center', va='center', transform=ax_r.transAxes)

fig.suptitle('ConvNetBN focus: our reproduction (left) vs paper Fig 4 (right)',
             fontsize=13, y=1.02)
fig.tight_layout()
out = OUT / 'fig4_convnetbn_only.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
plt.close(fig)

print()
for pair, pts in sorted(curves.items()):
    print(f'  ConvNetBN {pair}:')
    for x, y, n in pts:
        print(f'    @{x:.3f}%  ASR={y:.0f}%  n={n}')
