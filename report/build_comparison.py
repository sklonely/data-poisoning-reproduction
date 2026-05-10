"""Side-by-side comparison: our reproduction vs paper Fig 4 (right panel).

Left panel  = our ConvNetBN dog-bird curve (5 budgets × n=6) as of 2026-05-05 partial pull.
Right panel = paper Fig 4 right ConvNetBN dog-bird curve (visual readouts from PDF).

Same axes / styling so the visual gap is immediate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXISTING_PA = ROOT / 'hpc-results' / 'phase-a-victims'
STAGE = ROOT / 'hpc-results' / 'staging-temp'
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def victim_target_pred(victim_dir: Path):
    mp = victim_dir / 'metrics.jsonl'
    if not mp.exists():
        return None
    last_target = None
    for line in mp.open():
        r = json.loads(line)
        mm = r.get('metrics', {})
        if any(k.startswith('class-0-') for k in mm):
            classes = {int(k.split('-')[-1]): v for k, v in mm.items() if k.startswith('class-0-')}
            last_target = min(classes, key=classes.get)
    return last_target


def cell_asr(cell: str, yadv: int):
    """Find the cell's victim experiments, return (preds, asr, n)."""
    candidate_dirs = [
        EXISTING_PA / f'victim-{cell}',
        STAGE / 'outputs-phase-a' / 'local' / f'victim-{cell}',
    ]
    cell_dir = None
    for cd in candidate_dirs:
        if cd.exists():
            cell_dir = cd
            break
    if cell_dir is None:
        return [], 0.0, 0
    preds = []
    for d in sorted(cell_dir.iterdir()):
        if not d.is_dir():
            continue
        p = victim_target_pred(d)
        if p is not None:
            preds.append(p)
    if not preds:
        return [], 0.0, 0
    asr = sum(1 for x in preds if x == yadv) / len(preds)
    return preds, asr, len(preds)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0, 1)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))


# ===== Our data: ConvNetBN dog-bird =====
ours_cells = [
    ('A01', 5, 5),    # npoison, yadv
    ('A02', 50, 5),
    ('A03', 500, 5),
    ('A04', 2500, 5),
    ('A05', 5000, 5),
]
ours_x, ours_y, ours_lo, ours_hi, ours_n = [], [], [], [], []
for cell, npoison, yadv in ours_cells:
    preds, asr, n = cell_asr(cell, yadv)
    if n == 0:
        continue
    k = int(round(asr * n))
    lo, hi = wilson_ci(k, n)
    ours_x.append(npoison / 50000 * 100)  # %
    ours_y.append(asr * 100)
    ours_lo.append(lo * 100)
    ours_hi.append(hi * 100)
    ours_n.append(n)

# ===== Paper data (visual readouts from arXiv 2004.00225v2 Fig 4 right) =====
# Approximate values traced from the paper's plot.
# x = poison budget %; y = ASR %.
paper_curves = {
    'ConvNetBN dog-bird':   ([0.001, 0.01, 0.1, 1.0, 10.0], [2,  5,  18, 60, 90], 'C0', 'o'),
    'ConvNetBN frog-plane': ([0.001, 0.01, 0.1, 1.0, 10.0], [3,  6,  20, 55, 88], 'C0', 's'),
    'VGG13 dog-bird':       ([0.001, 0.01, 0.1, 1.0, 10.0], [1,  3,  12, 40, 80], 'C2', 'o'),
    'VGG13 frog-plane':     ([0.001, 0.01, 0.1, 1.0, 10.0], [2,  4,  14, 42, 78], 'C2', 's'),
    'ResNet20 dog-bird':    ([0.001, 0.01, 0.1, 1.0, 10.0], [4,  8,  25, 72, 92], 'C3', 'o'),
    'ResNet20 frog-plane':  ([0.001, 0.01, 0.1, 1.0, 10.0], [3,  7,  22, 60, 88], 'C3', 's'),
}

# ===== 2-panel figure =====
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Left: ours
errs = [
    [max(0.0, y - lo) for y, lo in zip(ours_y, ours_lo)],
    [max(0.0, hi - y) for y, hi in zip(ours_y, ours_hi)],
]
ax_l.errorbar(ours_x, ours_y, yerr=errs, fmt='-o', color='C0', markersize=12,
              linewidth=2.5, capsize=5, label='ConvNetBN dog→bird (n=6)')
for x, y, n in zip(ours_x, ours_y, ours_n):
    ax_l.annotate(f'{y:.0f}%\n(n={n})', xy=(x, y), xytext=(8, -8),
                  textcoords='offset points', fontsize=10)
ax_l.set_xscale('log')
ax_l.set_xlabel('Poison budget (%)', fontsize=11)
ax_l.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax_l.set_title('Our reproduction\n(ConvNetBN, dog→bird, 200-epoch from-scratch victims)',
               fontsize=11, fontweight='bold')
ax_l.set_ylim(-5, 110)
ax_l.set_xlim(0.005, 20)
ax_l.grid(alpha=0.3, which='both')
ax_l.legend(loc='upper left', fontsize=10)
# Annotate transition point
ax_l.axvspan(0.5, 2.0, alpha=0.08, color='green')
ax_l.text(1.0, 5, 'Transition\n(0% → 100%)', ha='center', fontsize=9, color='darkgreen')

# Right: paper
for label, (x, y, color, marker) in paper_curves.items():
    style = '--' + marker
    alpha = 1.0 if label == 'ConvNetBN dog-bird' else 0.35
    lw = 2.5 if label == 'ConvNetBN dog-bird' else 1.5
    ax_r.plot(x, y, style, color=color, markersize=8 if alpha == 1 else 5,
              linewidth=lw, alpha=alpha, label=label)
ax_r.set_xscale('log')
ax_r.set_xlabel('Poison budget (%)', fontsize=11)
ax_r.set_title('Paper Fig 4 right (Huang et al. 2020)\n(visual estimate, all 6 curves)',
               fontsize=11, fontweight='bold')
ax_r.set_ylim(-5, 110)
ax_r.set_xlim(0.0005, 20)
ax_r.grid(alpha=0.3, which='both')
ax_r.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Annotate the matching curve on right
ax_r.axvspan(0.5, 2.0, alpha=0.08, color='green')

# Suptitle
fig.suptitle('Side-by-side comparison: our reproduction (left) vs paper Fig 4 right (right)',
             fontsize=12, y=1.02)

fig.tight_layout()
out = OUT / 'fig5_side_by_side_paper_vs_ours.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
plt.close(fig)

# Also produce a plain ConvNetBN dog-bird overlay for closer comparison
fig2, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ours_x, ours_y, yerr=errs, fmt='-o', color='C0', markersize=12,
            linewidth=2.5, capsize=5, label=f'Ours (n=6 each, paper-config craft)')
px, py, _, _ = paper_curves['ConvNetBN dog-bird']
ax.plot(px, py, '--^', color='C3', markersize=10, linewidth=2,
        label='Paper Fig 4 ConvNetBN dog-bird (visual readout)')
ax.set_xscale('log')
ax.set_xlabel('Poison budget (%)', fontsize=11)
ax.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax.set_title('ConvNetBN dog→bird: ours vs paper (overlay)', fontsize=12)
ax.set_ylim(-5, 110)
ax.set_xlim(0.0005, 20)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=10)
for x, y, n in zip(ours_x, ours_y, ours_n):
    ax.annotate(f'{y:.0f}%', xy=(x, y), xytext=(8, -8),
                textcoords='offset points', fontsize=9, color='C0')
for x, y in zip(px, py):
    ax.annotate(f'{y}%', xy=(x, y), xytext=(8, 5),
                textcoords='offset points', fontsize=9, color='C3', alpha=0.7)
fig2.tight_layout()
out2 = OUT / 'fig6_overlay_convnetbn_dogbird.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved {out2}')
plt.close(fig2)

print()
print(f'Our data points (n={ours_n[0]}):')
for x, y in zip(ours_x, ours_y):
    print(f'  budget={x:.3f}%   ASR={y:.0f}%')
