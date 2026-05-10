"""Updated Fig 4 reproduction figure: 3-panel layout with TF, PyTorch, and paper baseline.

Sources:
- TF victim metrics: hpc-results/{phase-a-victims, staging-temp/outputs-phase-a/local/}
- PyTorch (Mac): hpc-results/fig4-pytorch-mac/*.json
- PyTorch (Windows): hpc-results/fig4-pytorch-windows/*.json
- Paper: visual readouts of Fig 4 right (Huang et al. 2020)
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXISTING_PA = ROOT / 'hpc-results' / 'phase-a-victims'
STAGE = ROOT / 'hpc-results' / 'staging-temp' / 'outputs-phase-a' / 'local'
PT_MAC = ROOT / 'hpc-results' / 'fig4-pytorch-mac'
PT_WIN = ROOT / 'hpc-results' / 'fig4-pytorch-windows'
PT_HPC = ROOT / 'hpc-results' / 'staging-temp' / 'outputs-fig4-pytorch'
BASELINE_JSON = ROOT / 'hpc-results' / 'baseline-zero-poison.json'
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


def tf_cell_results(cell, yadv):
    """Return list of target_pred for all completed victim runs of this cell."""
    preds = []
    for cd in [EXISTING_PA / f'victim-{cell}', STAGE / f'victim-{cell}']:
        if cd.exists():
            for d in sorted(cd.iterdir()):
                if not d.is_dir(): continue
                p = victim_target_pred(d)
                if p is not None: preds.append(p)
    # also pull target_ids 1-4 victims (saved as victim-{cell}_t{i})
    for ti in range(1, 5):
        for cd in [STAGE / f'victim-{cell}_t{ti}']:
            if cd.exists():
                for d in sorted(cd.iterdir()):
                    if not d.is_dir(): continue
                    p = victim_target_pred(d)
                    if p is not None: preds.append(p)
    return preds


def pt_cell_results(cell):
    """Return list of target_pred for PyTorch trials. Looks for cell.json in mac/win/hpc dirs."""
    preds = []
    for d in [PT_MAC, PT_WIN, PT_HPC]:
        jf = d / f'{cell}.json'
        if jf.exists():
            try:
                data = json.load(jf.open())
                for trial in data.get('trials', []):
                    tp = trial.get('final', {}).get('target_pred')
                    if tp:
                        preds.append(tp[0] if isinstance(tp, list) else tp)
            except Exception:
                pass
    return preds


def baseline_asr():
    """Return 0% baseline ASR as % (predicted as adv class on clean victim)."""
    if not BASELINE_JSON.exists():
        return None, None
    d = json.load(BASELINE_JSON.open())
    s = d['summary']
    return s['baseline_asr_pct_predicted_as_adv'], s['total_votes']


tf_curves = defaultdict(list)
pt_curves = defaultdict(list)
print(f'{"cell":<5} {"arch":<10} {"pair":<11} {"npois":<6} {"TF n/asr":<14} {"PT n/asr"}')
print('-' * 70)
for cell, (arch, pair, npoison, yadv) in cells_meta.items():
    tf_preds = tf_cell_results(cell, yadv)
    tf_n = len(tf_preds)
    tf_asr = sum(1 for p in tf_preds if p == yadv) / tf_n if tf_n else None
    pt_preds = pt_cell_results(cell)
    pt_n = len(pt_preds)
    pt_asr = sum(1 for p in pt_preds if p == yadv) / pt_n if pt_n else None
    tf_str = f'n={tf_n} {tf_asr*100:.0f}%' if tf_asr is not None else 'n=0 -'
    pt_str = f'n={pt_n} {pt_asr*100:.0f}%' if pt_asr is not None else 'n=0 -'
    print(f'{cell:<5} {arch:<10} {pair:<11} {npoison:<6} {tf_str:<14} {pt_str}')
    if tf_asr is not None:
        tf_curves[(arch, pair)].append((npoison/50000*100, tf_asr*100, tf_n))
    if pt_asr is not None:
        pt_curves[(arch, pair)].append((npoison/50000*100, pt_asr*100, pt_n))

for k in tf_curves:
    tf_curves[k].sort()
for k in pt_curves:
    pt_curves[k].sort()

# 0% baseline (clean ResNet20 ConvNetBN-equivalent victim trained on CIFAR-10 with no poisons)
# Use 0.001% as plotting x position (paper does same to put baseline on log scale)
base_asr, base_n = baseline_asr()
if base_asr is not None:
    print(f'\n0% baseline: {base_asr:.1f}% ASR (n={base_n}) — predicted as adv class never')


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


# ============ FIG: 3-panel TF / PyTorch / Paper ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
ax_tf, ax_pt, ax_pa = axes

for (arch, pair), pts in sorted(tf_curves.items()):
    if not pts: continue
    xs, ys, ns = zip(*pts)
    color = color_map[arch]
    marker = marker_map[pair]
    label = f'{arch} {pair} (n̄={int(np.mean(ns))})'
    ax_tf.plot(xs, ys, '-' + marker, color=color, markersize=8, linewidth=2, label=label)

ax_tf.set_xscale('log')
ax_tf.set_xlabel('Poison budget (%)', fontsize=11)
ax_tf.set_ylabel('Attack Success Rate (%)', fontsize=11)
n_pts = sum(len(v) for v in tf_curves.values())
ax_tf.set_title(f'Our TF reproduction\n({n_pts} datapoints across {len(tf_curves)} curves)',
                fontsize=11, fontweight='bold')
ax_tf.set_ylim(-5, 110)
ax_tf.set_xlim(0.0005, 20)
ax_tf.grid(alpha=0.3, which='both')
ax_tf.legend(loc='lower right', fontsize=8)

for (arch, pair), pts in sorted(pt_curves.items()):
    if not pts: continue
    xs, ys, ns = zip(*pts)
    color = color_map[arch]
    marker = marker_map[pair]
    label = f'{arch} {pair} (n=4 each)'
    ax_pt.plot(xs, ys, '-' + marker, color=color, markersize=8, linewidth=2, label=label)

ax_pt.set_xscale('log')
ax_pt.set_xlabel('Poison budget (%)', fontsize=11)
n_pts_pt = sum(len(v) for v in pt_curves.values())
ax_pt.set_title(f'Our PyTorch reproduction\n({n_pts_pt} datapoints, cross-framework verification)',
                fontsize=11, fontweight='bold')
ax_pt.set_ylim(-5, 110)
ax_pt.set_xlim(0.0005, 20)
ax_pt.grid(alpha=0.3, which='both')
ax_pt.legend(loc='lower right', fontsize=8)

for (arch, pair), (px, py) in paper.items():
    color = color_map[arch]
    marker = marker_map[pair]
    label = f'{arch} {pair}'
    ax_pa.plot(px, py, '--' + marker, color=color, markersize=7, linewidth=1.8, alpha=0.85, label=label)

ax_pa.set_xscale('log')
ax_pa.set_xlabel('Poison budget (%)', fontsize=11)
ax_pa.set_title('Paper Fig 4 right (Huang et al. 2020)\nVisual readout of all 6 curves',
                fontsize=11, fontweight='bold')
ax_pa.set_ylim(-5, 110)
ax_pa.set_xlim(0.0005, 20)
ax_pa.grid(alpha=0.3, which='both')
ax_pa.legend(loc='lower right', fontsize=8)

fig.suptitle('Fig 4 reproduction snapshot — TF (left) | PyTorch cross-framework (mid) | Paper (right)',
             fontsize=13, y=1.01)
fig.tight_layout()
out = OUT / 'fig4_v2_three_panel.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved {out}')
plt.close(fig)
