"""Analyze partial results pulled mid-run.

Inputs:
- hpc-results/staging-temp/outputs-ablation/{A,B,C}.json
- hpc-results/staging-temp/outputs-phase-d/fc-victims/*.json (163 done so far)
- hpc-results/staging-temp/outputs-phase-a/local/victim-A0{5,6,7}/*/metrics.jsonl (new since last pull)

Outputs:
- Console summary
- report/figures/fig4_phase_a_partial.png   (n=6 Fig 4 progress)
- report/figures/fig5_fc_partial.png         (FC baseline preview)
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
STAGE = ROOT / 'hpc-results' / 'staging-temp'
EXISTING_PA = ROOT / 'hpc-results' / 'phase-a-victims'  # has A01-A04
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def victim_target_pred(victim_dir):
    """Read TF victim metrics.jsonl, return final target prediction class (argmin of class-0-N)."""
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


# ============================================================
# 1. Augmentation ablation
# ============================================================
print('=' * 78)
print('AUGMENTATION ABLATION (job 20290350)')
print('=' * 78)
ABLATION = STAGE / 'outputs-ablation'
for name, label in [
    ('A_baseline_aug_mom_wd.json', 'A: aug + mom + wd  (= round 2 PyTorch)'),
    ('B_noaug_mom_wd.json',         'B: no-aug + mom + wd  (paper-faithful)'),
    ('C_noaug_plain_sgd.json',      'C: no-aug + plain SGD (closest to TF victim)'),
]:
    d = json.load((ABLATION / name).open())
    s = d['summary']
    asr = s['attack_success_rate_mean']
    acc = s['valid_acc_mean']
    n = s['n_trials']
    preds = [t['final']['target_pred'][0] for t in d['trials']]
    true = d['trials'][0]['final']['target_true'][0]
    adv = d['trials'][0]['final']['target_adv'][0]
    print(f'  {label:50s}  ASR={asr:.3f}  acc={acc:.3f}  preds={preds}  (true={true} adv={adv})')

# ============================================================
# 2. Phase A: re-aggregate all cells now that A05-A07 added
# ============================================================
print()
print('=' * 78)
print('PHASE A VICTIMS (n=6 Fig 4, target_id=0 only)')
print('=' * 78)

cells_meta = {
    'A01': ('ConvNetBN', 'dog-bird', 5, 5),
    'A02': ('ConvNetBN', 'dog-bird', 50, 5),
    'A03': ('ConvNetBN', 'dog-bird', 500, 5),
    'A04': ('ConvNetBN', 'dog-bird', 2500, 5),
    'A05': ('ConvNetBN', 'dog-bird', 5000, 5),
    'A06': ('ConvNetBN', 'frog-plane', 5, 6),
    'A07': ('ConvNetBN', 'frog-plane', 50, 6),
    'A08': ('ConvNetBN', 'frog-plane', 500, 6),
    'A09': ('ConvNetBN', 'frog-plane', 2500, 6),
    'A10': ('ConvNetBN', 'frog-plane', 5000, 6),
    'A11': ('VGG13BN', 'dog-bird', 5, 5),
    'A12': ('VGG13BN', 'dog-bird', 50, 5),
    'A13': ('VGG13BN', 'dog-bird', 500, 5),
    'A14': ('VGG13BN', 'dog-bird', 2500, 5),
    'A15': ('VGG13BN', 'dog-bird', 5000, 5),
    'A16': ('VGG13BN', 'frog-plane', 5, 6),
    'A17': ('VGG13BN', 'frog-plane', 50, 6),
    'A18': ('VGG13BN', 'frog-plane', 500, 6),
    'A19': ('VGG13BN', 'frog-plane', 2500, 6),
    'A20': ('VGG13BN', 'frog-plane', 5000, 6),
    'A21': ('ResNet', 'dog-bird', 5, 5),
    'A22': ('ResNet', 'dog-bird', 50, 5),
    'A23': ('ResNet', 'dog-bird', 500, 5),
    'A24': ('ResNet', 'dog-bird', 2500, 5),
    'A25': ('ResNet', 'dog-bird', 5000, 5),
    'A26': ('ResNet', 'frog-plane', 5, 6),
    'A27': ('ResNet', 'frog-plane', 50, 6),
    'A28': ('ResNet', 'frog-plane', 500, 6),
    'A29': ('ResNet', 'frog-plane', 2500, 6),
    'A30': ('ResNet', 'frog-plane', 5000, 6),
}

results = {}
for cell, (arch, pair, npoison, yadv) in cells_meta.items():
    # A01-A04 in old location, A05-A07 in staging
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
        print(f'  {cell}: no data')
        continue
    preds = []
    for d in sorted(cell_dir.iterdir()):
        if not d.is_dir():
            continue
        p = victim_target_pred(d)
        if p is not None:
            preds.append(p)
    if not preds:
        print(f'  {cell}: no data')
        continue
    asr = sum(1 for x in preds if x == yadv) / len(preds)
    results[cell] = (arch, pair, npoison, yadv, preds, asr)
    print(f'  {cell} {arch:<10} {pair:<11} n={npoison:<5} preds={preds}  ASR={asr*len(preds):.0f}/{len(preds)} = {asr:.0%}')

# ============================================================
# 3. FC baseline (Phase D)
# ============================================================
print()
print('=' * 78)
print('PHASE D FC VICTIMS (Feature Collision baseline, partial)')
print('=' * 78)

FC_VIC = STAGE / 'outputs-phase-d' / 'fc-victims'
fc_results = defaultdict(list)
for jf in sorted(FC_VIC.glob('*.json')):
    try:
        d = json.load(jf.open())
    except Exception:
        continue
    # victim_id format: D-fc-{pair}-{t#}-{n#}-{wm|nowm}-s{seed}
    name = jf.stem
    parts = name.split('-')
    # Example: ['D', 'fc', 'bird', 'dog', 't0', 'n5', 'wm', 's0']
    if len(parts) < 8:
        continue
    pair = f'{parts[2]}-{parts[3]}'
    target_idx = int(parts[4][1:])
    n_poison = int(parts[5][1:])
    wm = parts[6]  # 'wm' or 'nowm'
    seed = int(parts[7][1:])
    # FC victim json has different format from torch_victim.py — check what's inside
    summary = d.get('summary', d)
    asr = summary.get('attack_success_rate_mean', summary.get('asr', summary.get('final_asr')))
    final = d.get('final', {})
    if asr is None and 'asr' in final:
        asr = final['asr']
    if asr is None:
        # Fall back to checking if any trial succeeded
        trials = d.get('trials', [])
        if trials:
            asr = sum(1 for t in trials if t.get('final', {}).get('attack_success_rate', 0) > 0) / len(trials)
    fc_results[(pair, n_poison, wm)].append({'target': target_idx, 'seed': seed, 'asr': asr})

print(f'  Aggregated across {sum(len(v) for v in fc_results.values())} FC victim runs.')
print()
print(f'  {"pair":<11} {"npoison":<8} {"wm":<5} {"n_runs":<7} {"ASR mean":<10}')
print('  ' + '-' * 50)
for key in sorted(fc_results.keys(), key=lambda k: (k[0], k[2], k[1])):
    pair, n_poison, wm = key
    runs = fc_results[key]
    asrs = [r['asr'] for r in runs if r['asr'] is not None]
    if not asrs:
        continue
    mean = np.mean(asrs)
    print(f'  {pair:<11} {n_poison:<8} {wm:<5} {len(asrs):<7} {mean:.3f}')

# Sample one FC victim JSON to understand format
print()
print('=' * 78)
print('FC VICTIM JSON SAMPLE (one file structure)')
print('=' * 78)
sample = next(FC_VIC.glob('*.json'), None)
if sample:
    d = json.load(sample.open())
    print(f'  file: {sample.name}')
    print(f'  top-level keys: {list(d.keys())}')
    if 'summary' in d:
        print(f'  summary: {d["summary"]}')
    if 'trials' in d and d['trials']:
        print(f'  trial[0] keys: {list(d["trials"][0].keys())}')
        if 'final' in d['trials'][0]:
            print(f'  trial[0].final: {d["trials"][0]["final"]}')

# ============================================================
# 4. Build progress figure (Fig 4 partial)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
# ConvNetBN dog-bird curve
xs_db = []; ys_db = []; ns_db = []
for cell, (arch, pair, npoison, yadv, preds, asr) in results.items():
    if arch == 'ConvNetBN' and pair == 'dog-bird':
        xs_db.append(npoison / 50000 * 100)
        ys_db.append(asr * 100)
        ns_db.append(len(preds))
xs_fp = []; ys_fp = []; ns_fp = []
for cell, (arch, pair, npoison, yadv, preds, asr) in results.items():
    if arch == 'ConvNetBN' and pair == 'frog-plane':
        xs_fp.append(npoison / 50000 * 100)
        ys_fp.append(asr * 100)
        ns_fp.append(len(preds))
order_db = np.argsort(xs_db)
xs_db = [xs_db[i] for i in order_db]; ys_db = [ys_db[i] for i in order_db]; ns_db = [ns_db[i] for i in order_db]
order_fp = np.argsort(xs_fp)
xs_fp = [xs_fp[i] for i in order_fp]; ys_fp = [ys_fp[i] for i in order_fp]; ns_fp = [ns_fp[i] for i in order_fp]

ax.plot(xs_db, ys_db, '-o', color='C0', markersize=10, label=f'ConvNetBN dog→bird (n=6)')
ax.plot(xs_fp, ys_fp, '-s', color='C1', markersize=10, label=f'ConvNetBN frog→plane (n=6, partial: {len(xs_fp)}/5 budgets)')

ax.set_xscale('log')
ax.set_xlabel('Poison budget (%)')
ax.set_ylabel('Attack Success Rate (%)')
ax.set_title('Phase A reproduction (snapshot, in-progress)\nConvNetBN, paper Fig 4 right panel target architecture')
ax.set_ylim(-5, 110)
ax.grid(alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
out = OUT / 'fig4_phase_a_partial.png'
fig.savefig(out, dpi=150)
print()
print(f'Saved {out}')
plt.close(fig)
