"""Generate Phase D manifests for the Fig 3 fine-tuning comparison.

Two manifest files:
  manifest_phase_d_fc_crafts.csv   — FC poison crafts (one per (target, base_class, n_poison, watermark))
  manifest_phase_d_fc_victims.csv  — fine-tune victim runs (one per (craft, seed))

Scope (kept tight for GPU budget; paper used 30 targets × 7 budgets):
  - 5 target instances (target_idx in CIFAR test set within target_class)
  - 6 poison budgets {1, 5, 10, 25, 50, 100}
  - 2 class pairs: (target=bird, base=dog) and (target=plane, base=frog)
  - 2 watermark variants: {with, without} eps clip — 'with' uses eps=16/255 ≈ paper clean-label; 'without' = unbounded perturbation
  - 3 victim seeds per craft
  → 5 × 6 × 2 × 2 = 120 FC crafts,  120 × 3 = 360 fine-tune victim runs
"""
import csv
from pathlib import Path

# CIFAR-10 class indices: 0=plane 1=car 2=bird 3=cat 4=deer 5=dog 6=frog 7=horse 8=ship 9=truck
CLASS_PAIRS = [
    {'name': 'bird-dog',   'target_class': 2, 'base_class': 5},  # bird image misclassified as dog
    {'name': 'plane-frog', 'target_class': 0, 'base_class': 6},  # plane image misclassified as frog
]
TARGET_INDICES = list(range(0, 5))
NPOISON_BUDGETS = [1, 5, 10, 25, 50, 100]
WATERMARK_VARIANTS = [
    {'name': 'wm', 'eps': 16},        # eps in 0..255 image space; ~mimics clean-label budget
    {'name': 'nowm', 'eps': None},    # unbounded perturbation
]
VICTIM_SEEDS = list(range(0, 3))

OUT_DIR = Path(__file__).parent
crafts_path = OUT_DIR / 'manifest_phase_d_fc_crafts.csv'
victims_path = OUT_DIR / 'manifest_phase_d_fc_victims.csv'


def craft_uid(pair_name, target_idx, n_poison, wm_name):
    return f'D-fc-{pair_name}-t{target_idx}-n{n_poison}-{wm_name}'


def main():
    crafts = []
    for pair in CLASS_PAIRS:
        for target_idx in TARGET_INDICES:
            for npoison in NPOISON_BUDGETS:
                for wm in WATERMARK_VARIANTS:
                    uid = craft_uid(pair['name'], target_idx, npoison, wm['name'])
                    crafts.append({
                        'craft_uid': uid,
                        'method': 'fc',
                        'pair_name': pair['name'],
                        'target_class': pair['target_class'],
                        'base_class': pair['base_class'],
                        'target_idx': target_idx,
                        'n_poison': npoison,
                        'eps': '' if wm['eps'] is None else wm['eps'],
                        'watermark': wm['name'],
                    })

    with crafts_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'craft_uid', 'method', 'pair_name', 'target_class', 'base_class',
            'target_idx', 'n_poison', 'eps', 'watermark',
        ])
        w.writeheader()
        for c in crafts:
            w.writerow(c)
    print(f'Wrote {crafts_path} ({len(crafts)} rows)')

    victims = []
    for c in crafts:
        for seed in VICTIM_SEEDS:
            victims.append({
                'victim_id': f"{c['craft_uid']}-s{seed}",
                'craft_uid': c['craft_uid'],
                'pair_name': c['pair_name'],
                'target_class': c['target_class'],
                'base_class': c['base_class'],
                'target_idx': c['target_idx'],
                'n_poison': c['n_poison'],
                'watermark': c['watermark'],
                'seed_idx': seed,
            })
    with victims_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'victim_id', 'craft_uid', 'pair_name', 'target_class', 'base_class',
            'target_idx', 'n_poison', 'watermark', 'seed_idx',
        ])
        w.writeheader()
        for v in victims:
            w.writerow(v)
    print(f'Wrote {victims_path} ({len(victims)} rows)')


if __name__ == '__main__':
    main()
