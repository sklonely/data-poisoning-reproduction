"""Bump n_seed from 4 to 6 for new targets (1-4) to reach n=30 = 5 targets x 6 seeds.

Existing infrastructure:
- target_id=0 already has 6 seeds (job 20299699 producing 180 victims = 30 cells x 6 seeds)
- target_ids 1-4 currently have 4 seeds (job 20306986 producing 480 victims = 30 x 4 x 4)

This file generates the EXTRA victim runs: target_ids 1-4 with seeds 4 and 5 only,
30 cells x 4 targets x 2 extra seeds = 240 new victim tasks.
"""
import csv
from pathlib import Path

base_cells = [f'A{i:02d}' for i in range(1, 31)]
EXTRA_SEEDS = [4, 5]
NEW_TARGET_IDS = [1, 2, 3, 4]

out = Path(__file__).parent / 'manifest_phase_a_n30_extra_victims.csv'
with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for cell in base_cells:
        for tid in NEW_TARGET_IDS:
            craft_uid = f'{cell}_t{tid}'
            craftproj = f'craft-{craft_uid}'
            for s in EXTRA_SEEDS:
                victim_id = f'{craft_uid}_s{s}'
                w.writerow([victim_id, craft_uid, craftproj, cell, tid, s])

print(f'Wrote {out}: 240 extra victim rows (seeds 4+5 of targets 1-4 across 30 cells)')
