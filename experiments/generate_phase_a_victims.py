"""Generate victim manifest for Phase A: 30 cells × 10 target_ids × 6 seeds = 1800 victim runs."""
import csv
from pathlib import Path

# Same 30 base cells as crafts
base_cells = [f'A{i:02d}' for i in range(1, 31)]
target_ids = list(range(0, 10))
seeds = list(range(0, 6))

out_path = Path(__file__).parent / 'manifest_phase_a_victims.csv'
with out_path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for cell in base_cells:
        for tid in target_ids:
            craft_uid = cell if tid == 0 else f'{cell}_t{tid}'
            craftproj = f'craft-{craft_uid}'
            for s in seeds:
                victim_id = f'{craft_uid}_s{s}'
                w.writerow([victim_id, craft_uid, craftproj, cell, tid, s])

print(f'Wrote {out_path}')
