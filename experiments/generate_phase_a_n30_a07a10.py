"""Generate n30 extras manifest for cells A07-A10 only (4 cells × 4 targets × 2 seeds = 32 tasks)."""
import csv
from pathlib import Path

cells = ['A07', 'A08', 'A09', 'A10']
targets = [1, 2, 3, 4]
seeds = [4, 5]

out = Path(__file__).parent / 'manifest_phase_a_n30_a07a10_victims.csv'
with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for cell in cells:
        for tid in targets:
            cu = f'{cell}_t{tid}'
            for s in seeds:
                w.writerow([f'{cu}_s{s}', cu, f'craft-{cu}', cell, tid, s])
print(f'Wrote {out}: 32 rows')
