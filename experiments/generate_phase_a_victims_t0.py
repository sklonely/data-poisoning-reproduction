"""Generate a target_id=0-only victim manifest with sequential rows.

Lets us submit a SLURM array `--array=25-180` and actually get A05_s0..A30_s5 (the
intended A05-A30 × 6 seeds for target_id=0). The full manifest is cell-major, so
range queries over target_id=0 are non-contiguous in that file."""
import csv
from pathlib import Path

base_cells = [f'A{i:02d}' for i in range(1, 31)]
seeds = list(range(0, 6))

out_path = Path(__file__).parent / 'manifest_phase_a_victims_t0.csv'
with out_path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for cell in base_cells:
        for s in seeds:
            craft_uid = cell  # target_id=0 has craft_uid == cell_id
            craftproj = f'craft-{craft_uid}'
            victim_id = f'{craft_uid}_s{s}'
            w.writerow([victim_id, craft_uid, craftproj, cell, 0, s])

print(f'Wrote {out_path}')
