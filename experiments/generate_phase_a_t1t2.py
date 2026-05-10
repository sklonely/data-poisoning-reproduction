"""Generate sequential-row Phase A subsets for the t1+t2 expansion.

Splits into:
  manifest_phase_a_t1t2_crafts.csv               — 60 rows (30 cells × 2 targets)
  manifest_phase_a_t1t2_victims.csv              — 360 rows (60 crafts × 6 seeds)

The original manifest_phase_a_extra.csv covers t1..t9 (270 rows) but is cell-major,
so target_id=1..2 rows are scattered. With this subset, --array=N-M maps cleanly.
"""
import csv
from pathlib import Path

src = Path(__file__).parent / 'manifest_phase_a_extra.csv'
crafts_out = Path(__file__).parent / 'manifest_phase_a_t1t2_crafts.csv'
victims_out = Path(__file__).parent / 'manifest_phase_a_t1t2_victims.csv'

with src.open() as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r['target_id'] in {'1', '2'}]

with crafts_out.open('w', newline='') as f:
    fieldnames = ['cell_id', 'arch', 'target_class', 'poison_class', 'ytargetadv',
                  'target_id', 'npoison', 'ncraftstep', 'nreplay', 'objective', 'note']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f'Wrote {crafts_out} ({len(rows)} rows)')

# Build victim manifest from these crafts × 6 seeds
seeds = list(range(0, 6))
with victims_out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for r in rows:
        cell_id = r['cell_id'].split('_')[0]   # 'A01_t1' -> 'A01'
        craft_uid = r['cell_id']                # 'A01_t1'
        craftproj = f'craft-{craft_uid}'
        for s in seeds:
            w.writerow([f'{craft_uid}_s{s}', craft_uid, craftproj, cell_id, r['target_id'], s])
print(f'Wrote {victims_out} ({len(rows) * 6} rows)')
