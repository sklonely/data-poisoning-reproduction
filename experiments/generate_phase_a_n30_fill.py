"""Fill the gap to reach n=30 for ConvNetBN A01-A10.

Current state:
- A01: t0=6, t1=6, t2=6, t3=6, t4=0  → need t4 (6 seeds)
- A02-A10: t0=6, t1=6, t2=6, t3=0, t4=0 → need t3+t4 (12 seeds each)

Total: 6 + 9*12 = 114 victim runs.
"""
import csv
from pathlib import Path

rows = []
# A01 needs t4 only
for s in range(6):
    rows.append((f'A01_t4_s{s}', 'A01_t4', 'craft-A01_t4', 'A01', 4, s))
# A02-A10 need t3 + t4 (each 6 seeds)
for cell_num in range(2, 11):
    cell = f'A{cell_num:02d}'
    for tid in [3, 4]:
        cu = f'{cell}_t{tid}'
        for s in range(6):
            rows.append((f'{cu}_s{s}', cu, f'craft-{cu}', cell, tid, s))

out = Path(__file__).parent / 'manifest_phase_a_n30_fill_victims.csv'
with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    w.writerows(rows)
print(f'Wrote {out}: {len(rows)} rows')
