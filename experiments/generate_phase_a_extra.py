"""Generate manifest CSV for Phase A target IDs 1-9.

The original manifest (manifest_phase_a.csv) covers target_id=0 for 30 cells.
This generates 30 cells × 9 target_ids = 270 additional crafts to match paper Fig 4
(which averages over 10 target images × 6 seeds = 60 votes per cell).
"""
import csv
from pathlib import Path

base_cells = [
    # cell_id, arch, target_class, poison_class, ytargetadv, npoison
    ('A01', 'ConvNetBN', 2, 5, 5, 5),
    ('A02', 'ConvNetBN', 2, 5, 5, 50),
    ('A03', 'ConvNetBN', 2, 5, 5, 500),
    ('A04', 'ConvNetBN', 2, 5, 5, 2500),
    ('A05', 'ConvNetBN', 2, 5, 5, 5000),
    ('A06', 'ConvNetBN', 0, 6, 6, 5),
    ('A07', 'ConvNetBN', 0, 6, 6, 50),
    ('A08', 'ConvNetBN', 0, 6, 6, 500),
    ('A09', 'ConvNetBN', 0, 6, 6, 2500),
    ('A10', 'ConvNetBN', 0, 6, 6, 5000),
    ('A11', 'VGG13BN', 2, 5, 5, 5),
    ('A12', 'VGG13BN', 2, 5, 5, 50),
    ('A13', 'VGG13BN', 2, 5, 5, 500),
    ('A14', 'VGG13BN', 2, 5, 5, 2500),
    ('A15', 'VGG13BN', 2, 5, 5, 5000),
    ('A16', 'VGG13BN', 0, 6, 6, 5),
    ('A17', 'VGG13BN', 0, 6, 6, 50),
    ('A18', 'VGG13BN', 0, 6, 6, 500),
    ('A19', 'VGG13BN', 0, 6, 6, 2500),
    ('A20', 'VGG13BN', 0, 6, 6, 5000),
    ('A21', 'ResNet', 2, 5, 5, 5),
    ('A22', 'ResNet', 2, 5, 5, 50),
    ('A23', 'ResNet', 2, 5, 5, 500),
    ('A24', 'ResNet', 2, 5, 5, 2500),
    ('A25', 'ResNet', 2, 5, 5, 5000),
    ('A26', 'ResNet', 0, 6, 6, 5),
    ('A27', 'ResNet', 0, 6, 6, 50),
    ('A28', 'ResNet', 0, 6, 6, 500),
    ('A29', 'ResNet', 0, 6, 6, 2500),
    ('A30', 'ResNet', 0, 6, 6, 5000),
]

NCRAFTSTEP = 61
NREPLAY = 6  # nproc=4 × nreplay=6 → nmeta=24 (paper-exact)
OBJECTIVE = 'cwT'

out_path = Path(__file__).parent / 'manifest_phase_a_extra.csv'
with out_path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cell_id', 'arch', 'target_class', 'poison_class', 'ytargetadv',
                'target_id', 'npoison', 'ncraftstep', 'nreplay', 'objective', 'note'])
    for cell, arch, tc, pc, yadv, npoison in base_cells:
        for target_id in range(1, 10):
            uid = f'{cell}_t{target_id}'
            note = f'fig4-{"dogbird" if pc==5 else "frogplane"}-{arch}-npoison{npoison}-target{target_id}'
            w.writerow([uid, arch, tc, pc, yadv, target_id, npoison, NCRAFTSTEP, NREPLAY, OBJECTIVE, note])

print(f'Wrote {out_path}')
