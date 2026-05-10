"""Generate manifests for paper-aligned Fig 4 reproduction at n=20 (=1/3 of paper's n=60).

Design: 5 targets × 4 seeds = 20 votes per cell (preserves target diversity per paper §3.2).
- target_id=0 uses existing crafts (29 cells already have these; A22 also done now)
- target_ids 1, 2, 3, 4 need NEW crafts → 4 × 30 = 120 new crafts

Victim runs needed:
- existing job 20299699 produces target_id=0 × 6 seeds — we keep all, use first 4 for the n=20 plot
- NEW victims for target_ids 1-4 × 4 seeds × 30 cells = 480 runs
"""
import csv
from pathlib import Path

base_cells = [
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
NREPLAY = 6  # paper-aligned nmeta=24
OBJECTIVE = 'cwT'
NEW_TARGET_IDS = [1, 2, 3, 4]
SEEDS_PER_TARGET = 4

# === Crafts manifest (120 rows for target_ids 1-4) ===
out_crafts = Path(__file__).parent / 'manifest_phase_a_n20_crafts.csv'
with out_crafts.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['cell_id', 'arch', 'target_class', 'poison_class', 'ytargetadv',
                'target_id', 'npoison', 'ncraftstep', 'nreplay', 'objective', 'note'])
    for cell, arch, tc, pc, yadv, npoison in base_cells:
        for tid in NEW_TARGET_IDS:
            uid = f'{cell}_t{tid}'
            note = f'fig4-n20-{"dogbird" if pc==5 else "frogplane"}-{arch}-npoison{npoison}-target{tid}'
            w.writerow([uid, arch, tc, pc, yadv, tid, npoison, NCRAFTSTEP, NREPLAY, OBJECTIVE, note])
print(f'Wrote {out_crafts}: {120} craft rows')

# === Victim manifest (480 rows for target_ids 1-4 × 4 seeds × 30 cells) ===
out_victims = Path(__file__).parent / 'manifest_phase_a_n20_victims.csv'
with out_victims.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['victim_id', 'craft_uid', 'craftproj', 'cell_id', 'target_id', 'seed_idx'])
    for cell, arch, tc, pc, yadv, npoison in base_cells:
        for tid in NEW_TARGET_IDS:
            craft_uid = f'{cell}_t{tid}'
            craftproj = f'craft-{craft_uid}'
            for s in range(SEEDS_PER_TARGET):
                victim_id = f'{craft_uid}_s{s}'
                w.writerow([victim_id, craft_uid, craftproj, cell, tid, s])
print(f'Wrote {out_victims}: {480} victim rows')
