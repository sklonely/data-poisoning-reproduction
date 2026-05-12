# Experiment Cell Mapping (A01–A30) and Naming Convention

If you're picking this repo up for the first time, this is the decoder ring. Every
crafted-poison dataset and every victim run is labelled with a code like
`A03_t2_s5` — this doc explains exactly what that means.

---

## TL;DR

| Cell range | Architecture | Class pair      | Target class | Adv (poison) class |
|------------|--------------|-----------------|--------------|--------------------|
| A01–A05    | ConvNetBN    | dog → bird      | 2 (dog)      | 5 (bird)           |
| A06–A10    | ConvNetBN    | frog → airplane | 0 (airplane) | 6 (frog)           |
| A11–A15    | VGG13BN      | dog → bird      | 2 (dog)      | 5 (bird)           |
| A16–A20    | VGG13BN      | frog → airplane | 0 (airplane) | 6 (frog)           |
| A21–A25    | ResNet       | dog → bird      | 2 (dog)      | 5 (bird)           |
| A26–A30    | ResNet       | frog → airplane | 0 (airplane) | 6 (frog)           |

Within each block of 5, the columns scan over **poison budget** (Fig 4 x-axis):

| Offset | npoison | Budget (of 50 000 CIFAR-10 train) |
|--------|---------|-----------------------------------|
| +0     | 5       | 0.01%                             |
| +1     | 50      | 0.1%                              |
| +2     | 500     | 1%                                |
| +3     | 2 500   | 5%                                |
| +4     | 5 000   | 10%                               |

So for example:

- **A03** = ConvNetBN, dog→bird, **1 % budget** (500 poisons) — the cell that gives 100% ASR in our headline result.
- **A18** = VGG13BN, frog→airplane, **1 % budget**.
- **A25** = ResNet, dog→bird, **10 % budget**.

Source of truth: [experiments/manifest_phase_a.csv](../experiments/manifest_phase_a.csv).

---

## Class-index legend (CIFAR-10)

| Index | Class      |
|-------|------------|
| 0     | airplane   |
| 1     | automobile |
| 2     | bird       |
| 3     | cat        |
| 4     | deer       |
| 5     | dog        |
| 6     | frog       |
| 7     | horse      |
| 8     | ship       |
| 9     | truck      |

The paper's convention (which we follow) is:
- **Target class** = the class the *attacker wants the victim to misclassify* at test time. So "dog→bird" means the target images are **dogs** that the victim mislabels as **bird**.
- **Adv / yadv class** = the *label the attacker wants the victim to output*. Equivalently, **the class whose images get perturbed** in the training set.
- **Poison class** = same as adv class in this paper (poisons live in the adv-class bucket of the train set, mid-perturbed away from clean bird-of-bird examples).

So "dog→bird" poisons are images that *look like* birds in the training set but have been
nudged so that a victim trained on them will, at test time, classify a held-out
**dog photo** as a **bird**.

---

## The full naming convention

A victim run's full identifier looks like `A03_t2_s5`. Three parts:

```
A03      = cell ID (= arch + class pair + budget; see table above)
_t2      = target_id 2 (which test-set image is the target; we ran 0..4 = 5 targets)
_s5      = seed index 5 (which random seed for victim training; we ran 0..5 = 6 seeds)
```

Each cell × target × seed combination is **one ASR vote**. For ConvNetBN we
report **n=30 per cell** (5 targets × 6 seeds).

On disk, cells A01–A04 actually have **36** victim runs: an early target_id=0
batch (job 20259324) wrote 6 victims before the staging batch re-ran
target_id=0 with another 6, so target_id=0 ends up with 12 runs there (t1–t4
have 6 each). The figure builder
[report/build_fig4_convnetbn.py](../report/build_fig4_convnetbn.py)
**uses only the staging target_id=0 batch** (and caps every target at 6 seeds)
so all cells report a uniform n=30. The dropped early-batch votes agree with
the staging votes within each cell (all 12 predictions match), so this is a
cosmetic uniformity choice, not a result change — the only number that moves is
dog→bird @0.1%, which is 3/30 = 10% under the n=30 design vs 3/36 ≈ 8% if you
keep the extras.

Craft jobs are similarly named `craft-A03_t2` (one craft per target, reused across
all 6 seeds — crafting is the expensive part, victim training is cheap).

---

## Phase naming

| Phase | What it does                                | Manifest                                          |
|-------|---------------------------------------------|---------------------------------------------------|
| Phase A | Fig 4 — ASR vs poison budget, 3 archs × 2 pairs | `manifest_phase_a*.csv`                       |
| Phase B | Fig 5 — transfer matrix (planned, not run)  | n/a (would reuse Phase A 1 % crafts)              |
| Phase C | Fig 7 — alt schemes (self-conceal, multi-class) | n/a (planned)                                  |
| Phase D | Fig 3 — fine-tuning + FC baseline (Shafahi 2018) | `manifest_phase_d_fc_*.csv`                  |

This repo's headline result is **Phase A, ConvNetBN only** (A01–A10) at n=30.
VGG13BN and ResNet cells (A11–A30) have partial data on the OSU HPC share but were
out of scope for the final write-up.

---

## What lives where

```
experiments/manifest_phase_a.csv             # 30 base cells (arch × pair × budget)
experiments/manifest_phase_a_victims.csv     # 1800 (cell, target_id, seed) rows
experiments/manifest_phase_a_n30_*_victims.csv  # the cell × target × seed expansion we actually ran

hpc-results/phase-a-victims/victim-A01/...   # early A01-A04 metrics pulled from HPC
hpc-results/staging-temp/outputs-phase-a/local/victim-A{cell}_t{tid}/{run_uid}/
                                             # later cells, target_id-tagged dirs;
                                             # metrics.jsonl + meta.json per run
hpc-results/baseline-zero-poison.json        # ConvNetBN clean victim, dog-bird (0% baseline)
hpc-results/baseline-zero-poison-plane.json  # ConvNetBN clean victim, frog-plane (0% baseline)
```

The figure builder [report/build_fig4_convnetbn.py](../report/build_fig4_convnetbn.py)
walks both `phase-a-victims/` and `staging-temp/outputs-phase-a/local/` so it
doesn't matter where a given cell's data ended up.

---

## How to add a new cell

1. Pick an unused suffix (A31, A32, …).
2. Add a row to `experiments/manifest_phase_a.csv` with the canonical columns
   `cell_id,arch,target_class,poison_class,ytargetadv,npoison,ncraftstep,nreplay,objective,note`.
3. For target_id-fanout, generate rows in the corresponding victim manifest:
   one `(cell_id, target_id ∈ 0..4)` per craft, then six seeds per craft.
4. The SLURM array scripts in `slurm/` read from the manifest CSVs by row index,
   so a sequential array `--array=1-N` maps directly to your new rows.

---

## Common gotchas

- **target_class is the dog, adv_class is the bird** — easy to flip if you read
  paper §3 without looking at the code first.
- **`-targetclass` in the CLI maps to the paper's "target class"** (the class
  whose test images get attacked), not the "target prediction." Same convention,
  different word from how some other poisoning papers use the term.
- **`A22` was the original missing craft cell** in our HPC sweep — that's a
  scheduling artefact, not a methodology choice.
