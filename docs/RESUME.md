# Project Resume / Checkpoint Doc — MetaPoison Reproduction

**Last updated: 2026-05-05**

This document is the single source of truth for picking the project back up after a pause. It lists what is done, what is in-flight, what is left, and the exact commands/files to resume each piece.

---

## 1. Top-level state (one-glance summary)

| Aspect | Status |
|---|---|
| Paper studied | ✅ MetaPoison NeurIPS 2020 (arXiv 2004.00225v2) |
| Code patched & uploaded | ✅ on HPC at `/nfs/hpc/share/chanc7/metapoison/repo/` |
| HPC envs ready | ✅ split `.venv-tf` + `.venv-torch`, mpi4py linked to OpenMPI 3.1 |
| Round 1 (smoke baseline) | ✅ ASR=0 (intentional under-spec, validates pipeline) |
| Round 2 (paper-scale validation) | ✅ TF ASR 3/3=100%, PyTorch ASR 1/3=33% |
| Phase A (Fig 4 grid) crafts | 🟡 29/300 cells crafted (target_id=0 for all 30 except A22) |
| Phase A victims | 🟡 24/1800 victims completed (A01-A04 × 6 seeds) |
| Phase B (Fig 5 transfer) | ⏸ 0% — depends on Phase A ResNet/VGG13/ConvNetBN at 1% budget (have crafts; need victim runs) |
| Phase C (Fig 7 alt schemes) | ⏸ 0% — code already exists in repo (`-objective xentC` and `-multiclasspoison` flags) |
| Phase D (Fig 3 fine-tuning) | 🟡 **Code skeleton complete** (FC craft + fine-tune victim + pretrain script + manifests + sbatches all written and unit-tested); pretrain job 20299700 RUNNING |
| Final report | ✅ Checkpoint Presentation 2 draft (`checkpoints/CHECKPOINT_2_zh.md` aligned to new narrative arc 2026-05-05) |
| Augmentation ablation (Fig 5 §3.3 axis) | ✅ job 20290350 done (3 conditions, A=0/3 ASR aug-on, B/C=3/3 aug-off) |
| HPC active queue | 🟢 **2 jobs running**: 20299699 (Phase A victim t0 A05-A30, 156 tasks) + 20299700 (Phase D pretrain) |

---

## 2. What's done — detailed inventory

### 2.1 Code work (locally + on HPC)

**Patches to forked upstream `official-metapoison/`** (vs upstream `298f2c0` master):

| File | What changed | Why |
|---|---|---|
| `tf_compat.py` (NEW) | Re-export `tf.compat.v1`, polyfill `xavier_initializer*` | TF 2.x doesn't have `tf.contrib.layers.xavier_initializer*` |
| `tracking.py` (NEW) | `LocalExperiment`, `LocalAPI`, `LocalExperimentView` | Replaces Comet ML — local file-based experiment tracking |
| `local_mpi.py` (NEW) | Dummy MPI fallback | Lets the code run single-process when mpi4py isn't available |
| `data.py`, `main.py`, `victim.py`, `utils.py`, `recolor.py`, `meta.py` | All swap `import tensorflow as tf` → `from tf_compat import tf`; all swap `mpi4py` → `local_mpi`; replace Comet calls with `LocalExperiment` | TF 2.15 compat + Comet decoupling |
| `learners/resnet.py` | (a) tf_compat imports; (b) **fixed bug** in `construct_weights` where `num_filters_in` wasn't updated inside the inner block loop | Latent upstream bug — `tf.gradients` interpreted shape mismatch as grouped conv → CPU fallback → InvalidArgumentError |
| `learners/vgg.py` | tf_compat imports | upstream's custom VGG was disabled in the dispatcher |
| `meta.py` | Lazy imports per arch + **re-route `'VGG13'`/`'VGG13BN'` to `learners/vgg.py`** instead of broken Keras path | Original upstream sent `'VGG13'` to Keras Applications which doesn't fit CIFAR-10 32×32 |
| `parse.py` | Add `-artifactroot` / `-workspace` CLI flags | Tell `LocalAPI` where to put outputs |

**New code we wrote** (non-fork):

| File | What it does |
|---|---|
| `src/metapoison_hpc/cifar_models.py` | PyTorch CIFAR ResNet-20 with Basic blocks; `features()` exposes penultimate-layer activations for FC |
| `src/metapoison_hpc/torch_victim.py` | PyTorch from-scratch victim trainer; eats `poisondataset-N.pkl` exported from TF |
| `src/metapoison_hpc/pretrain_cifar.py` | (NEW 2026-05-05) Train clean CIFAR-10 ResNet-20 → checkpoint for FC feature extractor + fine-tune init |
| `src/metapoison_hpc/feature_collision.py` | (NEW 2026-05-05) Phase D / Fig 3 baseline. Implements Shafahi et al. 2018 forward-backward splitting (FBS) FC algorithm |
| `src/metapoison_hpc/torch_finetune_victim.py` | (NEW 2026-05-05) Fine-tune victim: load pretrained → train on (clean ∪ poisons) for 10 epochs → ASR |
| `src/metapoison_hpc/test_phase_d_smoke.py` | (NEW 2026-05-05) Smoke test for Phase D pipeline; verifies FC math drives feat_l2 down |
| `experiments/manifest_phase_a.csv` | 30 base cells (3 archs × 2 class pairs × 5 budgets) for target_id=0 |
| `experiments/manifest_phase_a_extra.csv` | 270 cells covering target_ids 1-9 (paper averages over 10 target images) |
| `experiments/manifest_phase_a_victims.csv` | 1800 (cell, target_id, seed) victim runs (cell-major layout) |
| `experiments/manifest_phase_a_victims_t0.csv` | (NEW 2026-05-05) target_id=0-only victim manifest with sequential 180 rows; lets `--array=N-M` map directly to A01..A30 × seeds |
| `experiments/manifest_phase_d_fc_crafts.csv` | (NEW 2026-05-05) 120 FC crafts: 5 targets × 6 budgets × 2 class pairs × 2 watermark variants |
| `experiments/manifest_phase_d_fc_victims.csv` | (NEW 2026-05-05) 360 fine-tune victim runs: 120 crafts × 3 seeds |
| `experiments/generate_phase_a_extra.py`, `generate_phase_a_victims.py`, `generate_phase_a_victims_t0.py`, `generate_phase_d.py` | Manifest generators |
| `hpc_run.sbatch` / `hpc_run_v2.sbatch` | Single-job pipelines (Round 1/2) |
| `hpc_array_craft.sbatch`, `hpc_array_craft_extra.sbatch` | SLURM array templates for Phase A craft |
| `hpc_array_victim.sbatch` | SLURM array for Phase A victim. Now reads `MANIFEST_PATH` env override (defaults to full cell-major manifest) |
| `hpc_pretrain.sbatch` | (NEW 2026-05-05) One-shot Phase D pretrain job |
| `hpc_array_phase_d_fc_craft.sbatch` | (NEW 2026-05-05) 120-task SLURM array for FC crafts |
| `hpc_array_phase_d_fc_victim.sbatch` | (NEW 2026-05-05) 360-task SLURM array for FC fine-tune victims |
| `hpc_ablation_torch.sbatch` | augmentation ablation (job 20290350, 2026-05-04) |
| `report/build_figures.py` | Result aggregator → produces Fig 1/2/3 |
| `checkpoints/CHECKPOINT_2_zh.md` | Slide-deck draft (aligned to new narrative arc 2026-05-05) |
| `report/figures/fig1..3.png` | Built figures |

### 2.2 Experimental runs completed

| Run | Job ID | Date | Spec | Result | Archive |
|---|---|---|---|---|---|
| **Round 1** | 20240192 | 2026-04-26 | 1×V100, nmeta=2, ResNet, dog-bird, 5000 poison | TF ASR 0/3, PyTorch ASR 0/3, 4h37m | `hpc-results/round1-job20240192/` |
| **Round 2** | 20241084 | 2026-04-27 | 4×V100 mpirun, nmeta=16, ResNet, dog-bird, 5000 poison | **TF ASR 3/3=100%, PyTorch ASR 1/3=33%**, 6h54m, valid_acc 84.3% | `hpc-results/round2-job20241084/` (with poison artifacts) |
| **Phase A craft batch 1** | 20253943 | 2026-04-29~30 | 30 cells × target_id=0, 4×V100 mpirun nmeta=24 | **29/30 craft cells done** (A22 cancelled before scheduled) | `outputs-phase-a/local/craft-A{01..30}/` on HPC |
| **Phase A craft batch 2** | 20255388 | 2026-04-29 | 270 cells × target_ids 1-9 | **0 done** (cancelled before any started) | n/a |
| **Phase A victim batch 1** | 20255418 | 2026-04-30 | 600 victim tasks (buggy skip-guard) | 6 done, 35 false-failed | partially in `outputs-phase-a/local/victim-A01/` |
| **Phase A victim batch 1 (fixed)** | 20259324 | 2026-04-30 | 600 victim tasks with `nullglob` guard | **24 done** (A01-A04 × 6 seeds), rest skipped or cancelled | `hpc-results/phase-a-victims/` (locally) |

### 2.3 Results we already pulled to local

- `hpc-results/round1-job20240192/` — full logs + metrics (no poison artifacts pulled — they're useless since attack failed)
- `hpc-results/round2-job20241084/` — full logs + metrics + **poison checkpoints + exported `poisondataset-60.pkl`** (704 MB)
- `hpc-results/phase-a-victims/` — 24 victim experiments with `metrics.jsonl` for A01-A04
- `hpc-results/poison-assets-both.tar` — was a corrupted intermediate; safe to delete

### 2.4 Results still on HPC share but not yet pulled

- 25 Phase A craft poison artifacts (cells A05-A21, A23-A30) — each ~60 MB × 7 checkpoints = ~400 MB per cell, total ~10 GB
- These are **vulnerable to share-space cleanup**. Next pause-resumption should rsync them down.

### 2.5 Memory entries (auto-loaded across sessions)

Saved at `~/.claude/projects/d--HW-CS-539-final-project/memory/`:
- `MEMORY.md` (index)
- `project_metapoison_repro.md` — project context + empirical results
- `reference_hpc_workspace.md` — HPC paths + venv layout + compatibility constraints
- `project_resnet_upstream_bug.md` — note about the resnet construct_weights fix

---

## 3. What's NOT done — task-by-task with resume command

### 3.1 Phase A: finish the Fig 4 grid

| Sub-task | Resume command (after `cd /nfs/hpc/share/chanc7/metapoison/`) |
|---|---|
| **A22 missing craft** (ResNet dog-bird 50 poisons, target_id=0) | `sbatch --array=22 hpc_array_craft.sbatch` |
| **270 t1-t9 crafts** | `sbatch slurm/hpc_array_craft_extra.sbatch` (already configured `--array=1-270%6`; the nullglob guard now works so re-submitting is safe — already-done cells skip immediately) |
| **Victims for cells A05-A30** (the cells we have crafts for but no victims yet) | `sbatch --export=ALL,VICTIM_OFFSET=0 hpc_array_victim.sbatch` and continue with offsets 600 / 1200 once quota frees |
| **Pull all artifacts back to local** | `tar -cf - outputs-phase-a/ | piped to local tar -xf -` (see `hpc_run_v2.sbatch` epilogue style) |

Cost estimate: ~75 GPU-day total for full Phase A (270 crafts × ~0.2 GPU-day each + 1776 victims × ~0.04 GPU-day each)

### 3.2 Phase B: Fig 5 transfer matrix + robustness — *NO new crafts needed*

Reuses Phase A crafts at 1% budget (cells A03 ConvNetBN, A13 VGG13BN, A23 ResNet for dog-bird; A08/A18/A28 for frog-plane).

**To do**:
- Write `manifest_phase_b_transfer.csv` with rows `(victim_arch, craft_uid, seed)` — 3 archs × 3 crafts × 6 class-pair × 60 victims = **~3000 victim runs**
- Write `manifest_phase_b_robustness.csv` — 8 hyperparam variants × 30 victims on A03 (ConvNetBN baseline) = **240 victim runs**
- Reuse `hpc_array_victim.sbatch` template; add `-Xnet $VICTIM_ARCH` and `-Xlrnrate / -Xbatchsize / -Xaugment / -Xweightdecay / -Xschedule` overrides

Cost: ~25 GPU-day (victim-only)

### 3.3 Phase C: Fig 7 alternative schemes — *code already exists, just config*

Self-concealment uses the existing `-objective xentC` (already in upstream meta.py line 209). Multi-class uses the existing `-multiclasspoison` flag (in parse.py line 29).

**To do**:
- Write `manifest_phase_c_self_concealment.csv` — 2 class pairs (bird-bird, plane-plane) × 10 target_ids = 20 crafts
- Write `manifest_phase_c_multiclass.csv` — 9 yadv values × 10 target_ids = 90 crafts
- Reuse `hpc_array_craft_extra.sbatch` with extra columns for `objective` and `multiclasspoison`
- Victim runs: 2×20 + 9×60 = ~580 victims

Cost: ~30 GPU-day

### 3.4 Phase D: Fig 3 fine-tuning comparison — **code skeleton landed 2026-05-05**

| Sub-task | Status | Resume command |
|---|---|---|
| Pretrained CIFAR-10 ResNet-20 classifier | 🟡 RUNNING (job 20299700, ampere, ~1-3 hr) | `sbatch slurm/hpc_pretrain.sbatch` |
| FC algorithm (Shafahi 2018 FBS) | ✅ implemented in `src/metapoison_hpc/feature_collision.py`; smoke test passes | — |
| Fine-tune victim trainer | ✅ `src/metapoison_hpc/torch_finetune_victim.py` (load pretrained → fine-tune on clean ∪ poisons → ASR) | — |
| Phase D manifests | ✅ 120 FC crafts + 360 victim runs (5 targets × 6 budgets × 2 class pairs × 2 watermark variants × 3 seeds) | — |
| FC craft array job | ⏸ launches once pretrain finishes | `sbatch slurm/hpc_array_phase_d_fc_craft.sbatch` |
| FC victim array job | ⏸ launches once FC crafts done | `sbatch slurm/hpc_array_phase_d_fc_victim.sbatch` |
| MetaPoison fine-tune-mode crafts (TF) | ⏸ TODO — `main.py -pretrain $PRETRAIN_KEY` invocation; sbatch not yet written | — |

Cost remaining: ~10 GPU-day for FC craft+victim sweep + ~5 GPU-day for MetaPoison fine-tune crafts.

Smoke test command (local, no GPU needed beyond CPU):
```
PYTHONPATH=src python -m metapoison_hpc.test_phase_d_smoke
```

### 3.5 Final write-up

**To do**:
- Run `report/build_figures.py` after each Phase completes — will pick up new metrics automatically
- Write paper-style write-up (intro, method, experiments, results, comparison to paper)
- Build a final figure pack matching paper Fig 3, 4, 5, 7
- Optional: convert `CHECKPOINT_2.md` → PowerPoint or PDF

---

## 4. Known gotchas (don't trip on these again)

1. **`set -euo pipefail` + `ls glob 2>/dev/null | wc -l`** triggers script failure when no files match (pipefail catches `ls` non-zero exit). Use `shopt -s nullglob; arr=(...); n=${#arr[@]}; shopt -u nullglob` instead. Both `hpc_array_craft_extra.sbatch` and `hpc_array_victim.sbatch` are now patched.
2. **SLURM `MaxArraySize=1001`** on this cluster — split big arrays into multiple submissions with an `OFFSET` env var (see `VICTIM_OFFSET` pattern).
3. **QOS `MaxSubmitJobsPerUser=1000` + `MaxJobsPerUser=400`** — submit in waves, not all at once. Skip-guards make this clean: submit oversized arrays and let them no-op until prerequisites are met.
4. **`MaxCpuRunMinsPerUser` ≈ 23000 cpu-min running concurrent** — round 2 craft (4×4×1440=23040) was at the limit. Use shorter `--time` to fit more concurrent jobs.
5. **`tensorflow[and-cuda]` and torch fight in same venv** — they pin different `nvidia-*-cu12` versions and one will silently fall back to CPU or torch will undef-symbol. Always split into `.venv-tf` and `.venv-torch`.
6. **TF 2.10 cannot see CUDA 12.2 (driver 580)** — must use TF 2.15 (last with `compat.v1.disable_v2_behavior()`).
7. **Upstream `learners/resnet.py` has a latent shape bug** at `#todo no ref`. Don't revert.
8. **Upstream `meta.py` dispatches `'VGG13'` to broken Keras path**, not the custom impl in `learners/vgg.py`. We patched the dispatcher.
9. **share space `/nfs/hpc/share/chanc7/` can be cleaned without warning** — pull artifacts back as soon as a phase completes.
10. **OSU SSH banner pollutes stdout in tar pipes** — use `-o LogLevel=QUIET` and grep banner lines out, or pipe straight into `tar -xf -` (banner goes to stderr in that case).

---

## 5. Quick-reference paths

```
Local:
  d:/HW/CS 539/final project/
    official-metapoison/      # TF code (patched fork)
    src/metapoison_hpc/       # PyTorch victim
    experiments/              # manifest CSVs + generators
    slurm/                    # hpc_*.sbatch + run_metapoison.sh
    hpc-results/              # pulled artifacts (~3 GB, gitignored heavy parts)
    report/                   # figures + build scripts
    checkpoints/              # CHECKPOINT_2.md / _zh.md / .pdf (slide-deck snapshots)
    docs/                     # RESUME.md (this), osu-hpc.md, README_HPC.md
    papers/                   # reference PDFs (gitignored)

HPC (chanc7@submit.hpc.engr.oregonstate.edu):
  /nfs/hpc/share/chanc7/metapoison/
    repo/                     # code (mirror of local; patches applied)
      .venv-tf/               # TF 2.15.1 + CUDA 12.2 bundled
      .venv-torch/            # torch 2.5.1+cu124
    experiments/              # uploaded manifests
    outputs/                  # round 1 (legacy)
    outputs-v2/               # round 2 (legacy)
    outputs-phase-a/          # Phase A craft + victim experiments
    logs/                     # SLURM stdout/stderr
    slurm/                    # uploaded SLURM templates
```

---

## 6. Resume sequence (recommended order)

When picking up after a pause:

1. `ssh osu-engr` — confirm SSH still works.
2. `ssh -J osu-engr chanc7@submit.hpc.engr.oregonstate.edu "ls /nfs/hpc/share/chanc7/metapoison/"` — confirm share space wasn't cleaned. If it was, restore from `hpc-results/` + re-run `setup-official` from `slurm/run_metapoison.sh`.
3. Pull any unpulled craft artifacts back to local: `tar -czf - outputs-phase-a/local | piped to local`.
4. Decide priority among remaining phases (recommend: Phase B reusable victims first since 0 new crafts needed; then Phase C; then Phase D coding).
5. Submit chosen phase's SLURM array.
6. Re-run `report/build_figures.py` periodically as new metrics arrive.
