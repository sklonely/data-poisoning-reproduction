# MetaPoison Reproduction — CS 539 Final Project

A reproduction of **MetaPoison** (Huang, Geiping, Fowl, Taylor, Goldstein — NeurIPS 2020,
[arXiv 2004.00225v2](https://arxiv.org/abs/2004.00225)): a clean-label data poisoning
attack that approximates the bilevel poison-crafting objective via meta-learning.

This repository contains the code, manifests, SLURM scripts, and aggregated results
behind our CS 539 final-project reproduction. The focus is **Figure 4** of the paper
(ASR vs poison budget), reproduced for **ConvNetBN** at paper-faithful scale.

---

## Headline result

ConvNetBN, CIFAR-10, **n=30 votes per cell** (5 target images × 6 victim seeds; paper uses n=60 = 10 × 6).
0% baseline is a separately-trained clean victim, not extrapolation.

| Class pair      | 0%  | 0.01% (5) | 0.1% (50) | 1% (500) | 5% (2500) | 10% (5000) |
|-----------------|-----|-----------|-----------|----------|-----------|------------|
| dog → bird      | 0%  |   0%      |   8%      |  100%    |  100%     |   90%      |
| frog → airplane | 0%  |   0%      |   7%      |   63%    |   80%     |   83%      |

Direction-of-effect matches the paper Fig 4 ConvNetBN curve closely. See
[report/figures/fig4_convnetbn_only.png](report/figures/fig4_convnetbn_only.png) for
our reproduction (left) side-by-side with the cropped paper figure (right).

**Collaborators**: read [docs/CELLS.md](docs/CELLS.md) first — it's the decoder
ring for `A01`–`A30` cell codes and the `A03_t2_s5` victim-run naming convention.

---

## Repository layout

```
official-metapoison/   Patched fork of Huang et al.'s TF code (the craft side)
src/metapoison_hpc/    Our PyTorch victim + Phase-D feature-collision baseline
experiments/           CSV manifests + generators for SLURM array sweeps
slurm/                 SLURM array templates + run_metapoison.sh
report/                Figure builders + figures/ + paper-image extractors
checkpoints/           Slide-deck snapshots (CHECKPOINT_2 markdown + PDF, EN + 中文)
hpc-results/           Aggregated metrics (heavy artifacts are gitignored)
docs/                  RESUME.md (project state), osu-hpc.md, legacy HPC notes
papers/                Reference PDFs (gitignored — not redistributable)
```

---

## Setup

Two **separate** virtualenvs because TF and PyTorch fight over `nvidia-*-cu12` pins:

```powershell
# Crafting side — TensorFlow 2.15 in compat.v1 mode
uv venv --python 3.10 .venv-tf-compat
uv pip install --python .venv-tf-compat\Scripts\python.exe `
  "tensorflow[and-cuda]==2.15.1" "tensorflow-probability==0.23" `
  "matplotlib<3.9" "pillow<11" "numpy<2"

# Victim side — PyTorch 2.5.1 + CUDA 12.4
uv venv --python 3.10 .venv-torch-local
uv pip install --python .venv-torch-local\Scripts\python.exe `
  "torch==2.5.1" "torchvision" "numpy<2" "matplotlib<3.9"
```

On HPC the equivalent venvs live under `/nfs/hpc/share/chanc7/metapoison/repo/.venv-tf`
and `.venv-torch`. See [docs/osu-hpc.md](docs/osu-hpc.md) for the partition/QOS quirks.

---

## What we reproduced

| Component                              | Status |
|----------------------------------------|--------|
| MetaPoison TF code (Huang et al.)      | ✅ patched fork (`official-metapoison/`) |
| ConvNetBN dog→bird Fig 4 (n=30)        | ✅ 5/5 budgets + real 0% baseline |
| ConvNetBN frog→airplane Fig 4 (n=30)   | ✅ 5/5 budgets + real 0% baseline |
| TF clean baselines (other 4 archs)     | 🟡 5/6 done (ResNet frog-plane timed out) |
| PyTorch victim reproduction            | ✅ identifies augmentation as cross-framework axis |
| VGG13BN / ResNet Fig 4 grid            | ⏸ paper-scale data partially gathered (out of scope for this report) |

The PyTorch victim was a **deliberate consistency check**, not a port-then-deploy.
The augmentation ablation (job 20290350) showed that the TF↔PyTorch ASR gap collapses
once augmentation is held constant; see [checkpoints/CHECKPOINT_2_zh.md](checkpoints/CHECKPOINT_2_zh.md) for the narrative.

---

## Upstream patches worth noting

These live inside `official-metapoison/` and are unrelated to the paper's contribution:

1. **`tf_compat.py`** — re-exports `tf.compat.v1` and polyfills `xavier_initializer`
   so the original TF1.14 code runs on TF 2.15.
2. **`tracking.py`** — local file-based experiment tracking that replaces Comet ML.
3. **`learners/resnet.py`** — fixes a latent shape bug in `construct_weights` where
   `num_filters_in` wasn't updated inside the inner block loop. Upstream had a
   `#todo no ref` marker on the same line. Don't revert this.
4. **`meta.py`** — routes `'VGG13'` / `'VGG13BN'` to `learners/vgg.py` instead of the
   broken Keras Applications path.
5. **`parse.py`** — adds `-artifactroot` and `-workspace` flags so `LocalAPI` knows
   where to write outputs.
6. **`clean_baseline.py`** (new) — TF clean-victim driver for the 0% datapoint in Fig 4.

---

## Reproducing the headline result

The TF craft and victim runs live behind these SLURM array scripts. Manifests are
checked in under `experiments/`.

```bash
# Phase A: craft the 30 cells (3 archs × 2 class pairs × 5 budgets) for target_id=0
sbatch slurm/hpc_array_craft.sbatch

# Phase A: 6-seed victim sweep for ConvNetBN cells (cell-major manifest)
sbatch slurm/hpc_array_phase_a_n20_victim.sbatch
sbatch slurm/hpc_array_phase_a_n30_fill.sbatch  # tops up to n=30 per cell

# 0% baselines (one TF clean victim per arch × class pair)
sbatch slurm/hpc_clean_baselines.sbatch

# Build the headline figure
.venv-torch-local/Scripts/python.exe report/build_fig4_convnetbn.py
```

The figure script `report/build_fig4_convnetbn.py` reads aggregated `metrics.jsonl`
files out of `hpc-results/staging-temp/outputs-phase-a/local/` and writes
`report/figures/fig4_convnetbn_only.png`.

---

## Known gotchas

These are written down at the bottom of [docs/RESUME.md](docs/RESUME.md), but the most
important three:

1. **CUDA 12.2 needs TF 2.15** — TF 2.10 silently falls back to CPU on the OSU dgx2
   nodes.
2. **Never share a venv between TF and PyTorch** — undefined NCCL symbols at import.
3. **OSU HPC array size cap is 1001** — split big arrays with an `OFFSET` env var.

---

## Paper / acknowledgements

- Original paper: Huang, W. R., Geiping, J., Fowl, L., Taylor, G., & Goldstein, T.
  *MetaPoison: Practical General-purpose Clean-label Data Poisoning.* NeurIPS 2020.
- Upstream code (forked here under `official-metapoison/`):
  https://github.com/wronnyhuang/metapoison
- Compute: OSU College of Engineering HPC (V100, A40, RTX 8000, H100 nodes).

---

## Files left out of git on purpose

`hpc-results/` is heavily gitignored — only the small `metrics.jsonl` and
`baseline-*.json` summaries are tracked. The 18 GB of poison checkpoints and the
1.1 GB of legacy round-1/2 dumps live on the OSU share. The paper PDFs are also
gitignored because they aren't ours to redistribute.
