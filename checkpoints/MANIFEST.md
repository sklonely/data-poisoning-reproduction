# A01–A10 Poisoned-Dataset Checkpoints

These are the **exported poisoned datasets** for the 10 ConvNetBN cells of our
Figure 4 reproduction. Each `.pkl` contains all 50 000 CIFAR-10 training images
with the npoison-many adversarial poisons substituted in, plus the held-out
target image. Drop one into a victim trainer and you get an ASR vote in ~15 min
on a single GPU — no crafting required.

## File list

| File                       | Cell | Architecture | Class pair      | Budget | npoison | Size  |
|----------------------------|------|--------------|-----------------|--------|---------|-------|
| `A01-poisondataset.pkl`    | A01  | ConvNetBN    | dog → bird      | 0.01%  | 5       | 704 MB |
| `A02-poisondataset.pkl`    | A02  | ConvNetBN    | dog → bird      | 0.1%   | 50      | 704 MB |
| `A03-poisondataset.pkl`    | A03  | ConvNetBN    | dog → bird      | 1%     | 500     | 704 MB |
| `A04-poisondataset.pkl`    | A04  | ConvNetBN    | dog → bird      | 5%     | 2 500   | 704 MB |
| `A05-poisondataset.pkl`    | A05  | ConvNetBN    | dog → bird      | 10%    | 5 000   | 704 MB |
| `A06-poisondataset.pkl`    | A06  | ConvNetBN    | frog → airplane | 0.01%  | 5       | 704 MB |
| `A07-poisondataset.pkl`    | A07  | ConvNetBN    | frog → airplane | 0.1%   | 50      | 704 MB |
| `A08-poisondataset.pkl`    | A08  | ConvNetBN    | frog → airplane | 1%     | 500     | 704 MB |
| `A09-poisondataset.pkl`    | A09  | ConvNetBN    | frog → airplane | 5%     | 2 500   | 704 MB |
| `A10-poisondataset.pkl`    | A10  | ConvNetBN    | frog → airplane | 10%    | 5 000   | 704 MB |

Total: **6.9 GB**. Integrity hashes: see [SHA256SUMS](SHA256SUMS).

For the full cell decoder ring (A11–A30, target_id / seed convention, class-index
legend), read [../docs/CELLS.md](../docs/CELLS.md).

## Cloud share

> **TODO**: drop the cloud URL here once uploaded.
>
> Example layout:
> - Google Drive folder: `<paste link>`
> - Or per-file mirrors: `A01 → <link>`, `A02 → <link>`, …

To verify a downloaded file is intact:

```bash
sha256sum -c SHA256SUMS
# expected: every line ends with "OK"
```

## Recommended use

The headline result of this repo only needs the **A03** (ConvNetBN dog→bird @1%)
and **A08** (frog→airplane @1%) pkls if you just want to confirm the "100% ASR
at 1% budget" claim — those are the two most striking cells. The rest let you
sweep the whole budget curve.

**Step-by-step single-GPU reproduction:** [../docs/REPRODUCE_LEVEL2.md](../docs/REPRODUCE_LEVEL2.md)
walks through downloading A03, verifying the SHA256, setting up the venv, running
the victim, and reading off the ASR. ~15 min for a single trial.

```bash
# Pull A03 only (the headline 1% cell)
# (download from cloud share, drop into checkpoints/)
sha256sum -c <(grep A03 SHA256SUMS)

# Train a TF victim on it
.venv-tf-compat\Scripts\python.exe official-metapoison\victim.py \
  my-A03-replication -artifactroot runs-local -workspace local \
  -gpu 0 -net ConvNetBN -targetclass 2 -ytargetadv 5 -targetids 0 \
  -nbatch 400 -batchsize 125 -npoison 500 \
  -loadpoisondataset checkpoints/A03-poisondataset.pkl

# Train a PyTorch victim on the same data (consistency check)
.venv-torch-local\Scripts\python.exe -m metapoison_hpc.torch_victim \
  --dataset checkpoints/A03-poisondataset.pkl \
  --arch resnet20 --epochs 200 --augment --trials 6 \
  --output checkpoints/A03-torch-results.json
```

## Provenance

Crafted on the OSU College of Engineering HPC cluster, jobs around
**2026-04-29 to 2026-05-08**. Each cell used `nmeta=24` surrogate models
(`4 GPUs × nreplay=6` via mpirun), `K=2` unroll, `ncraftstep=61`. The TF
patched fork under `official-metapoison/` is what produced them. Reproduce
from scratch with `slurm/hpc_array_craft.sbatch` (≈ 1.5 h per cell on 4×V100).
