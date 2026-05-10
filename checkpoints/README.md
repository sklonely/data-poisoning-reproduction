# Model checkpoints — dropzone

This folder is **gitignored** apart from this README. Drop poisoned-dataset
`.pkl` files, victim-model weights, and pretrained classifier checkpoints
here when you fetch them from HPC. The figure builders and victim trainers
will pick them up automatically.

## Where the real artefacts live

We don't push the heavy files to GitHub (700 MB per poisoned dataset × 10
ConvNetBN cells = ~7 GB). They live on the OSU HPC share:

```
/nfs/hpc/share/chanc7/metapoison/
  outputs-phase-a/local/craft-A{01..30}/<run-uid>/assets/*ckpt-*  # model + poison ckpts per craftstep
  exports/A{01..30}-poisondataset.pkl                              # exported poisoned datasets (704 MB each)
  pretrain/resnet20-cifar10.pt                                     # Phase-D pretrained classifier
```

## Pulling one cell to local

```bash
# Pull the poisoned dataset for ConvNetBN dog→bird @ 1% budget (cell A03)
rsync -avzP \
  chanc7@submit.hpc.engr.oregonstate.edu:/nfs/hpc/share/chanc7/metapoison/exports/A03-poisondataset.pkl \
  checkpoints/
```

## Using a pulled poisoned dataset

```bash
# TF victim (paper-faithful)
.venv-tf-compat\Scripts\python.exe official-metapoison\victim.py \
  A03-victim-local -artifactroot runs-local -workspace local \
  -gpu 0 -net ConvNetBN -targetclass 2 -ytargetadv 5 -targetids 0 \
  -nbatch 400 -batchsize 125 -npoison 500 \
  -loadpoisondataset checkpoints/A03-poisondataset.pkl

# PyTorch victim (consistency check)
.venv-torch-local\Scripts\python.exe -m metapoison_hpc.torch_victim \
  --dataset checkpoints/A03-poisondataset.pkl \
  --arch resnet20 --epochs 200 --augment --trials 6 \
  --output checkpoints/A03-torch-results.json
```

## What "checkpoint" means here

| Artefact                          | What it is                                      | Typical size |
|-----------------------------------|-------------------------------------------------|--------------|
| `A{cell}-poisondataset.pkl`       | All 50 000 CIFAR-10 train images with the npoison-many poisoned ones substituted in, plus the held-out target image | 704 MB |
| `outputs-phase-a/.../assets/poisoninputs-N` | Crafted poison pixels at craftstep N (intermediate; you usually want the final one) | ~60 MB / step |
| `outputs-phase-a/.../assets/weights-N` | Surrogate model weights at craftstep N | ~few MB |
| `pretrain/resnet20-cifar10.pt`    | Pretrained victim classifier (Phase D / fine-tune scheme only) | ~5 MB |

For reproducing **Figure 4 directly** (the headline result), you don't need
any of these — the aggregated `metrics.jsonl` files under `hpc-results/` are
enough to rebuild the plot. You only need the heavy artefacts if you want to
re-train a victim from a poisoned dataset, or re-craft from scratch.
