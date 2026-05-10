# Level 2 Reproduction — Verify One Cell on Your Own GPU

**Time budget:** ~15 min on a single GPU (single trial). ~1.5 h for a full
6-trial victim sweep that matches our reported n=30 (you do 6 of those).

**What you'll demonstrate:** the same poisoned dataset that gave us a high ASR
on OSU HPC will trip up a victim model on your machine too. The attack
**survives the transition** from crafting environment to deployment environment
— which is the entire point of MetaPoison.

This walkthrough uses cell **A03** (ConvNetBN dog→bird, 1% budget, **headline
100% ASR cell**). The same recipe works for any of A01–A10; just swap the
filename.

---

## 0. What you need

- 1 NVIDIA GPU with ≥4 GB VRAM (anything from a GTX 1060 onwards works for ResNet20)
- ~10 GB free disk: 704 MB pkl + CIFAR-10 (~340 MB) + a Python virtualenv (~3 GB)
- Python 3.10
- 15 min for one ASR data point; longer if you want to average over seeds

You **don't** need TensorFlow, OpenMPI, the 4-GPU mpirun setup, or anything
HPC-flavoured for the PyTorch path below. That's the whole point — crafting was
expensive, deployment is cheap.

---

## 1. Clone and set up the PyTorch venv

```bash
git clone https://github.com/sklonely/data-poisoning-reproduction.git
cd data-poisoning-reproduction

# uv is recommended (faster); falls back to plain python -m venv fine
uv venv --python 3.10 .venv-torch
uv pip install --python .venv-torch/bin/python \
    "torch==2.5.1" "torchvision" "numpy<2" "matplotlib<3.9"
```

Quick sanity check that CUDA sees your GPU:
```bash
.venv-torch/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA <your GPU>
```

---

## 2. Download cell A03 + verify integrity

```bash
# Pull A03 from the cloud share. Replace <A03 URL> with the link from MANIFEST.md.
cd checkpoints
curl -L -o A03-poisondataset.pkl "<A03 URL>"

# Verify the file matches what we uploaded
sha256sum -c <(grep A03 SHA256SUMS)
# expected: A03-poisondataset.pkl: OK
```

If the hash doesn't match, you got a corrupted download — re-fetch before
spending GPU time.

What's inside the pkl (FYI, you don't need to open it):
- `xtrain` (50 000, 32, 32, 3) — full CIFAR-10 train set with **500 of the
  bird-class images replaced by adversarial poisons**
- `ytrain` (50 000,) — labels, including the poisons labelled as "bird" (5)
- `xtarget`, `ytarget` (1, 32, 32, 3) + true label "dog" (2) — the target dog
  image we want misclassified
- `ytargetadv` — "bird" (5) — what we want the victim to predict instead
- `xvalid`, `yvalid` — CIFAR-10 test set for clean-acc reference

---

## 3. Train one victim and read off ASR

The PyTorch victim trains ResNet-20 from scratch on the poisoned data for 200
epochs (~12-15 min on a single 3060/4060-class GPU).

```bash
cd ..  # back to repo root

.venv-torch/bin/python -m metapoison_hpc.torch_victim \
    --dataset checkpoints/A03-poisondataset.pkl \
    --arch resnet20 \
    --epochs 200 \
    --batch-size 128 \
    --augment \
    --trials 1 \
    --seed 1337 \
    --output checkpoints/A03-my-replication.json
```

**Flag-by-flag, why these values:**

| Flag | Value | Why |
|------|-------|-----|
| `--dataset` | A03 pkl | Cell A03 = ConvNetBN dog→bird @1% budget (500 poisons) |
| `--arch` | `resnet20` | Only ResNet variants supported here; the original crafting was on ConvNetBN so expect partial cross-arch transfer (more on this below) |
| `--epochs 200` | paper-faithful victim training schedule |
| `--batch-size 128` | matches paper §3.1 |
| `--augment` | random crop + horizontal flip — **load-bearing**, see Caveats |
| `--trials 1` | one ASR vote; do `--trials 6` if you want our n=6 per cell |
| `--seed 1337` | deterministic across runs; bump it for different vote |

While it trains you'll see lines like:

```
[trial 0 | epoch  20] train_acc=0.612  valid_acc=0.685  loss=1.31
[trial 0 | epoch  40] train_acc=0.781  valid_acc=0.748  loss=0.74
[trial 0 | epoch 199] train_acc=0.945  valid_acc=0.837
  target prediction: class 5  (adv class = 5, true class = 2) → ATTACK SUCCESS
```

Two numbers to watch in the final output:

1. **`valid_acc`** (should land 82–86 %): if poisoning worked, **clean
   accuracy stays normal** — that's the whole point of clean-label attacks.
2. **target prediction**: if it equals `ytargetadv` (5 = bird for A03), the
   attack succeeded on this trial. ASR is the fraction of trials where this
   happens.

The JSON output (`checkpoints/A03-my-replication.json`) summarizes everything:

```json
{
  "dataset": "checkpoints/A03-poisondataset.pkl",
  "summary": {
    "n_trials": 1,
    "valid_acc_mean": 0.834,
    "target_pred_majority": 5,
    "asr": 1.0
  },
  "trials": [ ... per-trial details ... ]
}
```

---

## 4. (Optional) Replicate the n=6 ASR we report

```bash
.venv-torch/bin/python -m metapoison_hpc.torch_victim \
    --dataset checkpoints/A03-poisondataset.pkl \
    --arch resnet20 --epochs 200 --augment \
    --trials 6 \
    --output checkpoints/A03-my-replication-n6.json
```

That's ~1.5 h on one GPU. The `summary.asr` field in the JSON is your
single-architecture ASR estimate (denominator 6; we report 30, paper uses 60).

---

## Caveats — things that will surprise you

### Why our paper-side number is "100% ASR" but you might see lower

Our headline **100% ASR on A03** was measured with **TF + ConvNetBN victim**
— the architecture the poisons were crafted against. The PyTorch victim
above uses **ResNet-20**, so you're implicitly testing **cross-architecture
transfer**, which the paper itself reports as weaker (paper Fig 5).

Expect roughly:
- **ResNet-20 victim on A03 poisons + `--augment`**: 30–70 % ASR (single-seed
  variance is high; this is why we average over 6).
- **ResNet-20 victim on A03 poisons without `--augment`**: pushes higher,
  closer to 80–100 %. Augmentation is a *defence*, not a *requirement* — see
  next caveat.

### `--augment` makes the attack harder, not easier

Counterintuitive but real and reported in §3.3 of the paper (and in our
[slides/CHECKPOINT_2_zh.md](../slides/CHECKPOINT_2_zh.md)). Augmentation **smears the poisoned pixels around**, dilutes the
gradient signal. Our **augmentation ablation** (job 20290350) found:

- Aug ON (paper-faithful for victims): 0/3 ASR on a particular cell
- Aug OFF: 3/3 ASR on the same cell

So if you're trying to maximise the chance of a single trial showing the
attack, drop `--augment`. If you're trying to match what we (or the paper)
report, keep it on. The honest comparison is **both your numbers + ours
with the same flag**.

### Why no ConvNetBN in PyTorch

We didn't port ConvNetBN to PyTorch because (a) the PyTorch side of this
repo exists as a **consistency check**, not as a parallel implementation,
and (b) doing so would muddy the "is the attack itself reproducing or are
two implementations of the same arch disagreeing" question. If you want
to hit our exact 100% ASR number, you need the TF path (Level 2b below).

---

## Level 2b — TF + ConvNetBN victim (advanced, reproduces 100% ASR exactly)

The TF victim doesn't directly consume the `.pkl` — it expects the **craft
asset directory layout** that our SLURM scripts produce on HPC. To use one
of our pkls with TF locally, you need a small adapter:

```bash
# 1. Same clone as Level 2 Step 1, but install the TF venv instead
uv venv --python 3.10 .venv-tf
uv pip install --python .venv-tf/bin/python \
    "tensorflow[and-cuda]==2.15.1" "tensorflow-probability==0.23" \
    "matplotlib<3.9" "pillow<11" "numpy<2"

# 2. Run TF victim with a one-shot loader script (NOT YET WRITTEN — see Open issues)
#    For now, the only supported TF path is to re-craft from scratch on HPC.
```

> 🚧 **Open issue**: `pkl → TF craft asset directory` adapter not yet
> written. If you specifically want to reproduce the headline 100% ASR with
> TF + ConvNetBN locally, ping the repo author. Crafting from scratch with
> [slurm/hpc_array_craft.sbatch](../slurm/hpc_array_craft.sbatch) takes ~1.5
> h on 4×V100.

---

## What success looks like

You've reproduced cell A03 at Level 2 if:
- ✅ SHA256 of the downloaded pkl matches
- ✅ Victim training completes (clean valid_acc 0.82-0.86)
- ✅ At least 1 of your 6 trials produces `target_pred == ytargetadv` (= 5 for A03)

If your full-6-trial ASR lands somewhere between 30% and 100%, you've
**replicated the qualitative claim** of MetaPoison: a small set of
clean-label poisoned images can flip a held-out test image's prediction,
without harming clean accuracy.

If your ASR is exactly 0/6, something went wrong — most likely:
- Wrong `--augment` interpretation (try without)
- File corruption (re-verify SHA256)
- Wrong seed coincidence (try `--seed 1338 --trials 6` again)

---

## Cross-references

- Cell decoder ring: [../docs/CELLS.md](CELLS.md)
- Full project state: [RESUME.md](RESUME.md)
- The slide deck explaining the augmentation ablation: [../slides/CHECKPOINT_2_zh.md](../slides/CHECKPOINT_2_zh.md)
- Cloud-share manifest with all 10 A0x URLs: [../checkpoints/MANIFEST.md](../checkpoints/MANIFEST.md)
