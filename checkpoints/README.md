# Model checkpoints — A01–A10 poisoned datasets

This folder holds the **exported poisoned datasets** for the 10 ConvNetBN cells
behind our Figure 4 reproduction. Drop one into a victim trainer and you get an
ASR vote in ~15 min on a single GPU — no crafting required.

The actual `.pkl` files are gitignored (10 × 704 MB = 6.9 GB, too heavy for
plain git). They're distributed out-of-band via a cloud share.

## Quick start for collaborators

1. Read [MANIFEST.md](MANIFEST.md) — table of what each `A01..A10` cell is, plus the cloud-share URL.
2. Download the cells you want into this folder (`checkpoints/`).
3. Verify integrity: `sha256sum -c SHA256SUMS` (every line should end with `OK`).
4. Train a victim on a pulled cell — see the example commands in [MANIFEST.md](MANIFEST.md).

## What's in this folder

| Tracked in git  | What                                                                    |
|-----------------|-------------------------------------------------------------------------|
| ✅ `README.md`   | This file                                                               |
| ✅ `MANIFEST.md` | Cell → arch/pair/budget table, cloud URL, recommended-use guide         |
| ✅ `SHA256SUMS`  | Integrity hashes for the 10 pkl files                                   |
| ❌ `A*.pkl`      | Gitignored — pull from the cloud link in MANIFEST.md                    |

## Other artefact types (not yet distributed)

The crafted-poison checkpoints (intermediate craftsteps, surrogate model
weights, pretrained Phase-D classifier) all live on the OSU HPC share under
`/nfs/hpc/share/chanc7/metapoison/`. None of them are needed to reproduce the
headline Figure 4 — only the exported poisoned datasets (above) are. The
aggregated `metrics.jsonl` files in `hpc-results/` are enough to rebuild the
plot without re-training anything.
