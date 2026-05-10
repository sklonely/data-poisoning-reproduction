# MetaPoison HPC Setup

This workspace now supports two paths:

1. `official-metapoison/`: the original TensorFlow 1.14 MetaPoison code, patched to store weights and poison assets locally instead of Comet.
2. `src/metapoison_hpc/`: a PyTorch victim trainer that consumes a poisoned dataset exported from the official pipeline, so you can compare TensorFlow victim results against a PyTorch reproduction on the same poisoned data.

## Recommended HPC prerequisites

- Linux x86_64
- `uv`
- `mpirun` / OpenMPI on `PATH`
- NVIDIA driver plus a CUDA stack compatible with `tensorflow-gpu==1.14.0`
- A module or base image that can supply Python 3.7 for the official reproduction

## Main entrypoint

```bash
./run_metapoison.sh setup-official
./run_metapoison.sh setup-torch
./run_metapoison.sh quickstart demo001
```

## Local smoke test on Windows

For a minimal local verification, this repo also supports a compatibility path:

- Python 3.10
- `tensorflow==2.10.1` in `compat.v1` mode
- single-process fallback MPI when no system MPI runtime is installed

The exact commands used for the local smoke test were:

```powershell
uv venv --python 3.10 .venv-tf-compat
uv pip install --python .venv-tf-compat\Scripts\python.exe `
  "tensorflow==2.10.1" `
  "tensorflow-probability==0.18.0" `
  "mpi4py==4.0.3" `
  "matplotlib<3.9" `
  "pillow<11" `
  "numpy<2"

.venv-tf-compat\Scripts\python.exe official-metapoison\main.py localsmoke2 `
  -artifactroot runs-local -workspace local `
  -craftproj craft-local -victimproj victim-local `
  -gpu 0 -net ConvNetBN -nreplay 1 -nadapt 1 -ncraftstep 1 -logperiod 1 `
  -skipvictim -targetclass 2 -poisonclass 5 -ytargetadv 5 -targetids 0 `
  -nbatch 2 -batchsize 10 -npoison 1

.venv-tf-compat\Scripts\python.exe official-metapoison\victim.py localsmoke2 `
  -artifactroot runs-local -workspace local `
  -craftproj craft-local -victimproj victim-local `
  -gpu 0 -craftsteps 0 -savepoisondataset `
  -poisondatasetfile runs-local\local\exports\poisondataset

uv run python -m metapoison_hpc.torch_victim `
  --dataset runs-local\local\exports\poisondataset-0.pkl `
  --epochs 1 --batch-size 10 --trials 1 --num-workers 0 `
  --output runs-local\local\exports\torch-results.json
```

## Typical workflow

Craft poisons with the official implementation:

```bash
NP=4 ./run_metapoison.sh craft-official run001 \
  -nreplay=2 -victimproj=resnetrobust -net=ResNet \
  -targetclass=2 -poisonclass=5 -targetids 0 \
  -nbatch=400 -batchsize=125 -npoison=5000
```

Export a local poisoned dataset from a saved craft step:

```bash
./run_metapoison.sh export-dataset run001 \
  -craftsteps 60 \
  -poisondatasetfile runs/run001/poisondataset
```

Train a PyTorch victim on that exported dataset:

```bash
./run_metapoison.sh torch-compare \
  --dataset runs/run001/poisondataset-60.pkl \
  --arch resnet20 \
  --epochs 200 \
  --augment \
  --trials 3 \
  --output runs/run001/torch-results.json
```

## What to compare

- Final validation accuracy
- Target prediction and attack success rate
- Stability across repeated trials

If the PyTorch victim behaves similarly to the official TensorFlow victim on the same exported poisoned dataset, then you have a reasonable basis for shifting future victim-side experiments to PyTorch while keeping the poison generation anchored to the original paper implementation.
