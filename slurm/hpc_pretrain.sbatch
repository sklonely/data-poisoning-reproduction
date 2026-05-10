#!/bin/bash
#SBATCH --job-name=mp-pretrain
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/pretrain-%j.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/pretrain-%j.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
OUT=$PROJECT/outputs-phase-d/pretrained
DATA=$PROJECT/data

mkdir -p "$OUT" "$DATA"

source "$REPO/.venv-torch/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$REPO"
python -m metapoison_hpc.pretrain_cifar \
  --data-root "$DATA" \
  --arch resnet20 \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 --momentum 0.9 --weight-decay 2e-4 \
  --schedule 60,80 \
  --num-workers 4 \
  --output "$OUT/cifar_clean_resnet20.pt"

echo "==> pretrain done at $(date -Iseconds)"
