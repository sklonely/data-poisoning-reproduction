#!/bin/bash
#SBATCH --job-name=mp-ablation
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/ablation-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/ablation-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
OUT=$PROJECT/outputs-ablation
DATASET=$PROJECT/outputs-v2/poisondataset-60.pkl

mkdir -p "$OUT"

source "$REPO/.venv-torch/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Three conditions to disentangle "framework difference" vs "training pipeline difference"
# Round 2 PyTorch baseline used: --augment, momentum=0.9, weight_decay=2e-4 -> ASR 1/3
case $SLURM_ARRAY_TASK_ID in
  1)
    COND=A_baseline_aug_mom_wd
    OPTS="--augment --momentum 0.9 --weight-decay 2e-4"
    ;;
  2)
    COND=B_noaug_mom_wd
    OPTS="--momentum 0.9 --weight-decay 2e-4"
    ;;
  3)
    COND=C_noaug_plain_sgd
    OPTS="--momentum 0 --weight-decay 0"
    ;;
esac

echo "==> Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} condition ${COND}"
echo "==> $(date -Iseconds)"
echo "==> opts: ${OPTS}"

cd "$REPO"
python -m metapoison_hpc.torch_victim \
  --dataset "$DATASET" \
  --arch resnet20 --epochs 200 --trials 3 \
  --num-workers 4 \
  $OPTS \
  --output "$OUT/${COND}.json"

echo "==> ${COND} done at $(date -Iseconds)"
