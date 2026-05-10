#!/bin/bash
#SBATCH --job-name=mp-fc-victim
#SBATCH --account=eecs
#SBATCH --partition=ampere,dgxh,dgx2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=1-360%48
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/fcvictim-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/fcvictim-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=$PROJECT/experiments/manifest_phase_d_fc_victims.csv
CRAFT_OUT=$PROJECT/outputs-phase-d/fc-crafts
VICTIM_OUT=$PROJECT/outputs-phase-d/fc-victims
DONE_DIR=$PROJECT/outputs-phase-d/done
DATA=$PROJECT/data
PRETRAIN=$PROJECT/outputs-phase-d/pretrained/cifar_clean_resnet20.pt

mkdir -p "$VICTIM_OUT" "$DONE_DIR"

if [ ! -f "$PRETRAIN" ]; then
  echo "==> ABORT: pretrained checkpoint not found at $PRETRAIN."
  exit 2
fi

source "$REPO/.venv-torch/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + ${VICTIM_OFFSET:-0} + 1)) 'NR==n' "$MANIFEST")
VICTIM_ID=$(echo "$ROW" | cut -d, -f1)
CRAFT_UID=$(echo "$ROW" | cut -d, -f2)
SEED_IDX=$(echo "$ROW" | cut -d, -f9)

if [ -z "$VICTIM_ID" ]; then
  echo "==> SKIP: empty manifest row at index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

DONE_MARKER="$DONE_DIR/${VICTIM_ID}.done"
if [ -f "$DONE_MARKER" ]; then
  echo "==> SKIP $VICTIM_ID (already done)"
  exit 0
fi

POISON_PKL="$CRAFT_OUT/${CRAFT_UID}.pkl"
if [ ! -f "$POISON_PKL" ]; then
  echo "==> SKIP $VICTIM_ID (craft $CRAFT_UID not finished)"
  exit 0
fi

echo "==> Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} victim ${VICTIM_ID}"

cd "$REPO"
python -m metapoison_hpc.torch_finetune_victim \
  --pretrained "$PRETRAIN" \
  --poisons "$POISON_PKL" \
  --data-root "$DATA" \
  --epochs 10 \
  --lr 0.001 \
  --batch-size 128 \
  --clean-per-class 500 \
  --trials 1 \
  --seed $((1337 + SEED_IDX)) \
  --num-workers 2 \
  --reset-fc \
  --output "$VICTIM_OUT/${VICTIM_ID}.json"

touch "$DONE_MARKER"
echo "==> victim ${VICTIM_ID} done at $(date -Iseconds)"
