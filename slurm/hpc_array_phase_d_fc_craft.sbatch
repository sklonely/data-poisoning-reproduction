#!/bin/bash
#SBATCH --job-name=mp-fc-craft
#SBATCH --account=eecs
#SBATCH --partition=ampere,dgxh,dgx2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --array=1-120%48
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/fccraft-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/fccraft-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=$PROJECT/experiments/manifest_phase_d_fc_crafts.csv
OUT=$PROJECT/outputs-phase-d/fc-crafts
DATA=$PROJECT/data
PRETRAIN=$PROJECT/outputs-phase-d/pretrained/cifar_clean_resnet20.pt

mkdir -p "$OUT"

if [ ! -f "$PRETRAIN" ]; then
  echo "==> ABORT: pretrained checkpoint not found at $PRETRAIN. Run hpc_pretrain.sbatch first."
  exit 2
fi

source "$REPO/.venv-torch/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + ${CRAFT_OFFSET:-0} + 1)) 'NR==n' "$MANIFEST")
CRAFT_UID=$(echo "$ROW" | cut -d, -f1)
TARGET_CLASS=$(echo "$ROW" | cut -d, -f4)
BASE_CLASS=$(echo "$ROW" | cut -d, -f5)
TARGET_IDX=$(echo "$ROW" | cut -d, -f6)
N_POISON=$(echo "$ROW" | cut -d, -f7)
EPS_STR=$(echo "$ROW" | cut -d, -f8)

if [ -z "$CRAFT_UID" ]; then
  echo "==> SKIP: empty manifest row at index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

OUT_PKL="$OUT/${CRAFT_UID}.pkl"
if [ -f "$OUT_PKL" ]; then
  echo "==> SKIP $CRAFT_UID (already crafted)"
  exit 0
fi

EPS_FLAG=""
if [ -n "$EPS_STR" ]; then
  EPS_FLAG="--eps $EPS_STR"
fi

echo "==> Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} craft ${CRAFT_UID}"
echo "==> target_class=${TARGET_CLASS} base_class=${BASE_CLASS} target_idx=${TARGET_IDX} n_poison=${N_POISON} eps=${EPS_STR:-none}"

cd "$REPO"
python -m metapoison_hpc.feature_collision \
  --pretrained "$PRETRAIN" \
  --data-root "$DATA" \
  --target-class "$TARGET_CLASS" \
  --base-class "$BASE_CLASS" \
  --target-idx "$TARGET_IDX" \
  --n-poison "$N_POISON" \
  --iters 12000 --decay-iter 10000 \
  --lr 2.55 --beta 0.25 \
  $EPS_FLAG \
  --output "$OUT_PKL"

echo "==> craft ${CRAFT_UID} done at $(date -Iseconds)"
