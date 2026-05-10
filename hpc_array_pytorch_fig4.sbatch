#!/bin/bash
#SBATCH --job-name=mp-pyt-f4
#SBATCH --account=eecs
#SBATCH --partition=ampere,dgx2,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --array=1-30%24
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/pytf4-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/pytf4-%a-%A.err

set -uo pipefail
PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
EXPORTS=$PROJECT/exports
OUT=$PROJECT/outputs-fig4-pytorch
mkdir -p "$OUT"

source "$REPO/.venv-torch/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CELL_ID=$(printf "A%02d" $SLURM_ARRAY_TASK_ID)
PKL=$EXPORTS/${CELL_ID}-poisondataset.pkl
JSON_OUT=$OUT/${CELL_ID}.json

if [ -f "$JSON_OUT" ]; then
  echo "==> SKIP $CELL_ID (already done)"; exit 0
fi
if [ ! -f "$PKL" ]; then
  echo "==> SKIP $CELL_ID (no poison dataset)"; exit 0
fi

echo "==> Job ${SLURM_JOB_ID} task $SLURM_ARRAY_TASK_ID cell $CELL_ID"
cd "$REPO"
python -m metapoison_hpc.torch_victim \
  --dataset "$PKL" \
  --arch resnet20 \
  --epochs 200 --trials 4 --batch-size 128 \
  --lr 0.1 --momentum 0 --weight-decay 0 \
  --schedule "100,150" --num-workers 2 \
  --output "$JSON_OUT"
echo "==> $CELL_ID done at $(date -Iseconds)"
