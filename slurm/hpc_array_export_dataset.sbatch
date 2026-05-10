#!/bin/bash
#SBATCH --job-name=mp-export
#SBATCH --account=eecs
#SBATCH --partition=share,ampere,dgx2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --array=1-30%12
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/export-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/export-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
EXPORT_DIR=$PROJECT/exports
OUT=$PROJECT/outputs-phase-a

mkdir -p "$EXPORT_DIR"

source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

CELL_ID=$(printf "A%02d" $SLURM_ARRAY_TASK_ID)
PKL_PATH=$EXPORT_DIR/${CELL_ID}-poisondataset.pkl

if [ -f "$PKL_PATH" ]; then
  echo "==> SKIP $CELL_ID (already exported)"
  exit 0
fi

shopt -s nullglob
poison_files=("$OUT"/local/craft-${CELL_ID}/*/assets/*poisoninputs-60*)
if [ ${#poison_files[@]} -lt 1 ]; then
  echo "==> SKIP $CELL_ID (no poisoninputs-60 craft yet)"
  exit 0
fi
shopt -u nullglob

echo "==> Job ${SLURM_JOB_ID} task $SLURM_ARRAY_TASK_ID exporting cell $CELL_ID"

cd "$REPO/official-metapoison"

"$REPO/.venv-tf/bin/python" victim.py "$CELL_ID" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "craft-${CELL_ID}" -Xvictimproj "victim-${CELL_ID}-export" \
  -gpu 0 \
  -craftsteps 60 -savepoisondataset \
  -poisondatasetfile "$EXPORT_DIR/${CELL_ID}-poisondataset"

# Note: victim.py creates a `<file>-<craftstep>.pkl` so move/symlink
if [ -f "$EXPORT_DIR/${CELL_ID}-poisondataset-60.pkl" ]; then
  mv "$EXPORT_DIR/${CELL_ID}-poisondataset-60.pkl" "$PKL_PATH"
fi
echo "==> exported $CELL_ID -> $PKL_PATH"
