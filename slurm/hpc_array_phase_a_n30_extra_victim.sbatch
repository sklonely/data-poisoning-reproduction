#!/bin/bash
#SBATCH --job-name=mp-n30vic
#SBATCH --account=eecs
#SBATCH --partition=dgx2,ampere,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=1-240%24
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/n30vic-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/n30vic-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=$PROJECT/experiments/manifest_phase_a_n30_extra_victims.csv
OUT=$PROJECT/outputs-phase-a

mkdir -p "$OUT"
source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$MANIFEST")
VICTIM_ID=$(echo "$ROW" | cut -d, -f1 | tr -d '\r')
CRAFT_UID=$(echo "$ROW" | cut -d, -f2 | tr -d '\r')
CRAFTPROJ=$(echo "$ROW" | cut -d, -f3 | tr -d '\r')
CELL_ID=$(echo "$ROW" | cut -d, -f4 | tr -d '\r')
TARGET_ID=$(echo "$ROW" | cut -d, -f5 | tr -d '\r')
SEED_IDX=$(echo "$ROW" | cut -d, -f6 | tr -d '\r')

VICTIMPROJ="victim-${CRAFT_UID}"
DONE_MARKER="$OUT/done/${VICTIM_ID}.done"
if [ -f "$DONE_MARKER" ]; then
  echo "==> SKIP $VICTIM_ID (already done)"
  exit 0
fi

shopt -s nullglob
poison_files=("$OUT"/local/${CRAFTPROJ}/*/assets/*poisoninputs-60*)
EXISTING_POISON=${#poison_files[@]}
shopt -u nullglob
if [ "$EXISTING_POISON" -lt 1 ]; then
  echo "==> SKIP $VICTIM_ID (craft $CRAFT_UID not yet finished)"
  exit 0
fi

echo "==> Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} victim ${VICTIM_ID}"

cd "$REPO/official-metapoison"
"$REPO/.venv-tf/bin/python" victim.py "$CRAFT_UID" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "$CRAFTPROJ" -Xvictimproj "$VICTIMPROJ" \
  -gpu 0 \
  -craftsteps 60 -ntrial 1 -nvictimepoch 200 \
  -tag "${VICTIM_ID}_seed${SEED_IDX}"

mkdir -p "$OUT/done"
touch "$DONE_MARKER"
echo "==> victim ${VICTIM_ID} done at $(date -Iseconds)"
