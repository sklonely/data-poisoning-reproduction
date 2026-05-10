#!/bin/bash
#SBATCH --job-name=mp-victim
#SBATCH --account=eecs
#SBATCH --partition=dgx2,ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=1-600%24
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/victim-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/victim-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=${MANIFEST_PATH:-$PROJECT/experiments/manifest_phase_a_victims.csv}
OUT=$PROJECT/outputs-phase-a
LOGS=$PROJECT/logs

mkdir -p "$OUT" "$LOGS"

source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + ${VICTIM_OFFSET:-0} + 1)) 'NR==n' "$MANIFEST")
VICTIM_ID=$(echo "$ROW" | cut -d, -f1)
CRAFT_UID=$(echo "$ROW" | cut -d, -f2)
CRAFTPROJ=$(echo "$ROW" | cut -d, -f3)
CELL_ID=$(echo "$ROW" | cut -d, -f4)
TARGET_ID=$(echo "$ROW" | cut -d, -f5)
SEED_IDX=$(echo "$ROW" | cut -d, -f6)

VICTIMPROJ="victim-${CRAFT_UID}"

# Skip if this victim already has a recorded experiment with our seed marker
DONE_MARKER="$OUT/done/${VICTIM_ID}.done"
if [ -f "$DONE_MARKER" ]; then
  echo "==> SKIP $VICTIM_ID (already done)"
  exit 0
fi

# Check craft is finished — has poison checkpoints. nullglob makes empty match
# expand to nothing instead of triggering pipefail/-e.
shopt -s nullglob
poison_files=("$OUT"/local/${CRAFTPROJ}/*/assets/*poisoninputs-60*)
EXISTING_POISON=${#poison_files[@]}
shopt -u nullglob
if [ "$EXISTING_POISON" -lt 1 ]; then
  echo "==> SKIP $VICTIM_ID (craft $CRAFT_UID not yet finished — no poisoninputs-60)"
  exit 0
fi

echo "==> Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} victim ${VICTIM_ID}"
echo "==> craft_uid=${CRAFT_UID} target_id=${TARGET_ID} seed_idx=${SEED_IDX}"

cd "$REPO/official-metapoison"

# victim.py auto-resolves craftkey from uid via find_by_uid
"$REPO/.venv-tf/bin/python" victim.py "$CRAFT_UID" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "$CRAFTPROJ" -Xvictimproj "$VICTIMPROJ" \
  -gpu 0 \
  -craftsteps 60 -ntrial 1 -nvictimepoch 200 \
  -tag "${VICTIM_ID}_seed${SEED_IDX}"

mkdir -p "$OUT/done"
touch "$DONE_MARKER"
echo "==> victim ${VICTIM_ID} done at $(date -Iseconds)"
