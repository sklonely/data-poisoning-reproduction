#!/bin/bash
#SBATCH --job-name=mp-n20craft
#SBATCH --account=eecs
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --array=1-120%6
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/n20craft-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/n20craft-%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=$PROJECT/experiments/manifest_phase_a_n20_crafts.csv
OUT=$PROJECT/outputs-phase-a
LOGS=$PROJECT/logs

mkdir -p "$OUT" "$LOGS"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load openmpi/3.1
source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$MANIFEST")
CELL_ID=$(echo "$ROW" | cut -d, -f1 | tr -d '\r')
ARCH=$(echo "$ROW" | cut -d, -f2 | tr -d '\r')
TARGETCLASS=$(echo "$ROW" | cut -d, -f3 | tr -d '\r')
POISONCLASS=$(echo "$ROW" | cut -d, -f4 | tr -d '\r')
YTARGETADV=$(echo "$ROW" | cut -d, -f5 | tr -d '\r')
TARGET_ID=$(echo "$ROW" | cut -d, -f6 | tr -d '\r')
NPOISON=$(echo "$ROW" | cut -d, -f7 | tr -d '\r')
NCRAFTSTEP=$(echo "$ROW" | cut -d, -f8 | tr -d '\r')
NREPLAY=$(echo "$ROW" | cut -d, -f9 | tr -d '\r')
OBJECTIVE=$(echo "$ROW" | cut -d, -f10 | tr -d '\r')

# Skip if this cell already has its 7 poison checkpoints
shopt -s nullglob
existing_files=("$OUT"/local/craft-${CELL_ID}/*/assets/*poisoninputs*)
EXISTING=${#existing_files[@]}
shopt -u nullglob
if [ "$EXISTING" -ge 7 ]; then
  echo "==> SKIP cell $CELL_ID (already has $EXISTING checkpoints)"
  exit 0
fi

echo "==> Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} cell ${CELL_ID} on $(hostname)"
echo "==> $(date -Iseconds)"
echo "==> arch=${ARCH} target_class=${TARGETCLASS} poison_class=${POISONCLASS} yadv=${YTARGETADV} target_id=${TARGET_ID} npoison=${NPOISON}"

cd "$REPO/official-metapoison"

mpirun -np 4 "$REPO/.venv-tf/bin/python" main.py "$CELL_ID" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "craft-${CELL_ID}" -victimproj "victim-${CELL_ID}" \
  -gpu 0 1 2 3 -net "$ARCH" \
  -targetclass "$TARGETCLASS" -poisonclass "$POISONCLASS" -ytargetadv "$YTARGETADV" -targetids "$TARGET_ID" \
  -nbatch 400 -batchsize 125 -npoison "$NPOISON" \
  -ncraftstep "$NCRAFTSTEP" -nreplay "$NREPLAY" -logperiod 10 \
  -objective "$OBJECTIVE" \
  -skipvictim

echo "==> craft cell ${CELL_ID} done at $(date -Iseconds)"
