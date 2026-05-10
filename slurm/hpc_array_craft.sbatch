#!/bin/bash
#SBATCH --job-name=mp-craft
#SBATCH --account=eecs
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-30%2
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/craft-A%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/craft-A%a-%A.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
MANIFEST=$PROJECT/experiments/manifest_phase_a.csv
OUT=$PROJECT/outputs-phase-a
LOGS=$PROJECT/logs

mkdir -p "$OUT" "$LOGS"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load openmpi/3.1
source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# Read row N+1 from manifest (header is row 1)
ROW=$(awk -F, -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$MANIFEST")
CELL_ID=$(echo "$ROW" | cut -d, -f1)
ARCH=$(echo "$ROW" | cut -d, -f2)
TARGETCLASS=$(echo "$ROW" | cut -d, -f3)
POISONCLASS=$(echo "$ROW" | cut -d, -f4)
YTARGETADV=$(echo "$ROW" | cut -d, -f5)
NPOISON=$(echo "$ROW" | cut -d, -f6)
NCRAFTSTEP=$(echo "$ROW" | cut -d, -f7)
NREPLAY=$(echo "$ROW" | cut -d, -f8)
OBJECTIVE=$(echo "$ROW" | cut -d, -f9)
UID_TAG=${CELL_ID}

echo "==> Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID} (cell ${CELL_ID}) on $(hostname)"
echo "==> $(date -Iseconds)"
echo "==> arch=${ARCH} target=${TARGETCLASS} poison=${POISONCLASS} yadv=${YTARGETADV} npoison=${NPOISON}"

cd "$REPO/official-metapoison"

mpirun -np 4 "$REPO/.venv-tf/bin/python" main.py "$UID_TAG" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "craft-${CELL_ID}" -victimproj "victim-${CELL_ID}" \
  -gpu 0 1 2 3 -net "$ARCH" \
  -targetclass "$TARGETCLASS" -poisonclass "$POISONCLASS" -ytargetadv "$YTARGETADV" -targetids 0 \
  -nbatch 400 -batchsize 125 -npoison "$NPOISON" \
  -ncraftstep "$NCRAFTSTEP" -nreplay "$NREPLAY" -logperiod 10 \
  -objective "$OBJECTIVE" \
  -skipvictim

echo "==> craft cell ${CELL_ID} done at $(date -Iseconds)"
