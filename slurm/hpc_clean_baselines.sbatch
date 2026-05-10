#!/bin/bash
#SBATCH --job-name=mp-baseline
#SBATCH --account=eecs
#SBATCH --partition=dgx2,ampere,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=1-6%6
#SBATCH --requeue
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/baseline-%a-%A.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/baseline-%a-%A.err

set -uo pipefail
PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
OUT=$PROJECT/baselines
mkdir -p "$OUT"

source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# 6 baselines: 3 archs × 2 class pairs
case $SLURM_ARRAY_TASK_ID in
  1) ARCH=ConvNetBN; TC=2; YADV=5; PAIR=dogbird ;;
  2) ARCH=ConvNetBN; TC=0; YADV=6; PAIR=frogplane ;;
  3) ARCH=VGG13BN;   TC=2; YADV=5; PAIR=dogbird ;;
  4) ARCH=VGG13BN;   TC=0; YADV=6; PAIR=frogplane ;;
  5) ARCH=ResNet;    TC=2; YADV=5; PAIR=dogbird ;;
  6) ARCH=ResNet;    TC=0; YADV=6; PAIR=frogplane ;;
esac

JSON_OUT=$OUT/baseline-${ARCH}-${PAIR}.json
if [ -f "$JSON_OUT" ]; then
  echo "==> SKIP $ARCH $PAIR (already done)"; exit 0
fi

UID_TAG=baseline-${ARCH}-${PAIR}
echo "==> Running $UID_TAG"
cd "$REPO/official-metapoison"
"$REPO/.venv-tf/bin/python" clean_baseline.py "$UID_TAG" \
  -artifactroot "$OUT/scratch" -workspace local \
  -craftproj "baseline-craft" \
  -gpu 0 -net "$ARCH" \
  -targetclass "$TC" -poisonclass 0 -ytargetadv "$YADV" -targetids 0 1 2 3 4 \
  -nbatch 400 -batchsize 125 -npoison 1 \
  -ntrial 6 -nvictimepoch 200 \
  -output_json "$JSON_OUT"
echo "==> $UID_TAG done at $(date -Iseconds)"
