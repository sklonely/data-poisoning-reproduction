#!/bin/bash
#SBATCH --job-name=metapoison
#SBATCH --account=eecs
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/hpc/share/chanc7/metapoison/logs/%j.out
#SBATCH --error=/nfs/hpc/share/chanc7/metapoison/logs/%j.err

set -euo pipefail

PROJECT=/nfs/hpc/share/chanc7/metapoison
REPO=$PROJECT/repo
OUT=$PROJECT/outputs
LOGS=$PROJECT/logs
UID_TAG=run001
CRAFTPROJ=craft1
VICTIMPROJ=victim1

mkdir -p "$OUT" "$LOGS"

source "$REPO/.venv-tf/bin/activate"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

cd "$REPO/official-metapoison"

echo "==> Job ${SLURM_JOB_ID:-N/A} on $(hostname)"
echo "==> Started $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$REPO/.venv-tf/bin/python" -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF', tf.__version__, 'GPUs:', gpus)
assert gpus, 'TF cannot see any GPU; aborting'
"
"$REPO/.venv-torch/bin/python" -c "
import torch
print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), 'Torch cannot see any GPU; aborting'
"

# ===== [1/4] Craft poisons (TF) =====
echo
echo "==> [1/4] Crafting poisons (ResNet, npoison=5000, ncraftstep=61, nreplay=2)"
python main.py "$UID_TAG" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "$CRAFTPROJ" -victimproj "$VICTIMPROJ" \
  -gpu 0 -net ResNet \
  -targetclass 2 -poisonclass 5 -ytargetadv 5 -targetids 0 \
  -nbatch 400 -batchsize 125 -npoison 5000 \
  -ncraftstep 61 -nreplay 2 -logperiod 10 \
  -skipvictim

# ===== Resolve craft experiment key from artifact metadata =====
CRAFT_KEY=$(python - <<PYEOF
import sys
sys.path.insert(0, '$REPO/official-metapoison')
from tracking import LocalAPI
api = LocalAPI(root='$OUT', workspace='local')
expts = api.find_by_uid('$CRAFTPROJ', '$UID_TAG')
print(expts[-1].id)
PYEOF
)
echo "==> craft expt key: $CRAFT_KEY"

# Pick the largest logged craftstep that actually has a poisoninputs asset
LAST_STEP=$(python - <<PYEOF
import sys
sys.path.insert(0, '$REPO/official-metapoison')
from tracking import LocalAPI
api = LocalAPI(root='$OUT', workspace='local')
expt = api.get_experiment('local', '$CRAFTPROJ', '$CRAFT_KEY')
steps = sorted({a['step'] for a in expt.get_asset_list() if 'poisoninputs-' in a['fileName']})
print(steps[-1] if steps else 0)
PYEOF
)
echo "==> using craftstep $LAST_STEP for victim eval"

# ===== [2/4] TF Victim training (real evaluation) =====
echo
echo "==> [2/4] TF victim training (3 trials, 200 epochs)"
python victim.py "$CRAFT_KEY" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "$CRAFTPROJ" -Xvictimproj "$VICTIMPROJ" \
  -gpu 0 \
  -craftsteps "$LAST_STEP" -ntrial 3 -nvictimepoch 200

# ===== [3/4] Export poisoned dataset to pkl for PyTorch =====
echo
echo "==> [3/4] Exporting poisoned dataset"
python victim.py "$CRAFT_KEY" \
  -artifactroot "$OUT" -workspace local \
  -craftproj "$CRAFTPROJ" -Xvictimproj "${VICTIMPROJ}-export" \
  -gpu 0 \
  -craftsteps "$LAST_STEP" -savepoisondataset \
  -poisondatasetfile "$OUT/poisondataset"

# ===== [4/4] PyTorch victim training (separate venv) =====
echo
echo "==> [4/4] PyTorch victim training (3 trials, 200 epochs, augment)"
deactivate
source "$REPO/.venv-torch/bin/activate"
cd "$REPO"
python -m metapoison_hpc.torch_victim \
  --dataset "$OUT/poisondataset-${LAST_STEP}.pkl" \
  --arch resnet20 --epochs 200 --augment --trials 3 \
  --num-workers 4 \
  --output "$OUT/torch-results.json"

echo
echo "==> All done at $(date -Iseconds)"
echo "==> Craft expt key: $CRAFT_KEY"
echo "==> Outputs in: $OUT"
