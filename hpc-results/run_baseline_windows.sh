#!/bin/bash
set -uo pipefail
REPO='/d/HW/CS 539/final project'
VENV="$REPO/.venv-torch-local/Scripts/python.exe"
export PYTHONPATH="$REPO/src"

"$VENV" -m metapoison_hpc.clean_baseline \
  --data-root "$REPO/data" \
  --arch resnet20 \
  --epochs 200 --trials 6 \
  --batch-size 128 --lr 0.1 --momentum 0 --weight-decay 0 \
  --schedule "100,150" --num-workers 0 \
  --target-class 2 --adv-class 5 --n-targets 5 \
  --output "$REPO/hpc-results/baseline-zero-poison.json"
