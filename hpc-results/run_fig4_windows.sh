#!/bin/bash
# Bash dispatcher for Windows local Fig 4 PyTorch victim runs.
# Designed to run as a long-lived background process via Bash run_in_background.
# Picks up odd-numbered cells (A01, A03, ..., A29) from local exports/.

set -uo pipefail
REPO='/d/HW/CS 539/final project'
VENV="$REPO/.venv-torch-local/Scripts/python.exe"
EXPORTS="$REPO/hpc-results/exports"
OUT_DIR="$REPO/hpc-results/fig4-pytorch-windows"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$REPO/src"

CELLS=(A01 A03 A05 A07 A09 A11 A13 A15 A17 A19 A21 A23 A25 A27 A29)

for cell in "${CELLS[@]}"; do
  pkl="$EXPORTS/${cell}-poisondataset.pkl"
  out="$OUT_DIR/${cell}.json"
  log="$OUT_DIR/${cell}.log"
  if [ -f "$out" ]; then
    echo "=== $(date +%H:%M:%S) skip $cell (already done) ==="
    continue
  fi
  while [ ! -f "$pkl" ]; do
    echo "=== $(date +%H:%M:%S) waiting for $pkl ==="
    sleep 60
  done
  echo "=== $(date +%H:%M:%S) Windows running $cell ==="
  "$VENV" -m metapoison_hpc.torch_victim \
    --dataset "$pkl" \
    --arch resnet20 \
    --epochs 200 \
    --trials 4 \
    --batch-size 128 \
    --lr 0.1 \
    --momentum 0 \
    --weight-decay 0 \
    --schedule "100,150" \
    --num-workers 0 \
    --output "$out" 2>&1 | tee "$log"
  echo "=== $(date +%H:%M:%S) $cell done ==="
done
echo "=== ALL WINDOWS CELLS DONE $(date -Iseconds) ==="
