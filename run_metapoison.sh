#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="${ROOT_DIR}/official-metapoison"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT_DIR}/runs}"
TF_VENV="${ROOT_DIR}/.venv-tf1"
TORCH_UV_PYTHON="${TORCH_UV_PYTHON:-3.11}"
TF_UV_PYTHON="${TF_UV_PYTHON:-3.7}"

setup_official() {
  uv venv --python "${TF_UV_PYTHON}" "${TF_VENV}"
  "${TF_VENV}/bin/python" -m pip install --upgrade pip
  uv pip install --python "${TF_VENV}/bin/python" -r "${ROOT_DIR}/requirements-official-tf1-linux.txt"
}

setup_torch() {
  uv sync --python "${TORCH_UV_PYTHON}"
}

craft_official() {
  local uid="${1:?uid is required}"
  shift
  mpirun -np "${NP:-1}" "${TF_VENV}/bin/python" "${OFFICIAL_DIR}/main.py" "${uid}" \
    -artifactroot "${ARTIFACT_ROOT}" \
    "$@"
}

export_dataset() {
  local uid="${1:?uid is required}"
  shift
  "${TF_VENV}/bin/python" "${OFFICIAL_DIR}/victim.py" "${uid}" \
    -artifactroot "${ARTIFACT_ROOT}" \
    -savepoisondataset \
    "$@"
}

torch_compare() {
  uv run python -m metapoison_hpc.torch_victim "$@"
}

quickstart() {
  local uid="${1:-quickstart}"
  craft_official "${uid}" \
    -nreplay="${NREPLAY:-1}" \
    -victimproj="${VICTIM_PROJ:-quickstart-victim}" \
    -targetclass=2 -poisonclass=5 -ytargetadv=5 -targetids 0 \
    -nbatch="${NBATCH:-40}" -batchsize="${BATCHSIZE:-125}" \
    -npoison="${NPOISON:-50}" -ncraftstep="${NCRAFTSTEP:-5}" \
    -logperiod=1
  export_dataset "${uid}" -craftsteps "${CRAFTSTEP:-4}" -poisondatasetfile "${ARTIFACT_ROOT}/poisondataset"
  torch_compare --dataset "${ARTIFACT_ROOT}/poisondataset-${CRAFTSTEP:-4}.pkl" --augment
}

usage() {
  cat <<'EOF'
Usage:
  ./run_metapoison.sh setup-official
  ./run_metapoison.sh setup-torch
  ./run_metapoison.sh craft-official <uid> [official args...]
  ./run_metapoison.sh export-dataset <uid> [victim args...]
  ./run_metapoison.sh torch-compare [torch args...]
  ./run_metapoison.sh quickstart [uid]

Notes:
  - Official reproduction expects Linux, CUDA compatible with TensorFlow 1.14, and OpenMPI available on PATH.
  - Artifacts are stored locally under ARTIFACT_ROOT instead of Comet.
  - For a multi-GPU HPC run, export NP=<num_mpi_ranks> before craft-official.
EOF
}

case "${1:-}" in
  setup-official) shift; setup_official "$@" ;;
  setup-torch) shift; setup_torch "$@" ;;
  craft-official) shift; craft_official "$@" ;;
  export-dataset) shift; export_dataset "$@" ;;
  torch-compare) shift; torch_compare "$@" ;;
  quickstart) shift; quickstart "$@" ;;
  *) usage; exit 1 ;;
esac
