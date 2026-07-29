#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_PREFIX="${COLLECTION_CONDA_ENV_PREFIX:-${REPO_ROOT}/.conda/envs/object-memory-demo}"
MODEL_DIR="${COLLECTION_MODEL_DIR:-${REPO_ROOT}/weights}"
DATA_DIR="${COLLECTION_DATA_DIR:-${REPO_ROOT}/data}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${MODEL_DIR}/sam3/sam3.pt}"

export HF_HOME="${HF_HOME:-${MODEL_DIR}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available." >&2
  exit 2
fi

if [[ ! -f "${SAM3_CHECKPOINT}" ]]; then
  echo "ERROR: SAM3 checkpoint not found: ${SAM3_CHECKPOINT}" >&2
  exit 3
fi

mkdir -p "$(dirname -- "${ENV_PREFIX}")" "${MODEL_DIR}" "${DATA_DIR}" "${HF_HOME}"

eval "$(conda shell.bash hook)"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Updating Conda environment at ${ENV_PREFIX}"
  conda env update --prefix "${ENV_PREFIX}" --file "${REPO_ROOT}/environment.yml" --prune
else
  echo "Creating Conda environment at ${ENV_PREFIX}"
  conda env create --prefix "${ENV_PREFIX}" --file "${REPO_ROOT}/environment.yml"
fi

conda activate "${ENV_PREFIX}"

echo "Python: $(python --version 2>&1)"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "SAM3 checkpoint: ${SAM3_CHECKPOINT}"
echo "Model directory: ${MODEL_DIR}"
echo "Data directory: ${DATA_DIR}"

set +e
python "${REPO_ROOT}/scripts/check_server_env.py" \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --sam3-checkpoint "${SAM3_CHECKPOINT}"
CHECK_STATUS=$?
set -e

if [[ ${CHECK_STATUS} -ne 0 ]]; then
  echo "Environment setup finished, but the self-check still reports blockers."
  echo "Review environment/server_env_report.json before continuing."
  exit "${CHECK_STATUS}"
fi

echo "Environment setup and self-check completed."
