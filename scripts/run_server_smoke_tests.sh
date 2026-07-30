#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_server_smoke_tests.sh /path/to/image sam3_prompt" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="$1"
SAM3_PROMPT="$2"
ENV_PREFIX="${COLLECTION_CONDA_ENV_PREFIX:-${REPO_ROOT}/.conda/envs/object-memory-demo}"
MODEL_DIR="${COLLECTION_MODEL_DIR:-${REPO_ROOT}/weights}"
DATA_DIR="${COLLECTION_DATA_DIR:-${REPO_ROOT}/data}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${MODEL_DIR}/sam3/sam3.pt}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${REPO_ROOT}/runs/smoke/${RUN_ID}"

export HF_HOME="${COLLECTION_HF_HOME:-${MODEL_DIR}/qwen}"
export HF_HUB_CACHE="${COLLECTION_HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${COLLECTION_OMP_NUM_THREADS:-8}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available." >&2
  exit 3
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"
mkdir -p "${OUTPUT_ROOT}"

set +e
python "${REPO_ROOT}/scripts/smoke_sam3.py" \
  --image "${IMAGE_PATH}" \
  --checkpoint "${SAM3_CHECKPOINT}" \
  --prompt "${SAM3_PROMPT}" \
  --output-dir "${OUTPUT_ROOT}/sam3" \
  --report "${REPO_ROOT}/environment/sam3_smoke_report.json"
SAM3_STATUS=$?

python "${REPO_ROOT}/scripts/smoke_qwen.py" \
  --image "${IMAGE_PATH}" \
  --model "${QWEN_MODEL_ID}" \
  --output-dir "${OUTPUT_ROOT}/qwen" \
  --report "${REPO_ROOT}/environment/qwen_smoke_report.json"
QWEN_STATUS=$?

python "${REPO_ROOT}/scripts/check_server_env.py" \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --sam3-checkpoint "${SAM3_CHECKPOINT}"
ENV_STATUS=$?
set -e

echo "SAM3 smoke exit code: ${SAM3_STATUS}"
echo "Qwen smoke exit code: ${QWEN_STATUS}"
echo "Environment check exit code: ${ENV_STATUS}"
echo "Generated reports:"
echo "  environment/server_env_report.json"
echo "  environment/sam3_smoke_report.json"
echo "  environment/qwen_smoke_report.json"
echo "Ignored artifacts: ${OUTPUT_ROOT}"

if [[ ${SAM3_STATUS} -ne 0 || ${QWEN_STATUS} -ne 0 || ${ENV_STATUS} -ne 0 ]]; then
  exit 1
fi
