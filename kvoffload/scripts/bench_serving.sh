#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BACKEND="${BACKEND:-sglang}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-/data1/models/GLM-5.2-FP8}"

DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-200}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-2048}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.2}"
SEED="${SEED:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
OUTPUT_SPEED="${OUTPUT_SPEED:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-${RUN_ID}}"
RESULT_DIR="${RESULT_DIR:-kvoffload/results/${RUN_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${RESULT_DIR}/bench_results.jsonl}"
REQUEST_OUTPUTS_FILE="${REQUEST_OUTPUTS_FILE:-${RESULT_DIR}/${TAG}_request_outputs.jsonl}"

EXTRA_CLI_ARGS=()
while (($#)); do
  case "$1" in
    --tag)
      if (($# < 2)); then
        echo "Error: --tag requires a value" >&2
        exit 1
      fi
      TAG="$2"
      shift 2
      ;;
    --tag=*)
      TAG="${1#*=}"
      shift
      ;;
    *)
      EXTRA_CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${REQUEST_OUTPUTS_FILE}")"

CMD=(
  python3 -m sglang.bench_serving
  --backend "${BACKEND}"
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL}"
  --dataset-name "${DATASET_NAME}"
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --random-input-len "${RANDOM_INPUT_LEN}"
  --random-output-len "${RANDOM_OUTPUT_LEN}"
  --random-range-ratio "${RANDOM_RANGE_RATIO}"
  --seed "${SEED}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --output-file "${OUTPUT_FILE}"
  --output-details
  --save-request-outputs
  --request-outputs-file "${REQUEST_OUTPUTS_FILE}"
)

if [[ -n "${DATASET_PATH}" ]]; then
  CMD+=(--dataset-path "${DATASET_PATH}")
fi

if [[ -n "${OUTPUT_SPEED}" ]]; then
  CMD+=(--output-speed "${OUTPUT_SPEED}")
fi

if [[ -n "${TAG}" ]]; then
  CMD+=(--tag "${TAG}")
fi

if [[ -n "${EXTRA_BENCH_ARGS:-}" ]]; then
  # Intentionally split EXTRA_BENCH_ARGS by shell words for compatibility.
  # shellcheck disable=SC2206
  EXTRA_BENCH_ARGS_ARR=(${EXTRA_BENCH_ARGS})
  CMD+=("${EXTRA_BENCH_ARGS_ARR[@]}")
fi

if ((${#EXTRA_CLI_ARGS[@]} > 0)); then
  CMD+=("${EXTRA_CLI_ARGS[@]}")
fi

exec "${CMD[@]}"
