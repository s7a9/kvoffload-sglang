#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-/data1/models/GLM-5.2-FP8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-${RUN_ID}}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/kvoffload/results/${RUN_ID}}"
RUNS="${RUNS:-5}"
CONCURRENT="${CONCURRENT:-1}"
INPUT_TOKENS="${INPUT_TOKENS:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
RANDOM_PROMPT="${RANDOM_PROMPT:-0}"
REQUEST_RATE="${REQUEST_RATE:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
IGNORE_EOS="${IGNORE_EOS:-0}"
EXACT_INPUT_TOKENS="${EXACT_INPUT_TOKENS:-0}"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

CMD=(
    "${PYTHON_BIN}" kvoffload/llm_bench_1785131610781.py
    --url "http://${HOST}:${PORT}/v1/chat/completions"
    --model "${MODEL}"
    --runs "${RUNS}"
    --concurrent "${CONCURRENT}"
    --input-tokens "${INPUT_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --seed "${SEED}"
    --temperature "${TEMPERATURE}"
    --request-rate "${REQUEST_RATE}"
    --request-timeout "${REQUEST_TIMEOUT}"
    --tag "${TAG}"
    --output-dir "${RESULT_DIR}"
    --output-json "${RESULT_DIR}/${TAG}.json"
    --output-csv "${RESULT_DIR}/${TAG}.csv"
    --summary-csv "${RESULT_DIR}/benchmark_results.csv"
    --note "${TAG}"
)

if [[ "${RANDOM_PROMPT}" == "1" ]]; then
    CMD+=(--random-prompt)
fi

if [[ "${IGNORE_EOS}" == "1" ]]; then
    CMD+=(--ignore-eos)
fi

if [[ "${EXACT_INPUT_TOKENS}" == "1" ]]; then
    CMD+=(--exact-input-tokens)
fi

if (($# > 0)); then
    CMD+=("$@")
fi

exec "${CMD[@]}"
