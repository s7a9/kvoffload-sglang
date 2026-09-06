#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PORT="${PORT:-31000}"
TAG="${TAG:-pressure-ondemand}"
MODE="${MODE:-offload}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/kvoffload/results/glm52-16k8k-small-graph}"
SERVER_LOG="${SERVER_LOG:-${RESULT_DIR}/${TAG}-server.log}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
HEALTH_ENDPOINT_GENERATION="${HEALTH_ENDPOINT_GENERATION:-false}"

mkdir -p "${RESULT_DIR}"

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -INT "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${HEALTH_ENDPOINT_GENERATION}" \
MODE="${MODE}" \
TAG="${TAG}" \
PORT="${PORT}" \
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}" \
    bash kvoffload/scripts/start_server_glm52_fp8.sh >"${SERVER_LOG}" 2>&1 &
server_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until grep -q "The server is fired up and ready to roll!" "${SERVER_LOG}"; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "Server exited before becoming ready. See ${SERVER_LOG}" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Server did not become ready within ${STARTUP_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 2
done

RUN_ID="${TAG}" \
TAG="${TAG}" \
RESULT_DIR="${RESULT_DIR}" \
PORT="${PORT}" \
RUNS="${RUNS:-10}" \
CONCURRENT="${CONCURRENT:-10}" \
INPUT_TOKENS="${INPUT_TOKENS:-16384}" \
MAX_TOKENS="${MAX_TOKENS:-8192}" \
RANDOM_PROMPT="${RANDOM_PROMPT:-1}" \
EXACT_INPUT_TOKENS="${EXACT_INPUT_TOKENS:-1}" \
IGNORE_EOS="${IGNORE_EOS:-1}" \
REQUEST_RATE="${REQUEST_RATE:-0.1}" \
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}" \
    bash kvoffload/scripts/run_llm_bench.sh
