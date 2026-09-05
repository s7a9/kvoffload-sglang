#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_BIN="${SGLANG_BIN:-${REPO_ROOT}/.venv/bin/sglang}"

MODE="${MODE:-offload}"
TAG="${TAG:-glm52-${MODE}}"
MODEL_PATH="${MODEL_PATH:-/data1/models/GLM-5.2-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-1}"
HICACHE_SIZE="${HICACHE_SIZE:-40}"
KV_OFFLOAD_POLICY="${KV_OFFLOAD_POLICY:-paper_v1}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-EAGLE}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-1}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-2}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"

case "${MODE}" in
    baseline)
        unset SGLANG_ENABLE_SYNC_CACHE
        ;;
    offload)
        export SGLANG_ENABLE_SYNC_CACHE="1"
        ;;
    *)
        echo "Error: MODE must be baseline or offload, got: ${MODE}" >&2
        exit 2
        ;;
esac

export SGLANG_RUN_TAG="${TAG}"
cd "${REPO_ROOT}"

CMD=(
    "${SGLANG_BIN}" serve
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
    --ep-size "${EP_SIZE}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --random-seed "${SERVER_RANDOM_SEED}"
    --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --trust-remote-code
)

if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
    CMD+=(--skip-server-warmup)
fi

if [[ "${MODE}" == "offload" ]]; then
    CMD+=(
        --enable-hierarchical-cache
        --hicache-size "${HICACHE_SIZE}"
        --kv-offload-policy "${KV_OFFLOAD_POLICY}"
    )
fi

echo "Starting GLM-5.2-FP8 mode=${MODE} tag=${TAG} port=${PORT}" >&2

if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_SERVER_ARGS_ARR=(${EXTRA_SERVER_ARGS})
    CMD+=("${EXTRA_SERVER_ARGS_ARR[@]}")
fi

if (($# > 0)); then
    CMD+=("$@")
fi

exec "${CMD[@]}"
