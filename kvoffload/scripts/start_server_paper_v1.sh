#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TAG="${TAG:-}"
MODEL_PATH="${MODEL_PATH:-MiniMaxAI/MiniMax-M2.77}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
DTYPE="${DTYPE:-auto}"

KV_OFFLOAD_POLICY="${KV_OFFLOAD_POLICY:-paper_v1}"
# export SGLANG_ENABLE_SYNC_CACHE="1"
# export SGLANG_KV_OFFLOAD_POLICY_DEBUG_LOG=1

cd "${REPO_ROOT}"

CMD=(
	python -m sglang.launch_server
	--model-path "${MODEL_PATH}"
	--host "${HOST}"
	--port "${PORT}"
	--tp-size "${TP_SIZE}"
	--ep-size "${EP_SIZE:-1}"
	--dtype "${DTYPE}"
	--kv-offload-policy "${KV_OFFLOAD_POLICY}"
	--enable-hierarchical-cache
	--trust-remote-code
    --tool-call-parser minimax-m2
    --reasoning-parser minimax-append-think
	--hicache-size 20
	--max-running-requests 128
)

if [[ -n "${TAG}" ]]; then
	export SGLANG_RUN_TAG="${TAG}"
	echo "Starting paper_v1 server with tag=${TAG}" >&2
else
	echo "Starting paper_v1 server" >&2
fi

if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
	# shellcheck disable=SC2206
	EXTRA_SERVER_ARGS_ARR=(${EXTRA_SERVER_ARGS})
	CMD+=("${EXTRA_SERVER_ARGS_ARR[@]}")
fi

if (($# > 0)); then
	CMD+=("$@")
fi

exec "${CMD[@]}"
