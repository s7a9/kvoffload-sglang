#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TAG="${TAG:-}"
MODEL_PATH="${MODEL_PATH:-/storage/nas/dch/models/Qwen--Qwen3-4B-Instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-auto}"

PAPER_V1_ATTENTION_BACKEND="${PAPER_V1_ATTENTION_BACKEND:-fa3}"
KV_OFFLOAD_POLICY="${KV_OFFLOAD_POLICY:-paper_v1}"
export SGLANG_ENABLE_SYNC_CACHE="1"

cd "${REPO_ROOT}"

CMD=(
	python -m sglang.launch_server
	--model-path "${MODEL_PATH}"
	--host "${HOST}"
	--port "${PORT}"
	--tp-size "${TP_SIZE}"
	--dtype "${DTYPE}"
	--attention-backend "${PAPER_V1_ATTENTION_BACKEND}"
	--kv-offload-policy "${KV_OFFLOAD_POLICY}"
	--enable-hierarchical-cache
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
