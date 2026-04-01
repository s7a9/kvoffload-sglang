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

cd "${REPO_ROOT}"

CMD=(
	python -m sglang.launch_server
	--model-path "${MODEL_PATH}"
	--host "${HOST}"
	--port "${PORT}"
	--tp-size "${TP_SIZE}"
	--dtype "${DTYPE}"
)

if [[ -n "${TAG}" ]]; then
	export SGLANG_RUN_TAG="${TAG}"
	echo "Starting default server with tag=${TAG}" >&2
else
	echo "Starting default server" >&2
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
