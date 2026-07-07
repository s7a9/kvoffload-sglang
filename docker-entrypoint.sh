#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_PATH:-/models/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
export PYTHONPATH="/workspace/sglang/python:${PYTHONPATH:-}"

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    $SGLANG_ARGS
