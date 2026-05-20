#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-build-adjectives
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-build-adjectives.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-build-adjectives.err
# Stage 2: Build lexical adjective word lists + contrastive phrases.
#
# Without LLM (deterministic fallback phrases):
#   bash run_build_adjectives.sh --vocab-dir data/coco_TIMESTAMP --name COCO
#
# With LLM (launches a vLLM server automatically):
#   bash run_build_adjectives.sh --vocab-dir data/coco_TIMESTAMP --name COCO \
#       --model google/gemma-4-26B-A4B-it
set -euo pipefail

# ---------------------------------------------------------------------------
# Parse --model from args to decide whether to launch vLLM
# ---------------------------------------------------------------------------
VLLM_MODEL=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--model" ]]; then
        VLLM_MODEL="$arg"
    elif [[ "$arg" == --model=* ]]; then
        VLLM_MODEL="${arg#--model=}"
    fi
    prev="$arg"
done

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Shutting down vLLM server (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        # Wait up to 30s for graceful shutdown, then SIGKILL.
        for _ in $(seq 1 30); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM did not exit after SIGTERM; sending SIGKILL..."
            kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -n "$VLLM_MODEL" ]]; then
    echo "Launching vLLM server: $VLLM_MODEL on port $VLLM_PORT..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --served-model-name "$(basename "$VLLM_MODEL")" \
        --port "$VLLM_PORT" \
        --trust-remote-code &
    VLLM_PID=$!

    echo "Waiting for vLLM server..."
    for _ in $(seq 1 120); do
        curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
        sleep 5
    done
    curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
        || { echo "ERROR: vLLM server not ready after 10 minutes."; exit 1; }
    echo "vLLM server ready."
fi

EXTRA_ARGS=()
if [[ -n "$VLLM_MODEL" ]]; then
    EXTRA_ARGS+=(--api-base "http://localhost:${VLLM_PORT}/v1")
fi

python -m apps.benchmark_creation.pipeline.lexical.build_adjectives "${EXTRA_ARGS[@]}" "$@"
