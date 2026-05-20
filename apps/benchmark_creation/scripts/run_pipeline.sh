#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-pipeline
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-pipeline.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-pipeline.err
# ==========================================================================
# run_pipeline.sh — Run the full Machine-DevBench creation pipeline.
#
# Stages run in dependency order. Independent stages run in parallel.
# Vocabulary extraction reads the per-corpus manifests configured in
# apps/benchmark_creation/configs/paths.yaml.
#
# Submit with sbatch — NOT bash. The #SBATCH directives above are only
# honored by sbatch; running with `bash` executes locally on the login node
# with no GPU allocation. Make sure MachineDevBench_logs/ exists first
# (otherwise SLURM silently drops stdout/stderr).
#
# Usage (from the repo root):
#   mkdir -p MachineDevBench_logs   # one-time, before first submission
#
#   # Full pipeline from a corpus (e.g. COCO)
#   sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
#       --dataset coco --name COCO
#
#   # Test mode (10 items per task)
#   sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
#       --dataset coco --name COCO --test
#
#   # Custom output directory
#   sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
#       --output-dir data/my_benchmark \
#       --dataset coco --name COCO
#
#   # Skip vocab extraction, provide your own vocab_sorted.csv
#   sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
#       --name COCO --vocab-csv path/to/vocab_sorted.csv
#
# Monitor:
#   squeue -u $USER
#   tail -f MachineDevBench_logs/slurm-<jobid>-pipeline.out
#
# Environment variables:
#   VLLM_PORT        vLLM server port (default: 8000)
#   SKIP_IMAGES      set to 1 to skip image generation stages
#   SKIP_FILTERING   set to 1 to skip post-filtering
#   FILTER_MODEL     VLM model for grammatical post-filtering
# ==========================================================================
set -euo pipefail

# Under SLURM the script is copied to /var/spool, so BASH_SOURCE points there.
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked) to resolve paths.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
    # Compute nodes may lack internet — use cached HF models.
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VOCAB_CSV=""
# Default matches outputs_root in configs/paths.yaml, resolved relative to
# the apps/benchmark_creation/ package root.  Under SLURM BASH_SOURCE points to
# /var/spool, so fall back to SLURM_SUBMIT_DIR.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    _REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    _REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

# Resolve pixi env pythons up front. The orchestrator runs Python tasks under
# the `dev` env (transformers/diffusers/nltk/openai) and only the vLLM server
# under the `vllm` env (which has no torch). Falling back to bare `python`
# would pick up whatever happens to be on PATH and miss most of the deps.
DEV_PYTHON="${_REPO_ROOT}/.pixi/envs/dev/bin/python"
VLLM_PYTHON="${_REPO_ROOT}/.pixi/envs/vllm/bin/python"
for p in "$DEV_PYTHON" "$VLLM_PYTHON"; do
    if [[ ! -x "$p" ]]; then
        echo "ERROR: pixi env python missing: $p" >&2
        echo "  Run \`pixi install -e dev\` and \`pixi install -e vllm\` from the repo root." >&2
        exit 1
    fi
done

OUTPUT_DIR="${_REPO_ROOT}/MachineDevBench"
DATASET=""
NAME="Dataset"
MODEL="google/gemma-4-26B-A4B-it"
STYLES="realistic cartoon"
IMG_MODEL="black-forest-labs/FLUX.2-klein-4B"
NUM_GPUS=4
TEST_MODE=0

# Size controls (overridden by --test)
MAX_NOUNS_PER_CATEGORY=240    # --max-per-category for build_nouns
MAX_ADJECTIVES=80             # --max-words for build_adjectives
ITEMS_PER_GRAM_CATEGORY=250   # --items-per-category for build_benchmark


while [[ $# -gt 0 ]]; do
    case "$1" in
        --vocab-csv)                VOCAB_CSV="$2"; shift 2 ;;
        --output-dir)               OUTPUT_DIR="$2"; shift 2 ;;
        --dataset)                  DATASET="$2"; shift 2 ;;
        --name)                     NAME="$2"; shift 2 ;;
        --model)                    MODEL="$2"; shift 2 ;;
        --styles)                   STYLES="$2"; shift 2 ;;
        --img-model)                IMG_MODEL="$2"; shift 2 ;;
        --num-gpus)                 NUM_GPUS="$2"; shift 2 ;;
        --test)                     TEST_MODE=1; shift ;;
        --max-nouns-per-category)   MAX_NOUNS_PER_CATEGORY="$2"; shift 2 ;;
        --max-adjectives)           MAX_ADJECTIVES="$2"; shift 2 ;;
        --items-per-gram-category)  ITEMS_PER_GRAM_CATEGORY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$VOCAB_CSV" && -z "$DATASET" ]]; then
    cat >&2 <<'EOF'
Usage: run_pipeline.sh --dataset DATASET --name NAME [OPTIONS]

Required:
  --dataset DATASET                 Source corpus: coco, howto100m, ego4d, babyview, all
  --name NAME                       Dataset label (e.g. COCO)

  OR provide a pre-computed vocabulary:
  --name NAME                       Dataset label
  --vocab-csv PATH                  Path to existing vocab_sorted.csv

Options:
  --output-dir PATH                 Output directory (default: ./MachineDevBench)
  --model MODEL                     LLM model (default: google/gemma-4-26B-A4B-it)
  --img-model MODEL                 Image model (default: FLUX.2-klein-4B)
  --styles "s1 s2"                  Image styles (default: "realistic cartoon")
  --num-gpus N                      GPUs for image generation (default: 4)
  --test                            Test mode: 10 items per task
  --max-nouns-per-category N        Nouns per category (default: 240)
  --max-adjectives N                Total adjectives (default: 80)
  --items-per-gram-category N       Grammatical items per category (default: 250)

Environment:
  VLLM_PORT=8000                    vLLM server port
  SKIP_IMAGES=1                     Skip image generation
  SKIP_FILTERING=1                  Skip post-filtering
  FILTER_MODEL=...                  VLM for grammatical filtering
EOF
    exit 1
fi

# Apply test mode: small counts + debug flags for image generation
if [[ "$TEST_MODE" == "1" ]]; then
    MAX_NOUNS_PER_CATEGORY=10
    MAX_ADJECTIVES=10
    ITEMS_PER_GRAM_CATEGORY=10
    echo "*** TEST MODE: 10 items per task ***"
fi

VLLM_PORT="${VLLM_PORT:-8000}"
SKIP_IMAGES="${SKIP_IMAGES:-0}"
SKIP_FILTERING="${SKIP_FILTERING:-0}"
FILTER_MODEL="${FILTER_MODEL:-$MODEL}"

SCRIPT_DIR="${_REPO_ROOT}/apps/benchmark_creation/scripts"

# Read styles into an array
read -ra STYLE_ARGS <<< "$STYLES"
STYLE_FLAGS=(--styles "${STYLE_ARGS[@]}")

# Debug flag for image generation in test mode
IMG_DEBUG_FLAG=()
if [[ "$TEST_MODE" == "1" ]]; then
    IMG_DEBUG_FLAG=(--debug)
fi

# ---------------------------------------------------------------------------
# Background process tracking
#
# Every backgrounded child must be registered via track_pid() so that
# cleanup() can guarantee they are reaped on early exit. Without this,
# a `set -e` failure in one waited job leaves siblings holding their
# GPUs, wedging the SLURM node.
# ---------------------------------------------------------------------------
BG_PIDS=()

track_pid() {
    BG_PIDS+=("$1")
}

kill_bg_pids() {
    local pid
    for pid in "${BG_PIDS[@]:-}"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing background PID $pid..." >&2
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # Give them a few seconds to exit, then SIGKILL whatever survives.
    local i
    for i in $(seq 1 10); do
        local any_alive=0
        for pid in "${BG_PIDS[@]:-}"; do
            [[ -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then any_alive=1; fi
        done
        [[ "$any_alive" == "0" ]] && break
        sleep 1
    done
    for pid in "${BG_PIDS[@]:-}"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "  SIGKILL background PID $pid..." >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    BG_PIDS=()
}

# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------
VLLM_PID=""

start_vllm() {
    local model="$1"; shift
    local extra_args=("$@")

    if [[ -z "$model" ]]; then return; fi
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        stop_vllm
    fi

    echo "  Launching vLLM server: $model on port $VLLM_PORT..."
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --served-model-name "$(basename "$model")" \
        --port "$VLLM_PORT" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code "${extra_args[@]}" &
    VLLM_PID=$!

    for _ in $(seq 1 120); do
        curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
        sleep 5
    done
    if ! curl -s --max-time 5 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "ERROR: vLLM server not ready after 10 minutes." >&2; exit 1
    fi
    echo "  vLLM server ready."
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  Stopping vLLM server (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        # Wait up to 30s for graceful shutdown, then SIGKILL.
        local i
        for i in $(seq 1 30); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "  vLLM did not exit after SIGTERM; sending SIGKILL..." >&2
            kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

cleanup() {
    # Order matters: kill workload children first so they release the GPU,
    # then shut down the vLLM server.
    kill_bg_pids
    stop_vllm
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

API_BASE="http://localhost:${VLLM_PORT}/v1"
LLM_ARGS=()
if [[ -n "$MODEL" ]]; then
    LLM_ARGS=(--api-base "$API_BASE" --model "$(basename "$MODEL")")
fi

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
banner "Machine-DevBench Pipeline"
echo "  Dataset:         ${DATASET:-'(using --vocab-csv)'}"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Name:            $NAME"
echo "  Vocab CSV:       ${VOCAB_CSV:-'(will be computed from corpus)'}"
echo "  LLM model:       $MODEL"
echo "  Image model:     $IMG_MODEL"
echo "  Filter model:    $FILTER_MODEL"
echo "  Styles:          $STYLES"
echo "  GPUs:            $NUM_GPUS"
echo "  Test mode:       $( [[ $TEST_MODE == 1 ]] && echo 'YES' || echo 'no' )"
echo "  Nouns/category:  $MAX_NOUNS_PER_CATEGORY"
echo "  Adjectives:      $MAX_ADJECTIVES"
echo "  Gram/category:   $ITEMS_PER_GRAM_CATEGORY"

# ==========================================================================
# Stage 0: Vocabulary Extraction (if no --vocab-csv provided)
# ==========================================================================
if [[ -z "$VOCAB_CSV" ]]; then
    banner "Stage 0: Vocabulary Extraction (--dataset $DATASET)"

    VOCAB_OUTPUT_DIR="${OUTPUT_DIR}/vocab_coverage"
    # Capture stdout so we can parse the "Results saved to:" line, which
    # tells us exactly which timestamped subdirectory the python script
    # created on this run. This avoids two bugs:
    #   1) `find ... | head -1` races with SIGPIPE under `set -o pipefail`
    #      when there are many matches (job 4612768 died with exit 13 here).
    #   2) `find` may return a stale vocab_sorted.csv from a prior run.
    VOCAB_LOG=$(mktemp)
    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.vocab_coverage \
        --dataset "$DATASET" \
        --output-root "$VOCAB_OUTPUT_DIR" 2>&1 | tee "$VOCAB_LOG"

    VOCAB_DIR=$(grep -m1 "Results saved to: " "$VOCAB_LOG" | sed 's/.*Results saved to: //' || true)
    rm -f "$VOCAB_LOG"
    if [[ -z "$VOCAB_DIR" ]]; then
        echo "ERROR: vocab_coverage did not print 'Results saved to:'" >&2
        exit 1
    fi
    VOCAB_CSV="$VOCAB_DIR/vocab_sorted.csv"
    if [[ ! -f "$VOCAB_CSV" ]]; then
        echo "ERROR: vocab_sorted.csv not found at $VOCAB_CSV" >&2
        exit 1
    fi
    echo "  Vocab CSV: $VOCAB_CSV"
else
    banner "Stage 0: Vocabulary Extraction (SKIPPED — using $VOCAB_CSV)"
fi

# ==========================================================================
# Stage 1: Vocabulary Curation
# ==========================================================================
banner "Stage 1: Vocabulary Curation"

"$DEV_PYTHON" -m apps.benchmark_creation.pipeline.create_vocabulary \
    --vocab-csv "$VOCAB_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --name "$NAME"

# Resolve the timestamped output directory created by create_vocabulary
DATA_DIR=$(ls -dt "${OUTPUT_DIR}/${NAME}"_* 2>/dev/null | head -1)
if [[ -z "$DATA_DIR" || ! -d "$DATA_DIR" ]]; then
    DATA_DIR="$OUTPUT_DIR"
fi
echo "  Data directory: $DATA_DIR"

# ==========================================================================
# Stage 2A + 3A: Build word lists + grammatical pairs (parallel)
# ==========================================================================
banner "Stage 2A + 3A: Build Word Lists & Grammatical Pairs"

if [[ -n "$MODEL" ]]; then
    start_vllm "$MODEL"
fi

"$DEV_PYTHON" -m apps.benchmark_creation.pipeline.lexical.build_nouns \
    --vocab-dir "$DATA_DIR" --name "$NAME" \
    --max-per-category "$MAX_NOUNS_PER_CATEGORY" \
    "${LLM_ARGS[@]}" &
PID_NOUNS=$!; track_pid "$PID_NOUNS"

"$DEV_PYTHON" -m apps.benchmark_creation.pipeline.lexical.build_adjectives \
    --vocab-dir "$DATA_DIR" --name "$NAME" \
    --max-words "$MAX_ADJECTIVES" \
    "${LLM_ARGS[@]}" &
PID_ADJ=$!; track_pid "$PID_ADJ"

PID_GRAM=""
if [[ -n "$MODEL" ]]; then
    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.grammatical.build_benchmark \
        --vocab-dir "$DATA_DIR" --name "$NAME" \
        --items-per-category "$ITEMS_PER_GRAM_CATEGORY" \
        "${LLM_ARGS[@]}" &
    PID_GRAM=$!; track_pid "$PID_GRAM"
fi

wait "$PID_NOUNS" || { echo "ERROR: build_nouns failed" >&2; exit 1; }
echo "  Nouns done."
wait "$PID_ADJ" || { echo "ERROR: build_adjectives failed" >&2; exit 1; }
echo "  Adjectives done."
if [[ -n "$PID_GRAM" ]]; then
    wait "$PID_GRAM" || { echo "ERROR: build_benchmark failed" >&2; exit 1; }
    echo "  Grammatical done."
fi

stop_vllm

# ==========================================================================
# Stage 2B + 3B: Image Generation (parallel, no vLLM needed)
# ==========================================================================
if [[ "$SKIP_IMAGES" != "1" ]]; then
    banner "Stage 2B + 3B: Image Generation"

    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.lexical.generate_noun_images \
        --data-dir "$DATA_DIR" --model-id "$IMG_MODEL" \
        --num-gpus "$NUM_GPUS" "${STYLE_FLAGS[@]}" "${IMG_DEBUG_FLAG[@]}" &
    PID_NOUN_IMG=$!; track_pid "$PID_NOUN_IMG"

    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.lexical.generate_adj_images \
        --data-dir "$DATA_DIR" --model-id "$IMG_MODEL" \
        --num-gpus "$NUM_GPUS" "${STYLE_FLAGS[@]}" "${IMG_DEBUG_FLAG[@]}" &
    PID_ADJ_IMG=$!; track_pid "$PID_ADJ_IMG"

    PID_GRAM_IMG=""
    if [[ -n "$MODEL" ]]; then
        "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.grammatical.generate_images \
            --data-dir "$DATA_DIR" --model-id "$IMG_MODEL" \
            --num-gpus "$NUM_GPUS" "${STYLE_FLAGS[@]}" "${IMG_DEBUG_FLAG[@]}" &
        PID_GRAM_IMG=$!; track_pid "$PID_GRAM_IMG"
    fi

    wait "$PID_NOUN_IMG" || { echo "ERROR: generate_noun_images failed" >&2; exit 1; }
    echo "  Noun images done."
    wait "$PID_ADJ_IMG" || { echo "ERROR: generate_adj_images failed" >&2; exit 1; }
    echo "  Adjective images done."
    if [[ -n "$PID_GRAM_IMG" ]]; then
        wait "$PID_GRAM_IMG" || { echo "ERROR: generate_grammatical_images failed" >&2; exit 1; }
        echo "  Grammatical images done."
    fi
else
    banner "Stage 2B + 3B: Image Generation (SKIPPED)"
fi

# ==========================================================================
# Stage 4: Post-Filtering (parallel)
# ==========================================================================
if [[ "$SKIP_FILTERING" != "1" && "$SKIP_IMAGES" != "1" ]]; then
    banner "Stage 4: Post-Filtering"

    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical \
        --data-dir "$DATA_DIR" "${STYLE_FLAGS[@]}" --write-filtered &
    PID_FILT_LEX=$!; track_pid "$PID_FILT_LEX"

    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical_hard \
        --data-dir "$DATA_DIR" "${STYLE_FLAGS[@]}" &
    PID_FILT_LEX_HARD=$!; track_pid "$PID_FILT_LEX_HARD"

    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.filtering.compute_distributions \
        --data-dir "$DATA_DIR" "${STYLE_FLAGS[@]}" &
    PID_DIST=$!; track_pid "$PID_DIST"

    PID_FILT_GRAM=""
    if [[ -n "$FILTER_MODEL" && -n "$MODEL" ]]; then
        start_vllm "$FILTER_MODEL" --limit-mm-per-prompt '{"image":2}'

        "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.filtering.post_filter_grammatical \
            --data-dir "$DATA_DIR" "${STYLE_FLAGS[@]}" \
            --api-base "$API_BASE" --model "$(basename "$FILTER_MODEL")" \
            --write-filtered &
        PID_FILT_GRAM=$!; track_pid "$PID_FILT_GRAM"
    fi

    wait "$PID_FILT_LEX" || { echo "WARNING: post_filter_lexical failed" >&2; }
    wait "$PID_FILT_LEX_HARD" || { echo "WARNING: post_filter_lexical_hard failed" >&2; }
    wait "$PID_DIST" || { echo "WARNING: compute_distributions failed" >&2; }
    if [[ -n "$PID_FILT_GRAM" ]]; then
        wait "$PID_FILT_GRAM" || { echo "WARNING: post_filter_grammatical failed" >&2; }
    fi
    echo "  Post-filtering done."

    stop_vllm
else
    banner "Stage 4: Post-Filtering (SKIPPED)"
fi

# ==========================================================================
# Stage 5: Manifest Generation (parallel)
# ==========================================================================
banner "Stage 5: Manifest Generation"

"$DEV_PYTHON" -m apps.benchmark_creation.pipeline.manifests.generate_lexical \
    --data-dir "$DATA_DIR" --tasks nouns adjectives "${STYLE_FLAGS[@]}" &
PID_MAN_LEX=$!; track_pid "$PID_MAN_LEX"

PID_MAN_GRAM=""
if [[ -n "$MODEL" ]]; then
    "$DEV_PYTHON" -m apps.benchmark_creation.pipeline.manifests.generate_grammatical \
        --data-dir "$DATA_DIR" "${STYLE_FLAGS[@]}" &
    PID_MAN_GRAM=$!; track_pid "$PID_MAN_GRAM"
fi

wait "$PID_MAN_LEX" || { echo "ERROR: generate_manifests_lexical failed" >&2; exit 1; }
echo "  Lexical manifests done."
if [[ -n "$PID_MAN_GRAM" ]]; then
    wait "$PID_MAN_GRAM" || { echo "ERROR: generate_manifests_grammatical failed" >&2; exit 1; }
    echo "  Grammatical manifests done."
fi

# ==========================================================================
# Done
# ==========================================================================
banner "Pipeline Complete"
echo "  Output: $DATA_DIR"
echo ""
