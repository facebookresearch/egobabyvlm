#!/bin/bash -l

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=llava-phase0-gpt2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/phase0_gpt2_%j.log
#SBATCH --error=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/phase0_gpt2_%j.err

# Phase 0: Train GPT-2 from scratch on COCO captions.
#
# This script trains a GPT-2 language model from scratch on a plain-text
# corpus (one caption per line). The trained model is then used as the
# language backbone for LLaVA Phase 1 / Phase 2 training.
#
# Submit with sbatch (the #SBATCH directives are honoured only by sbatch):
#   sbatch --qos=<your_qos> --account=<your_account> \
#       apps/baselines/lm_training/scripts/phase0_train_gpt2.sh
#
# Tunables via env vars (all optional, sensible defaults below):
#   EGOBABYVLM_DATA_DIR    Directory containing coco_captions_{train,val}.txt
#                            (default: ./egobabyvlm_data)
#   EGOBABYVLM_CKPT_DIR    Where to write checkpoints
#                            (default: ./egobabyvlm_checkpoints)
#   EGOBABYVLM_LOG_DIR     Where SLURM writes %j logs
#                            (default: ./egobabyvlm_logs)
#   PHASE0_FORMAT            Optional caption format suffix (image_description,
#                            yes_no, random); selects coco_captions_${FMT}_{train,val}.txt
#   TOKENIZER_MODE           "custom" (default; trains a fresh BPE) or "mistral"
#                            (uses llava-hf/llava-v1.6-mistral-7b-hf tokenizer)
#   SEED                     Random seed (default: 42)

set -euo pipefail

DATA_DIR="${EGOBABYVLM_DATA_DIR:-./egobabyvlm_data}"
CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"
LOG_DIR="${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}"
mkdir -p "$LOG_DIR"

PHASE0_FORMAT="${PHASE0_FORMAT:-}"
TOKENIZER_MODE="${TOKENIZER_MODE:-custom}"
SEED="${SEED:-42}"

if [ -n "$PHASE0_FORMAT" ]; then
    TRAIN_FILE="${DATA_DIR}/coco_captions_${PHASE0_FORMAT}_train.txt"
    VAL_FILE="${DATA_DIR}/coco_captions_${PHASE0_FORMAT}_val.txt"
    FORMAT_TAG="${PHASE0_FORMAT}"
else
    TRAIN_FILE="${DATA_DIR}/coco_captions_train.txt"
    VAL_FILE="${DATA_DIR}/coco_captions_val.txt"
    FORMAT_TAG="none"
fi

if [ "$TOKENIZER_MODE" = "mistral" ]; then
    TOKENIZER_ARGS="--tokenizer_name llava-hf/llava-v1.6-mistral-7b-hf"
    TOK_TAG="mistral"
else
    TOKENIZER_ARGS="--vocab_size 52000"
    TOK_TAG="custom"
fi

LR="${LR:-1e-4}"
BS="${BS:-16}"
GACC="${GACC:-4}"
EPOCHS="${EPOCHS:-30}"

if [ "$SEED" != "42" ]; then
    SEED_TAG="_seed${SEED}"
else
    SEED_TAG=""
fi
OUTPUT_DIR="${CKPT_BASE}/phase0_gpt2/gpt2_${TOK_TAG}_${FORMAT_TAG}${SEED_TAG}"
export WANDB_NAME="gpt2_${TOK_TAG}_${FORMAT_TAG}${SEED_TAG}"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Training data not found at $TRAIN_FILE"
    echo "Set EGOBABYVLM_DATA_DIR or generate the corpus first."
    exit 1
fi

echo "=============================================="
echo "Phase 0: GPT-2 training on COCO captions"
echo "  Tokenizer:  ${TOK_TAG}"
echo "  Format tag: ${FORMAT_TAG}"
echo "  Train:      ${TRAIN_FILE}"
echo "  Val:        ${VAL_FILE}"
echo "  LR:         ${LR}"
echo "  Batch:      ${BS} (gacc=${GACC})"
echo "  Epochs:     ${EPOCHS}"
echo "  Seed:       ${SEED}"
echo "  Output:     ${OUTPUT_DIR}"
echo "=============================================="

python -m apps.baselines.lm_training.train.train_gpt2 \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    $TOKENIZER_ARGS \
    --do_train \
    --do_eval \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size "$BS" \
    --gradient_accumulation_steps "$GACC" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --weight_decay 0.3 \
    --lr_scheduler_type linear \
    --bf16 \
    --optim adamw_torch \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --seed "$SEED" \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --overwrite_cache \
    --report_to wandb

echo "Phase 0 training complete; model saved to ${OUTPUT_DIR}"
