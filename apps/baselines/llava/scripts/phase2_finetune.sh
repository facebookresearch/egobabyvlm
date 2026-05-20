#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Phase 2: full finetune (LLM + projector unfrozen, vision tower frozen).
#
# Submit with sbatch (the script intentionally has no #SBATCH directives —
# choose your own GPUs/QoS/account on the command line):
#   sbatch --gpus=1 --mem=64G --time=12:00:00 \
#       --qos=<your_qos> --account=<your_account> \
#       apps/baselines/llava/scripts/phase2_finetune.sh
#
# Tunables via env vars (all optional, sensible defaults below):
#   GPT2_MODEL              Phase 0 GPT-2 checkpoint dir (REQUIRED)
#   PRETRAIN_PROJECTOR      Phase 1 mm_projector.bin (REQUIRED)
#   VISION_TOWER            torch.hub identifier (default: dinov2_vitb14)
#   VISION_TOWER_PATH       Optional path to a custom DINOv2 ckpt
#   DATA_PATH               LLaVA-format COCO JSON (REQUIRED)
#   IMAGE_FOLDER            COCO image root (REQUIRED)
#   PERMUTATION_PERCENT     0..100 — see phase1_pretrain.sh
#   SEED                    Random seed (default: 42)
#   LR                      Learning rate (default: 2e-3)
#   EPOCHS                  Number of epochs (default: 5)
#   EGOBABYVLM_CKPT_DIR   Output base dir (default: ./egobabyvlm_checkpoints)

set -euo pipefail

GPT2_MODEL="${GPT2_MODEL:?Set GPT2_MODEL to the Phase 0 GPT-2 checkpoint directory}"
PRETRAIN_PROJECTOR="${PRETRAIN_PROJECTOR:?Set PRETRAIN_PROJECTOR to the Phase 1 mm_projector.bin path}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a LLaVA-format JSON manifest}"
IMAGE_FOLDER="${IMAGE_FOLDER:?Set IMAGE_FOLDER to the COCO image root}"

VISION_TOWER="${VISION_TOWER:-dinov2_vitb14}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-}"
PERMUTATION_PERCENT="${PERMUTATION_PERCENT:-0}"
SEED="${SEED:-42}"
LR="${LR:-2e-3}"
EPOCHS="${EPOCHS:-5}"
CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"

if [ -n "$VISION_TOWER_PATH" ]; then
    VT_TAG="custom"
    VISION_TOWER_PATH_ARG="--vision_tower_path $VISION_TOWER_PATH"
else
    VT_TAG="offshelf"
    VISION_TOWER_PATH_ARG=""
fi

if [ "$PERMUTATION_PERCENT" -gt 0 ]; then
    PERM_TAG="_p${PERMUTATION_PERCENT}"
else
    PERM_TAG=""
fi

OUTPUT_DIR="${CKPT_BASE}/phase2_finetune_coco/${VT_TAG}${PERM_TAG}_lr${LR}_ep${EPOCHS}"

echo "=============================================="
echo "Phase 2: full finetune"
echo "  GPT-2:        ${GPT2_MODEL}"
echo "  Projector:    ${PRETRAIN_PROJECTOR}"
echo "  Vision tower: ${VISION_TOWER} (${VT_TAG})"
echo "  Data:         ${DATA_PATH}"
echo "  Images:       ${IMAGE_FOLDER}"
echo "  Permutation:  ${PERMUTATION_PERCENT}%"
echo "  LR:           ${LR}"
echo "  Epochs:       ${EPOCHS}"
echo "  Output:       ${OUTPUT_DIR}"
echo "=============================================="

if [ ! -d "$GPT2_MODEL" ]; then
    echo "GPT-2 model not found at $GPT2_MODEL"
    exit 1
fi
if [ ! -f "$PRETRAIN_PROJECTOR" ]; then
    echo "Phase 1 projector not found at $PRETRAIN_PROJECTOR"
    exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "Training data not found at $DATA_PATH"
    exit 1
fi

deepspeed --module apps.baselines.llava.train.train \
    --deepspeed apps/baselines/llava/scripts/deepspeed_zero2_config.json \
    --model_name_or_path "$GPT2_MODEL" \
    --vision_tower "$VISION_TOWER" \
    $VISION_TOWER_PATH_ARG \
    --pretrain_mm_mlp_adapter "$PRETRAIN_PROJECTOR" \
    --mm_projector_type mlp2x_gelu \
    --freeze_vision_tower True \
    --freeze_llm_backbone False \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --bf16 False \
    --fp16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb

echo "Phase 2 training complete; checkpoint saved to ${OUTPUT_DIR}"
