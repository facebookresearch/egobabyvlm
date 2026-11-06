#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# EgoVLP contrastive fine-tune for one (setting, seed). The paper runs are the
# 6 settings x 3 seeds below, each on one GPU.
#
# Submit with sbatch (the script intentionally has no #SBATCH directives —
# choose your own GPUs/QoS/account on the command line):
#   SETTING=coco_p50 SEED=42 EGOVLP_CHECKPOINT=/path/to/egovlp.pth \
#   EXTRACTED_ROOT=/path/to/scratch/coco_metaclip/extracted \
#   sbatch --gpus=1 --cpus-per-gpu=8 --mem=128G --time=12:00:00 \
#       --qos=<your_qos> --account=<your_account> \
#       apps/baselines/egovlp/scripts/train_egovlp.sh
#
# Or run all 18 paper runs as one array (index -> setting = idx % 6, seed = idx / 6):
#   sbatch --array=0-17 ... apps/baselines/egovlp/scripts/train_egovlp.sh
#
# Tunables via env vars:
#   EGOVLP_CHECKPOINT     Released EgoClip egovlp.pth (REQUIRED)
#   SETTING               coco_p0|coco_p25|coco_p50|coco_p75|coco_p100|babyview
#                         (REQUIRED unless SLURM_ARRAY_TASK_ID is set)
#   SEED                  Training seed (default: 42; paper: 42 43 44)
#   EXTRACTED_ROOT        Staged MetaCLIP jpegs (COCO settings; see README)
#   COCO_MC_MANIFEST, COCO_MC_SPLIT_DIR, COCO_MC_PERMUTATION_DIR, COCO_IMAGE_DIR
#                         COCO-MC data (COCO settings; see README)
#   BABYVIEW_ROOT         BabyView manifests + frames (babyview setting)
#   LR                    Learning rate (default: 3e-5)
#   BATCH_SIZE            Batch size (default: 32)
#   EPOCHS                Number of epochs (default: 10)
#   PYTHON                Python interpreter (default: python)
#   EGOBABYVLM_CKPT_DIR   Output base dir (default: ./egobabyvlm_checkpoints)

set -euo pipefail

SETTINGS=(coco_p0 coco_p25 coco_p50 coco_p75 coco_p100 babyview)
SEEDS=(42 43 44)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${SETTING:-}" ]; then
    SETTING="${SETTINGS[$((SLURM_ARRAY_TASK_ID % ${#SETTINGS[@]}))]}"
    SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#SETTINGS[@]}))]}"
fi

EGOVLP_CHECKPOINT="${EGOVLP_CHECKPOINT:?Set EGOVLP_CHECKPOINT to the released EgoClip egovlp.pth}"
SETTING="${SETTING:?Set SETTING (one of: ${SETTINGS[*]}) or submit as a SLURM array}"
SEED="${SEED:-42}"
LR="${LR:-3e-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
PYTHON="${PYTHON:-python}"
CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"

OUTPUT_DIR="${CKPT_BASE}/egovlp/${SETTING}/seed${SEED}"

case "$SETTING" in
    coco_p0|coco_p25|coco_p50|coco_p75|coco_p100)
        EXTRACTED_ROOT="${EXTRACTED_ROOT:?Set EXTRACTED_ROOT to the staged MetaCLIP jpegs (see README)}"
        DATA_ARGS=(--extracted_root "$EXTRACTED_ROOT")
        ;;
    babyview)
        : "${BABYVIEW_ROOT:?Set BABYVIEW_ROOT to the BabyView manifest + frames root}"
        DATA_ARGS=()
        ;;
    *)
        echo "Unknown SETTING=${SETTING} (expected one of: ${SETTINGS[*]})" >&2
        exit 1
        ;;
esac

if [ ! -f "$EGOVLP_CHECKPOINT" ]; then
    echo "EgoClip checkpoint not found at $EGOVLP_CHECKPOINT" >&2
    exit 1
fi

echo "=============================================="
echo "EgoVLP fine-tune"
echo "  Setting:    ${SETTING}"
echo "  Seed:       ${SEED}"
echo "  LR:         ${LR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs:     ${EPOCHS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "=============================================="

"$PYTHON" -m apps.baselines.egovlp.training.train \
    --dataset "$SETTING" \
    --seed "$SEED" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --pretrained_checkpoint "$EGOVLP_CHECKPOINT" \
    "${DATA_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR"

echo "EgoVLP fine-tune complete; checkpoints (best.pth, latest.pth) in ${OUTPUT_DIR}"
