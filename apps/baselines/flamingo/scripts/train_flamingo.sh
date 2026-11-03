#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Flamingo training for one (condition, seed) with the paper's settings
# (configs/ defaults): 4 GPUs x batch 32 = 128, 10 epochs.
#
# Submit with sbatch (the script intentionally has no #SBATCH directives —
# choose your own QoS/account on the command line). One task per GPU:
#   CONDITION=p50 SEED=42 \
#   COCOMC_VISION_TOWER_PATH=/path/to/coco_mc_dinov2/teacher_checkpoint.pth \
#   COCOMC_DECODER_INIT_PATH=/path/to/coco_mc_phase0_gpt2 \
#   EXTRACTED_ROOT=/path/to/scratch/coco_metaclip/extracted \
#   sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=8 \
#       --mem=256G --time=12:00:00 --qos=<your_qos> --account=<your_account> \
#       apps/baselines/flamingo/scripts/train_flamingo.sh
#
# Or run all 18 paper runs as one array (condition = idx % 6, seed = idx / 6):
#   sbatch --array=0-17 ... apps/baselines/flamingo/scripts/train_flamingo.sh
#
# Outside SLURM the trainer runs as a single process (for smoke tests).
#
# Tunables via env vars:
#   CONDITION                  p0|p25|p50|p75|p100 (COCO-MC, % captions shuffled)
#                              or babyview (REQUIRED unless SLURM_ARRAY_TASK_ID is set)
#   SEED                       Training seed (default: 42; paper: 42 43 44)
#   COCOMC_VISION_TOWER_PATH   DINOv2 teacher checkpoint pretrained on COCO-MC
#   COCOMC_DECODER_INIT_PATH   Phase-0 GPT-2 dir pretrained on COCO-MC
#   BABYVIEW_VISION_TOWER_PATH DINOv2 teacher checkpoint pretrained on BabyView
#   BABYVIEW_DECODER_INIT_PATH Phase-0 GPT-2 dir pretrained on BabyView
#   EXTRACTED_ROOT             Staged MetaCLIP jpegs (COCO-MC; see the EgoVLP README)
#   COCO_MC_MANIFEST, COCO_MC_SPLIT_DIR, COCO_MC_PERMUTATION_DIR, COCO_IMAGE_DIR
#                              COCO-MC data (see apps/baselines/egovlp/README.md)
#   BABYVIEW_ROOT              BabyView manifests + frames (babyview)
#   WANDB_PROJECT              Enable W&B logging to this project (default: off)
#   MAX_STEPS                  Cap optimizer steps (smoke tests)
#   PYTHON                     Python interpreter (default: python)
#   EGOBABYVLM_CKPT_DIR        Output base dir (default: ./egobabyvlm_checkpoints)

set -euo pipefail

CONDITIONS=(p0 p25 p50 p75 p100 babyview)
SEEDS=(42 43 44)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${CONDITION:-}" ]; then
    CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID % ${#CONDITIONS[@]}))]}"
    SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#CONDITIONS[@]}))]}"
fi

CONDITION="${CONDITION:?Set CONDITION (one of: ${CONDITIONS[*]}) or submit as a SLURM array}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"

OVERRIDES=()
case "$CONDITION" in
    p0|p25|p50|p75|p100)
        PERCENT="${CONDITION#p}"
        NAME="flamingo_cocomc_${CONDITION}_seed${SEED}"
        VISION="${COCOMC_VISION_TOWER_PATH:?Set COCOMC_VISION_TOWER_PATH}"
        DECODER="${COCOMC_DECODER_INIT_PATH:?Set COCOMC_DECODER_INIT_PATH}"
        EXTRACTED_ROOT="${EXTRACTED_ROOT:?Set EXTRACTED_ROOT to the staged MetaCLIP jpegs}"
        if [ "$PERCENT" = "0" ]; then
            OVERRIDES+=(data=coco_mc_aligned)
        else
            OVERRIDES+=(data=coco_mc_shuffled "data.dataset.percent=${PERCENT}")
        fi
        OVERRIDES+=("data.dataset.extracted_root=${EXTRACTED_ROOT}")
        ;;
    babyview)
        NAME="flamingo_babyview_seed${SEED}"
        VISION="${BABYVIEW_VISION_TOWER_PATH:?Set BABYVIEW_VISION_TOWER_PATH}"
        DECODER="${BABYVIEW_DECODER_INIT_PATH:?Set BABYVIEW_DECODER_INIT_PATH}"
        : "${BABYVIEW_ROOT:?Set BABYVIEW_ROOT to the BabyView manifest + frames root}"
        OVERRIDES+=(data=babyview)
        ;;
    *)
        echo "Unknown CONDITION=${CONDITION} (expected one of: ${CONDITIONS[*]})" >&2
        exit 1
        ;;
esac

for p in "$VISION" "$DECODER"; do
    if [ ! -e "$p" ]; then
        echo "Encoder path not found: $p" >&2
        exit 1
    fi
done

if [ -n "${WANDB_PROJECT:-}" ]; then
    OVERRIDES+=(wandb.enabled=true "wandb.project=${WANDB_PROJECT}")
fi
if [ -n "${MAX_STEPS:-}" ]; then
    OVERRIDES+=("max_steps=${MAX_STEPS}")
fi

SAVE_DIR="${CKPT_BASE}/flamingo/${NAME}/checkpoints"

echo "=============================================="
echo "Flamingo training"
echo "  Condition:    ${CONDITION}"
echo "  Seed:         ${SEED}"
echo "  Vision tower: ${VISION}"
echo "  Decoder init: ${DECODER}"
echo "  Output:       ${SAVE_DIR}"
echo "=============================================="

TRAIN_CMD=("$PYTHON" -m apps.baselines.flamingo.training.train
    "name=${NAME}"
    "seed=${SEED}"
    "model.vision_tower_path=${VISION}"
    "model.decoder_init_path=${DECODER}"
    "checkpoint.save_dir=${SAVE_DIR}"
    "${OVERRIDES[@]}")

if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
    # One process per GPU; the trainer reads RANK / LOCAL_RANK / WORLD_SIZE.
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
    MASTER_PORT="$((20000 + SLURM_JOB_ID % 40000))"
    export MASTER_ADDR MASTER_PORT
    srun bash -c '
        export RANK="${SLURM_PROCID}" LOCAL_RANK="${SLURM_LOCALID}" WORLD_SIZE="${SLURM_NTASKS}"
        exec "$@"
    ' _ "${TRAIN_CMD[@]}"
else
    "${TRAIN_CMD[@]}"
fi

echo "Flamingo training complete; checkpoints in ${SAVE_DIR} (final epoch: latest.pt)"
