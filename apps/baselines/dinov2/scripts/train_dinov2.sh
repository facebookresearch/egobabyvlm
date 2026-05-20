#!/bin/bash -l

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=dinov2-ssl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=8
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/dinov2_ssl_%j.log
#SBATCH --error=${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}/dinov2_ssl_%j.err

# DINOv2 self-supervised pretraining on a vision corpus (ImageNet, COCO, Ego4D,
# HowTo, BabyView, ...).
#
# Submit with sbatch (the #SBATCH directives are honoured only by sbatch);
# pick QoS / account on the command line:
#   sbatch --qos=<your_qos> --account=<your_account> \
#       apps/baselines/dinov2/scripts/train_dinov2.sh
#
# Required env vars:
#   CONFIG_FILE              Path to the DINOv2 config YAML (e.g. one of the
#                            shipped ones under
#                            apps/baselines/dinov2/third_party/dinov2/configs/train/)
#
# Optional env vars (sensible defaults below):
#   OUTPUT_DIR               Output checkpoint dir
#                            (default: ${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}/dinov2_ssl)
#   EGOBABYVLM_CKPT_DIR      Output base (default: ./egobabyvlm_checkpoints)
#   EGOBABYVLM_LOG_DIR       SLURM %j log dir (default: ./egobabyvlm_logs)
#   OPTS                     Extra Hydra-style ``key=value`` overrides forwarded
#                            to the trainer (e.g. ``train.batch_size_per_gpu=64``).
#                            Quote them as a single string.

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:?Set CONFIG_FILE to a DINOv2 training YAML}"
CKPT_BASE="${EGOBABYVLM_CKPT_DIR:-./egobabyvlm_checkpoints}"
LOG_DIR="${EGOBABYVLM_LOG_DIR:-./egobabyvlm_logs}"
mkdir -p "$LOG_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_BASE}/dinov2_ssl}"
OPTS="${OPTS:-}"

echo "=============================================="
echo "DINOv2 SSL training"
echo "  Config:  ${CONFIG_FILE}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  GPUs:    ${SLURM_GPUS_PER_NODE:-?} per node"
[ -n "$OPTS" ] && echo "  Opts:   ${OPTS}"
echo "=============================================="

# torchrun handles per-rank launch within this SLURM allocation; ntasks-per-node
# is set above so SLURM allocates one process per GPU.
python -m apps.baselines.dinov2.training.train \
    --config-file "$CONFIG_FILE" \
    train.output_dir="$OUTPUT_DIR" \
    $OPTS

echo "DINOv2 SSL training complete; checkpoint saved to ${OUTPUT_DIR}"
