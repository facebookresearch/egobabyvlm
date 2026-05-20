#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-gen-gram-imgs
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-gen-gram-imgs.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-gen-gram-imgs.err
# Stage 3: Generate images for grammatical 2-AFC benchmark.
# Usage: bash run_generate_grammatical_images.sh --data-dir data/coco_TIMESTAMP --styles realistic cartoon [--num-gpus 4]
set -euo pipefail

python -m apps.benchmark_creation.pipeline.grammatical.generate_images "$@"
