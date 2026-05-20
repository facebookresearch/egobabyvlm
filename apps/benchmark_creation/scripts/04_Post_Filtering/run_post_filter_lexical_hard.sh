#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-filter-lexical-hard
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-filter-lexical-hard.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-filter-lexical-hard.err
# Stage 4: Hard post-filter for lexical tasks (stricter SigLIP2 threshold).
# Usage: bash run_post_filter_lexical_hard.sh --data-dir data/coco_TIMESTAMP [--write-filtered]
set -euo pipefail

python -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical_hard "$@"
