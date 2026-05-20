#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-filter-lexical
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-filter-lexical.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-filter-lexical.err
# Stage 4: SigLIP2-based post-filtering for lexical images.
# Usage: bash run_post_filter_lexical.sh --data-dir data/coco_TIMESTAMP [--write-filtered]
set -euo pipefail

python -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical "$@"
