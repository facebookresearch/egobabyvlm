#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-manifests-grammatical
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-manifests-grammatical.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-manifests-grammatical.err
# Stage 5: Generate per-category JSON manifests for grammatical evaluation.
# Usage: bash run_generate_manifests_grammatical.sh --data-dir data/coco_TIMESTAMP [--styles realistic cartoon]
set -euo pipefail

python -m apps.benchmark_creation.pipeline.manifests.generate_grammatical "$@"
