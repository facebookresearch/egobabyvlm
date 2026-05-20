#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=devbench-manifests-lexical
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=MachineDevBench_logs/slurm-%j-manifests-lexical.out
#SBATCH --error=MachineDevBench_logs/slurm-%j-manifests-lexical.err
# Stage 5: Generate per-task JSON manifests for lexical evaluation.
# Usage: bash run_generate_manifests_lexical.sh --data-dir data/coco_TIMESTAMP [--tasks nouns adjectives --styles realistic cartoon]
set -euo pipefail

python -m apps.benchmark_creation.pipeline.manifests.generate_lexical "$@"
