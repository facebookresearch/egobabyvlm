# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Submitit-based SLURM submission entrypoint for the DINOv2 SSL trainer.

Wraps the upstream ``dinov2.run.train.train.main`` so the OSS-facing
module path matches the rest of ``apps/baselines/``::

    python -m apps.baselines.dinov2.training.submit \\
        --config-file <path> \\
        --partition <slurm_partition> \\
        --output-dir <path> \\
        --ngpus <int>
"""

from __future__ import annotations

import sys

from dinov2.run.train.train import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
