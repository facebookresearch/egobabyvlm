# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flamingo training package."""

from apps.baselines.flamingo.training.checkpoint import load_checkpoint, save_checkpoint
from apps.baselines.flamingo.training.loop import FlamingoTrainer
from apps.baselines.flamingo.training.losses import (
    FlamingoTrainingModule,
    ObjectiveWeights,
    sequence_ll,
)
from apps.baselines.flamingo.training.optim import (
    build_adamw,
    build_cosine_scheduler,
    build_param_groups,
)

__all__ = [
    "FlamingoTrainer",
    "FlamingoTrainingModule",
    "ObjectiveWeights",
    "build_adamw",
    "build_cosine_scheduler",
    "build_param_groups",
    "load_checkpoint",
    "save_checkpoint",
    "sequence_ll",
]
