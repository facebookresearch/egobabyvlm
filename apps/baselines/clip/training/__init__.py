# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer scaffolding."""

from apps.baselines.clip.training.checkpoint import load_checkpoint, save_checkpoint
from apps.baselines.clip.training.interleave import InterleaveScheduler
from apps.baselines.clip.training.loop import ContrastiveTrainer
from apps.baselines.clip.training.optim import build_adamw, build_cosine_scheduler

__all__ = [
    "ContrastiveTrainer",
    "InterleaveScheduler",
    "build_adamw",
    "build_cosine_scheduler",
    "load_checkpoint",
    "save_checkpoint",
]
