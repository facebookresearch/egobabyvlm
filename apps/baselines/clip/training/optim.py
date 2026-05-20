# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer + LR scheduler factories for the contrastive trainer.

The trainer instantiates one optimizer per loss head: the contrastive optimizer
covers the multimodal model's parameters; the MLM optimizer (if active)
covers text encoder + MLM head; the DINOv2 optimizer is owned by
:class:`apps.baselines.clip.modeling.DINOv2SSL` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

if TYPE_CHECKING:
    from collections.abc import Iterable


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Plain AdamW factory."""
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LRScheduler:
    """Cosine LR decay over ``total_steps`` with optional non-zero floor.

    Args:
        optimizer: Optimizer whose LR(s) will be annealed.
        total_steps: Number of steps over which to anneal.
        min_lr_ratio: Final LR as a fraction of the initial LR.

    Returns:
        A :class:`torch.optim.lr_scheduler.LRScheduler` whose ``.step()``
        should be called once per training step.
    """
    eta_min = next(g["lr"] for g in optimizer.param_groups) * min_lr_ratio
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
