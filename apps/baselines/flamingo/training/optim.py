# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer factory for the Flamingo trainer.

A single AdamW group with weight decay over *all* trainable params would decay
learnable tanh gate scalars toward 0 (``tanh(0) = 0``), pushing the visual path
closed. :func:`build_param_groups` therefore splits parameters into two groups:

* **decay** (``weight_decay = optim.weight_decay``): ≥2-D weight matrices
  (Linear / attention projections, embeddings).
* **no-decay** (``weight_decay = 0.0``): all gate scalars (``*_gate``), all 1-D
  params (LayerNorm weight/bias, biases), and the perceiver ``latents`` / ``time_pos_emb``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Return True if ``param`` should NOT be weight-decayed."""
    return (
        param.ndim <= 1  # LayerNorm weights/biases, all biases, scalar gates
        or name.endswith(("_gate", "latents", "time_pos_emb"))
    )


def build_param_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    split_weight_decay: bool = True,
) -> list[dict]:
    """Build AdamW param groups with the gate/1-D/latents no-decay split.

    Only parameters with ``requires_grad=True`` are included (the frozen DINOv2
    tower and ``frozen_open`` gates are excluded automatically).
    """
    if not split_weight_decay:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "weight_decay": weight_decay}]

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if _is_no_decay(name, param) else decay).append(param)

    logger.info(
        "AdamW param groups: %d decayed (wd=%s), %d no-decay (wd=0.0)",
        len(decay),
        weight_decay,
        len(no_decay),
    )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_adamw(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    split_weight_decay: bool = True,
) -> torch.optim.AdamW:
    """AdamW over the (optionally split) param groups."""
    groups = build_param_groups(model, weight_decay=weight_decay, split_weight_decay=split_weight_decay)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LRScheduler:
    """Cosine LR decay over ``total_steps`` with an optional non-zero floor."""
    eta_min = next(g["lr"] for g in optimizer.param_groups) * min_lr_ratio
    return CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=eta_min)
