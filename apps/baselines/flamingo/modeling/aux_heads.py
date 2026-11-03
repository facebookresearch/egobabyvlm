# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Auxiliary heads for the optional (b) ITM + contrastive objectives.

Config-gated and **default OFF** (loss weights default to 0). These are only
constructed when at least one of the ITM / contrastive weights is > 0, so a
default Flamingo run has no unused parameters (clean DDP).

See :mod:`apps.baselines.flamingo.training.losses` for the loss plumbing and
``FLAMINGO_c_options_and_ab_sketches.md` §Q4(b) for the recipe.
"""

from __future__ import annotations

import torch
from torch import nn


class FlamingoAuxHeads(nn.Module):
    """ITM classifier + image/text projection heads for contrastive alignment.

    Args:
        hidden_size: Decoder hidden dim (image + text summaries live here).
        proj_dim: Contrastive projection dim.
        init_temp: Initial contrastive temperature (stored as ``log_temp``).
    """

    def __init__(self, hidden_size: int, *, proj_dim: int = 256, init_temp: float = 0.07) -> None:
        super().__init__()
        self.itm_head = nn.Linear(hidden_size, 2)
        self.img_proj = nn.Linear(hidden_size, proj_dim)
        self.txt_proj = nn.Linear(hidden_size, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``(B, S, H)`` over valid ``(B, S)`` positions → ``(B, H)``."""
    m = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
