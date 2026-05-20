# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal projector builder for EgoBabyLLaVA."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class IdentityMap(nn.Module):
    """Identity projector that passes features through unchanged."""

    def forward(self, x: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return x

    @property
    def config(self) -> dict[str, str]:
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    """Simple residual block with LayerNorm and MLP."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(
    config: PretrainedConfig,
    *,
    delay_load: bool = False,  # noqa: ARG001 -- kept for upstream LLaVA call-site parity
    **_kwargs: Any,  # noqa: ANN401
) -> nn.Module:
    """Build a vision projector from a config.

    Supported ``mm_projector_type`` values:

    * ``"linear"`` — single linear layer (``mm_hidden_size`` → ``hidden_size``).
    * ``"mlpNx_gelu"`` — N-layer MLP with GELU between layers (e.g. ``mlp2x_gelu``).
    * ``"identity"`` — pass-through, no projection.
    """
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules: list[nn.Module] = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
