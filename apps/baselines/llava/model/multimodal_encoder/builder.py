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

"""Vision encoder builder for EgoBabyLLaVA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dinov2_encoder import DINOv2ViTB14VisionTower

if TYPE_CHECKING:
    from torch import nn


def build_vision_tower(vision_tower_cfg: object, **kwargs: Any) -> nn.Module:  # noqa: ANN401
    """Build a vision tower from a config or model-args namespace.

    Recognizes ``mm_vision_tower`` or ``vision_tower`` (name or path) and
    routes any DINOv2 / ViT-B/14 identifier to :class:`DINOv2ViTB14VisionTower`.
    """
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))

    if vision_tower is None:
        raise ValueError("No vision tower specified in config")

    if "dinov2" in vision_tower.lower() or "vitb14" in vision_tower.lower():
        return DINOv2ViTB14VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(
        f"Unknown vision tower: {vision_tower}. EgoBabyLLaVA only supports DINOv2 ViT-B/14 ('dinov2_vitb14').",
    )
