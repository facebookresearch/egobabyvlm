# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen DINOv2 vision tower for the native Flamingo baseline.

Reuses egobabyvlm's own DINOv2 ViT-B/14 tower
(:class:`apps.baselines.llava.model.multimodal_encoder.dinov2_encoder.DINOv2ViTB14VisionTower`)
so Flamingo shares the exact same encoder + image processor + custom-checkpoint
loader as the CLIP+/LLaVA baselines.

The tower loads a custom DINOv2 teacher checkpoint (``vision_tower_path``) via
:func:`~apps.baselines.llava.model.multimodal_encoder.dinov2_encoder.build_dinov2_from_checkpoint`
(``OmegaConf.merge(dinov2_default_config, <run>/config.yaml)`` ->
``build_model_from_cfg(only_teacher=True)`` -> ``load_pretrained_weights(...,
'teacher')``). That architecture is NON-STANDARD (``ffn_layer=swiglufused``,
``block_chunks=2``, ``num_register_tokens=0``) so the torch.hub ``dinov2_vitb14``
weights are INCOMPATIBLE and are only a guarded fallback (see the warning in
:meth:`FrozenDINOv2VisionTower.__init__`). ``dinov2`` resolves to the in-tree
fork at ``apps/baselines/dinov2/third_party`` (editable-install source root).

The underlying tower returns ``(B, 257, 768)`` (CLS + 256 patch tokens) at
224x224; Flamingo feeds only the ``x_norm_patchtokens`` patch grid
``(B, 256, 768)`` to the Perceiver Resampler (the CLS token is dropped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from apps.baselines.llava.model.multimodal_encoder.dinov2_encoder import (
    DINOv2ImageProcessor,
    DINOv2ViTB14VisionTower,
)

logger = logging.getLogger(__name__)


@dataclass
class _VisionTowerArgs:
    """Minimal args shim for :class:`DINOv2ViTB14VisionTower`."""

    vision_tower_path: str | None = None
    freeze_vision_tower: bool = True


class FrozenDINOv2VisionTower(nn.Module):
    """Thin frozen wrapper around egobabyvlm's DINOv2 ViT-B/14 tower.

    Args:
        model_name: torch.hub DINOv2 name used ONLY as the guarded fallback
            (default ``"dinov2_vitb14"``).
        vision_tower_path: Custom COCO-MC DINOv2 teacher checkpoint
            (``teacher_checkpoint.pth``); its sibling ``config.yaml`` is
            auto-resolved. A path that does not exist raises. If it is ``None``,
            the tower uses torch.hub ImageNet weights (a different architecture,
            not usable with checkpoints trained on a custom tower).
        freeze: Freeze the backbone (default ``True`` — Flamingo keeps DINOv2
            frozen and trains only the resampler / cross-attn / decoder).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        *,
        vision_tower_path: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if vision_tower_path and not Path(vision_tower_path).exists():
            # A silent torch.hub fallback would build an incompatible architecture
            # (the custom DINOv2 uses swiglufused FFNs, block_chunks=2).
            msg = f"Flamingo DINOv2 vision_tower_path {vision_tower_path!r} does not exist."
            raise FileNotFoundError(msg)
        args = _VisionTowerArgs(vision_tower_path=vision_tower_path, freeze_vision_tower=freeze)
        self.tower = DINOv2ViTB14VisionTower(model_name, args, delay_load=False)
        self.freeze = freeze
        if freeze:
            self.tower.requires_grad_(requires_grad=False)
            self.tower.eval()

    @property
    def hidden_size(self) -> int:
        """Vision feature dim (768 for DINOv2 ViT-B/14)."""
        return int(self.tower.hidden_size)

    @property
    def image_processor(self) -> DINOv2ImageProcessor:
        """The bundled DINOv2 image processor (ImageNet normalization)."""
        return self.tower.image_processor

    def train(self, mode: bool = True) -> FrozenDINOv2VisionTower:  # noqa: FBT001, FBT002
        """Keep the frozen backbone in eval mode regardless of trainer mode."""
        super().train(mode)
        if self.freeze:
            self.tower.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return DINOv2 patch tokens ``(B, 256, 768)`` (``x_norm_patchtokens``).

        The underlying tower returns ``(B, 257, 768)`` = ``[CLS] + 256`` patch
        tokens; Flamingo feeds only the patch grid to the Perceiver Resampler,
        so we drop the leading CLS token.
        """
        tokens = self.tower(pixel_values)
        return tokens[:, 1:, :]


def build_vision_tower(
    model_name: str = "dinov2_vitb14",
    *,
    vision_tower_path: str | None = None,
    freeze: bool = True,
    **_kwargs: Any,  # noqa: ANN401 -- config passthrough tolerance
) -> FrozenDINOv2VisionTower:
    """Factory for :class:`FrozenDINOv2VisionTower` (Hydra-friendly)."""
    return FrozenDINOv2VisionTower(model_name, vision_tower_path=vision_tower_path, freeze=freeze)
