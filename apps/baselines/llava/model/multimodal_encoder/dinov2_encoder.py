# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023 Haotian Liu
# Copyright 2024 Meta Platforms, Inc.
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

"""DINOv2 ViT-B/14 vision encoder for EgoBabyLLaVA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from transformers import CLIPImageProcessor

# Standard DINOv2 ViT-B/14 architecture constants.
_DEFAULT_HIDDEN_SIZE = 768
_DEFAULT_PATCH_SIZE = 14
_DEFAULT_IMAGE_SIZE = 224

_HUB_MODEL_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")
_HUB_EMBED_DIMS: dict[str, int] = {
    "vits": 384,
    "vitb": 768,
    "vitl": 1024,
    "vitg": 1536,
}

# ImageNet normalization statistics used by the DINOv2 image processor.
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class DINOv2ImageProcessor(CLIPImageProcessor):
    """Image processor for DINOv2 with ImageNet normalization."""

    def __init__(self, image_size: int = _DEFAULT_IMAGE_SIZE, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self.size = {"height": image_size, "width": image_size}
        self.image_mean = _IMAGENET_MEAN
        self.image_std = _IMAGENET_STD
        self.do_center_crop = False
        self.do_resize = True
        self.do_normalize = True
        # PIL.Image.Resampling.BILINEAR (Pillow 9.1+); BILINEAR is the legacy alias.
        self.resample = Image.Resampling.BILINEAR


def build_dinov2_from_checkpoint(
    checkpoint_path: str,
    *,
    only_teacher: bool = True,
) -> tuple[nn.Module, int, str]:
    """Build a DINOv2 model from a teacher checkpoint and its sibling ``config.yaml``.

    The checkpoint path is expected to be a teacher checkpoint emitted by the
    DINOv2 trainer; ``config.yaml`` is resolved relative to the run directory
    (``<run>/eval/<step>/teacher_checkpoint.pth`` → ``<run>/config.yaml``).

    Returns the loaded model, embedding dimension, and architecture string.
    """
    from dinov2.configs import dinov2_default_config
    from dinov2.models import build_model_from_cfg
    from dinov2.utils import utils as dinov2_utils
    from omegaconf import OmegaConf

    # The checkpoint typically lives under <run>/eval/<step>/teacher_checkpoint.pth;
    # the config.yaml sits at the run root.
    base_dir = checkpoint_path.split("/eval")[0]
    config_path = Path(base_dir) / "config.yaml"

    if not config_path.exists():
        msg = f"Config file not found at {config_path}. Expected config.yaml next to the checkpoint."
        raise FileNotFoundError(msg)

    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(str(config_path))
    cfg = OmegaConf.merge(default_cfg, cfg)

    model, embed_dim = build_model_from_cfg(cfg, only_teacher=only_teacher)
    dinov2_utils.load_pretrained_weights(model, checkpoint_path, "teacher")

    arch = cfg.student.arch + str(cfg.student.patch_size)

    return model, embed_dim, arch


def build_dinov2_from_hub(model_name: str = "dinov2_vitb14") -> tuple[nn.Module, int, str]:
    """Build a DINOv2 model from torch.hub (ImageNet pretrained).

    Coordinates rank-0 download under ``torch.distributed`` to avoid races.
    """
    import torch.distributed as dist

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if is_distributed and rank != 0:
        dist.barrier()

    model = torch.hub.load("facebookresearch/dinov2", model_name)

    if is_distributed and rank == 0:
        dist.barrier()

    embed_dim: int = _DEFAULT_HIDDEN_SIZE
    for prefix, dim in _HUB_EMBED_DIMS.items():
        if prefix in model_name:
            embed_dim = dim
            break
    else:
        embed_dim = int(model.embed_dim)

    return model, embed_dim, model_name


class DINOv2ViTB14VisionTower(nn.Module):
    """DINOv2 ViT-B/14 vision tower for LLaVA.

    Loads either a egobabyvlm DINOv2 checkpoint (with sibling ``config.yaml``) or
    an off-the-shelf torch.hub model. Output is ``(B, num_patches + 1, hidden_size)``
    — i.e. ``(B, 257, 768)`` at the default 224x224 resolution.
    """

    def __init__(
        self,
        vision_tower: str,
        args: object,
        *,
        delay_load: bool = False,
        **_kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_path: str | None = getattr(args, "vision_tower_path", None) or getattr(
            args,
            "mm_vision_tower_path",
            None,
        )
        self.freeze: bool = getattr(args, "freeze_vision_tower", True)

        self._hidden_size = _DEFAULT_HIDDEN_SIZE
        self._patch_size = _DEFAULT_PATCH_SIZE
        self._image_size = _DEFAULT_IMAGE_SIZE
        self._num_patches = (self._image_size // self._patch_size) ** 2

        if not delay_load:
            self.load_model()

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the vision encoder (768 for ViT-B)."""
        return self._hidden_size

    @property
    def num_patches(self) -> int:
        """Number of image patches (256 for 224x224 with patch_size=14)."""
        return self._num_patches

    @property
    def num_patches_per_side(self) -> int:
        """Number of patches per side (16 for 224x224 with patch_size=14)."""
        return self._image_size // self._patch_size

    def load_model(self, device_map: object = None) -> None:  # noqa: ARG002 -- upstream call-site parity
        """Load the ViT-B/14 model and DINOv2 weights."""
        if self.is_loaded:
            return

        self.image_processor = DINOv2ImageProcessor(image_size=self._image_size)

        if self.vision_tower_path and Path(self.vision_tower_path).exists():
            self.vision_tower, embed_dim, _arch = build_dinov2_from_checkpoint(self.vision_tower_path)
            self._hidden_size = embed_dim

        elif self.vision_tower_name in _HUB_MODEL_NAMES:
            self.vision_tower, embed_dim, _arch = build_dinov2_from_hub(self.vision_tower_name)
            self._hidden_size = embed_dim

        else:
            msg = (
                f"vision_tower_path '{self.vision_tower_path}' does not exist and "
                f"vision_tower_name '{self.vision_tower_name}' is not a valid torch.hub model. "
                f"Valid torch.hub models: {', '.join(_HUB_MODEL_NAMES)}"
            )
            raise ValueError(msg)

        self.vision_tower.requires_grad_(requires_grad=not self.freeze)

        self.is_loaded = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run images through the frozen-or-trainable DINOv2 backbone."""
        images = images.to(device=self.device, dtype=self.dtype)

        if self.freeze:
            with torch.no_grad():
                features = self._forward_impl(images)
        else:
            features = self._forward_impl(images)

        return features

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        """Return all token features (CLS + patches) without pooling.

        DINOv2 ``forward_features`` returns a dict with ``x_norm_clstoken`` and
        ``x_norm_patchtokens``; if either is missing we fall through to whatever
        the backbone returns.
        """
        try:
            outputs = self.vision_tower.forward_features(images)  # type: ignore[operator]  # forward_features is on the DINOv2 backbone
            if isinstance(outputs, dict):
                cls_token = outputs.get("x_norm_clstoken")
                patch_tokens = outputs.get("x_norm_patchtokens")
                if cls_token is not None and patch_tokens is not None:
                    features: torch.Tensor = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
                else:
                    features = outputs  # type: ignore[assignment]
            else:
                features = outputs
        except AttributeError:
            features = self.vision_tower(images)

        return features

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the model parameters."""
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.vision_tower.parameters()).device
