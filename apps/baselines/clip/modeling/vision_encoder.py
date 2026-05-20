# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vision encoders for the contrastive trainer.

Three explicit variants, each Hydra-instantiable via its own ``_target_``:

* :class:`HubDINOv2VisionEncoder` — pretrained DINOv2 from torch.hub (ImageNet22k).
* :class:`CustomDINOv2VisionEncoder` — DINOv2 SSL teacher checkpoint produced
  by the DINOv2 training pipeline at ``apps/baselines/dinov2/``. Both the
  ``.pth`` weights and the ``config.yaml`` describing the architecture
  must be passed explicitly.
* :class:`RandomViTVisionEncoder` — torchvision ``VisionTransformer`` with
  random initialization. Useful when training the vision tower from scratch.

All three expose a uniform API:

* ``__call__(images: Tensor[B, C, H, W]) -> Tensor[B, embedding_dim]``.
* ``output_dim: int`` — backbone hidden size before the projection head.
* ``arch: str`` — the architecture identifier (e.g. ``"vit_base_patch14"``);
  used by the trainer to validate compatibility between the contrastive
  vision encoder and the DINOv2 SSL student when both are active.
* ``image_size: int`` — input resolution the backbone was built for.
"""

from __future__ import annotations

from pathlib import Path

# Pin ``sys.modules["dinov2"]`` to our in-tree fork before any
# ``torch.hub.load("facebookresearch/dinov2", ...)`` call below: torch.hub
# transiently puts its cache dir on ``sys.path`` while running upstream's
# ``hubconf.py`` and leaves the upstream ``dinov2.*`` submodules cached in
# ``sys.modules``, which would shadow our copy on every later import.
import dinov2  # noqa: F401
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from core.modeling.freeze import freeze

# Backbone hidden dim per ViT size. Used by all three encoder variants for
# their projection-head input dim.
_BACKBONE_DIM = {
    "vits": 384,
    "vit_small": 384,
    "vitb": 768,
    "vit_base": 768,
    "vitl": 1024,
    "vit_large": 1024,
    "vitg": 1536,
    "vit_giant": 1536,
}

# Random-init ViT presets: (hidden_dim, num_heads, depth).
_VIT_PRESETS = {
    "vits": (384, 6, 12),
    "vitb": (768, 12, 12),
    "vitl": (1024, 16, 24),
    "vitg": (1536, 24, 40),
}

_HUB_DINOV2_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")


def _backbone_dim(arch: str) -> int:
    for prefix, dim in _BACKBONE_DIM.items():
        if prefix in arch:
            return dim
    raise ValueError(f"Unknown vision arch: {arch!r}")


class _ProjectedBackbone(nn.Module):
    """Common base: holds a backbone + linear projection to ``embedding_dim``.

    Subclasses populate ``self.backbone``, ``self.arch``, ``self.image_size``,
    and ``self.output_dim`` in their ``__init__`` then call ``_build_projection()``.
    The forward pass is uniform: backbone → projection.
    """

    backbone: nn.Module
    arch: str
    image_size: int
    output_dim: int

    def __init__(self, *, embedding_dim: int, freeze: bool = False) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze = freeze

    def _build_projection(self) -> None:
        self.projection = nn.Linear(self.output_dim, self.embedding_dim)
        if self.freeze:
            freeze(self.backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))


class HubDINOv2VisionEncoder(_ProjectedBackbone):
    """DINOv2 backbone pulled from ``facebookresearch/dinov2`` via ``torch.hub``.

    Args:
        model_name: One of ``dinov2_vits14``, ``dinov2_vitb14``, ``dinov2_vitl14``,
            ``dinov2_vitg14``.
        embedding_dim: Output dim after the linear projection head.
        freeze: Freeze the backbone parameters (projection stays trainable).
        image_size: The Hub model is trained at 518x518 but interpolates
            positional embeddings on the fly; we report 224 by default since
            that's what the contrastive trainer typically feeds.
    """

    def __init__(
        self,
        model_name: str,
        *,
        embedding_dim: int = 512,
        freeze: bool = False,
        image_size: int = 224,
    ) -> None:
        if model_name not in _HUB_DINOV2_NAMES:
            raise ValueError(f"Unknown Hub DINOv2 model {model_name!r}; expected one of {_HUB_DINOV2_NAMES}")
        super().__init__(embedding_dim=embedding_dim, freeze=freeze)

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.arch = model_name
        self.image_size = image_size
        self.output_dim = _backbone_dim(model_name)
        self._build_projection()


class CustomDINOv2VisionEncoder(_ProjectedBackbone):
    """DINOv2 backbone built from a custom architecture spec.

    Accepts the SSL training config either as an on-disk ``config.yaml`` path
    (``config_path``) or as an embedded dict (``dinov2_config``); the latter
    is what the contrastive trainer pickles into checkpoints so the resulting
    ``.pt`` is fully self-contained — no SSL workspace needs to be reachable
    at load time.

    The optional ``checkpoint_path`` only matters at the start of contrastive
    training: it seeds the backbone with the SSL teacher weights before
    contrastive fine-tuning takes over. When loading a trained contrastive
    checkpoint we omit it (the embedded ``image_embed.backbone.*`` weights
    overwrite anything we'd seed from disk).

    Args:
        config_path: On-disk DINOv2 ``config.yaml`` path. Mutually exclusive
            with ``dinov2_config``.
        dinov2_config: Embedded SSL config (dict / OmegaConf). Mutually
            exclusive with ``config_path``.
        checkpoint_path: Optional ``teacher_checkpoint.pth`` to seed weights
            from. Skipped when ``None``.
        embedding_dim: Output dim after the linear projection head.
        freeze: Freeze the backbone parameters.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        dinov2_config: dict | DictConfig | None = None,
        checkpoint_path: str | Path | None = None,
        embedding_dim: int = 512,
        freeze: bool = False,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, freeze=freeze)

        from dinov2.models import build_model_from_cfg
        from dinov2.utils import utils as dinov2_utils

        if (config_path is None) == (dinov2_config is None):
            msg = "exactly one of config_path / dinov2_config must be provided"
            raise ValueError(msg)
        if config_path is not None:
            cfg_path = Path(config_path)
            if not cfg_path.is_file():
                raise FileNotFoundError(f"DINOv2 config not found: {cfg_path}")
            config = OmegaConf.load(cfg_path)
        else:
            config = OmegaConf.create(dinov2_config) if not isinstance(dinov2_config, DictConfig) else dinov2_config

        self.backbone, _ = build_model_from_cfg(config, only_teacher=True)

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if not ckpt.is_file():
                raise FileNotFoundError(f"DINOv2 checkpoint not found: {ckpt}")
            dinov2_utils.load_pretrained_weights(self.backbone, str(ckpt), "teacher")

        self.arch = f"{config.student.arch}{config.student.patch_size}"
        self.image_size = int(config.crops.global_crops_size)
        self.output_dim = _backbone_dim(self.arch)
        self._build_projection()


class RandomViTVisionEncoder(_ProjectedBackbone):
    """torchvision ``VisionTransformer`` with random initialization.

    Args:
        arch: One of ``vits14``, ``vitb14``, ``vitl14``, ``vitg14`` (or their
            ``patch_size=16`` variants ``vits16``, ``vitb16``, etc.).
        embedding_dim: Output dim after the linear projection head.
        image_size: Input resolution.
        freeze: Freeze the backbone parameters.
    """

    def __init__(
        self,
        arch: str,
        *,
        embedding_dim: int = 512,
        image_size: int = 224,
        freeze: bool = False,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim, freeze=freeze)

        from torchvision.models.vision_transformer import VisionTransformer

        prefix = next((p for p in _VIT_PRESETS if p in arch), None)
        if prefix is None:
            raise ValueError(f"Unsupported random-init ViT arch: {arch!r}; expected prefix in {tuple(_VIT_PRESETS)}")
        hidden_dim, num_heads, depth = _VIT_PRESETS[prefix]
        patch_size = 14 if "14" in arch else 16

        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=hidden_dim * 4,
        )
        # torchvision's VisionTransformer ends in `model.heads` (a Sequential
        # classifier); strip it so the backbone outputs raw features.
        self.backbone.heads = nn.Identity()
        self.arch = arch
        self.image_size = image_size
        self.output_dim = _backbone_dim(arch)
        self._build_projection()
