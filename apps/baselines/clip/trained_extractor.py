# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Feature extractor for the contrastive trainer checkpoint format.

Loads a self-contained ``.pt`` produced by the contrastive trainer,
instantiates the matching :class:`MultiModalModel` from the embedded
config, and exposes the standard
:class:`core.protocols.MultiModalFeatureExtractor` API used by the eval
pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision import transforms

from apps.baselines.clip.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from apps.baselines.clip.modeling import MultiModalModel

logger = logging.getLogger(__name__)


class ContrastiveFeatureExtractor(torch.nn.Module):
    """Image+text feature extractor wrapping a trained ``MultiModalModel``.

    Inherits from :class:`torch.nn.Module` so callers can use ``.eval()`` /
    ``.to(device)`` (the eval-pipeline conventions) without special-casing.
    The underlying ``MultiModalModel`` is registered as a submodule, so its
    state moves with the wrapper.

    Args:
        checkpoint_path: Path to a ``.pt`` produced by the contrastive trainer.
            Must have ``model_state_dict`` and ``config.model`` keys.
        device: Where to load the model.
        normalize: Whether to L2-normalize returned features (independent of
            the model's own ``normalize_features`` setting; eval pipelines
            typically want this).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.normalize = normalize

        payload: Mapping = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(payload["config"])

        text_encoder = instantiate(cfg.model.text_encoder)
        vision_encoder = instantiate(cfg.model.vision_encoder)
        self.model = MultiModalModel(
            vision_encoder,
            text_encoder,
            normalize_features=cfg.model.normalize_features,
            temperature=cfg.model.temperature,
            fix_temperature=cfg.model.fix_temperature,
        )
        missing, unexpected = self.model.load_state_dict(payload["model_state_dict"], strict=False)
        if missing or unexpected:
            logger.warning(
                "Loaded checkpoint with %d missing and %d unexpected keys; first missing=%s, first unexpected=%s",
                len(missing),
                len(unexpected),
                missing[:3],
                unexpected[:3],
            )
        self.model.eval()
        self.model = self.model.to(self._device)

        self._feature_dim = int(cfg.model.embedding_dim)
        self._image_size = int(getattr(vision_encoder, "image_size", 224))
        self._image_transform = transforms.Compose(
            [
                transforms.Resize((self._image_size, self._image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ],
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args: Any, **kwargs: Any) -> ContrastiveFeatureExtractor:  # type: ignore[override]  # noqa: ANN401
        """Override to track the device the eval pipeline moves us onto."""
        super().to(*args, **kwargs)
        # Re-read the current device from a parameter so we stay in sync
        # regardless of how to() was called (string, dtype, etc.).
        self._device = next(self.parameters()).device
        return self

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def input_size(self) -> tuple[int, int]:
        return (self._image_size, self._image_size)

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        return {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}

    def extract_image_features(self, images: torch.Tensor | list) -> torch.Tensor:
        """Encode pre-normalized images. Accepts a tensor of shape ``(B, C, H, W)``
        or a list of PIL images (which we normalize via the trainer's transform)."""
        if isinstance(images, list):
            images = torch.stack([self._image_transform(img.convert("RGB")) for img in images])
        with torch.no_grad():
            features = self.model.encode_image(images.to(self._device))
        return F.normalize(features, p=2, dim=-1) if self.normalize else features

    def extract_text_features(self, text: list[str]) -> torch.Tensor:
        """Encode raw caption strings. Pre-tokenized tensor input is not supported
        (the new ``TextEncoder`` is raw-text only)."""
        if not isinstance(text, list):
            raise TypeError(
                f"ContrastiveFeatureExtractor takes raw text strings; got {type(text).__name__}",
            )
        with torch.no_grad():
            features, _ = self.model.encode_text(text)
        return F.normalize(features, p=2, dim=-1) if self.normalize else features

    def extract_features(
        self,
        inputs: torch.Tensor | list[str] | Mapping[str, object],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Dispatch on input shape.

        * Image tensor or PIL list           → ``(B, D)`` image features.
        * Raw text list                      → ``(B, D)`` text features.
        * Dict with ``"image"`` / ``"text"`` → ``{"image_features", "text_features"}``
          (the eval pipeline / DevBench convention).
        """
        if isinstance(inputs, Mapping):
            out: dict[str, torch.Tensor] = {}
            if "image" in inputs:
                out["image_features"] = self.extract_image_features(cast("torch.Tensor | list[Any]", inputs["image"]))
            if "text" in inputs:
                out["text_features"] = self.extract_text_features(cast("list[str]", inputs["text"]))
            return out
        if isinstance(inputs, torch.Tensor):
            return self.extract_image_features(inputs)
        return self.extract_text_features(inputs)

    def compute_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Cosine similarity of ``a`` against ``b``.

        Args:
            a, b: Feature tensors of shape ``(N, D)`` and ``(M, D)``.
            normalize: Override per-call whether to L2-normalize first.
                Defaults to the inverse of :attr:`self.normalize` — if features
                are already normalized, skip; otherwise normalize here.
        """
        do_norm = (not self.normalize) if normalize is None else normalize
        if do_norm:
            a = F.normalize(a, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)
        return a @ b.T

    def tokenize(self, text: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Pass through to the underlying TextEncoder's tokenizer for callers that need it."""
        target = device or self.device
        return self.model.text_embed.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(target)
