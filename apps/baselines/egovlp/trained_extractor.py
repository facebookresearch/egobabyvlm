# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DevBench feature extractor for a fine-tuned EgoVLP checkpoint.

Loads an EgoVLP-native ``.pth`` (produced by ``training/train.py``) via
:func:`apps.baselines.egovlp.model.load_egovlp_model` and exposes the standard
:class:`core.protocols.MultiModalFeatureExtractor` API that
``evaluation/multimodal/machine_devbench/base.py`` consumes (cosine-similarity
path, exactly like the CLIP baseline's ``ContrastiveFeatureExtractor``).

Because this extractor and the trainer import the **same** model module, the
embedding space (256-d projection, single-frame video encoding, DistilBERT text
encoding, L2 normalization) is guaranteed to match what was trained.

Registered by ``evaluation/configs/model/egovlp.yaml`` via
``_target_: apps.baselines.egovlp.trained_extractor.EgoVLPFeatureExtractor``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

from apps.baselines.egovlp.data.transforms import (
    EGOVLP_MEAN,
    EGOVLP_STD,
    build_egovlp_transform,
)
from apps.baselines.egovlp.model import load_egovlp_model

logger = logging.getLogger(__name__)


class EgoVLPFeatureExtractor(torch.nn.Module):
    """Image+text feature extractor wrapping a fine-tuned EgoVLP model.

    Args:
        checkpoint_path: EgoVLP-native ``.pth`` from the fine-tune entrypoint.
        device: Where to load the model.
        normalize: L2-normalize returned features (DevBench wants this).
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

        self.model, self.egovlp_config = load_egovlp_model(self.checkpoint_path, device=self._device)
        self._feature_dim = int(self.egovlp_config.embed_dim)
        self._image_size = int(self.egovlp_config.image_size)
        self._image_transform = build_egovlp_transform(image_size=self._image_size, augment=False)

    # ------------------------------------------------------------------ props
    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args: Any, **kwargs: Any) -> EgoVLPFeatureExtractor:  # type: ignore[override]  # noqa: ANN401
        super().to(*args, **kwargs)
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
        return {"mean": list(EGOVLP_MEAN), "std": list(EGOVLP_STD)}

    # ------------------------------------------------------------------ protocol
    def extract_image_features(self, images: torch.Tensor | list) -> torch.Tensor:
        """Encode images as single-frame clips.

        Accepts a list of PIL images (normalized via the EgoVLP transform) or a
        ``(B, C, H, W)`` / ``(B, T, C, H, W)`` tensor. The model turns a
        4D input into a T=1 clip internally.
        """
        if isinstance(images, list):
            images = torch.stack([self._image_transform(img.convert("RGB")) for img in images])
        with torch.no_grad():
            features = self.model.encode_image(images.to(self._device))
        return F.normalize(features, p=2, dim=-1) if self.normalize else features

    def extract_text_features(self, text: list[str]) -> torch.Tensor:
        if not isinstance(text, list):
            msg = f"EgoVLPFeatureExtractor takes raw text strings; got {type(text).__name__}"
            raise TypeError(msg)
        with torch.no_grad():
            features = self.model.encode_text(text)
        return F.normalize(features, p=2, dim=-1) if self.normalize else features

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a ``(B, T, C, H, W)`` clip (native EgoVLP video path)."""
        with torch.no_grad():
            features = self.model.encode_image(video.to(self._device))
        return F.normalize(features, p=2, dim=-1) if self.normalize else features

    def extract_features(
        self,
        inputs: torch.Tensor | list[str] | Mapping[str, object],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
        """Cosine similarity ``a @ b.T``.

        ``normalize`` defaults to the inverse of ``self.normalize`` (skip if the
        features are already normalized). Matches the CLIP baseline semantics.
        """
        do_norm = (not self.normalize) if normalize is None else normalize
        if do_norm:
            a = F.normalize(a, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)
        return a @ b.T
