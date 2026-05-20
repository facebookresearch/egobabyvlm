# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Eval-pipeline image feature extractor for trained EgoBabyLLaVA checkpoints.

Wraps a `LlavaGPT2ForCausalLM` checkpoint as a
:class:`core.protocols.ImageFeatureExtractor` so the existing
`evaluation/vision/` tasks (KNN, linear, ABX, depth, segmentation) can score
the model's vision tower in isolation.

Only the vision tower is exposed — the projector and language model are loaded
but not invoked, since the standard vision-eval suite operates directly on
backbone features.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class LlavaVisionFeatureExtractor(nn.Module):
    """Image feature extractor backed by the vision tower of a trained LLaVA checkpoint.

    Loads the full LLaVA model via ``apps.baselines.llava.model.builder``, then
    delegates ``extract_features`` to the vision tower's ``forward`` (returning
    CLS-pooled or patch-mean-pooled features depending on ``pooling``).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        vision_tower_path: str | None = None,
        pooling: str = "cls",
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Load a trained LLaVA checkpoint and pin its vision tower for feature extraction.

        Args:
            model_path: Path to a HuggingFace-style checkpoint dir produced by
                ``apps.baselines.llava.train.train``.
            vision_tower_path: Optional override for the DINOv2 vision tower
                checkpoint. Falls back to the path embedded in the LLaVA
                checkpoint's ``mm_vision_tower_path`` config field, then to
                torch.hub if neither is set.
            pooling: ``"cls"`` returns the [CLS] token embedding (B, D);
                ``"mean_patch"`` averages the patch tokens (B, D); ``"all"``
                returns every token (B, N+1, D).
            device: Target device for the model. ``None`` selects CUDA if
                available, else CPU.
            torch_dtype: Vision tower dtype. Defaults to fp16 for inference
                throughput; pass ``torch.float32`` for numerical comparisons.
        """
        super().__init__()
        # Local import: the builder pulls in transformers + the full LLaVA arch.
        from apps.baselines.llava.model.builder import load_pretrained_model

        if pooling not in ("cls", "mean_patch", "all"):
            msg = f"Unknown pooling={pooling!r}; expected 'cls', 'mean_patch', or 'all'."
            raise ValueError(msg)

        self.pooling = pooling
        self._device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        _, model, image_processor, _ = load_pretrained_model(
            model_path=str(model_path),
            device_map={"": str(self._device)},
            device=str(self._device),
            vision_tower_path=vision_tower_path,
            torch_dtype=torch_dtype,
        )
        self._model = model
        self._image_processor = image_processor

        vision_tower = model.get_vision_tower()  # type: ignore[operator]  # get_vision_tower exists on LlavaMetaForCausalLM subclasses
        self._vision_tower = vision_tower
        self._feature_dim = int(vision_tower.hidden_size)
        self._image_size = int(vision_tower._image_size)
        self._normalize_params: dict[str, list[float]] = {
            "mean": list(image_processor.image_mean),
            "std": list(image_processor.image_std),
        }

        logger.info(
            "Loaded LlavaVisionFeatureExtractor: feature_dim=%d, image_size=%d, pooling=%s",
            self._feature_dim,
            self._image_size,
            self.pooling,
        )

    @property
    def feature_dim(self) -> int:
        """Vision tower hidden size (e.g. 768 for DINOv2 ViT-B/14)."""
        return self._feature_dim

    @property
    def input_size(self) -> tuple[int, int]:
        """Expected ``(H, W)`` of input images."""
        return (self._image_size, self._image_size)

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        """ImageNet-style normalization used by the bundled image processor."""
        return self._normalize_params

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision-tower features.

        Args:
            images: ``(B, 3, H, W)`` already normalized to the dtype/device the
                model was loaded on.

        Returns:
            ``(B, D)`` for ``pooling`` in ``{"cls", "mean_patch"}`` or
            ``(B, N+1, D)`` for ``pooling="all"``.
        """
        images = images.to(device=self._device, dtype=self._vision_tower.dtype)
        tokens = self._vision_tower(images)  # (B, N+1, D), CLS at index 0
        if self.pooling == "cls":
            return tokens[:, 0, :]
        if self.pooling == "mean_patch":
            return tokens[:, 1:, :].mean(dim=1)
        return tokens
