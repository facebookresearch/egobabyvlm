# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal CLIP-style model: image + text encoders + InfoNCE contrastive loss.

The encoders both project to ``embedding_dim`` and the model handles only
the temperature-scaled similarity + symmetric InfoNCE loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from apps.baselines.clip.modeling.text_encoder import TextEncoder


@dataclass(frozen=True)
class ContrastiveOutput:
    """Output of :meth:`MultiModalModel.compute_contrastive_loss`."""

    loss: torch.Tensor
    image_accuracy: torch.Tensor
    text_accuracy: torch.Tensor
    image_entropy: torch.Tensor
    text_entropy: torch.Tensor
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor


def _entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=dim)
    return (F.softmax(log_p, dim=dim) * -log_p).sum(dim=dim)


class MultiModalModel(nn.Module):
    """Image+text encoders with a learnable temperature and InfoNCE loss.

    Args:
        vision_encoder: Module mapping ``(B, C, H, W)`` → ``(B, embedding_dim)``.
        text_encoder: Module mapping ``list[str]`` → ``((B, embedding_dim), (B, L, D))``.
        normalize_features: L2-normalize embeddings before the dot product.
        temperature: Initial softmax temperature (typical: 0.07).
        fix_temperature: If ``True``, temperature is a buffer; otherwise a Parameter.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: TextEncoder,
        *,
        normalize_features: bool = False,
        temperature: float = 0.07,
        fix_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.image_embed = vision_encoder
        self.text_embed = text_encoder
        self.normalize_features = normalize_features

        # Stored as ``-log(t)`` so that ``exp(.)`` recovers the inverse temperature.
        log_inv_temp = torch.tensor(-math.log(temperature))
        if fix_temperature:
            self.register_buffer("logit_neg_log_temperature", log_inv_temp)
        else:
            self.logit_neg_log_temperature = nn.Parameter(log_inv_temp)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_embed(images)
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)
        return features

    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        features, hidden = self.text_embed(texts)
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)
        return features, hidden

    def forward(self, images: torch.Tensor, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled image-text similarity logits.

        Returns:
            ``(logits_per_image, logits_per_text)`` with shape ``(B, B)``.
        """
        image_features = self.encode_image(images)
        text_features, _ = self.encode_text(texts)
        logit_scale = self.logit_neg_log_temperature.exp()
        match = image_features @ text_features.T
        return match * logit_scale, match.T * logit_scale

    def compute_contrastive_loss(self, images: torch.Tensor, texts: list[str]) -> ContrastiveOutput:
        """Symmetric InfoNCE loss + diagnostic accuracy / entropy stats."""
        logits_per_image, logits_per_text = self(images, texts)
        batch_size = logits_per_image.size(0)
        target = torch.arange(batch_size, device=logits_per_image.device)

        loss = (F.cross_entropy(logits_per_image, target) + F.cross_entropy(logits_per_text, target)) / 2

        image_pred = logits_per_image.argmax(dim=-1)
        text_pred = logits_per_text.argmax(dim=-1)
        image_accuracy = (image_pred == target).float().mean()
        text_accuracy = (text_pred == target).float().mean()

        return ContrastiveOutput(
            loss=loss,
            image_accuracy=image_accuracy,
            text_accuracy=text_accuracy,
            image_entropy=_entropy(logits_per_image).mean(),
            text_entropy=_entropy(logits_per_text).mean(),
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
        )
