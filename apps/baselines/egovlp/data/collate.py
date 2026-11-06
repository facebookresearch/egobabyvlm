# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collate for EgoVLP contrastive fine-tuning.

Datasets emit ``(image_tensor, caption_str, sample_id)``. The collate stacks
images into a **1-frame clip** tensor ``(B, T=1, C, H, W)`` — the single-frame
input EgoVLP's video encoder expects — and keeps captions as a raw ``list[str]``
(the text encoder tokenizes internally).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# Batch dict keys are stable across the training loop.
KEY_CLIP = "clip"
KEY_CAPTIONS = "captions"
KEY_IDS = "ids"


def egovlp_collate(
    batch: Sequence[tuple[torch.Tensor, str, str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate ``(image, caption, id)`` samples into an EgoVLP batch.

    Returns ``{"clip": (B, 1, C, H, W), "captions": list[str], "ids": list[str]}``.
    """
    images, captions, ids = zip(*batch, strict=True)
    clip = torch.stack(list(images), dim=0).unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)
    return {KEY_CLIP: clip, KEY_CAPTIONS: list(captions), KEY_IDS: list(ids)}
