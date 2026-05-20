# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collate functions for caption datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data.dataloader import default_collate

if TYPE_CHECKING:
    import torch


def image_captions_collate_fn(
    batch: list[tuple[Any, str]],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Collate a batch of ``(image, caption)`` pairs into ``(image_tensor, captions)``."""
    transposed = list(zip(*batch, strict=False))
    imgs = default_collate(list(transposed[0]))
    texts = transposed[1]
    return imgs, texts
