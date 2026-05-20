# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collate functions for the contrastive trainer.

The contrastive collate just stacks image tensors and keeps captions as a
plain list of strings (the :class:`TextEncoder.forward` signature). No
pre-tokenization happens here — that's the encoder's responsibility.
"""

from __future__ import annotations

import torch


def contrastive_collate(batch: list[tuple[torch.Tensor, str, str]]) -> dict[str, list[str] | torch.Tensor]:
    """Collate ``(image, caption, sample_id)`` tuples into a batch dict.

    Returns:
        Dict with ``images`` (Tensor of shape ``(B, C, H, W)``), ``captions``
        (list of B strings), and ``ids`` (list of B sample-id strings).
    """
    images, captions, ids = zip(*batch, strict=True)
    return {
        "images": torch.stack(list(images)),
        "captions": list(captions),
        "ids": list(ids),
    }
