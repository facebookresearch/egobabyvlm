# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pair-up two caption datasets for semantic textual similarity scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.utils.data

if TYPE_CHECKING:
    from .base import CaptionsPathDataset


class TextPairDataset(torch.utils.data.Dataset):
    """Yield ``(text_a, text_b, media_id)`` triples joined on ``media_id``."""

    def __init__(self, dataset_a: CaptionsPathDataset, dataset_b: CaptionsPathDataset) -> None:
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        text_a_map = {sample.media_id: sample.text for sample in (dataset_a[i] for i in range(len(dataset_a)))}
        self.samples: list[tuple[str, str, str | int]] = [
            (text_a_map[sample.media_id], sample.text, sample.media_id)
            for sample in (dataset_b[i] for i in range(len(dataset_b)))
            if sample.media_id in text_a_map
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, str, str | int]:
        return self.samples[index]
