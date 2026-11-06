# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Collate for the Flamingo trainer.

Stacks ``pixel_values`` and right-pads ``input_ids`` / ``attention_mask`` /
``labels`` to the batch max. Also emits **in-batch hard negatives**
(``neg_*`` = the batch rolled by one) so the optional hard-negative-margin
objective (a) can score the *same image* against a *wrong caption* with no extra
data plumbing. The negatives are cheap to produce and ignored unless the
corresponding loss weight is > 0.
"""

from __future__ import annotations

from typing import Any

import torch

from apps.baselines.flamingo.constants import IGNORE_INDEX


def _pad_stack(seqs: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to the max length and stack."""
    max_len = max(int(s.shape[0]) for s in seqs)
    out = seqs[0].new_full((len(seqs), max_len), pad_value)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def flamingo_collate(batch: list[dict[str, Any]], *, pad_token_id: int = 0) -> dict[str, Any]:
    """Collate tokenized Flamingo samples into a batch dict.

    Args:
        batch: List of samples from :class:`FlamingoCaptionDataset`.
        pad_token_id: Token id used to pad ``input_ids`` (attention_mask=0 there).

    Returns:
        Dict with ``pixel_values`` ``(B,3,H,W)``, ``input_ids`` /
        ``attention_mask`` / ``labels`` ``(B, L)``, ``captions`` / ``ids`` lists,
        plus rolled ``neg_input_ids`` / ``neg_attention_mask`` / ``neg_labels``.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = _pad_stack([b["input_ids"] for b in batch], pad_token_id)
    attention_mask = _pad_stack([b["attention_mask"] for b in batch], 0)
    labels = _pad_stack([b["labels"] for b in batch], IGNORE_INDEX)
    captions = [b["caption"] for b in batch]
    ids = [b["id"] for b in batch]

    # In-batch hard negatives: roll captions by one so image i is paired with
    # caption i-1 (a guaranteed mismatch for the margin objective).
    roll = torch.roll(torch.arange(input_ids.shape[0]), shifts=1, dims=0)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "captions": captions,
        "ids": ids,
        "neg_input_ids": input_ids[roll].clone(),
        "neg_attention_mask": attention_mask[roll].clone(),
        "neg_labels": labels[roll].clone(),
    }
