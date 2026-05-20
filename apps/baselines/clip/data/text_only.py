# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only dataset + collate for BERT MLM co-training.

The BERT MLM head co-trains alongside the contrastive loss on a separate
stream of raw text. Captions for MLM can come from a different distribution
than the contrastive pairs (e.g. a larger general-domain corpus), so this
dataset is intentionally decoupled from the image-caption datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from apps.baselines.clip.modeling.mlm_head import MLM_IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MLM_PROBABILITY = 0.15


class TextOnlyDataset(Dataset):
    """One raw text per line; pre-tokenized to fixed length at item time.

    Args:
        text_file: Plain-text file with one example per line. Empty lines
            are filtered out at load time.
        tokenizer: HuggingFace tokenizer (typically the same one held by
            :class:`apps.baselines.clip.modeling.TextEncoder`).
        max_seq_len: Pad/truncate to this length.
    """

    def __init__(
        self,
        text_file: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with Path(text_file).open() as f:
            self.texts = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class MLMCollator:
    """Picklable collator that applies BERT-style MLM masking.

    Implemented as a class (rather than a ``functools.partial`` closure) so
    DataLoader workers using ``spawn`` can pickle it.

    Args:
        tokenizer: HuggingFace tokenizer.
        mlm_probability: Fraction of non-special tokens to mask per sample.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = DEFAULT_MLM_PROBABILITY) -> None:
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        special = (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
        self.special_token_ids = torch.tensor([tid for tid in special if tid is not None])
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        # Maskable positions: non-special tokens within the attention window.
        is_special = torch.isin(input_ids, self.special_token_ids)
        maskable = (~is_special) & (attention_mask == 1)

        # Sample mask positions.
        mask_tokens = (torch.rand(input_ids.shape) < self.mlm_probability) & maskable

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_tokens] = self.mask_token_id

        labels = input_ids.clone()
        labels[~mask_tokens] = MLM_IGNORE_INDEX

        return {
            "input_ids": masked_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def _collated_fields() -> tuple[str, ...]:
    """Public list of field names returned by :class:`MLMCollator`. For tests."""
    return ("input_ids", "labels", "attention_mask")


__all__ = ["MLM_IGNORE_INDEX", "MLMCollator", "TextOnlyDataset"]
