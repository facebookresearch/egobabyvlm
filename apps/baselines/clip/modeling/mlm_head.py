# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BERT MLM head for the interleaved-LM and triple training modes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

if TYPE_CHECKING:
    from apps.baselines.clip.modeling.text_encoder import TextEncoder

# HuggingFace's standard ignore_index for masked-LM labels: positions with this
# value are excluded from the loss and accuracy computation.
MLM_IGNORE_INDEX = -100


class MLMHead(nn.Module):
    """BERT MLM prediction head + cross-entropy loss helper.

    If ``text_encoder.hf_model_name`` is a local HuggingFace checkpoint
    directory containing ``cls.*`` weights, those are loaded into the head.
    The prediction decoder is tied to the encoder's word-embedding matrix
    in either case (matching ``BertForMaskedLM``).

    Args:
        text_encoder: The :class:`TextEncoder` whose backbone hidden states
            this head will consume.
    """

    def __init__(self, text_encoder: TextEncoder) -> None:
        super().__init__()
        self.head = BertOnlyMLMHead(text_encoder.config)
        self._maybe_load_pretrained_cls(text_encoder)
        self._tie_decoder_to_embeddings(text_encoder)

    def _maybe_load_pretrained_cls(self, text_encoder: TextEncoder) -> None:
        """Load ``cls.*`` weights from a local HF checkpoint if present."""
        path = Path(text_encoder.hf_model_name)
        if not path.is_dir():
            return
        candidates: list[tuple[Path, str]] = [
            (path / "model.safetensors", "safetensors"),
            (path / "pytorch_model.bin", "torch"),
        ]
        weights_path: Path | None = None
        loader: str | None = None
        for cand, kind in candidates:
            if cand.is_file():
                weights_path, loader = cand, kind
                break
        if weights_path is None:
            return

        if loader == "safetensors":
            from safetensors.torch import load_file as load_safetensors

            full_state = load_safetensors(str(weights_path))
        else:
            full_state = torch.load(str(weights_path), map_location="cpu", weights_only=True)

        cls_state = {k.removeprefix("cls."): v for k, v in full_state.items() if k.startswith("cls.")}
        if not cls_state:
            return
        # ``strict=False``: ``cls.predictions.decoder.weight`` is omitted from
        # disk under the standard tied-embedding layout; we tie it next.
        self.head.load_state_dict(cls_state, strict=False)

    def _tie_decoder_to_embeddings(self, text_encoder: TextEncoder) -> None:
        """Tie ``cls.predictions.decoder.weight`` to the encoder word embeddings."""
        word_embeddings = text_encoder.model.embeddings.word_embeddings
        self.head.predictions.decoder.weight = word_embeddings.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``(B, L, vocab_size)`` prediction logits."""
        return self.head(hidden_states)

    @staticmethod
    def loss(prediction_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Masked cross-entropy loss with ``ignore_index=-100`` (HF convention)."""
        return F.cross_entropy(
            prediction_scores.view(-1, prediction_scores.size(-1)),
            labels.view(-1),
            ignore_index=MLM_IGNORE_INDEX,
        )

    @staticmethod
    def accuracy(prediction_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Token-level accuracy over the masked positions only."""
        predictions = prediction_scores.argmax(dim=-1)
        mask = labels != MLM_IGNORE_INDEX
        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction_scores.device)
        return (predictions == labels).float()[mask].mean()
