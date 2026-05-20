# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text encoder: HF BERT (or compatible) backbone + projection head, raw-text input.

Replaces an older ``TextEncoder`` variant that took vocab-indexed tensors,
mapped them back to strings, and re-tokenized with the BERT tokenizer. This
encoder takes raw caption strings directly and runs them through the HF
tokenizer + BERT backbone + projection head, eliminating the lossy roundtrip
entirely.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

#: Pooling strategies for collapsing the per-token hidden states into a single
#: sentence embedding before the projection head.
Pooling = Literal["cls", "mean"]


class TextEncoder(nn.Module):
    """Raw-text encoder: tokenizer → backbone → pooled embedding → projection.

    Args:
        hf_model_name: HF model name or local directory (e.g. ``bert-base-uncased``).
        embedding_dim: Output dimension after the linear projection head.
        dropout: Dropout applied to the projected embedding and the per-token
            hidden states before they're returned.
        freeze: If ``True``, freeze the backbone parameters (the projection
            stays trainable).
        pooling: ``"cls"`` (default; standard BERT) takes the first token of
            the last hidden state. ``"mean"`` mean-pools all token states using
            the attention mask — better for encoder-only models without a
            dedicated CLS token.
    """

    def __init__(
        self,
        hf_model_name: str,
        *,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        freeze: bool = False,
        pooling: Pooling = "cls",
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")

        self.hf_model_name = hf_model_name
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        self.pooling: Pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(hf_model_name)
        # ``add_pooling_layer=False`` matches the MLM-pretrained BERT checkpoint shape.
        self.model = AutoModel.from_pretrained(hf_model_name, add_pooling_layer=False)

        if "gpt2" in hf_model_name.lower() and hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.config, "hidden_size"):
            self.output_dim = self.config.hidden_size
        elif hasattr(self.config, "n_embd"):
            self.output_dim = self.config.n_embd
        else:
            raise ValueError(f"Could not determine hidden size for {hf_model_name!r}")

        self.projection = nn.Linear(self.output_dim, embedding_dim)
        self.output_dropout = nn.Dropout(dropout)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of raw caption strings.

        Args:
            texts: List of *B* caption strings.

        Returns:
            ``(projected, hidden_states)`` where ``projected`` is
            ``(B, embedding_dim)`` (post-dropout) and ``hidden_states`` is
            ``(B, L, output_dim)`` (post-dropout, useful for an MLM head).
        """
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            embeddings = hidden_states[:, 0]
        else:  # "mean"
            mask = encoded.attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        projected = self.output_dropout(self.projection(embeddings))
        return projected, self.output_dropout(hidden_states)
