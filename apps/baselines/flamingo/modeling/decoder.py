# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The model structure (frozen vision encoder -> Perceiver Resampler -> language
# model with a gated cross-attention block before every ``cross_attn_every``-th
# layer) follows flamingo-pytorch's ``flamingo_palm.py``
# (https://github.com/lucidrains/flamingo-pytorch, MIT License, Copyright (c)
# 2022 Phil Wang; full notice in ``perceiver_resampler.py``).

"""GPT-2 decoder with interleaved gated cross-attention (native Flamingo).

Wraps a HuggingFace ``GPT2Model``'s components (``wte``/``wpe``/``drop``/``h``/
``ln_f``) and inserts a :class:`GatedCrossAttentionBlock` *before* every
``cross_attn_every``-th transformer block (every block in the paper runs).

The decoder always builds an explicit ``(B, 1, S, S)`` causal + key-padding
mask for the GPT-2 blocks. They are called directly (not through
``GPT2Model.forward``), so a padding-only mask would make attention
non-causal under sdpa; see ``_causal_attention_mask``.

The forward returns per-layer hidden states so the eval adapter can run a
logit-lens: ``hidden_states`` is a tuple of length ``n_layer + 1`` whose last
element is **post-``ln_f``** (so ``lm_head(hidden_states[-1])`` == the returned
``logits``) and whose earlier elements are pre-``ln_f`` — matching the LLaVA
layerwise adapter convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from apps.baselines.flamingo.modeling.gated_cross_attention import GatedCrossAttentionBlock

logger = logging.getLogger(__name__)

#: Allowed vocab-size gap between a Phase-0 checkpoint and the target tokenizer
#: (added special tokens such as ``<image>``).
_VOCAB_MISMATCH_TOLERANCE = 8


def build_gpt2_backbone(
    *,
    decoder_init: str,
    decoder_init_path: str | None,
    vocab_size: int,
    max_position_embeddings: int = 1024,
    hidden_size: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
) -> GPT2Model:
    """Build the GPT-2 backbone according to ``decoder_init``.

    * ``scratch``      — random init from a fresh ``GPT2Config``.
    * ``hf_gpt2``      — pretrained ``gpt2`` from HuggingFace.
    * ``phase0_ckpt``  — egobabyvlm Phase-0 from-scratch GPT-2 (V2-style target-
      corpus init). ``decoder_init_path`` is a standard HF model directory
      (``config.json`` with ``architectures=["GPT2LMHeadModel"]`` +
      ``model.safetensors`` + tokenizer files), loaded via
      ``GPT2Model.from_pretrained(decoder_init_path)`` (the LM head is dropped;
      the decoder ties its own head to ``wte``). The checkpoint was trained with
      its OWN COCO-MC BPE tokenizer (vocab=22358); the caller
      (:func:`build_flamingo_model_and_tokenizer`) loads that matching tokenizer
      from ``decoder_init_path`` so ``vocab_size`` here equals 22358 + the added
      ``<image>`` media token. ``decoder_init_path`` is required (no default).

    The token embeddings are resized to ``vocab_size`` (accounts for the added
    ``<image>`` media token).
    """
    if decoder_init == "hf_gpt2":
        gpt2 = GPT2Model.from_pretrained("gpt2")
    elif decoder_init == "phase0_ckpt":
        if not decoder_init_path:
            msg = "decoder_init='phase0_ckpt' requires model.decoder_init_path to be set (no default)."
            raise ValueError(msg)
        gpt2 = GPT2Model.from_pretrained(decoder_init_path)
    elif decoder_init == "scratch":
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_position_embeddings,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
        )
        gpt2 = GPT2Model(cfg)
    else:
        msg = f"decoder_init must be one of {{scratch, hf_gpt2, phase0_ckpt}}, got {decoder_init!r}"
        raise ValueError(msg)

    if gpt2.get_input_embeddings().num_embeddings != vocab_size:
        loaded_vocab = gpt2.get_input_embeddings().num_embeddings
        # A large gap between the loaded vocab and the target usually means the
        # tokenizer does NOT match the decoder checkpoint (e.g. phase0_ckpt paired
        # with the standard gpt2 tokenizer) — the pretrained embedding rows would
        # then be aligned to the WRONG token ids. build_flamingo_model_and_tokenizer
        # loads the matching tokenizer for phase0_ckpt; warn if that invariant slips.
        if decoder_init == "phase0_ckpt" and abs(loaded_vocab - vocab_size) > _VOCAB_MISMATCH_TOLERANCE:
            logger.warning(
                "phase0_ckpt decoder vocab (%d) differs from target vocab (%d) by more than the "
                "added special tokens; ensure the tokenizer is loaded from decoder_init_path so "
                "embeddings stay aligned.",
                loaded_vocab,
                vocab_size,
            )
        gpt2.resize_token_embeddings(vocab_size)
    return gpt2


@dataclass
class FlamingoDecoderOutput:
    """Container for the decoder forward pass."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    last_hidden_state: torch.Tensor


class FlamingoGPT2Decoder(nn.Module):
    """GPT-2 decoder + interleaved gated cross-attention + tied LM head.

    Args:
        gpt2: A constructed :class:`~transformers.GPT2Model` (embeddings resized
            to the target vocab).
        cross_attn_every: Insert a gated cross-attn block before layer ``i``
            when ``i % cross_attn_every == 0``.
        cross_attn_heads: Heads for the gated cross-attn blocks.
        cross_attn_dim_head: Dim per head for the gated cross-attn blocks.
        gate_init: Initial tanh-gate value (``learnable`` mode).
        gate_mode: ``learnable`` / ``frozen_open`` / ``none``.
        only_attend_immediate_media: Cross-attn masking strategy.
    """

    def __init__(
        self,
        gpt2: GPT2Model,
        *,
        cross_attn_every: int = 1,
        cross_attn_heads: int = 12,
        cross_attn_dim_head: int = 64,
        gate_init: float = 0.0,
        gate_mode: str = "frozen_open",
        only_attend_immediate_media: bool = True,
    ) -> None:
        super().__init__()
        self.wte = gpt2.wte
        self.wpe = gpt2.wpe
        self.drop = gpt2.drop
        self.h = gpt2.h
        self.ln_f = gpt2.ln_f
        self.hidden_size = gpt2.config.n_embd
        self.n_layer = gpt2.config.n_layer
        self.cross_attn_every = cross_attn_every

        self.gated_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=self.hidden_size,
                    dim_head=cross_attn_dim_head,
                    heads=cross_attn_heads,
                    only_attend_immediate_media=only_attend_immediate_media,
                    gate_init=gate_init,
                    gate_mode=gate_mode,
                )
                if (i % cross_attn_every) == 0
                else None
                for i in range(self.n_layer)
            ]
        )

        # Tied LM head (GPT-2 convention).
        self.lm_head = nn.Linear(self.hidden_size, self.wte.num_embeddings, bias=False)
        self.lm_head.weight = self.wte.weight

    def _causal_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive ``(B, 1, S, S)`` causal + padding mask.

        The ``GPT2Block``s are called directly (not through ``GPT2Model.forward``),
        so nothing upstream builds the causal mask. Passing only a ``(B, 1, 1, S)``
        padding mask makes the sdpa attention path run *non-causally* (it sets
        ``is_causal=False`` whenever a mask is given), letting every token see the
        future. The mask is always built — also when ``attention_mask`` is
        ``None`` (single-sequence eval) — so training and eval attend identically.
        """
        allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        allowed = allowed[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
        if attention_mask is not None:
            allowed = allowed & attention_mask.to(dtype=torch.bool, device=device)[:, None, None, :]
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def num_gated_blocks(self) -> int:
        """Number of inserted gated cross-attention blocks."""
        return sum(1 for b in self.gated_attn_layers if b is not None)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        image_embeds: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> FlamingoDecoderOutput:
        """Run the interleaved decoder.

        Args:
            input_ids: ``(B, S)`` token ids.
            image_embeds: Perceiver latents ``(B, T, M, D)`` or ``None`` (text-only).
            media_locations: Boolean ``(B, S)`` of ``<image>`` positions.
            attention_mask: ``(B, S)`` 0/1 padding mask.

        Returns:
            :class:`FlamingoDecoderOutput`.
        """
        b, s = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, s)

        hidden = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        ext_mask = self._causal_attention_mask(attention_mask, b, s, hidden.dtype, device)

        hidden_states: list[torch.Tensor] = [hidden]
        for i, block in enumerate(self.h):
            gated = self.gated_attn_layers[i]
            if gated is not None and image_embeds is not None:
                hidden = gated(hidden, image_embeds, media_locations=media_locations)
            outputs = block(hidden, attention_mask=ext_mask)
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            hidden_states.append(hidden)

        last_hidden_state = self.ln_f(hidden)
        logits = self.lm_head(last_hidden_state)

        # Replace the final (pre-ln_f) hidden state with the post-ln_f one so
        # lm_head(hidden_states[-1]) == logits (logit-lens parity).
        hs_tuple = (*hidden_states[:-1], last_hidden_state)
        return FlamingoDecoderOutput(
            logits=logits,
            hidden_states=hs_tuple,
            last_hidden_state=last_hidden_state,
        )
