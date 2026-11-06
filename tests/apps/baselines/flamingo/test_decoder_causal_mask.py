# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Causality regression test for the native Flamingo GPT-2 decoder.

Runs the decoder the way the *training* path does (``FlamingoTrainingModule``
passes a right-padded batch with a ``(B, S)`` padding ``attention_mask`` plus
image latents / media locations), perturbs the token at position ``t`` and
asserts the logits at positions ``< t`` are unchanged. Before the causal-mask
fix this failed under sdpa: a padding-only mask switched GPT-2 attention to
non-causal and every token could see the future.

The Phase-0 checkpoint test runs only when ``$FLAMINGO_PHASE0_GPT2`` points at a
Phase-0 GPT-2 HF model dir.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2Model

from apps.baselines.flamingo.modeling.decoder import FlamingoGPT2Decoder, build_gpt2_backbone

MEDIA_ID = 5
PHASE0_GPT2 = os.environ.get("FLAMINGO_PHASE0_GPT2", "")


def _assert_causal(decoder: FlamingoGPT2Decoder, vocab_size: int, dim: int) -> None:
    torch.manual_seed(0)
    b, s = 2, 16
    input_ids = torch.randint(MEDIA_ID + 1, vocab_size, (b, s))
    input_ids[:, 0] = MEDIA_ID
    attention_mask = torch.ones(b, s, dtype=torch.long)
    attention_mask[1, 12:] = 0  # right padding, as in training
    media_locations = input_ids == MEDIA_ID
    image_embeds = torch.randn(b, 1, 8, dim)

    def logits(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return decoder(
                ids,
                image_embeds=image_embeds,
                media_locations=media_locations,
                attention_mask=attention_mask,
            ).logits

    base = logits(input_ids)
    for t in (5, 11):
        perturbed = input_ids.clone()
        perturbed[:, t] = (input_ids[:, t] - MEDIA_ID) % (vocab_size - MEDIA_ID - 1) + MEDIA_ID + 1
        out = logits(perturbed)
        torch.testing.assert_close(out[:, :t], base[:, :t], atol=1e-5, rtol=0)
        assert not torch.allclose(out[0, t:], base[0, t:]), "perturbation had no effect (test is vacuous)"


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_training_forward_is_causal_tiny(attn_implementation: str) -> None:
    config = GPT2Config(n_embd=64, n_head=4, n_layer=3, vocab_size=100, attn_implementation=attn_implementation)
    decoder = FlamingoGPT2Decoder(
        GPT2Model(config),
        cross_attn_every=1,
        cross_attn_heads=4,
        cross_attn_dim_head=16,
        gate_mode="frozen_open",
    ).eval()
    _assert_causal(decoder, vocab_size=100, dim=64)


@pytest.mark.integration
@pytest.mark.skipif(not (PHASE0_GPT2 and Path(PHASE0_GPT2).exists()), reason="$FLAMINGO_PHASE0_GPT2 not set")
def test_training_forward_is_causal_phase0_checkpoint() -> None:
    vocab_size = GPT2Config.from_pretrained(PHASE0_GPT2).vocab_size + 1  # + the <image> token
    gpt2 = build_gpt2_backbone(decoder_init="phase0_ckpt", decoder_init_path=PHASE0_GPT2, vocab_size=vocab_size)
    decoder = FlamingoGPT2Decoder(gpt2, cross_attn_every=1, gate_mode="frozen_open").eval()
    _assert_causal(decoder, vocab_size=vocab_size, dim=gpt2.config.n_embd)
