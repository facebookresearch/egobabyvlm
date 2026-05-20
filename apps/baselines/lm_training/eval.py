# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``lm_eval`` causal-LM wrappers that work with our trained GPT-2 checkpoints.

The upstream ``lm_eval.models.huggingface.AutoCausalLM`` hard-codes
``use_fast=False`` in ``_create_auto_tokenizer``, which forces it to load a
slow ``GPT2Tokenizer`` and therefore needs ``vocab.json`` + ``merges.txt`` on
disk. Our ``apps.baselines.lm_training.train.train_gpt2`` writes only the
fast ``tokenizer.json`` (the standard HF "fast" format), so the upstream
class can't load our checkpoints without converting the tokenizer first.

:class:`FastAutoCausalLM` flips ``use_fast=True`` so the fast
``PreTrainedTokenizerFast`` is constructed directly from ``tokenizer.json``.
Functionally identical to ``AutoCausalLM`` for everything downstream evals
call into (``loglikelihood`` / ``_model_call``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lm_eval.models.huggingface import AutoCausalLM

if TYPE_CHECKING:
    import transformers


class FastAutoCausalLM(AutoCausalLM):
    """``AutoCausalLM`` that loads the fast tokenizer (default in transformers)."""

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: str | None = None,
        trust_remote_code: bool | None = False,
    ) -> transformers.PreTrainedTokenizerBase:
        # Skip ``AutoCausalLM._create_auto_tokenizer`` (which forces use_fast=False)
        # and call the grand-parent directly with use_fast=True.
        tok = super(AutoCausalLM, self)._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        tok.padding_side = "left"
        return tok
