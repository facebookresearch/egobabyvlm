# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 backbone wired into LLaVA's multimodal architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
)

from apps.baselines.llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput
    from transformers.modeling_outputs import CausalLMOutputWithPast


class LlavaGPT2Config(GPT2Config):
    """Configuration class for LLaVA-GPT2."""

    model_type = "llava_gpt2"


class LlavaGPT2Model(LlavaMetaModel, GPT2Model):
    """LlavaMetaModel mixed into ``GPT2Model`` (vision tower + projector + GPT-2)."""

    config_class = LlavaGPT2Config  # type: ignore[assignment]

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)

    @property
    def embed_tokens(self) -> nn.Module:
        """Return input embeddings (compatibility with LlavaMetaForCausalLM)."""
        return self.get_input_embeddings()


class LlavaGPT2ForCausalLM(GPT2LMHeadModel, LlavaMetaForCausalLM):  # type: ignore[misc]  # HF base classes have conflicting generation_config types
    """GPT-2 LM head with LLaVA's multimodal forward / generate path."""

    config_class = LlavaGPT2Config  # type: ignore[assignment]

    def __init__(self, config: LlavaGPT2Config) -> None:
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = LlavaGPT2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model_parallel = False

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return input embeddings from the transformer."""
        return self.model.get_input_embeddings()

    @property
    def model(self) -> LlavaGPT2Model:
        """Return the underlying transformer model."""
        return self.transformer

    def get_model(self) -> LlavaGPT2Model:
        """Return the underlying model (for LlavaMetaForCausalLM compatibility)."""
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        return_dict: bool | None = None,
        **_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Any, ...] | CausalLMOutputWithPast:
        """Forward pass with optional image conditioning."""
        if inputs_embeds is None:
            assert input_ids is not None, "Either input_ids or inputs_embeds must be provided"
            (
                input_ids_out,
                position_ids_out,
                attention_mask_out,
                past_key_values_out,
                inputs_embeds_out,
                labels_out,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                # HF carries list[list[int]] here; downstream only uses .shape via getattr-guards.
                cast("torch.Tensor | None", image_sizes),
            )
            input_ids = cast("torch.LongTensor | None", input_ids_out)
            position_ids = cast("torch.LongTensor | None", position_ids_out)
            attention_mask = attention_mask_out
            past_key_values = cast("list[torch.FloatTensor] | None", past_key_values_out)
            inputs_embeds = cast("torch.FloatTensor | None", inputs_embeds_out)
            labels = cast("torch.LongTensor | None", labels_out)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        inputs: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> GenerateOutput | torch.LongTensor:
        """Generate text with optional image conditioning."""
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs_embeds: torch.Tensor | None
        if images is not None:
            assert inputs is not None, "`inputs` is required when `images` is provided"
            if inputs.shape[-1] == 1:
                # The lone <image> token: encode the image directly.
                inputs_embeds = self.encode_images(images)
                if attention_mask is not None:
                    attention_mask = torch.ones(
                        inputs_embeds.shape[0:2],
                        dtype=torch.bool,
                        device=inputs_embeds.device,
                    )
            else:
                (
                    inputs_out,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds_out,
                    _,
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
                inputs = inputs_out  # type: ignore[assignment]  # may be None when image features replace token inputs
                inputs_embeds = inputs_embeds_out
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any = None,  # noqa: ANN401 -- HF cache type varies
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Prepare inputs for generation, preserving image information."""
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_gpt2", LlavaGPT2Config)
AutoModelForCausalLM.register(LlavaGPT2Config, LlavaGPT2ForCausalLM)
