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

"""LLaVA architecture base classes for multimodal vision+language fusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from apps.baselines.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

if TYPE_CHECKING:
    # At type-check time, treat the mixins as PreTrainedModel subclasses so attrs
    # like ``self.config``, ``self.device``, ``self.resize_token_embeddings()``,
    # ``self.get_input_embeddings()``, and ``self.get_output_embeddings()`` resolve.
    # At runtime, ``LlavaMetaModel`` and ``LlavaMetaForCausalLM`` remain pure
    # mixins that get composed with concrete HF models via MRO.
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    _MetaModelBase = PreTrainedModel
    _MetaForCausalLMBase = PreTrainedModel
else:
    _MetaModelBase = object
    _MetaForCausalLMBase = ABC

# An image tensor with one extra leading dim is treated as a (B, T, C, H, W) batch.
_IMAGE_BATCH_NDIM = 5
# A single 3-D image tensor lacks the batch dim and needs unsqueezing.
_IMAGE_HWC_NDIM = 3
# initialize_vision_tokenizer assumes exactly two new image-marker tokens.
_NUM_IMAGE_MARKER_TOKENS = 2

# Tuple returned by prepare_inputs_labels_for_multimodal: (input_ids, position_ids,
# attention_mask, past_key_values, inputs_embeds, labels).
PreparedMultimodalInputs = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    Any,
    torch.Tensor | None,
    torch.Tensor | None,
]


class LlavaMetaModel(_MetaModelBase):
    """Mixin: vision tower + projector wiring for the underlying transformer."""

    # Declared for mypy: nn.Module's __getattr__ returns Tensor | Module, which
    # confuses precise typing. The runtime values come from __init__/instance assigns.
    vision_tower: nn.Module | list[nn.Module]
    mm_projector: nn.Module

    def __init__(self, config: Any) -> None:  # noqa: ANN401 -- HF config types vary
        super().__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self) -> nn.Module | None:
        """Return the vision tower, unwrapping FSDP's list wrapper if present."""
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(
        self,
        model_args: Any,  # noqa: ANN401 -- arbitrary HF args namespace
        fsdp: list[Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build / freeze the vision tower + projector based on ``model_args``."""
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = getattr(model_args, "mm_patch_merge_type", "flat")

        self.config.mm_vision_tower = vision_tower

        # Persist vision_tower_path so saved config.json records whether the
        # checkpoint used an off-the-shelf or custom DINOv2, and where it lives.
        vision_tower_path = getattr(model_args, "vision_tower_path", None)
        if vision_tower_path is not None:
            self.config.mm_vision_tower_path = vision_tower_path

        if self.get_vision_tower() is None:
            new_vt = build_vision_tower(model_args, **kwargs)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [new_vt]
            else:
                self.vision_tower = new_vt
        else:
            existing_vt = self.vision_tower
            new_vt = existing_vt[0] if isinstance(existing_vt, list) else existing_vt
            new_vt.load_model()  # type: ignore[operator]  # load_model is on DINOv2ViTB14VisionTower

        freeze_vision_tower = getattr(model_args, "freeze_vision_tower", True)
        if freeze_vision_tower:
            current_vt = self.get_vision_tower()
            assert current_vt is not None, "vision_tower must be initialized before freezing"
            current_vt.requires_grad_(requires_grad=False)
            new_vt = current_vt

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = new_vt.hidden_size  # type: ignore[union-attr]  # hidden_size on DINOv2ViTB14VisionTower
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # Re-enable projector grads in case LoRA / earlier code froze them.
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights: dict[str, torch.Tensor], keyword: str) -> dict[str, torch.Tensor]:
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))


class LlavaMetaForCausalLM(_MetaForCausalLMBase):
    """Mixin: image encoding + multimodal input prep for a causal LM head."""

    @abstractmethod
    def get_model(self) -> Any:  # noqa: ANN401 -- backbone type varies per subclass
        """Return the underlying model (e.g., GPT2Model)."""

    def get_vision_tower(self) -> nn.Module | None:
        """Return the vision tower from the underlying model."""
        return self.get_model().get_vision_tower()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Run images through the vision tower and projector → ``(B, num_patches, hidden_size)``."""
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(  # noqa: PLR0915 -- single-pass fusion logic, splitting hurts clarity
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Any,  # noqa: ANN401 -- HF cache type varies
        labels: torch.Tensor | None,
        images: torch.Tensor | list[torch.Tensor] | None,
        image_sizes: torch.Tensor | None = None,  # noqa: ARG002 -- LLaVA call-site parity
    ) -> PreparedMultimodalInputs:
        """Replace ``IMAGE_TOKEN_INDEX`` positions in ``input_ids`` with image-feature embeddings."""
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(images, list) or images.ndim == _IMAGE_BATCH_NDIM:
            if isinstance(images, list):
                images = [x.unsqueeze(0) if x.ndim == _IMAGE_HWC_NDIM else x for x in images]
            concat_images = torch.cat(list(images), dim=0)
            encoded = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            split_features: tuple[torch.Tensor, ...] = torch.split(encoded, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            if mm_patch_merge_type == "flat":
                image_features = torch.cat(split_features, dim=0)
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # Image start/end tokens are not yet wired into pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Add dummy tensors so we don't have to special-case None throughout.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Strip padding via the attention_mask.
        input_ids_list: list[torch.Tensor] = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=False)
        ]
        labels_list: list[torch.Tensor] = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask, strict=False)
        ]

        new_input_embeds: list[torch.Tensor] = []
        new_labels: list[torch.Tensor] = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels_list[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [
                -1,
                *torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist(),
                cur_input_ids.shape[0],
            ]
            cur_input_ids_noim = []
            cur_labels = labels_list[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            chunk_input_embeds: list[torch.Tensor] = []
            chunk_labels: list[torch.Tensor] = []

            for i in range(num_images + 1):
                chunk_input_embeds.append(cur_input_embeds_no_im[i])
                chunk_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    chunk_input_embeds.append(cur_image_features)
                    chunk_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        ),
                    )

            chunk_input_embeds = [x.to(self.device) for x in chunk_input_embeds]

            cur_new_input_embeds_t = torch.cat(chunk_input_embeds)
            cur_new_labels_t = torch.cat(chunk_labels)

            new_input_embeds.append(cur_new_input_embeds_t)
            new_labels.append(cur_new_labels_t)

        # Truncate; image embeddings can extend the sequence past the LM's max length.
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded: list[torch.Tensor] = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels, strict=False)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    ),
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    ),
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds_t = torch.stack(new_input_embeds_padded, dim=0)

        new_labels_out: torch.Tensor | None = None if _labels is None else new_labels_padded

        attention_mask_out: torch.Tensor | None = (
            None if _attention_mask is None else attention_mask.to(dtype=_attention_mask.dtype)
        )

        position_ids_out: torch.Tensor | None = None if _position_ids is None else position_ids

        return None, position_ids_out, attention_mask_out, past_key_values, new_input_embeds_t, new_labels_out

    def initialize_vision_tokenizer(
        self,
        model_args: Any,  # noqa: ANN401 -- arbitrary HF args namespace
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Add image-related special tokens (and re-init their embeddings)."""
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings: torch.Tensor = self.get_input_embeddings().weight.data  # type: ignore[union-attr,assignment]  # nn.Module __getattr__ returns Tensor|Module; .weight is Tensor here
                output_embeddings: torch.Tensor = self.get_output_embeddings().weight.data  # type: ignore[union-attr,assignment]  # same as above

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == _NUM_IMAGE_MARKER_TOKENS
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    msg = (
                        f"Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. "
                        f"Number of new tokens: {num_new_tokens}."
                    )
                    raise ValueError(msg)
        elif model_args.mm_use_im_patch_token and model_args.tune_mm_mlp_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False
