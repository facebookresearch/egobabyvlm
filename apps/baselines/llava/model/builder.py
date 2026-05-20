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

"""Loader for trained EgoBabyLLaVA (GPT-2 + DINOv2) checkpoints.

The vision tower is built with ``delay_load=True`` during ``from_pretrained``,
so ``transformers`` reports its weights as "UNEXPECTED" and discards them. We
re-attach the inner DINOv2 ``nn.Module`` here and reload its weights from the
checkpoint so finetuned vision encoders are restored correctly.

Vision-tower path resolution (highest priority first):

1. Explicit ``vision_tower_path`` argument.
2. ``mm_vision_tower_path`` in the checkpoint's ``config.json``.
3. torch.hub default (ImageNet-pretrained DINOv2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Default LM context length when neither max_sequence_length nor n_positions is set.
_DEFAULT_CONTEXT_LEN = 512


def _load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer from ``<model_path>/tokenizer/`` if present, else from ``model_path``."""
    tokenizer_subdir = Path(model_path) / "tokenizer"
    tok_path: str
    if tokenizer_subdir.is_dir() and (tokenizer_subdir / "tokenizer_config.json").exists():
        tok_path = str(tokenizer_subdir)
    else:
        tok_path = model_path
    try:
        return AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    except ValueError as e:
        # Older checkpoints set ``tokenizer_class: TokenizersBackend`` in
        # tokenizer_config.json — that class doesn't exist in modern transformers,
        # so AutoTokenizer rejects the lookup. Fall back to constructing a generic
        # PreTrainedTokenizerFast directly from the bundled tokenizer.json.
        if "TokenizersBackend" not in str(e):
            raise
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(tok_path) / "tokenizer.json"))
        # Replay the standard special-token assignments stored in tokenizer_config.json.
        import json as _json

        with (Path(tok_path) / "tokenizer_config.json").open() as _f:
            _cfg = _json.load(_f)
        for _k in ("bos_token", "eos_token", "pad_token", "unk_token"):
            if _v := _cfg.get(_k):
                setattr(tok, _k, _v)
        tok.model_max_length = int(_cfg.get("model_max_length", 1024))
        return tok


def _reload_vision_tower_weights(model: PreTrainedModel, model_path: str) -> None:
    """Reload vision-tower weights from ``model_path`` into ``model``.

    Reads ``model*.safetensors`` (preferred) or ``pytorch_model*.bin`` from the
    checkpoint dir, filters to vision-tower keys, and assigns them. Shape-mismatched
    parameters (typically ``pos_embed`` after a resolution change) are reassigned
    manually because ``load_state_dict(strict=False)`` still raises on shape mismatch.
    """
    from torch import nn  # local: avoid double-import surface in module header

    safetensor_files = sorted(Path(model_path).glob("model*.safetensors"))
    bin_files = sorted(Path(model_path).glob("pytorch_model*.bin"))

    ckpt_state: dict[str, torch.Tensor] = {}
    if safetensor_files:
        from safetensors.torch import load_file

        for f in safetensor_files:
            ckpt_state.update(load_file(str(f), device="cpu"))
    elif bin_files:
        for f in bin_files:
            ckpt_state.update(torch.load(str(f), map_location="cpu", weights_only=True))
    else:
        return

    vt_prefix = "transformer.vision_tower.vision_tower."
    vt_weights = {k[len(vt_prefix) :]: v for k, v in ckpt_state.items() if k.startswith(vt_prefix)}

    if not vt_weights:
        return

    vision_tower = model.get_model().get_vision_tower()  # type: ignore[operator]  # get_model is on LlavaGPT2ForCausalLM, not base PreTrainedModel
    inner_model: nn.Module = vision_tower.vision_tower

    # load_state_dict(strict=False) still raises RuntimeError on shape mismatches,
    # so split the keys up-front.
    current_state = inner_model.state_dict()
    compatible_weights: dict[str, torch.Tensor] = {}
    mismatched_weights: dict[str, torch.Tensor] = {}

    for key, ckpt_val in vt_weights.items():
        if key not in current_state:
            continue
        if current_state[key].shape != ckpt_val.shape:
            mismatched_weights[key] = ckpt_val
        else:
            compatible_weights[key] = ckpt_val

    if compatible_weights:
        inner_model.load_state_dict(compatible_weights, strict=False)

    # Manually assign shape-mismatched parameters (e.g. pos_embed when the
    # checkpoint was trained at 224x224 but torch.hub loaded 518x518).
    for key, ckpt_val in mismatched_weights.items():
        parts = key.rsplit(".", 1)
        if len(parts) == 2:  # noqa: PLR2004 -- "name has a parent" check, the literal is self-explanatory
            parent_name, param_name = parts
            parent: nn.Module = inner_model
            for attr in parent_name.split("."):
                parent = getattr(parent, attr)
            setattr(parent, param_name, nn.Parameter(ckpt_val))
        else:
            setattr(inner_model, key, nn.Parameter(ckpt_val))

    # When pos_embed shape changed, propagate the new image-resolution metadata.
    if "pos_embed" in mismatched_weights:
        ckpt_n_tokens = mismatched_weights["pos_embed"].shape[1]
        ckpt_n_patches = ckpt_n_tokens - 1
        patch_size = vision_tower._patch_size
        new_image_size = int(ckpt_n_patches**0.5) * patch_size
        vision_tower._num_patches = ckpt_n_patches
        vision_tower._image_size = new_image_size


def load_pretrained_model(
    model_path: str,
    model_name: str | None = None,
    device_map: str | dict[str, str] = "auto",
    device: str = "cuda",
    vision_tower_path: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, Any, int]:
    """Load a pretrained EgoBabyLLaVA model from ``model_path``.

    Returns ``(tokenizer, model, image_processor, context_len)``. The image
    processor is ``None`` when the checkpoint has no vision tower (e.g. Phase 0
    LM-only checkpoints).
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = torch.float16

    if model_name is None:
        model_name = Path(model_path).name.lower()

    # AutoConfig dispatches to LlavaGPT2ForCausalLM when config.json has
    # model_type="llava_gpt2", regardless of the checkpoint dir name.
    tokenizer = _load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor: Any = None

    has_vision_tower = hasattr(model, "get_vision_tower") and model.get_vision_tower() is not None  # type: ignore[operator]  # get_vision_tower exists on LlavaMetaForCausalLM subclasses

    if has_vision_tower:
        vision_tower = model.get_vision_tower()  # type: ignore[operator]  # see above

        resolved_vt_path = vision_tower_path or getattr(model.config, "mm_vision_tower_path", None)

        if resolved_vt_path:
            # Inject so DINOv2ViTB14VisionTower.load_model uses build_dinov2_from_checkpoint
            # instead of torch.hub.
            vision_tower.vision_tower_path = resolved_vt_path

        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)

        # Required for finetuned VT checkpoints (whose weights differ from the
        # base DINOv2 loaded above); harmless no-op for frozen VT checkpoints.
        _reload_vision_tower_weights(model, model_path)

        if device_map != "auto":
            # device_map at this point is ``{"": device}`` (set above when ``device != "cuda"``).
            # ``Module.to`` expects a real torch.device, not the accelerate-style dict.
            vision_tower.to(device=device, dtype=torch.float16)

        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "n_positions"):
        context_len = model.config.n_positions
    else:
        context_len = _DEFAULT_CONTEXT_LEN

    return tokenizer, model, image_processor, context_len
