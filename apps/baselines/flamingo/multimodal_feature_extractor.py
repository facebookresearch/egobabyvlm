# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Native MachineDevBench adapter for the in-repo Flamingo VLM.

Exposes a :class:`apps.baselines.flamingo.modeling.FlamingoVLM` checkpoint as a
:class:`core.protocols.MultiModalFeatureExtractor`, loaded via
``apps.baselines.flamingo.model.builder.load_pretrained_model``.

Flamingo is generative, so there is no shared embedding space:
``extract_*_features`` stash the raw PIL images / caption strings as handles,
and :meth:`compute_similarity` returns an ``(N_img, N_txt)`` matrix of caption
**mean log-likelihood** (``-CE``), higher = better — the same direction cosine
similarity has, so ``machine_devbench/base.py``'s decision rules work unchanged.

:meth:`compute_similarity_layerwise` runs a **logit-lens**: one
``hidden_states`` forward per pair, decoding every decoder hidden state through
the tied ``ln_f`` + ``lm_head`` and scoring ``-CE`` per layer. ``ln_f`` is
applied to layers ``0..L-2`` but not the last (already post-``ln_f`` in the
model output) so ``[-1]`` equals :meth:`compute_similarity`. This is ready for
the layerwise MDB pipeline when it lands (discovered via ``getattr`` there).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from apps.baselines.flamingo.constants import DEFAULT_PROMPT_TEMPLATE, IGNORE_INDEX

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class FlamingoMultiModalAdapter(nn.Module):
    """MachineDevBench adapter for a native Flamingo checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        vision_tower_path: str | None = None,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = torch.float32,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        """Load a native Flamingo checkpoint for log-likelihood MDB scoring.

        Args:
            model_path: A ``.pt`` file (e.g. ``.../best.pt``) or a dir containing
                one, produced by the native Flamingo trainer / ``save_flamingo_model``.
            vision_tower_path: Optional DINOv2 tower checkpoint override.
            device: ``"cuda"`` / ``"cpu"`` (falls back to CPU if no CUDA).
            torch_dtype: Model dtype (``"float32"``/``"float16"`` or a dtype).
            prompt_template: Prompt with a trailing ``{text}`` placeholder.
        """
        super().__init__()
        # Local import: pulls in the full native Flamingo stack + transformers.
        from apps.baselines.flamingo.model.builder import load_pretrained_model

        if "{text}" not in prompt_template:
            msg = f"prompt_template must contain a trailing '{{text}}', got {prompt_template!r}"
            raise ValueError(msg)
        self.prompt_template = prompt_template

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            device=device,
            vision_tower_path=vision_tower_path,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self._context_len = context_len
        self._device = next(model.parameters()).device
        self._dtype = torch_dtype
        self._feature_dim = int(model.config.hidden_size)

        logger.info(
            "Loaded FlamingoMultiModalAdapter: model_path=%s device=%s prompt=%r feature_dim=%d",
            model_path,
            self._device,
            self.prompt_template,
            self._feature_dim,
        )

    def to(self, *args: Any, **kwargs: Any) -> FlamingoMultiModalAdapter:  # type: ignore[override]  # noqa: ANN401
        """Move the model and keep the cached input device in sync (the eval harness calls ``.to``)."""
        super().to(*args, **kwargs)
        self._device = next(self._model.parameters()).device
        return self

    # ------------------------------------------------------------------ protocol

    @property
    def feature_dim(self) -> int:
        """Decoder hidden size (unused by MDB; kept for the protocol)."""
        return self._feature_dim

    def extract_image_features(self, images: Any) -> Any:  # noqa: ANN401
        """Stash raw images as a handle (no shared embedding space)."""
        return list(images)

    def extract_text_features(self, text: Any) -> Any:  # noqa: ANN401
        """Stash raw captions as a handle."""
        return list(text)

    def extract_video_features(self, video: Any) -> Any:  # noqa: ANN401
        """Flamingo image scoring does not support video."""
        msg = "FlamingoMultiModalAdapter does not support video features."
        raise NotImplementedError(msg)

    def extract_features(
        self,
        inputs: torch.Tensor | list[str] | Mapping[str, Any],
    ) -> Any:  # noqa: ANN401
        """Dispatch on input shape (MDB uses the Mapping branch)."""
        if isinstance(inputs, dict):
            return {
                "image_features": self.extract_image_features(inputs["image"]),
                "text_features": self.extract_text_features(inputs["text"]),
            }
        if isinstance(inputs, (list, tuple)) and (not inputs or isinstance(inputs[0], str)):
            return self.extract_text_features(inputs)
        return self.extract_image_features(inputs)

    def compute_similarity(
        self,
        a: Any,  # noqa: ANN401
        b: Any,  # noqa: ANN401
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Return the ``(N_img, N_txt)`` log-likelihood matrix (``-CE``)."""
        if normalize:
            logger.debug("compute_similarity: normalize=%s ignored for log-likelihood scoring.", normalize)
        images = list(a)
        texts = list(b)
        scores = torch.empty((len(images), len(texts)), dtype=torch.float32)
        with torch.no_grad():
            for i, img in enumerate(images):
                for j, txt in enumerate(texts):
                    scores[i, j] = self._score_pair(img, txt)
        return scores

    def compute_similarity_layerwise(
        self,
        a: Any,  # noqa: ANN401
        b: Any,  # noqa: ANN401
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Return ``(n_layers, N_img, N_txt)`` logit-lens ``-CE`` scores."""
        if normalize:
            logger.debug("compute_similarity_layerwise: normalize=%s ignored.", normalize)
        images = list(a)
        texts = list(b)
        per_pair: list[list[list[float]]] = []
        n_layers: int | None = None
        with torch.no_grad():
            rows: list[list[list[float]]] = []
            for img in images:
                row: list[list[float]] = []
                for txt in texts:
                    layer_scores = self._score_pair_layerwise(img, txt)
                    if n_layers is None:
                        n_layers = len(layer_scores)
                    row.append(layer_scores)
                rows.append(row)
            per_pair = rows
        assert n_layers is not None
        # (N_img, N_txt, L) -> (L, N_img, N_txt)
        tensor = torch.tensor(per_pair, dtype=torch.float32)
        return tensor.permute(2, 0, 1).contiguous()

    # ------------------------------------------------------------------ scoring

    def _build_inputs(self, image: Any, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: ANN401
        """Return ``(pixel_values, input_ids, labels)`` for one (image, caption)."""
        processed = self._image_processor(image)
        pixel_values = processed["pixel_values"][0]
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)
        pixel_values = pixel_values.unsqueeze(0).to(device=self._device, dtype=self._dtype)

        prompt = self.prompt_template.format(text=text)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)

        target_ids = self.tokenizer.encode(text, add_special_tokens=False)
        n_target = min(len(target_ids), input_ids.shape[1])
        labels = input_ids.clone()
        if n_target > 0:
            labels[0, : input_ids.shape[1] - n_target] = IGNORE_INDEX
        else:
            labels[:] = IGNORE_INDEX
        return pixel_values, input_ids, labels

    def _score_pair(self, image: Any, text: str) -> float:  # noqa: ANN401
        """Mean caption log-likelihood (``-CE``) of ``text`` given ``image``."""
        pixel_values, input_ids, labels = self._build_inputs(image, text)
        out = self._model(input_ids=input_ids, labels=labels, pixel_values=pixel_values)
        return float(-out.loss.item())

    def _score_pair_layerwise(self, image: Any, text: str) -> list[float]:  # noqa: ANN401
        """Per-layer logit-lens ``-CE`` scores for one (image, caption) pair."""
        pixel_values, input_ids, labels = self._build_inputs(image, text)
        out = self._model(input_ids=input_ids, pixel_values=pixel_values)
        hidden_states = out.hidden_states
        n = len(hidden_states)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        scores: list[float] = []
        for idx, hs in enumerate(hidden_states):
            # Last element is already post-ln_f in the model output; earlier ones
            # are pre-ln_f, so normalize them before the tied head.
            normed = hs if idx == n - 1 else self._model.ln_f(hs)
            logits = self._model.lm_head(normed)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
            scores.append(float(-ce.item()))
        return scores
