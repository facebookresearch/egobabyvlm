# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flamingo caption dataset: prompt + label masking + fail-loud image loading.

:class:`FlamingoCaptionDataset` wraps any base dataset that yields
``(image_tensor, caption, sample_id)`` (e.g. the COCO-MC / BabyView readers in
``apps.baselines.egovlp.data`` or the CLIP caption datasets) and turns each sample into a tokenized
``{pixel_values, input_ids, attention_mask, labels}`` record:

* The prompt is ``prompt_template.format(text=caption)`` (default
  ``"<image> {text}"``), so the ``<image>`` media token sits at the front and
  drives cross-attention.
* Labels mask everything except the caption tokens to ``IGNORE_INDEX`` (the
  "count target tokens from the end" convention, BPE-boundary safe).
* **Fail-loud loading:** on an image-load error we *log the path*,
  increment a failure counter, and either **skip to the next sample**
  (``on_load_error="skip"``) or **raise** (``on_load_error="raise"``). Once the
  counter exceeds ``max_load_failures`` a ``RuntimeError`` is raised so a broken
  manifest fails fast. **A black image is NEVER silently substituted.**
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

from apps.baselines.flamingo.constants import DEFAULT_PROMPT_TEMPLATE, IGNORE_INDEX

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ON_LOAD_ERROR_MODES = ("skip", "raise")


class FlamingoCaptionDataset(Dataset):
    """Tokenizing, fail-loud wrapper around an ``(image, caption, id)`` dataset.

    Args:
        base: Dataset returning ``(image_tensor, caption, sample_id)``.
        tokenizer: The Flamingo tokenizer (must contain the media token).
        media_token: Media placeholder token string (default ``"<image>"``).
        prompt_template: Prompt with a trailing ``{text}`` placeholder.
        max_length: Max tokenized length (truncation).
        on_load_error: ``"skip"`` (log + skip to next) or ``"raise"`` (strict).
        max_load_failures: Fuse; raise once cumulative failures exceed this.
    """

    def __init__(
        self,
        base: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        *,
        media_token: str = "<image>",  # noqa: S107 - not a secret
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_length: int = 64,
        on_load_error: str = "skip",
        max_load_failures: int = 100,
    ) -> None:
        super().__init__()
        if on_load_error not in ON_LOAD_ERROR_MODES:
            msg = f"on_load_error must be one of {ON_LOAD_ERROR_MODES}, got {on_load_error!r}"
            raise ValueError(msg)
        if "{text}" not in prompt_template:
            msg = f"prompt_template must contain a trailing '{{text}}', got {prompt_template!r}"
            raise ValueError(msg)
        self.base = base
        self.tokenizer = tokenizer
        self.media_token = media_token
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.on_load_error = on_load_error
        self.max_load_failures = max_load_failures
        self._load_failures = 0

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def _tokenize(self, image: torch.Tensor, caption: str, sample_id: str) -> dict[str, Any]:
        prompt = self.prompt_template.format(text=caption)
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # Mask everything except the caption tokens (count from the end).
        target_ids = self.tokenizer.encode(caption, add_special_tokens=False)
        n_target = min(len(target_ids), input_ids.shape[0])
        labels = input_ids.clone()
        if n_target > 0:
            labels[: input_ids.shape[0] - n_target] = IGNORE_INDEX
        else:
            labels[:] = IGNORE_INDEX

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,
            "id": sample_id,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        attempt_idx = idx
        for _ in range(n):
            try:
                image, caption, sample_id = self.base[attempt_idx]
            except Exception as exc:
                self._load_failures += 1
                logger.warning(
                    "image load failed at base idx %d (failures=%d): %s",
                    attempt_idx,
                    self._load_failures,
                    exc,
                )
                if self.on_load_error == "raise" or self._load_failures > self.max_load_failures:
                    msg = (
                        f"image load failures ({self._load_failures}) exceeded max_load_failures "
                        f"({self.max_load_failures}) or on_load_error='raise'; failing fast."
                    )
                    raise RuntimeError(msg) from exc
                attempt_idx = (attempt_idx + 1) % n
                continue
            return self._tokenize(image, caption, str(sample_id))

        msg = "all samples failed to load; check the manifest / image roots."
        raise RuntimeError(msg)
