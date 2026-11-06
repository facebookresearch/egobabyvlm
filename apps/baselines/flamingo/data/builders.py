# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset builders that wrap base ``(image, caption, id)`` datasets for Flamingo.

These are the Hydra ``_target_``s referenced by ``configs/data/*.yaml``. The
trainer's ``train.py`` injects the live ``tokenizer`` + ``transform`` objects as
kwargs to ``hydra.utils.instantiate`` (the same pattern the CLIP trainer uses to
inject ``transform``), and each builder returns a
``(train_ds, val_ds)`` tuple of :class:`FlamingoCaptionDataset`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from apps.baselines.egovlp.data.babyview import (
    BABYVIEW_IMAGE_ROOT,
    BABYVIEW_TRAIN_JSON,
    BABYVIEW_VAL_JSON,
    BabyViewCaptionsDataset,
)
from apps.baselines.egovlp.data.coco_metaclip import build_coco_metaclip_datasets
from apps.baselines.flamingo.constants import DEFAULT_MEDIA_TOKEN, DEFAULT_PROMPT_TEMPLATE
from apps.baselines.flamingo.data.dataset import FlamingoCaptionDataset

if TYPE_CHECKING:
    from torchvision import transforms
    from transformers import PreTrainedTokenizerBase


def build_flamingo_coco_metaclip(
    *,
    tokenizer: PreTrainedTokenizerBase,
    transform: transforms.Compose,
    val_transform: transforms.Compose | None = None,
    percent: int = 0,
    seed: int = 42,
    extracted_root: str | None = None,
    check_train_size: bool = True,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_length: int = 64,
    media_token: str = DEFAULT_MEDIA_TOKEN,
    on_load_error: str = "skip",
    max_load_failures: int = 100,
) -> tuple[FlamingoCaptionDataset, FlamingoCaptionDataset]:
    """Build ``(train, val)`` Flamingo datasets over COCO-MC.

    ``percent`` selects the caption-permutation (alignment) level: ``0`` = aligned
    (true image-caption pairs), ``100`` = fully shuffled (control). ``seed`` is
    the permutation seed (fixed 42 in the standard recipe).
    """
    train_base, val_base = build_coco_metaclip_datasets(
        transform,
        val_transform=val_transform,
        percent=percent,
        seed=seed,
        extracted_root=extracted_root,
        check_train_size=check_train_size,
    )

    def _wrap(base: object) -> FlamingoCaptionDataset:
        return FlamingoCaptionDataset(
            base,  # type: ignore[arg-type]
            tokenizer,
            media_token=media_token,
            prompt_template=prompt_template,
            max_length=max_length,
            on_load_error=on_load_error,
            max_load_failures=max_load_failures,
        )

    return _wrap(train_base), _wrap(val_base)


def _wrap_caption_dataset(
    base: object,
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt_template: str,
    max_length: int,
    media_token: str,
    on_load_error: str,
    max_load_failures: int,
) -> FlamingoCaptionDataset:
    """Wrap an ``(image, caption, id)`` base dataset for the Flamingo trainer."""
    return FlamingoCaptionDataset(
        base,  # type: ignore[arg-type]
        tokenizer,
        media_token=media_token,
        prompt_template=prompt_template,
        max_length=max_length,
        on_load_error=on_load_error,
        max_load_failures=max_load_failures,
    )


def build_flamingo_babyview(
    *,
    tokenizer: PreTrainedTokenizerBase,
    transform: transforms.Compose,
    val_transform: transforms.Compose | None = None,
    manifest_train: str | None = BABYVIEW_TRAIN_JSON,
    manifest_val: str | None = BABYVIEW_VAL_JSON,
    image_root: str | None = BABYVIEW_IMAGE_ROOT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_length: int = 64,
    media_token: str = DEFAULT_MEDIA_TOKEN,
    on_load_error: str = "skip",
    max_load_failures: int = 100,
) -> tuple[FlamingoCaptionDataset, FlamingoCaptionDataset]:
    """Build ``(train, val)`` Flamingo datasets over BabyView (aligned).

    Uses the EgoVLP baseline's :class:`BabyViewCaptionsDataset` reader (defaults
    from ``$BABYVIEW_ROOT``). Train samples a random frame per utterance
    (``multiple_frames=True``); val is deterministic (first frame). BabyView
    captions are true image-caption pairs (no permutation control).
    """
    if not (manifest_train and manifest_val and image_root):
        msg = "Set $BABYVIEW_ROOT or pass manifest_train / manifest_val / image_root."
        raise ValueError(msg)
    train_base = BabyViewCaptionsDataset(
        manifest_train,
        image_root,
        transform,
        multiple_frames=True,
    )
    val_base = BabyViewCaptionsDataset(
        manifest_val,
        image_root,
        val_transform or transform,
        multiple_frames=False,
    )
    _wrap = partial(
        _wrap_caption_dataset,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=max_length,
        media_token=media_token,
        on_load_error=on_load_error,
        max_load_failures=max_load_failures,
    )
    return _wrap(train_base), _wrap(val_base)
