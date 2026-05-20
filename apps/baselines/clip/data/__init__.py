# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data layer for the contrastive trainer."""

from apps.baselines.clip.data.captions import (
    CaptionsDataset,
    CocoCaptionsDataset,
    Ego4DCaptionsDataset,
    HowToCaptionsDataset,
)
from apps.baselines.clip.data.collate import contrastive_collate
from apps.baselines.clip.data.text_only import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MLM_PROBABILITY,
    MLMCollator,
    TextOnlyDataset,
)
from apps.baselines.clip.data.transforms import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
    denormalize_imagenet,
)

__all__ = [
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_MLM_PROBABILITY",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SIZE",
    "CaptionsDataset",
    "CocoCaptionsDataset",
    "Ego4DCaptionsDataset",
    "HowToCaptionsDataset",
    "MLMCollator",
    "TextOnlyDataset",
    "build_eval_transform",
    "build_train_transform",
    "contrastive_collate",
    "denormalize_imagenet",
]
