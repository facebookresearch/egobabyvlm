# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoVLP data loaders + transforms."""

from __future__ import annotations

from apps.baselines.egovlp.data.babyview import (
    BABYVIEW_IMAGE_ROOT,
    BABYVIEW_TRAIN_JSON,
    BABYVIEW_VAL_JSON,
    BabyViewCaptionsDataset,
)
from apps.baselines.egovlp.data.coco_metaclip import (
    EXPECTED_TRAIN_PAIRS,
    PERMUTATION_PERCENTS,
    CocoMetaclipCaptionsDataset,
    build_coco_metaclip_datasets,
    permutation_metadata_path,
)
from apps.baselines.egovlp.data.collate import egovlp_collate
from apps.baselines.egovlp.data.transforms import (
    EGOVLP_IMAGE_SIZE,
    EGOVLP_MEAN,
    EGOVLP_STD,
    build_egovlp_transform,
)

__all__ = [
    "BABYVIEW_IMAGE_ROOT",
    "BABYVIEW_TRAIN_JSON",
    "BABYVIEW_VAL_JSON",
    "EGOVLP_IMAGE_SIZE",
    "EGOVLP_MEAN",
    "EGOVLP_STD",
    "EXPECTED_TRAIN_PAIRS",
    "PERMUTATION_PERCENTS",
    "BabyViewCaptionsDataset",
    "CocoMetaclipCaptionsDataset",
    "build_coco_metaclip_datasets",
    "build_egovlp_transform",
    "egovlp_collate",
    "permutation_metadata_path",
]
