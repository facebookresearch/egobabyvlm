# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flamingo data package.

The underlying ``(image, caption, id)`` readers for COCO-MC and BabyView are
shared with the EgoVLP baseline (``apps.baselines.egovlp.data``); the builders
here wrap them with the Flamingo tokenization / prompt template.
"""

from apps.baselines.flamingo.data.builders import (
    build_flamingo_babyview,
    build_flamingo_coco_metaclip,
)
from apps.baselines.flamingo.data.collate import flamingo_collate
from apps.baselines.flamingo.data.dataset import FlamingoCaptionDataset

__all__ = [
    "FlamingoCaptionDataset",
    "build_flamingo_babyview",
    "build_flamingo_coco_metaclip",
    "flamingo_collate",
]
