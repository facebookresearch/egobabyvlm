# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Caption-style datasets for the alignment-scoring pipelines."""

from .base import (
    CaptionsMediaDataset,
    CaptionsMediaSample,
    CaptionsPathDataset,
    CaptionsPathSample,
    MultiCaptionPathSample,
)
from .coco import (
    CocoCaptionsDataset,
    CocoCaptionsPathDataset,
    KarpathyCocoCaptionsDataset,
    KarpathyCocoCaptionsPathDataset,
    load_coco_samples,
    load_preprocessed_karpathy_coco_samples,
)
from .collate import image_captions_collate_fn
from .text_pair import TextPairDataset
from .video import VideoCaptionsDataset, VideoCaptionsPathDataset

__all__ = [
    "CaptionsMediaDataset",
    "CaptionsMediaSample",
    "CaptionsPathDataset",
    "CaptionsPathSample",
    "CocoCaptionsDataset",
    "CocoCaptionsPathDataset",
    "KarpathyCocoCaptionsDataset",
    "KarpathyCocoCaptionsPathDataset",
    "MultiCaptionPathSample",
    "TextPairDataset",
    "VideoCaptionsDataset",
    "VideoCaptionsPathDataset",
    "image_captions_collate_fn",
    "load_coco_samples",
    "load_preprocessed_karpathy_coco_samples",
]
