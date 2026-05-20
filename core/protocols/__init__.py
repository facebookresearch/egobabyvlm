# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Structural protocols for the evaluation pipeline."""

from core.protocols.feature_extractor import (
    FeatureExtractor,
    ImageFeatureExtractor,
    MultiModalFeatureExtractor,
    TextFeatureExtractor,
    VideoFeatureExtractor,
)

__all__ = [
    "FeatureExtractor",
    "ImageFeatureExtractor",
    "MultiModalFeatureExtractor",
    "TextFeatureExtractor",
    "VideoFeatureExtractor",
]
