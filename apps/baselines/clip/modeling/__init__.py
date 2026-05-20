# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modeling components for the contrastive trainer."""

from apps.baselines.clip.modeling.dinov2_ssl import DINOv2SSL
from apps.baselines.clip.modeling.mlm_head import MLM_IGNORE_INDEX, MLMHead
from apps.baselines.clip.modeling.multimodal_model import ContrastiveOutput, MultiModalModel
from apps.baselines.clip.modeling.text_encoder import TextEncoder
from apps.baselines.clip.modeling.vision_encoder import (
    CustomDINOv2VisionEncoder,
    HubDINOv2VisionEncoder,
    RandomViTVisionEncoder,
)

__all__ = [
    "MLM_IGNORE_INDEX",
    "ContrastiveOutput",
    "CustomDINOv2VisionEncoder",
    "DINOv2SSL",
    "HubDINOv2VisionEncoder",
    "MLMHead",
    "MultiModalModel",
    "RandomViTVisionEncoder",
    "TextEncoder",
]
