# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoVLP model package."""

from __future__ import annotations

from apps.baselines.egovlp.model.frozen import (
    DEFAULT_EMBED_DIM,
    DEFAULT_TEXT_MODEL,
    EgoVLPConfig,
    EgoVLPModel,
    build_egovlp_model,
    load_egovlp_model,
    save_egovlp_checkpoint,
)

__all__ = [
    "DEFAULT_EMBED_DIM",
    "DEFAULT_TEXT_MODEL",
    "EgoVLPConfig",
    "EgoVLPModel",
    "build_egovlp_model",
    "load_egovlp_model",
    "save_egovlp_checkpoint",
]
