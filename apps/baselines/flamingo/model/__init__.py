# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flamingo model loader package."""

from apps.baselines.flamingo.model.builder import load_pretrained_model, save_flamingo_model

__all__ = ["load_pretrained_model", "save_flamingo_model"]
