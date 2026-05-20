# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration dataclasses for the evaluation pipeline."""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class EvalDatasetConfig:
    """Hydra-instantiable evaluation dataset config."""

    _target_: str = MISSING

    #: Human-readable dataset name (used in result paths and logging).
    name: str = MISSING

    #: Keyword arguments forwarded to the dataset class.
    kwargs: dict[str, Any] = MISSING


@dataclass
class EvalModelConfig:
    """Hydra-instantiable evaluation model config."""

    _target_: str = MISSING

    #: Human-readable model name (used in result paths and logging).
    name: str = MISSING

    #: Keyword arguments forwarded to the model class.
    kwargs: dict[str, Any] = MISSING


@dataclass
class VisionBackboneConfig:
    """Vision backbone spec separated from sweep parameters."""

    _target_: str = MISSING

    #: Human-readable backbone name.
    name: str = MISSING

    #: Base kwargs (weights path, config, etc.).
    kwargs: dict[str, Any] = MISSING


@dataclass
class PoolingStrategy:
    """A single pooling configuration for sweeping over feature extraction strategies."""

    #: Pooling type, e.g. ``"cls"``, ``"concat_cls"``, ``"semantic_segmentation"``.
    pooling: str = MISSING

    #: Number of layers to use for feature extraction.
    last_n_layers: int = 1

    #: Optional suffix; auto-generated if empty from pooling and layers.
    name_suffix: str = ""
