# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""YAML serialization for nested non-primitive structures."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def _to_primitive(obj: Any) -> Any:  # noqa: ANN401, C901, PLR0911
    """Recursively reduce ``obj`` to OmegaConf-compatible primitives.

    Args:
        obj: Arbitrary value to reduce.

    Returns:
        A primitive value, list, or dict suitable for OmegaConf.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return _to_primitive(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return _to_primitive(obj.detach().cpu().tolist())
    if isinstance(obj, DictConfig):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_primitive(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def to_yaml(d: dict[str, Any] | DictConfig) -> str:
    """Render a possibly nested mapping as a YAML string.

    Args:
        d: Mapping or OmegaConf DictConfig. Non-primitive leaves (Path, NumPy,
            torch.Tensor, dataclass-like objects) are converted to primitives
            before rendering.

    Returns:
        YAML representation of ``d``.
    """
    return OmegaConf.to_yaml(OmegaConf.create(_to_primitive(d)))
