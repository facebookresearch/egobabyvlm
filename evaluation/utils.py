# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation-only utility helpers.

The metric tracking helpers (:class:`MetricLogger`, :class:`SmoothedValue`) and
:func:`unwrap_model` live in :mod:`core.utils`; we re-export them here so legacy
``from evaluation.utils import ...`` callsites keep working.
"""

from typing import TYPE_CHECKING, Any

from core.utils.distributed import unwrap_model
from core.utils.metrics import MetricLogger, SmoothedValue

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MetricLogger", "SmoothedValue", "numpy_to_py_types", "unwrap_model"]


def numpy_to_py_types(obj: object) -> object:
    """Recursively convert NumPy scalars and arrays to plain Python types."""
    import numpy as np

    if isinstance(obj, (np.floating, np.integer, np.ndarray, np.str_)):
        type_map: dict[type, Callable[[Any], Any]] = {
            np.floating: float,
            np.integer: int,
            np.ndarray: lambda x: x.tolist(),
            np.str_: str,
        }
        for numpy_type, converter in type_map.items():
            if isinstance(obj, numpy_type):
                return converter(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_py_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        result = [numpy_to_py_types(item) for item in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    return obj
