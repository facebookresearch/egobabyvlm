# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from core.utils.yaml import to_yaml


def test_to_yaml_round_trips_mixed_types() -> None:
    data = {
        "path": Path("/tmp/x"),
        "array": np.array([1.0, 2.0]),
        "scalar_f": np.float32(1.5),
        "scalar_i": np.int64(3),
        "scalar_b": np.bool_(True),  # noqa: FBT003
        "tensor": torch.tensor([4.0, 5.0]),
        "nested": {"k": (1, 2)},
    }
    parsed = OmegaConf.create(to_yaml(data))
    assert parsed.path == "/tmp/x"
    assert list(parsed.array) == [1.0, 2.0]
    assert parsed.scalar_f == 1.5
    assert parsed.scalar_i == 3
    assert parsed.scalar_b is True
    assert list(parsed.tensor) == [4.0, 5.0]
    assert list(parsed.nested.k) == [1, 2]


def test_to_yaml_handles_object_with_dict() -> None:
    class Holder:
        def __init__(self) -> None:
            self.public = 1
            self._private = 2

    parsed = OmegaConf.create(to_yaml({"obj": Holder()}))
    assert parsed.obj.public == 1
    assert "_private" not in parsed.obj
