# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RNG seeding."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch.

    Args:
        seed: Seed value applied to all three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
