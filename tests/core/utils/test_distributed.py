# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from core.utils.distributed import (
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)


def test_defaults_when_dist_not_initialized() -> None:
    assert is_dist_avail_and_initialized() is False
    assert get_world_size() == 1
    assert get_rank() == 0
    assert is_main_process() is True
