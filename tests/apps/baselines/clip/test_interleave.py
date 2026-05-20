# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the InterleaveScheduler."""

from __future__ import annotations

import pytest

from apps.baselines.clip.training.interleave import InterleaveScheduler


def test_single_mode_cycles_forever() -> None:
    sched = InterleaveScheduler({"contrastive": 1})
    for _ in range(5):
        mode, advanced = sched.step()
        assert mode == "contrastive"
        assert advanced is True


def test_two_mode_round_robin() -> None:
    sched = InterleaveScheduler({"a": 4, "b": 1})
    sequence = [sched.step()[0] for _ in range(10)]
    assert sequence == ["a"] * 4 + ["b"] + ["a"] * 4 + ["b"]


def test_advanced_flag_only_on_last_step_of_block() -> None:
    sched = InterleaveScheduler({"a": 3})
    flags = [sched.step()[1] for _ in range(6)]
    assert flags == [False, False, True, False, False, True]


def test_zero_budget_modes_dropped() -> None:
    sched = InterleaveScheduler({"a": 2, "skip_me": 0, "b": 1})
    assert sched.modes == ("a", "b")


def test_empty_schedule_raises() -> None:
    with pytest.raises(ValueError, match="positive budget"):
        InterleaveScheduler({})

    with pytest.raises(ValueError, match="positive budget"):
        InterleaveScheduler({"a": 0, "b": -1})


def test_state_dict_round_trip() -> None:
    sched = InterleaveScheduler({"a": 3, "b": 2})
    for _ in range(4):
        sched.step()
    state = sched.state_dict()

    restored = InterleaveScheduler({"a": 3, "b": 2})
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.current_mode == sched.current_mode
