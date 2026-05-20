# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Round-robin scheduler that decides which mode runs each step.

Each mode (e.g. ``"contrastive"``, ``"mlm"``, ``"dinov2"``) gets a step budget;
once the current mode's budget is exhausted the scheduler advances to the
next mode. ``contrastive``-only training is a degenerate case where the
schedule is ``{"contrastive": N}`` for any positive ``N``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class InterleaveScheduler:
    """Cycles through modes according to per-mode step budgets.

    The order of modes is the insertion order of ``schedule``: each mode runs
    for its budget, then the next mode runs for its budget, etc. After every
    mode has run, the cycle repeats.

    Args:
        schedule: Dict mapping mode name → number of consecutive steps to run
            before advancing. Modes with budget ≤ 0 are dropped.

    Raises:
        ValueError: If no mode has a positive budget.
    """

    def __init__(self, schedule: dict[str, int]) -> None:
        active = OrderedDict((name, int(budget)) for name, budget in schedule.items() if int(budget) > 0)
        if not active:
            raise ValueError(
                f"InterleaveScheduler requires at least one mode with a positive budget; got {schedule!r}",
            )
        self._budgets = active
        self._modes: list[str] = list(active.keys())
        self._mode_index = 0
        self._mode_step = 0

    @property
    def modes(self) -> tuple[str, ...]:
        """All active modes, in cycle order."""
        return tuple(self._modes)

    @property
    def current_mode(self) -> str:
        """Mode that should run on the next call to :meth:`step`."""
        return self._modes[self._mode_index]

    def step(self) -> tuple[str, bool]:
        """Advance one step.

        Returns:
            ``(mode_just_run, advanced)`` where ``advanced`` is ``True`` iff
            this step exhausted the current mode's budget and the next call
            will run a different mode. Useful for triggering mode-exit hooks
            (e.g. cross-tower weight sync after a DINOv2 block).
        """
        mode = self._modes[self._mode_index]
        self._mode_step += 1
        advanced = self._mode_step >= self._budgets[mode]
        if advanced:
            self._mode_index = (self._mode_index + 1) % len(self._modes)
            self._mode_step = 0
        return mode, advanced

    def state_dict(self) -> dict[str, Any]:
        return {
            "budgets": dict(self._budgets),
            "mode_index": self._mode_index,
            "mode_step": self._mode_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Accept either the original mode list (from ``budgets``) or just the
        # cursor positions; the budgets are owned by the constructor.
        self._mode_index = int(state["mode_index"])
        self._mode_step = int(state["mode_step"])
