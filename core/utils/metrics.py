# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Metric tracking helpers shared by trainers across the repo.

- :class:`SmoothedValue` — windowed-moving + global running stats over a single
  scalar series. Cheap, stdlib only, supports cross-rank synchronization.
- :class:`MetricLogger` — collects multiple named ``SmoothedValue`` meters,
  prints them periodically with ETA + memory + per-step timing.
"""

from __future__ import annotations

import datetime
import logging
import time
from collections import defaultdict, deque
from collections.abc import Sized
from typing import TYPE_CHECKING

import torch

from core.utils.distributed import is_dist_avail_and_initialized

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class SmoothedValue:
    """Windowed median + running global stats for a single scalar series."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        """Push ``value`` (with multiplicity ``n``) into the rolling window + running totals."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Cross-rank reduce ``count`` and ``total`` (the rolling deque is *not* synced)."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        self.count = int(t.tolist()[0])
        self.total = float(t.tolist()[1])

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
            total=self.total,
        )


class MetricLogger:
    """Collect named :class:`SmoothedValue` meters and print them periodically."""

    def __init__(self, delimiter: str = "\t") -> None:
        self.meters: dict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.header = ""
        self.delimiter = delimiter

    def update(self, **kwargs: float) -> None:
        for name, raw in kwargs.items():
            if raw is None:
                continue
            value = raw.item() if isinstance(raw, torch.Tensor) else raw
            assert isinstance(value, (float, int))
            self.meters[name].update(value)

    def __getattr__(self, attr: str) -> object:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        return self.delimiter.join(
            f"{name}: {meter!s}" for name, meter in self.meters.items() if not name.startswith("_")
        )

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def set_header(self, header: str) -> None:
        self.header = header

    def log_every(
        self,
        iterable: Iterable,
        print_freq: int,
        header: str | None = None,
    ) -> Iterable:
        """Wrap ``iterable`` and log meters + ETA + per-step timing every ``print_freq`` steps."""
        if header is not None:
            self.set_header(header)

        if isinstance(iterable, Sized):
            total: int | None = len(iterable)
        else:
            total = None  # iterator without a known length — skip ETA and total
        cuda_available = torch.cuda.is_available()
        mb = 1024.0 * 1024.0

        total_str = str(total) if total is not None else "?"
        space_fmt = ":" + str(len(total_str)) + "d"
        log_msg_parts = [
            self.header,
            "[{0" + space_fmt + "}/" + total_str + "]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if cuda_available:
            log_msg_parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg_parts)

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        last_i = -1

        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            should_log = (i % print_freq == 0) or (total is not None and i == total - 1)
            if should_log:
                eta_seconds = iter_time.global_avg * (total - i) if total is not None else 0.0
                fmt_kwargs: dict[str, object] = {
                    "eta": str(datetime.timedelta(seconds=int(eta_seconds))) if total is not None else "?",
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time),
                }
                if cuda_available:
                    fmt_kwargs["memory"] = torch.cuda.max_memory_allocated() / mb
                logger.info(log_msg.format(i, **fmt_kwargs))
            end = time.time()
            last_i = i

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        n_done = last_i + 1
        logger.info(
            "%s Total time: %s (%.4f s / it)",
            self.header,
            total_time_str,
            total_time / max(n_done, 1),
        )
