"""Stub for upstream `core/distributed.py` — only `get_is_master` is needed."""

from __future__ import annotations

import torch.distributed as dist


def get_is_master() -> bool:
    """Return True if the current process is rank 0 (or no distributed init yet)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
