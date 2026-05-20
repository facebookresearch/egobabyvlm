"""Stub for upstream `core/probe.py` — `log_stats` is a no-op for inference."""

from __future__ import annotations

import torch


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:  # noqa: ARG001
    """No-op replacement for the upstream tensor-stats logger."""
    return x
