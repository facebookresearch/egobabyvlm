# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Freeze helper used by eval tasks that work with frozen backbones."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch


class _HasParameters(Protocol):
    """Minimal surface freeze() needs — anything that exposes ``parameters()``."""

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...  # noqa: FBT001, FBT002


def freeze(model: _HasParameters) -> None:
    """Set ``requires_grad = False`` on every parameter of ``model``.

    Args:
        model: Any module-like with a ``.parameters()`` iterator (covers both
            ``torch.nn.Module`` subclasses and the feature-extractor Protocols
            in ``core.protocols``).
    """
    for param in model.parameters():
        param.requires_grad = False
