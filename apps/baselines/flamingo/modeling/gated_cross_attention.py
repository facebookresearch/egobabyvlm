# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from flamingo-pytorch (https://github.com/lucidrains/flamingo-pytorch),
# distributed under the following license:
#
# MIT License
#
# Copyright (c) 2022 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Gated cross-attention for the native Flamingo baseline.

Text tokens attend to the Perceiver's visual latents through a tanh-gated
cross-attention + feed-forward block, inserted every ``cross_attn_every``
decoder layers.

Adapted from flamingo-pytorch's ``MaskedCrossAttention`` /
``GatedCrossAttentionBlock`` (see header), rewritten with plain ``torch`` ops
(no einops). The gate behaviour is a config knob (``gate_mode``); the paper runs
use ``frozen_open``. Gates are also excluded from weight decay by the optimizer
param-group split (:mod:`apps.baselines.flamingo.training.optim`):

* ``learnable``    — the original Flamingo gate: ``x + attn(x) * tanh(gate)``
  with ``gate`` a trainable scalar initialised to ``gate_init``.
* ``frozen_open``  — ``gate`` is set so ``tanh(gate) ≈ 1`` and frozen
  (``requires_grad=False``); the visual path is always (nearly) fully open and
  neither the optimizer nor weight decay can close it.
* ``none``         — the tanh gate is dropped: ``x + attn(x)`` (multiply by
  ``1.0``); no gate parameters are created.

Gate parameters are named ``attn_gate`` / ``ff_gate`` so the WD param-group
split can select them by name.
"""

from __future__ import annotations

import torch
from torch import nn

from apps.baselines.flamingo.modeling.perceiver_resampler import feed_forward

#: Allowed values for the ``gate_mode`` knob.
GATE_MODES = ("learnable", "frozen_open", "none")

#: Pre-tanh value used for ``frozen_open`` (``tanh(3.0) ≈ 0.995 ≈ fully on``).
FROZEN_OPEN_GATE_VALUE = 3.0


class MaskedCrossAttention(nn.Module):
    """Cross-attention where text queries attend to media KV.

    Supports ``media_locations`` masking so each text token attends only to the
    most recent preceding image (``only_attend_immediate_media=True``) or to all
    previous images.

    Args:
        dim: Model dimension.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        only_attend_immediate_media: Attention masking strategy.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.only_attend_immediate_media = only_attend_immediate_media

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, S, H*Dh) -> (B, H, S, Dh)``."""
        b, s, _ = x.shape
        return x.view(b, s, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend text hidden states over media latents.

        Args:
            x: Text hidden states ``(B, S, D)``.
            media: Perceiver output ``(B, T, M, D)``.
            media_locations: Boolean ``(B, S)`` marking ``<image>`` positions.

        Returns:
            Cross-attention output ``(B, S, D)``.
        """
        b, t, m = media.shape[:3]

        x = self.norm(x)
        q = self._split_heads(self.to_q(x))  # (B, H, S, Dh)

        media_flat = media.reshape(b, t * m, media.shape[-1])
        k, v = self.to_kv(media_flat).chunk(2, dim=-1)
        k = self._split_heads(k)  # (B, H, T*M, Dh)
        v = self._split_heads(v)

        q = q * self.scale
        sim = torch.matmul(q, k.transpose(-1, -2))  # (B, H, S, T*M)

        if media_locations is not None:
            text_time = media_locations.cumsum(dim=-1)  # (B, S)
            media_time = torch.arange(t, device=x.device) + 1  # (T,)
            # Each media time index covers M latents.
            media_time_expanded = media_time.repeat_interleave(m)  # (T*M,)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(
                text_time[:, None, :, None],  # (B, 1, S, 1)
                media_time_expanded[None, None, None, :],  # (1, 1, 1, T*M)
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if media_locations is not None and self.only_attend_immediate_media:
            text_without_media_mask = (media_locations.cumsum(dim=-1) == 0)[:, None, :, None]
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = torch.matmul(attn, v)  # (B, H, S, Dh)
        b_, h_, s_, dh_ = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b_, s_, h_ * dh_)
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross-attention + feed-forward block.

    Args:
        dim: Model dimension.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        ff_mult: Feed-forward expansion factor.
        only_attend_immediate_media: Attention masking strategy.
        gate_init: Initial value for the tanh gates (``learnable`` mode).
        gate_mode: One of :data:`GATE_MODES` — ``learnable`` / ``frozen_open`` /
            ``none``.
    """

    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
        gate_init: float = 0.0,
        gate_mode: str = "learnable",
    ) -> None:
        super().__init__()
        if gate_mode not in GATE_MODES:
            msg = f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}"
            raise ValueError(msg)
        self.gate_mode = gate_mode

        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.ff = feed_forward(dim, mult=ff_mult)

        if gate_mode == "none":
            # No gate parameters — the visual path is always fully on (x 1.0).
            self.attn_gate = None
            self.ff_gate = None
        else:
            init = FROZEN_OPEN_GATE_VALUE if gate_mode == "frozen_open" else gate_init
            self.attn_gate = nn.Parameter(torch.tensor([init]))
            self.ff_gate = nn.Parameter(torch.tensor([init]))
            if gate_mode == "frozen_open":
                self.attn_gate.requires_grad_(False)  # noqa: FBT003 - torch stub is positional
                self.ff_gate.requires_grad_(False)  # noqa: FBT003

    def gate_values(self) -> tuple[float, float]:
        """Return the current ``(tanh(attn_gate), tanh(ff_gate))`` for logging."""
        if self.attn_gate is None or self.ff_gate is None:  # gate_mode == "none"
            return (1.0, 1.0)
        return (float(self.attn_gate.tanh().item()), float(self.ff_gate.tanh().item()))

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply gated cross-attention then gated feed-forward (residual)."""
        attn_out = self.attn(x, media, media_locations=media_locations)
        ff_scale: torch.Tensor | float
        attn_scale: torch.Tensor | float
        if self.attn_gate is None or self.ff_gate is None:  # gate_mode == "none"
            attn_scale = 1.0
            ff_scale = 1.0
        else:
            attn_scale = self.attn_gate.tanh()
            ff_scale = self.ff_gate.tanh()
        x = attn_out * attn_scale + x
        x = self.ff(x) * ff_scale + x
        return x
