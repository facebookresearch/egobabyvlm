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

"""Perceiver Resampler for the native Flamingo baseline.

Compresses variable-length visual features (e.g. 257 tokens from DINOv2
ViT-B/14) into a fixed number of latent tokens using cross-attention.

Adapted from flamingo-pytorch's ``PerceiverResampler`` (see header), rewritten
with plain ``torch`` tensor ops (no einops dependency).

Parameter names are chosen so the weight-decay param-group split
(:func:`apps.baselines.flamingo.training.optim.build_param_groups`) can select
the resampler's ``latents`` / ``time_pos_emb`` by name.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    """LayerNorm → Linear → GELU → Linear feed-forward block (no bias)."""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """Cross-attention where latents query concatenated ``(media, latents)`` KV.

    Operates on ``(B, T, N, D)`` tensors (T is the time / media dimension).
    """

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, S, H*Dh) -> (B, T, H, S, Dh)``."""
        b, t, s, _ = x.shape
        x = x.view(b, t, s, self.heads, self.dim_head)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Attend ``latents`` over concatenated ``(x, latents)``.

        Args:
            x: Media features ``(B, T, N, D)``.
            latents: Latent queries ``(B, T, M, D)``.

        Returns:
            Updated latents ``(B, T, M, D)``.
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, t, m, _ = latents.shape

        q = self._split_heads(self.to_q(latents))  # (B, T, H, M, Dh)
        kv_input = torch.cat((x, latents), dim=-2)  # (B, T, N+M, D)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        k = self._split_heads(k)  # (B, T, H, N+M, Dh)
        v = self._split_heads(v)

        q = q * self.scale
        sim = torch.matmul(q, k.transpose(-1, -2))  # (B, T, H, M, N+M)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.matmul(attn, v)  # (B, T, H, M, Dh)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, t, m, self.heads * self.dim_head)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Compress visual features into a fixed set of latent tokens.

    Takes ``(B, N, D)`` or ``(B, T, N, D)`` visual features and produces
    ``(B, T, num_latents, dim)`` output.

    Args:
        dim: Model dimension (must match the decoder hidden dim).
        depth: Number of Perceiver attention + feed-forward layers.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        num_latents: Number of learnable latent tokens.
        num_time_embeds: Number of time positional embeddings.
        ff_mult: Feed-forward expansion factor.
        inp_dim: Input feature dim; if different from ``dim`` a linear projection
            is applied first. ``None`` means the input already has ``dim``.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_time_embeds: int = 4,
        ff_mult: int = 4,
        inp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))
        self.inp_linear = nn.Linear(inp_dim, dim, bias=False) if inp_dim is not None else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        feed_forward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resample visual features.

        Args:
            x: Visual features ``(B, N, D)`` or ``(B, T, N, D)``.

        Returns:
            Latent tokens ``(B, T, num_latents, dim)``.
        """
        if x.ndim == 3:  # noqa: PLR2004 - (B, N, D) without a time axis
            x = x.unsqueeze(1)  # (B, 1, N, D)

        if self.inp_linear is not None:
            x = self.inp_linear(x)

        times = x.shape[1]
        x = x + self.time_pos_emb[:times]

        b, t = x.shape[0], x.shape[1]
        latents = self.latents.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)  # (B, T, M, D)

        for layer in self.layers:
            attn, ff = cast("nn.ModuleList", layer)
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
