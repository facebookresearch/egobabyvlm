# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from EgoVLP (https://github.com/showlab/EgoVLP, MIT License), whose
# SpaceTimeTransformer and FrozenInTime are in turn adapted from Frozen-in-Time
# (https://github.com/m-bain/frozen-in-time), distributed under the following
# license:
#
# MIT License
#
# Copyright (c) 2021 Max Bain
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
#
# As in Frozen-in-Time, the SpaceTimeTransformer builds on the Vision Transformer
# in timm (https://github.com/huggingface/pytorch-image-models) and is inspired
# by lucidrains' TimeSformer-pytorch (https://github.com/lucidrains/TimeSformer-pytorch).
# timm is distributed under the following license:
#
# Copyright 2019 Ross Wightman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored, modernized EgoVLP SpaceTimeTransformer + FrozenInTime.

Ported from the upstream ``showlab/EgoVLP`` repo
(``model/video_transformer.py`` = SpaceTimeTransformer, ``model/model.py`` =
FrozenInTime) so the released EgoClip checkpoint (``egovlp.pth``) can be loaded
and run under a **modern** stack (torch 2.8 / timm 1.0 / transformers 4.57)
*without* importing the upstream repo (which is pinned to
python3.8 / timm0.4.12 / torch1.12 and whose ``import timm`` chain breaks under
timm 1.0.23 — ``RotaryEmbedding`` moved).

Only what is needed for ``compute_video`` / ``compute_text`` + the 256-d
projection is ported. Parameter names match the checkpoint EXACTLY (post
``module.`` strip): ``video_model.*`` (SpaceTimeTransformer),
``text_model.*`` (DistilBERT), ``vid_proj.0.*``, ``txt_proj.1.*`` — so
``egovlp.pth`` loads with ``strict=True``.

The SpaceTimeTransformer is built with ``num_frames=16`` (the pretraining
temporal resolution => ``temporal_embed`` is ``(1, 16, 768)``, matching the
checkpoint exactly, no positional-embed inflation required). The forward pass
handles fewer input frames (BabyView/COCO-MC feed single-frame ``T=1`` clips)
by tiling/slicing positional embeds, identical to upstream.

The only timm dependency is ``timm.layers`` (``DropPath``, ``to_2tuple``,
``trunc_normal_``), which is import-stable under timm 1.0 (upstream imported
these from the now-broken ``timm.models.layers``).
"""

from __future__ import annotations

import logging
import sys
import types
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn

logger = logging.getLogger(__name__)

#: EgoVLP pretraining temporal resolution (size of ``video_model.temporal_embed``).
EGOVLP_NUM_FRAMES = 16


def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    sim = einsum("b i d, b j d -> b i j", q, k)
    a = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", a, v)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VideoPatchEmbed(nn.Module):
    """Video to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=8) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = x.shape  # noqa: N806
        assert self.num_frames >= F
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        return x


class VarAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,  # noqa: FBT002 - upstream signature
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        initialize="random",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == "zeros":
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal
            # attention inputs are multiplied by 0*0, hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, einops_from: str, einops_to: str, **einops_dims: int) -> torch.Tensor:
        h = self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        q *= self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = ((t[:, 0:1], t[:, 1:]) for t in (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)
        # rearrange across time or space
        q_, k_, v_ = (rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims) for t in (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = (repeat(t, "b () d -> (b r) () d", r=r) for t in (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class SpaceTimeBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,  # noqa: FBT002 - upstream signature
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        time_init="zeros",
        attention_style="frozen-in-time",
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.timeattn = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            initialize=time_init,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.attention_style = attention_style

    def forward(
        self,
        x,
        einops_from_space,
        einops_to_space,
        einops_from_time,
        einops_to_time,
        time_n,
        space_f,
    ) -> torch.Tensor:
        time_output = self.timeattn(self.norm3(x), einops_from_time, einops_to_time, n=time_n)
        time_residual = x + time_output
        space_output = self.attn(self.norm1(time_residual), einops_from_space, einops_to_space, f=space_f)
        if self.attention_style == "frozen-in-time":
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError
        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))
        return x


class SpaceTimeTransformer(nn.Module):
    """Space-Time Transformer from Frozen-in-Time (Max Bain).

    Ported verbatim from upstream ``model/video_transformer.py`` (only the timm
    import path changed: ``timm.layers`` instead of the broken
    ``timm.models.layers``). Parameter names are unchanged, so the checkpoint's
    ``video_model.*`` keys map 1:1.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,  # noqa: FBT002 - upstream signature
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        num_frames=8,
        time_init="rand",
        attention_style="frozen-in-time",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if hybrid_backbone is not None:
            raise NotImplementedError("hybrid backbone not implemented")
        self.patch_embed = VideoPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1, embed_dim)
        )  # remember to take pos_embed[1:] for tiling over time
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                SpaceTimeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    time_init=time_init,
                    attention_style=attention_style,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.pre_logits: nn.Module
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([("fc", nn.Linear(embed_dim, representation_size)), ("act", nn.Tanh())])
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        if num_frames == 1:
            self.apply(self._init_weights)

        self.einops_from_space = "b (f n) d"
        self.einops_to_space = "(b f) n d"
        self.einops_from_time = "b (f n) d"
        self.einops_to_time = "(b n) f d"

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:  # noqa: ARG002 - timm API
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b, curr_frames, channels, _, _ = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]  # noqa: N806
        cls_tokens = self.cls_token.expand(BF, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # positional embed tiled per frame ([1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed repeated within each frame ([1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames

        for blk in self.blocks:
            x = blk(
                x,
                self.einops_from_space,
                self.einops_to_space,
                self.einops_from_time,
                self.einops_to_time,
                time_n=n,
                space_f=f,
            )

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _load_egovlp_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load the released EgoVLP checkpoint's ``state_dict`` (``module.`` stripped).

    The released ``.pth`` pickles a top-level ``config`` object of type
    ``parse_config.ConfigParser`` (an upstream module absent here), so a throwaway
    stub is registered before unpickling. Only tensor weights are used downstream.
    """
    if "parse_config" not in sys.modules:
        stub = types.ModuleType("parse_config")

        class _ConfigParserStub:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __setstate__(self, state: Any) -> None:  # noqa: ANN401
                if isinstance(state, dict):
                    self.__dict__.update(state)

        stub.ConfigParser = _ConfigParserStub  # type: ignore[attr-defined]
        sys.modules["parse_config"] = stub

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    return {(k.removeprefix("module.")): v for k, v in state_dict.items()}


class FrozenInTime(nn.Module):
    """Vendored EgoVLP dual encoder (SpaceTimeTransformer + DistilBERT + 256-d proj).

    Modern-stack port of upstream ``model/model.py::FrozenInTime`` restricted to
    the inference/fine-tune path we need. Parameter layout matches the released
    checkpoint exactly:

    * ``video_model.*`` — :class:`SpaceTimeTransformer` (head/pre_logits are
      ``Identity`` so no extra params).
    * ``text_model.*`` — HuggingFace ``distilbert-base-uncased``.
    * ``vid_proj.0.*`` — ``Linear(768, projection_dim)``.
    * ``txt_proj.1.*`` — ``ReLU`` + ``Linear(768, projection_dim)``.

    Args:
        projection_dim: Shared embedding dim (EgoVLP uses 256).
        text_model_name: HF id for the DistilBERT text encoder.
        load_checkpoint: Released ``egovlp.pth`` to init from (``strict=True``).
            ``None`` => random ViT + pretrained-HF DistilBERT (weights are
            expected to be overwritten by an external ``state_dict`` load).
        pretrained_text: Load HF pretrained DistilBERT weights (cached locally).
            Ignored effectively when ``load_checkpoint`` overwrites them.
    """

    def __init__(
        self,
        *,
        projection_dim: int = 256,
        text_model_name: str = "distilbert-base-uncased",
        load_checkpoint: str | None = None,
        pretrained_text: bool = True,
        num_frames: int = EGOVLP_NUM_FRAMES,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.text_model_name = text_model_name

        if pretrained_text:
            self.text_model = AutoModel.from_pretrained(text_model_name)
        else:
            self.text_model = AutoModel.from_config(AutoConfig.from_pretrained(text_model_name))
        self.text_model.train()

        model = SpaceTimeTransformer(
            num_frames=num_frames,
            time_init="zeros",
            attention_style="frozen-in-time",
        )
        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        self.video_model = model
        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        txt_dim = self.text_model.config.hidden_size
        self.txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(txt_dim, projection_dim))
        self.vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))

        if load_checkpoint:
            state_dict = _load_egovlp_state_dict(load_checkpoint)
            missing, unexpected = self.load_state_dict(state_dict, strict=True)
            logger.info(
                "Loaded EgoVLP checkpoint %s (missing=%d unexpected=%d)",
                load_checkpoint,
                len(missing),
                len(unexpected),
            )

    def compute_video(self, video_data: torch.Tensor) -> torch.Tensor:
        video_embeddings = self.video_model(video_data)
        return self.vid_proj(video_embeddings)

    def compute_text(self, text_data: dict[str, torch.Tensor]) -> torch.Tensor:
        # DistilBERT: use the first-token ([CLS]-equivalent) representation.
        text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        return self.txt_proj(text_embeddings)

    def forward(
        self, data: dict[str, Any], *, return_embeds: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        text_embeddings = self.compute_text(data["text"])
        video_embeddings = self.compute_video(data["video"])
        if return_embeds:
            return text_embeddings, video_embeddings
        return sim_matrix(text_embeddings, video_embeddings)


def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine-similarity matrix with an eps for numerical stability."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return torch.mm(a_norm, b_norm.transpose(0, 1))
