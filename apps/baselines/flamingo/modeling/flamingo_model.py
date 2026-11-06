# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The model structure (frozen vision encoder -> Perceiver Resampler -> language
# model with a gated cross-attention block before every ``cross_attn_every``-th
# layer) follows flamingo-pytorch's ``flamingo_palm.py``
# (https://github.com/lucidrains/flamingo-pytorch, MIT License, Copyright (c)
# 2022 Phil Wang; full notice in ``perceiver_resampler.py``).

"""Native Flamingo VLM: frozen DINOv2 → Perceiver → gated cross-attn GPT-2.

``FlamingoVLM.forward(pixel_values, input_ids, labels, ...)`` returns a
:class:`FlamingoOutput` with ``.loss`` (teacher-forced caption cross-entropy,
prefix masked to ``IGNORE_INDEX``), ``.logits`` and ``.hidden_states`` (for the
eval adapter's logit-lens). The ``<image>`` media token drives cross-attention
masking via ``media_locations = input_ids == media_token_id``; ``media_token_id``
is stored in :class:`FlamingoConfig` and therefore in the checkpoint.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from apps.baselines.flamingo.constants import DEFAULT_MEDIA_TOKEN, IGNORE_INDEX
from apps.baselines.flamingo.modeling.aux_heads import FlamingoAuxHeads
from apps.baselines.flamingo.modeling.decoder import FlamingoGPT2Decoder, build_gpt2_backbone
from apps.baselines.flamingo.modeling.gated_cross_attention import GatedCrossAttentionBlock
from apps.baselines.flamingo.modeling.perceiver_resampler import PerceiverResampler
from apps.baselines.flamingo.modeling.vision_tower import FrozenDINOv2VisionTower

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class FlamingoConfig:
    """Serializable config for :class:`FlamingoVLM` (stored in the checkpoint)."""

    # --- Tokenizer / media wiring ---
    tokenizer_name: str = "gpt2"
    media_token: str = DEFAULT_MEDIA_TOKEN
    #: Filled by :func:`build_flamingo_model_and_tokenizer`.
    media_token_id: int | None = None
    vocab_size: int | None = None

    # --- Vision tower (frozen DINOv2) ---
    vision_encoder: str = "dinov2_vitb14"
    vision_tower_path: str | None = None
    freeze_vision: bool = True

    # --- Perceiver Resampler (visual capacity knobs) ---
    perceiver_num_latents: int = 64
    perceiver_depth: int = 2
    perceiver_heads: int = 8
    perceiver_dim_head: int = 64
    perceiver_num_time_embeds: int = 4
    perceiver_ff_mult: int = 4

    # --- Gated cross-attention ---
    cross_attn_every: int = 1
    cross_attn_heads: int = 12
    cross_attn_dim_head: int = 64
    only_attend_immediate_media: bool = True
    gate_init: float = 0.0
    #: ``learnable`` | ``frozen_open`` | ``none`` (see gated_cross_attention).
    gate_mode: str = "frozen_open"

    # --- Decoder (GPT-2) ---
    #: ``scratch`` | ``hf_gpt2`` | ``phase0_ckpt``.
    decoder_init: str = "scratch"
    #: Required when ``decoder_init == "phase0_ckpt"`` (no fabricated default).
    decoder_init_path: str | None = None
    hidden_size: int = 768
    n_layer: int = 12
    n_head: int = 12
    max_position_embeddings: int = 1024
    #: Adapter-only training (off in the paper). Freeze the pretrained GPT-2 LM (embeddings,
    #: transformer blocks, final LN, tied LM head) and train only the Perceiver +
    #: gated cross-attention adapter. See :meth:`FlamingoVLM._freeze_language_decoder`.
    freeze_decoder: bool = False

    # --- Optional aux heads (objectives (a)/(b), default OFF) ---
    use_aux_heads: bool = False
    aux_proj_dim: int = 256
    contrastive_init_temp: float = 0.07


@dataclass
class FlamingoOutput:
    """Forward output (HF-``CausalLMOutput``-like surface)."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class FlamingoVLM(nn.Module):
    """Native Flamingo generative VLM."""

    def __init__(self, config: FlamingoConfig) -> None:
        super().__init__()
        if config.media_token_id is None or config.vocab_size is None:
            msg = (
                "FlamingoConfig.media_token_id / vocab_size are unset. Build the model via "
                "build_flamingo_model_and_tokenizer(...) which fills them from the tokenizer."
            )
            raise ValueError(msg)
        self.config = config
        self.media_token_id = int(config.media_token_id)

        self.vision_tower = FrozenDINOv2VisionTower(
            config.vision_encoder,
            vision_tower_path=config.vision_tower_path,
            freeze=config.freeze_vision,
        )
        vision_hidden = self.vision_tower.hidden_size

        self.perceiver = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.perceiver_depth,
            dim_head=config.perceiver_dim_head,
            heads=config.perceiver_heads,
            num_latents=config.perceiver_num_latents,
            num_time_embeds=config.perceiver_num_time_embeds,
            ff_mult=config.perceiver_ff_mult,
            inp_dim=vision_hidden if vision_hidden != config.hidden_size else None,
        )

        gpt2 = build_gpt2_backbone(
            decoder_init=config.decoder_init,
            decoder_init_path=config.decoder_init_path,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
        )
        self.decoder = FlamingoGPT2Decoder(
            gpt2,
            cross_attn_every=config.cross_attn_every,
            cross_attn_heads=config.cross_attn_heads,
            cross_attn_dim_head=config.cross_attn_dim_head,
            gate_init=config.gate_init,
            gate_mode=config.gate_mode,
            only_attend_immediate_media=config.only_attend_immediate_media,
        )

        self.aux_heads: FlamingoAuxHeads | None = (
            FlamingoAuxHeads(
                config.hidden_size,
                proj_dim=config.aux_proj_dim,
                init_temp=config.contrastive_init_temp,
            )
            if config.use_aux_heads
            else None
        )

        # Optional adapter-only training: freeze the pretrained GPT-2 LM (off in the paper).
        if config.freeze_decoder:
            self._freeze_language_decoder()

    def _freeze_language_decoder(self) -> None:
        """Freeze the pretrained GPT-2 language decoder (adapter-only training).

        Sets ``requires_grad=False`` on every decoder parameter that is NOT part
        of the gated cross-attention adapter — i.e. the token/position embeddings
        (``wte``/``wpe``), all GPT-2 transformer blocks (``h``), the final
        LayerNorm (``ln_f``), and the tied LM head (``lm_head.weight`` IS
        ``wte.weight`` — the same Parameter — so it is frozen with the embedding).
        The gated cross-attention blocks (``decoder.gated_attn_layers``) and the
        Perceiver Resampler (a separate top-level module) stay trainable, so only
        the adapter learns. Frozen params are excluded from the AdamW groups
        automatically because :func:`optim.build_param_groups` skips
        ``requires_grad=False`` params.
        """
        for name, param in self.decoder.named_parameters():
            if not name.startswith("gated_attn_layers"):
                param.requires_grad_(False)  # noqa: FBT003 - torch stub is positional

    # ------------------------------------------------------------------ helpers

    @property
    def ln_f(self) -> nn.Module:
        """Final LayerNorm (for the logit-lens)."""
        return self.decoder.ln_f

    @property
    def lm_head(self) -> nn.Module:
        """Tied LM head (for the logit-lens)."""
        return self.decoder.lm_head

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``pixel_values (B,3,H,W) -> perceiver latents (B, T, M, D)``."""
        feats = self.vision_tower(pixel_values)
        return self.perceiver(feats)

    def gate_summary(self) -> dict[str, float]:
        """Mean ``tanh(gate)`` across inserted blocks (for W&B logging)."""
        attn_vals: list[float] = []
        ff_vals: list[float] = []
        for block in self.decoder.gated_attn_layers:
            if not isinstance(block, GatedCrossAttentionBlock):  # None placeholder
                continue
            a, f = block.gate_values()
            attn_vals.append(a)
            ff_vals.append(f)
        if not attn_vals:
            return {}
        return {
            "gate/attn_tanh_mean": sum(attn_vals) / len(attn_vals),
            "gate/ff_tanh_mean": sum(ff_vals) / len(ff_vals),
        }

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
    ) -> FlamingoOutput:
        """Teacher-forced forward.

        Args:
            input_ids: ``(B, S)`` token ids (prompt + caption).
            attention_mask: ``(B, S)`` padding mask.
            labels: ``(B, S)`` with ``IGNORE_INDEX`` on prompt / padding.
            pixel_values: ``(B, 3, H, W)`` images (encoded here) — or ``None``
                when ``image_embeds`` is supplied directly.
            image_embeds: Pre-computed perceiver latents ``(B, T, M, D)`` (lets
                aux losses reuse a mismatched image without re-encoding).

        Returns:
            :class:`FlamingoOutput`.
        """
        if image_embeds is None and pixel_values is not None:
            image_embeds = self.encode_images(pixel_values)

        media_locations = input_ids == self.media_token_id

        dec = self.decoder(
            input_ids,
            image_embeds=image_embeds,
            media_locations=media_locations,
            attention_mask=attention_mask,
        )

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = dec.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return FlamingoOutput(loss=loss, logits=dec.logits, hidden_states=dec.hidden_states)


def build_tokenizer(
    tokenizer_name: str = "gpt2",
    *,
    media_token: str = DEFAULT_MEDIA_TOKEN,
) -> tuple[PreTrainedTokenizerBase, int]:
    """Load a tokenizer, add the ``<image>`` media token, set pad=eos.

    Returns ``(tokenizer, media_token_id)``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if media_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [media_token]})
    media_token_id = int(tokenizer.convert_tokens_to_ids(media_token))
    return tokenizer, media_token_id


def build_flamingo_model_and_tokenizer(
    config: FlamingoConfig | dict[str, Any],
) -> tuple[FlamingoVLM, PreTrainedTokenizerBase]:
    """Build a :class:`FlamingoVLM` + tokenizer with media-token wiring filled in.

    Accepts a :class:`FlamingoConfig` or a plain dict (e.g. an OmegaConf-derived
    mapping). Fills ``media_token_id`` and ``vocab_size`` from the tokenizer.

    When ``decoder_init == "phase0_ckpt"`` the tokenizer is loaded from
    ``decoder_init_path`` (the Phase-0 GPT-2's OWN COCO-MC BPE vocab, size 22358)
    so the decoder embeddings and the tokenizer stay aligned; ``tokenizer_name``
    is only used for the ``scratch`` / ``hf_gpt2`` decoders (standard gpt2).
    """
    if isinstance(config, dict):
        config = FlamingoConfig(**{k: v for k, v in config.items() if k in FlamingoConfig().__dict__})
    tokenizer_source = (
        config.decoder_init_path
        if config.decoder_init == "phase0_ckpt" and config.decoder_init_path
        else config.tokenizer_name
    )
    tokenizer, media_token_id = build_tokenizer(tokenizer_source, media_token=config.media_token)
    config.media_token_id = media_token_id
    config.vocab_size = len(tokenizer)
    model = FlamingoVLM(config)
    return model, tokenizer


def config_to_dict(config: FlamingoConfig) -> dict[str, Any]:
    """Serialize a :class:`FlamingoConfig` for checkpointing."""
    return asdict(config)
