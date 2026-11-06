# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hydra dataclass schemas for the native Flamingo trainer.

Mirrors ``apps/baselines/clip/configs.py`` and adds the Flamingo-specific knobs
(visual capacity, gate mode, decoder init) plus the objective weights for the
optional (a)/(b) auxiliary losses (default OFF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from core.utils import LauncherConfig


@dataclass
class ModelConfig:
    """Flamingo architecture knobs (maps onto ``modeling.FlamingoConfig``)."""

    # Tokenizer / media wiring
    tokenizer_name: str = "gpt2"
    media_token: str = "<image>"  # noqa: S105 - not a secret

    # Frozen DINOv2 vision tower
    vision_encoder: str = "dinov2_vitb14"
    vision_tower_path: str | None = None
    freeze_vision: bool = True

    # Perceiver Resampler
    perceiver_num_latents: int = 64
    perceiver_depth: int = 2
    perceiver_heads: int = 8
    perceiver_dim_head: int = 64
    perceiver_num_time_embeds: int = 4
    perceiver_ff_mult: int = 4

    # Gated cross-attention (paper: a gated block before every GPT-2 layer)
    cross_attn_every: int = 1
    cross_attn_heads: int = 12
    cross_attn_dim_head: int = 64
    only_attend_immediate_media: bool = True
    gate_init: float = 0.0
    #: ``learnable`` | ``frozen_open`` | ``none``. Paper: ``frozen_open``
    #: (tanh gates pinned at tanh(3) ~= 0.995).
    gate_mode: str = "frozen_open"

    # Decoder (GPT-2)
    #: ``scratch`` | ``hf_gpt2`` | ``phase0_ckpt``. Paper: ``phase0_ckpt``, the
    #: Phase-0 GPT-2 (``apps/baselines/lm_training``) pretrained on the same
    #: corpus; its tokenizer is loaded from the same directory.
    decoder_init: str = "phase0_ckpt"
    #: Phase-0 GPT-2 HF model dir (required for ``phase0_ckpt``).
    decoder_init_path: str | None = None
    hidden_size: int = 768
    n_layer: int = 12
    n_head: int = 12
    max_position_embeddings: int = 1024

    #: Adapter-only training (off in the paper). When True, freeze the pretrained GPT-2
    #: language decoder — token/position embeddings, all transformer blocks, the
    #: final LayerNorm, and the tied LM head (``lm_head`` shares ``wte``'s weight,
    #: so it is frozen with it) — and train ONLY the newly-added adapter (the
    #: Perceiver Resampler + the gated cross-attention blocks). The frozen params
    #: get ``requires_grad=False`` and are therefore excluded from the AdamW param
    #: groups automatically (see ``optim.build_param_groups``). Default False
    #: leaves the base recipe unchanged.
    freeze_decoder: bool = False

    # Aux heads (constructed only when an aux loss weight > 0; see ObjectiveConfig)
    aux_proj_dim: int = 256
    contrastive_init_temp: float = 0.07


@dataclass
class DataConfig:
    """Dataset builder + loader knobs."""

    #: Hydra-instantiable builder returning ``(train_ds, val_ds)`` — the trainer
    #: injects ``tokenizer`` / ``transform`` / ``val_transform`` at call time.
    #: All other builder kwargs (percent, seed, prompt_template, max_length,
    #: on_load_error, max_load_failures, ...) live in the dataset YAML itself.
    dataset: Any = MISSING

    batch_size: int = 32
    val_batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    augment: bool = False
    image_size: int = 224


@dataclass
class OptimConfig:
    """AdamW with a weight-decay param-group split (see ``training/optim.py``)."""

    lr: float = 1.0e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    min_lr_ratio: float = 0.0
    grad_clip: float | None = 1.0

    #: When True, gate scalars / all 1-D params (LayerNorm, biases) / perceiver
    #: ``latents`` + ``time_pos_emb`` get ``weight_decay=0.0`` (the WD-on-gates
    #: fix). Leave True.
    split_weight_decay: bool = True


@dataclass
class ObjectiveConfig:
    """Loss weights. Aux terms default to 0 (the paper's caption-LL objective)."""

    #: Teacher-forced caption cross-entropy (the generative head).
    caption_loss_weight: float = 1.0

    #: (a) Hard-negative caption margin (same image, wrong caption). No params.
    hard_negative_loss_weight: float = 0.0
    hard_negative_margin: float = 1.0

    #: (b) ITM classifier (match vs. shuffled image). Needs aux heads.
    itm_loss_weight: float = 0.0

    #: (b) In-batch InfoNCE between pooled image latents and caption states.
    contrastive_loss_weight: float = 0.0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    run_name: str | None = None
    mode: str = "online"


@dataclass
class CheckpointConfig:
    save_dir: str = MISSING
    save_every: int = 1
    keep_last: int = 3
    resume_from: str | None = None


@dataclass
class FlamingoTrainerConfig:
    """Top-level config for the native Flamingo trainer."""

    name: str = MISSING
    seed: int = 42
    epochs: int = 10
    log_interval: int = 50
    #: Optional hard cap on the number of optimizer steps. ``None`` (default) =
    #: unlimited (run the full ``epochs``). When set, training stops after this
    #: many optimizer steps and the normal checkpoint save path is triggered.
    #: Flat top-level knob (like ``epochs``/``log_interval``); override on the
    #: CLI as ``max_steps=20``.
    max_steps: int | None = None

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = MISSING
    optim: OptimConfig = field(default_factory=OptimConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    checkpoint: CheckpointConfig = MISSING
    wandb: WandbConfig = field(default_factory=WandbConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
