# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hydra dataclass schemas for the contrastive trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from core.utils import LauncherConfig


@dataclass
class TextEncoderConfig:
    """Hydra-instantiable text encoder."""

    _target_: str = "apps.baselines.clip.modeling.TextEncoder"
    hf_model_name: str = "bert-base-uncased"
    embedding_dim: int = 512
    dropout: float = 0.1
    freeze: bool = False

    #: ``"cls"`` (default; standard BERT) or ``"mean"`` (pooled hidden states).
    pooling: str = "cls"


@dataclass
class ModelConfig:
    """Multimodal model: text + vision encoders + temperature."""

    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)

    #: Left ``Any`` so callers can pick one of the three concrete encoder classes
    #: (``HubDINOv2VisionEncoder``, ``CustomDINOv2VisionEncoder``,
    #: ``RandomViTVisionEncoder``) via Hydra composition without forcing a common
    #: dataclass schema. Each encoder declares its own constructor args.
    vision_encoder: Any = MISSING

    #: Shared embedding dimension. Must match both encoders' projections.
    embedding_dim: int = 512

    normalize_features: bool = True
    temperature: float = 0.07
    fix_temperature: bool = False


@dataclass
class DataConfig:
    """Train + val datasets and loader knobs."""

    #: Hydra-instantiable dataset config (e.g. ``CocoCaptionsDataset`` or
    #: ``HowToCaptionsDataset``); the dataset schema isn't pinned here so each
    #: subclass can declare its own fields (``multiple_captions`` for COCO,
    #: ``multiple_frames`` for video). The constructor's signature is the source
    #: of truth.
    train_dataset: Any = MISSING

    val_dataset: Any | None = None
    batch_size: int = 64
    val_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    #: Whether to apply random horizontal flip + Gaussian blur to train images.
    augment: bool = False


@dataclass
class TextOnlyDataConfig:
    """Optional raw-text corpus for the BERT MLM head."""

    #: Path to a plain-text file (one example per line).
    train_file: str = MISSING

    val_file: str | None = None
    max_seq_len: int = 512
    mlm_probability: float = 0.15
    batch_size: int = 64
    num_workers: int = 2


@dataclass
class OptimConfig:
    """AdamW + cosine LR schedule shared by contrastive and (if active) MLM."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    #: Final LR as a fraction of the initial LR for the cosine schedule.
    min_lr_ratio: float = 0.0

    grad_clip: float | None = None


@dataclass
class ModeConfig:
    """Which losses run, and how often.

    The trainer cycles through ``interleave`` in dict order, running each mode
    for its budget before advancing. Modes with budget 0 are dropped.
    """

    #: One of ``contrastive``, ``interleaved_lm``, ``interleaved_dino``, ``triple``.
    name: str = MISSING

    #: Per-mode step budget, e.g. ``{"contrastive": 4, "mlm": 1}``.
    interleave: dict[str, int] = field(default_factory=lambda: {"contrastive": 1})

    #: If True, copy DINOv2 teacher backbone → contrastive vision encoder after
    #: each DINOv2 block. Requires the SSL student and the vision encoder to share
    #: the same architecture and image_size; otherwise the trainer raises at
    #: construction.
    sync_vision_from_dinov2: bool = False


@dataclass
class DINOv2Config:
    """DINOv2 SSL config (loaded explicitly from a YAML on disk)."""

    #: Absolute or relative path to a DINOv2 SSL training YAML. The configs
    #: under ``apps/baselines/dinov2/third_party/dinov2/configs/train/`` are
    #: the standard source of these — but any YAML composable with
    #: ``ssl_default_config.yaml`` works.
    config_path: str = MISSING

    #: Optional nested overrides applied on top of the loaded config (e.g.
    #: ``{train: {OFFICIAL_EPOCH_LENGTH: 100}}`` for short smoke runs). Typed as
    #: ``Any`` so callers can pass nested DictConfigs without struct-mode
    #: constraints.
    overrides: Any = field(default_factory=dict)

    #: Optional path to a directory containing a DINOv2 SSL FSDP checkpoint
    #: (``last_checkpoint.rank_0`` + the matching ``model_*.rank_0.pth``).
    #: Required when ``mode.sync_vision_from_dinov2=true``.
    pretrained_dir: str | None = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    run_name: str | None = None
    mode: str = "online"


@dataclass
class CheckpointConfig:
    save_dir: str = MISSING

    #: Save a checkpoint every N epochs.
    save_every: int = 1

    #: Retain only the most recent N ``epoch_*.pt`` checkpoints. ``best.pt``,
    #: ``latest.pt``, and ``interrupted.pt`` are always preserved. Set to 0
    #: to keep every per-epoch checkpoint.
    keep_last: int = 3

    #: Path to a checkpoint to resume training from.
    resume_from: str | None = None


@dataclass
class ContrastiveTrainerConfig:
    """Top-level config for the contrastive trainer."""

    #: Run name (used for output/checkpoint paths).
    name: str = MISSING

    seed: int = 42
    epochs: int = 30
    log_interval: int = 50

    mode: ModeConfig = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = MISSING
    optim: OptimConfig = field(default_factory=OptimConfig)

    #: Required for modes that include MLM (``interleaved_lm``, ``triple``).
    text_only_data: TextOnlyDataConfig | None = None

    #: Required for modes that include DINOv2 (``interleaved_dino``, ``triple``).
    dinov2: DINOv2Config | None = None

    checkpoint: CheckpointConfig = MISSING
    wandb: WandbConfig = field(default_factory=WandbConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
