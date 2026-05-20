# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hydra dataclass schemas for the alignment-scoring pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from core.utils import LauncherConfig


@dataclass
class DatasetConfig:
    """Hydra-instantiable caption dataset (COCO, video CSV, ...)."""

    #: Fully qualified class name of the dataset (e.g. ``CocoCaptionsDataset``).
    _target_: str = MISSING

    #: Path to the manifest file (COCO JSON or CSV with ``clip_filename`` + ``utterance``).
    manifest_path: str = MISSING

    #: Directory containing the images or video clips referenced by the manifest.
    dataset_dir: str = MISSING


@dataclass
class DataConfig:
    """Dataset + DataLoader knobs used by every pipeline."""

    dataset: DatasetConfig = MISSING
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LoraConfig:
    """PEFT LoRA config (instantiated via ``hydra.utils.instantiate``)."""

    _target_: str = "peft.LoraConfig"
    r: int = 8
    lora_alpha: int = 16
    target_modules: Any = "all-linear"
    lora_dropout: float = 0.1
    bias: str = "none"


@dataclass
class ModelConfig:
    """An open_clip model spec, optionally LoRA-wrapped."""

    #: open_clip model identifier (e.g. ``"ViT-B-16-quickgelu"``).
    model_name: str = MISSING

    #: ``"eval"`` freezes; ``"train"`` enables grads and uses train preprocess.
    mode: str = "eval"

    #: Pretrained tag (e.g. ``"openai"`` or ``"meta"``).
    pretrained: str | None = None

    #: LoRA config — set to enable adapter wrapping (used by finetune_lora).
    lora: LoraConfig | None = None

    gradient_checkpointing: bool = False

    #: Whether the encoder is video-aware. False = mean-pool image features per clip.
    is_video_model: bool = True


@dataclass
class CLIPOptimConfig:
    """AdamW optimizer config used by the LoRA finetune trainer."""

    lr: float = 8e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    start_epoch: int = 0
    epochs: int = 50
    warmup_epochs: int = 5
    clip_grad_norm: float | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    eps: float = 1e-8
    bias_wd: bool = False
    dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 1
    logit_scale_lr_scale: float = 1.0

    #: Steps already completed in the current epoch — set when resuming mid-epoch.
    epoch_offset: int | None = None

    #: Stop training when val loss hasn't improved for ``patience`` epochs.
    early_stopping: bool = False

    #: Number of epochs without improvement before early stop kicks in.
    patience: int = 5


# ---------------------------------------------------------------------------
# Per-pipeline configs.
# ---------------------------------------------------------------------------


@dataclass
class CLIPProcessorConfig:
    """One side of the CLIP-scoring pipeline (matched OR shuffled pairs)."""

    name: str = MISSING
    data: DataConfig = MISSING
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_name="PE-Core-bigG-14-448",
            pretrained="meta",
        ),
    )

    #: Max items per Stopes job-array chunk.
    num_items_per_chunk: int = 2000


@dataclass
class CLIPScoringPipelineConfig:
    """Full CLIP scoring pipeline: schedule a matched + shuffled processor and aggregate."""

    name: str = MISSING
    matched_processor: CLIPProcessorConfig = MISSING
    shuffled_processor: CLIPProcessorConfig = MISSING
    output_dir: str = MISSING
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


@dataclass
class SonarSTSProcessorConfig:
    """One side of the STS-scoring pipeline (joins two text manifests on media_id)."""

    name: str = MISSING
    dataset_a: DatasetConfig = MISSING
    dataset_b: DatasetConfig = MISSING
    encoder: str = "text_sonar_basic_encoder"
    source_lang: str = "eng_Latn"
    batch_size: int = 32
    num_items_per_chunk: int = 2000


@dataclass
class SonarSTSScoringPipelineConfig:
    """STS scoring with matched + shuffled processors and JS-divergence aggregation."""

    name: str = MISSING
    matched_processor: SonarSTSProcessorConfig = MISSING
    shuffled_processor: SonarSTSProcessorConfig = MISSING
    output_dir: str = MISSING
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


@dataclass
class PLMGenerationConfig:
    """Perception-LM generation config used by captioning + VQA scoring."""

    name: str = MISSING

    #: HuggingFace repo ID or local checkpoint dir.
    ckpt: str = "facebook/Perception-LM-8B"

    dataset: DatasetConfig = MISSING

    #: The prompt fed to PLM (e.g. ``"Describe this image in detail."``).
    question: str = MISSING

    num_items_per_chunk: int = 500
    max_gen_len: int = 256
    temperature: float = 0.6
    top_p: float | None = None
    top_k: int | None = None
    dtype: str = "bf16"

    #: Which dataset field to feed PLM (``"image"`` or ``"video"``).
    media_field: str = "image"

    #: If True, the prompt template uses ``{text}`` and we score P(Yes) instead of generating.
    vqa_scoring: bool = False


@dataclass
class CaptioningPipelineConfig:
    name: str = MISSING
    generation: PLMGenerationConfig = MISSING
    output_dir: str = MISSING
    output_manifest_path: str = MISSING
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


@dataclass
class VQAScoringPipelineConfig:
    """VQA-style alignment scoring: P(Yes) over matched + shuffled, then JS divergence."""

    name: str = MISSING
    matched_processor: PLMGenerationConfig = MISSING
    shuffled_processor: PLMGenerationConfig = MISSING
    yes_token: str = "Yes"  # noqa: S105 — literal token name, not a credential
    output_dir: str = MISSING
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


@dataclass
class WandbConfig:
    """W&B logging knobs (used by the LoRA finetune trainer)."""

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    run_id: str | None = None
    log_code: bool = False


@dataclass
class FinetuneLoraConfig:
    """Single-flow CLIP/PE LoRA finetune config (DDP-aware)."""

    name: str = MISSING
    data_train: DataConfig = MISSING

    #: Validation manifest — usually a held-out split. Reuse data_train for smoke tests.
    data_val: DataConfig = MISSING

    model: ModelConfig = MISSING
    optim: CLIPOptimConfig = field(default_factory=CLIPOptimConfig)
    output_dir: str = MISSING

    #: Steps between mid-epoch metric prints.
    log_interval: int = 50

    #: Steps between mid-epoch validation passes (very large = epoch-end only).
    eval_interval: int = 1_000_000

    #: Bound on retained checkpoint files (oldest evicted first).
    max_checkpoints: int | None = 3

    #: Explicit checkpoint path / dir to resume from. Defaults to the latest in output_dir.
    resume: str | None = None

    seed: int = 42

    wandb: WandbConfig = field(default_factory=WandbConfig)
