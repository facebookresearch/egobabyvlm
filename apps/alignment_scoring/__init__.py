# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Alignment-scoring pipelines: clip / sts / vqa / captioning + LoRA finetuning."""

from hydra.core.config_store import ConfigStore

from .configs import (
    CaptioningPipelineConfig,
    CLIPProcessorConfig,
    CLIPScoringPipelineConfig,
    FinetuneLoraConfig,
    SonarSTSProcessorConfig,
    SonarSTSScoringPipelineConfig,
    VQAScoringPipelineConfig,
)

_cs = ConfigStore.instance()
_cs.store(name="base_clip_scoring_pipeline", node=CLIPScoringPipelineConfig)
_cs.store(name="base_clip_processor", node=CLIPProcessorConfig)
_cs.store(name="base_sts_scoring_pipeline", node=SonarSTSScoringPipelineConfig)
_cs.store(name="base_sts_processor", node=SonarSTSProcessorConfig)
_cs.store(name="base_captioning_pipeline", node=CaptioningPipelineConfig)
_cs.store(name="base_vqa_scoring_pipeline", node=VQAScoringPipelineConfig)
_cs.store(name="base_finetune_lora", node=FinetuneLoraConfig)
