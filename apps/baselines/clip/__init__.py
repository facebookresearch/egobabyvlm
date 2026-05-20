# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLIP-style baseline: contrastive trainer + feature extractors.

Importing this package registers Hydra ConfigStore nodes for the contrastive
trainer's schema so YAML configs under ``apps/baselines/clip/configs/`` can
extend them with ``defaults: [base_contrastive_trainer, ...]``.
"""

from hydra.core.config_store import ConfigStore

from apps.baselines.clip.configs import (
    CheckpointConfig,
    ContrastiveTrainerConfig,
    DataConfig,
    DINOv2Config,
    ModeConfig,
    ModelConfig,
    OptimConfig,
    TextEncoderConfig,
    TextOnlyDataConfig,
    WandbConfig,
)

_cs = ConfigStore.instance()
_cs.store(name="base_contrastive_trainer", node=ContrastiveTrainerConfig)
_cs.store(name="base_mode", group="mode", node=ModeConfig)
_cs.store(name="base_model", group="model", node=ModelConfig)
_cs.store(name="base_data", group="data", node=DataConfig)
_cs.store(name="base_text_encoder", group="text_encoder", node=TextEncoderConfig)
_cs.store(name="base_optim", group="optim", node=OptimConfig)
_cs.store(name="base_text_only_data", group="text_only_data", node=TextOnlyDataConfig)
_cs.store(name="base_dinov2", group="dinov2", node=DINOv2Config)
_cs.store(name="base_checkpoint", group="checkpoint", node=CheckpointConfig)
_cs.store(name="base_wandb", group="wandb", node=WandbConfig)
