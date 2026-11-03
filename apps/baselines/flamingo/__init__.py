# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Native Flamingo baseline: model + trainer + eval adapter.

Importing this package registers Hydra ConfigStore nodes for the Flamingo
trainer's schema so YAML configs under ``apps/baselines/flamingo/configs/`` can
extend them with ``defaults: [base_flamingo_trainer, ...]``.
"""

from hydra.core.config_store import ConfigStore

from apps.baselines.flamingo.training.configs import (
    CheckpointConfig,
    DataConfig,
    FlamingoTrainerConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimConfig,
    WandbConfig,
)

_cs = ConfigStore.instance()
_cs.store(name="base_flamingo_trainer", node=FlamingoTrainerConfig)
_cs.store(name="base_model", group="model", node=ModelConfig)
_cs.store(name="base_data", group="data", node=DataConfig)
_cs.store(name="base_optim", group="optim", node=OptimConfig)
_cs.store(name="base_objective", group="objective", node=ObjectiveConfig)
_cs.store(name="base_checkpoint", group="checkpoint", node=CheckpointConfig)
_cs.store(name="base_wandb", group="wandb", node=WandbConfig)
