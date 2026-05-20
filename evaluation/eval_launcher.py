# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING
from submitit.helpers import clean_env

from core.utils import LauncherConfig, resolve_and_print_config, setup_logging, to_yaml
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalModelConfig, VisionBackboneConfig
from evaluation.multimodal.devbench.base import DevBenchTaskEvalConfig
from evaluation.multimodal.devbench.pipeline import DevBenchPipelineConfig
from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalConfig
from evaluation.multimodal.machine_devbench.pipeline import MachineDevBenchPipelineConfig
from evaluation.text.ltswap import LTSwapEvalConfig
from evaluation.text.pipeline import TextPipelineConfig
from evaluation.text.zorro import ZorroEvalConfig
from evaluation.vision.abx import ABXEvalModuleConfig
from evaluation.vision.depth_estimation import DepthEstimationEvalModuleConfig
from evaluation.vision.knn import KNNEvalModuleConfig
from evaluation.vision.linear import LinearEvalModuleConfig
from evaluation.vision.pipeline import VisionPipelineConfig
from evaluation.vision.semantic_segmentation import (
    SemanticSegmentationEvalModuleConfig,
    SemanticSegmentationRunConfig,
    SemanticSegmentationSweepConfig,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from stopes.core import Launcher

logger = logging.getLogger(__name__)


@dataclass
class EvalLauncherConfig:
    eval: EvalConfig = MISSING
    launcher: LauncherConfig = field(default_factory=LauncherConfig)

    #: Model for standalone tasks — override with: model=gpt2.
    model: Any | None = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=EvalLauncherConfig)
cs.store(name="abx", group="eval", node=ABXEvalModuleConfig)
cs.store(name="linear", group="eval", node=LinearEvalModuleConfig)
cs.store(name="knn", group="eval", node=KNNEvalModuleConfig)
cs.store(name="depth_estimation", group="eval", node=DepthEstimationEvalModuleConfig)
cs.store(name="semantic_segmentation", group="eval", node=SemanticSegmentationEvalModuleConfig)
cs.store(name="semantic_segmentation_sweep", group="eval", node=SemanticSegmentationSweepConfig)
cs.store(name="semantic_segmentation_run", group="eval", node=SemanticSegmentationRunConfig)
cs.store(name="devbench", group="eval", node=DevBenchTaskEvalConfig)
cs.store(name="devbench_pipeline_base", group="eval", node=DevBenchPipelineConfig)
cs.store(name="machine_devbench", group="eval", node=MachineDevBenchEvalConfig)
cs.store(name="machine_devbench_pipeline_base", group="eval", node=MachineDevBenchPipelineConfig)
cs.store(name="zorro_base", group="eval", node=ZorroEvalConfig)
cs.store(name="ltswap_base", group="eval", node=LTSwapEvalConfig)
cs.store(name="text_pipeline_base", group="eval", node=TextPipelineConfig)
cs.store(name="vision_pipeline_base", group="eval", node=VisionPipelineConfig)
cs.store(name="model_base", group="model", node=EvalModelConfig)
cs.store(name="backbone_base", group="model", node=VisionBackboneConfig)


async def eval_pipeline(config: EvalLauncherConfig) -> None:
    with clean_env():
        launcher: Launcher = instantiate(config.launcher)

        eval_module = EvalModule.build(config.eval)

        results = await launcher.schedule(eval_module)

    logger.info("Results for evaluation module %s:\n%s", eval_module.name(), to_yaml(results))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: EvalLauncherConfig) -> None:
    setup_logging()
    resolve_and_print_config(cast("DictConfig", config))

    try:
        asyncio.run(eval_pipeline(config))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
