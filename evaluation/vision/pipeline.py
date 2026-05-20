# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from stopes.core import Launcher, Requirements
from submitit.helpers import clean_env

from core.utils import LauncherConfig, setup_logging, to_yaml
from evaluation.base import PipelineError, process_task_results, to_path
from evaluation.base.eval_module import EvalConfig, EvalModule

logger = logging.getLogger(__name__)


@dataclass
class VisionPipelineConfig(EvalConfig):
    """Configuration for Vision evaluation pipeline."""

    _target_: str = "evaluation.vision.pipeline.VisionPipeline"

    name: str = "vision_pipeline"

    #: Required. Must have a 'name' field for result organization.
    model: dict[str, Any] = field(default_factory=dict)

    #: Maps task names to their evaluation configurations.
    tasks: dict[str, Any] = field(default_factory=dict)

    launcher: LauncherConfig = field(default_factory=lambda: LauncherConfig(cluster="slurm"))

    #: Whether to compute aggregate vision score.
    aggregate_results: bool = True


# Constants for normalization
_COCO_STUFF_NUM_CLASSES = 171
# mIoU chance for uniform random predictions over K classes:
#   E[IoU_i] = (1/K²) / ((2K-1)/K²) = 1/(2K-1).
# (Using 1/K is wrong — that's the per-pixel hit rate, not IoU.)
_SEGMENTATION_CHANCE = 1.0 / (2 * _COCO_STUFF_NUM_CLASSES - 1)  # 1/341 ≈ 0.00293
_ABX_CHANCE = 0.5
_DEPTH_RMSE_MAX = 2.0  # Assumed max RMSE for normalization


class VisionPipeline(EvalModule):
    """Pipeline for running all vision evaluations."""

    def __init__(self, config: VisionPipelineConfig) -> None:
        """Initialize the Vision Pipeline.

        Args:
            config: Configuration for the pipeline.
        """
        super().__init__(config, VisionPipelineConfig)

        if not self.config.model or "name" not in self.config.model:
            raise ValueError("VisionPipeline requires a model configuration with a 'name' field.")

        model_name = self.config.model["name"]
        self.results_dir = (
            to_path(self.config.output_dir) / "vision_pipeline_results" / model_name / f"seed{self.config.seed}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        """Requirements for the pipeline coordinator job."""
        return Requirements(
            nodes=1,
            mem_gb=16,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        """Return the name of the pipeline."""
        return self.config.name

    async def run_tasks(self) -> dict[str, Any]:
        """Run all vision tasks asynchronously.

        Returns:
            Dictionary containing results from all evaluations.

        Raises:
            PipelineError: If one or more tasks fail.
        """
        logger.info("Starting Vision Pipeline evaluation")

        task_modules = {}
        for task_name, task_config in self.config.tasks.items():
            logger.info("Building module for task: %s", task_name)
            task_modules[task_name] = EvalModule.build(task_config)

        if not task_modules:
            logger.warning("No tasks configured for vision pipeline")
            return {}

        logger.info("Tasks to run: %s", list(task_modules.keys()))

        launcher: Launcher = instantiate(self.config.launcher)
        logger.info("Launcher type: %s", type(launcher).__name__)

        with clean_env():
            tasks_to_run = []
            for task_name, module in task_modules.items():
                logger.info("Scheduling task: %s", task_name)
                task_future = launcher.schedule(module)
                tasks_to_run.append((task_name, task_future))

            logger.info("All %d tasks scheduled, waiting for completion...", len(tasks_to_run))

            task_names = [task_name for task_name, _ in tasks_to_run]
            task_futures = [task_future for _, task_future in tasks_to_run]

            task_results = await asyncio.gather(*task_futures, return_exceptions=True)

            successful_results, task_errors = process_task_results(task_names, task_results, self.config.name)

            if task_errors:
                raise PipelineError(self.config.name, task_errors)

            return successful_results

    def compute_aggregate_score(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compute aggregate vision score from task results.

        All scores normalized to 0-100 scale accounting for chance levels:
          - Classification (kNN, Linear): S = acc (already 0-100)
          - ABX: S = 100 * (acc - 0.5) / 0.5  (chance = 0.5)
          - Depth: S = 100 * (1 - RMSE / 2.0)  (lower RMSE = better)
          - Segmentation: S = 100 * (mIoU - 1/(2K-1)) / (1 - 1/(2K-1))  (K=171, chance ≈ 1/341)

          Final score: arithmetic mean of normalized scores.

          Args:
              results: Dictionary containing results from all tasks.

          Returns:
              Dictionary containing component scores and aggregate vision_score.
        """
        aggregate = {}
        normalized_scores = []

        for task_name, result in results.items():
            logger.info("Processing task %s", task_name)
            if isinstance(result, dict) and "error" in result:
                logger.warning("Skipping task %s due to error", task_name)
                continue

            # kNN tasks (e.g., knn_imagenet, knn_mnist)
            if task_name.startswith("knn"):
                target_key = "full_k20"
                if target_key in result and isinstance(result[target_key], dict):
                    raw_acc = result[target_key]["top-1"]  # 0-1 scale
                    # Convert to 0-100 scale
                    raw_acc_pct = raw_acc * 100.0
                    normalized = raw_acc_pct
                    normalized_scores.append(normalized)
                    aggregate[f"{task_name}_acc"] = raw_acc_pct
                    aggregate[f"{task_name}_normalized"] = normalized
                    logger.info("kNN %s (k=20, full): acc=%.2f%%, normalized=%.2f", task_name, raw_acc_pct, normalized)
                else:
                    logger.warning("kNN result %s missing expected key '%s', skipping", task_name, target_key)

            # Linear tasks (e.g., linear_imagenet, linear_mnist, linear_countbench)
            elif task_name.startswith("linear") and "summary" in result:
                raw_acc = result["summary"].get("best_val_accuracy", 0)  # 0-1 scale
                # Convert to 0-100 scale
                raw_acc_pct = raw_acc * 100.0
                normalized = raw_acc_pct
                normalized_scores.append(normalized)
                aggregate[f"{task_name}_acc"] = raw_acc_pct
                aggregate[f"{task_name}_normalized"] = normalized
                logger.info("Linear %s: acc=%.2f%%, normalized=%.2f", task_name, raw_acc_pct, normalized)

            # ABX tasks (e.g., abx_imagenet, abx_mnist, abx_countbench)
            # ABX: S = 100 * (acc - 0.5) / 0.5
            elif task_name.startswith("abx") and "accuracy" in result:
                raw_acc = result["accuracy"]  # 0-1 scale
                normalized = 100.0 * (raw_acc - _ABX_CHANCE) / (1.0 - _ABX_CHANCE)
                normalized = max(0.0, min(100.0, normalized))  # Clamp to [0, 100]
                normalized_scores.append(normalized)
                aggregate[f"{task_name}_acc"] = raw_acc
                aggregate[f"{task_name}_normalized"] = normalized
                logger.info("ABX %s: acc=%.4f, normalized=%.2f", task_name, raw_acc, normalized)

            # Semantic segmentation: S = 100 * (mIoU - 1/92) / (1 - 1/92)
            elif task_name.startswith("semantic_seg") and "best_config" in result:
                best_config = result["best_config"]
                if best_config and "best_val_metrics" in best_config:
                    raw_miou = best_config["best_val_metrics"]["miou"]  # 0-1 scale
                    normalized = 100.0 * (raw_miou - _SEGMENTATION_CHANCE) / (1.0 - _SEGMENTATION_CHANCE)
                    normalized = max(0.0, min(100.0, normalized))  # Clamp to [0, 100]
                    normalized_scores.append(normalized)
                    aggregate[f"{task_name}_miou"] = raw_miou
                    aggregate[f"{task_name}_normalized"] = normalized
                    logger.info("Segmentation %s: mIoU=%.4f, normalized=%.2f", task_name, raw_miou, normalized)

            # Depth: S = 100 * (1 - RMSE / 2.0)
            elif task_name.startswith("depth") and "best_val_metrics" in result:
                raw_rmse = result["best_val_metrics"].get("rmse", _DEPTH_RMSE_MAX)
                normalized = 100.0 * (1.0 - raw_rmse / _DEPTH_RMSE_MAX)
                normalized = max(0.0, min(100.0, normalized))  # Clamp to [0, 100]
                normalized_scores.append(normalized)
                aggregate[f"{task_name}_rmse"] = raw_rmse
                aggregate[f"{task_name}_normalized"] = normalized
                logger.info("Depth %s: RMSE=%.4f, normalized=%.2f", task_name, raw_rmse, normalized)

            else:
                logger.warning("Unknown task type or missing expected keys for task %s, skipping", task_name)

        if normalized_scores:
            aggregate["vision_score"] = sum(normalized_scores) / len(normalized_scores)
            aggregate["num_components"] = len(normalized_scores)
            logger.info(
                "Aggregate vision score: %.2f (from %d components)",
                aggregate["vision_score"],
                len(normalized_scores),
            )
        else:
            aggregate["vision_score"] = None
            aggregate["num_components"] = 0
            logger.warning("No valid scores found for aggregation")

        return aggregate

    def run(
        self,
        iteration_value: object = None,
        iteration_index: int = 0,
    ) -> dict[str, Any]:
        """Run the Vision Pipeline.

        Args:
            iteration_value: Unused, for compatibility with base class.
            iteration_index: Unused, for compatibility with base class.

        Returns:
            Dictionary containing all results including aggregate score.

        Raises:
            PipelineError: If one or more tasks fail (after saving partial results).
        """
        setup_logging()

        pipeline_error = None
        try:
            results = asyncio.run(self.run_tasks())
        except PipelineError as e:
            results = e.get_results_with_errors({})
            pipeline_error = e

        raw_results_path = str(self.results_dir / "raw_results.yaml")
        with Path(raw_results_path).open("w") as f:
            f.write(to_yaml(results))
        logger.info("Raw results saved to %s", raw_results_path)

        if self.config.aggregate_results:
            aggregate = self.compute_aggregate_score(results)
            results["aggregate"] = aggregate

            aggregate_path = str(self.results_dir / "aggregate_results.yaml")
            with Path(aggregate_path).open("w") as f:
                f.write(to_yaml(aggregate))
            logger.info("Aggregate results saved to %s", aggregate_path)

            if aggregate.get("vision_score") is not None:
                logger.info("%s", "\n" + "=" * 80)
                logger.info("VISION PIPELINE COMPLETE")
                logger.info("=" * 80)
                logger.info("Aggregate vision_score: %.4f", aggregate["vision_score"])
                logger.info("Components: %d", aggregate["num_components"])
                for key, value in aggregate.items():
                    if key not in ("vision_score", "num_components"):
                        logger.info("  %s: %.4f", key, value)
                logger.info("=" * 80)
            else:
                logger.warning("Vision Pipeline complete but aggregate vision_score could not be computed")

        if pipeline_error:
            raise pipeline_error

        return results
