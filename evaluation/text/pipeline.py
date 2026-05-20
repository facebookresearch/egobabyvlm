# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text Pipeline module for running Zorro and LT-Swap evaluations together."""

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
class TextPipelineConfig(EvalConfig):
    """Configuration for Text Pipeline that runs Zorro and LT-Swap evaluations."""

    _target_: str = "evaluation.text.pipeline.TextPipeline"

    name: str = "text_pipeline"
    """Name of the pipeline"""

    model: dict[str, Any] = field(default_factory=dict)
    """Model configuration (required). Must have a 'name' field for result organization."""

    tasks: dict[str, Any] = field(default_factory=dict)
    """Mapping from task names to their evaluation configurations"""

    launcher: LauncherConfig = field(default_factory=lambda: LauncherConfig(cluster="slurm"))
    """Stopes launcher configuration."""

    aggregate_results: bool = True
    """Whether to aggregate results into a combined text score."""


class TextPipeline(EvalModule):
    """Pipeline for running Zorro and LT-Swap text evaluations together."""

    def __init__(self, config: TextPipelineConfig) -> None:
        """Initialize the Text Pipeline.

        Args:
            config: Configuration for the pipeline.
        """
        super().__init__(config, TextPipelineConfig)

        if not self.config.model or "name" not in self.config.model:
            raise ValueError("TextPipeline requires a model configuration with a 'name' field.")

        model_name = self.config.model["name"]
        self.results_dir = (
            to_path(self.config.output_dir) / "text_pipeline_results" / model_name / f"seed{self.config.seed}"
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
        """Run all text evaluation tasks asynchronously.

        Returns:
            Dictionary containing results from all evaluations.

        Raises:
            PipelineError: If any task fails (after all tasks complete).
        """
        logger.info("Starting Text Pipeline evaluation")

        task_modules = {}
        for task_name, task_config in self.config.tasks.items():
            logger.info("Building module for task: %s", task_name)
            task_modules[task_name] = EvalModule.build(task_config)

        if not task_modules:
            logger.warning("No tasks configured for text pipeline")
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

            for task_error in task_errors:
                logger.error(
                    "Task %s failed: %s: %s\n%s",
                    task_error.task_name,
                    task_error.error_type,
                    task_error.message,
                    task_error.traceback,
                )

            for task_name in successful_results:
                logger.info("Task %s completed", task_name)

            # If there were errors, raise PipelineError with all error details
            if task_errors:
                raise PipelineError(self.config.name, task_errors)

        return successful_results

    def compute_aggregate_score(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compute aggregate text score from individual results.

        The aggregate score is computed as the equal-weighted mean of:
        - zorro_aggregate / 100 (scaled from 0-100 to 0-1)
        - wordswap_avg_accuracy
        - inflswap_avg_accuracy
        - agrswap_avg_accuracy
        - visual_avg_accuracy

        Args:
            results: Dictionary containing 'zorro' and 'ltswap' results.

        Returns:
            Dictionary containing component scores and aggregate text_score.
        """
        aggregate = {}
        component_scores = []

        if "zorro" in results and "aggregate_score" in results["zorro"]:
            zorro_score = results["zorro"]["aggregate_score"] / 100.0
            aggregate["zorro_score"] = results["zorro"]["aggregate_score"]
            aggregate["zorro_normalized"] = zorro_score
            component_scores.append(zorro_score)
        else:
            logger.warning("Zorro aggregate_score not found in results")

        if "ltswap" in results:
            ltswap_results = results["ltswap"]

            for task_type in ["wordswap", "visual", "agrswap", "inflswap"]:
                if task_type in ltswap_results and "avg_accuracy" in ltswap_results[task_type]:
                    acc = ltswap_results[task_type]["avg_accuracy"]
                    aggregate[f"{task_type}_accuracy"] = acc
                    component_scores.append(acc)
                else:
                    logger.warning("LT-Swap %s avg_accuracy not found in results", task_type)

        if component_scores:
            aggregate["text_score"] = sum(component_scores) / len(component_scores)
            aggregate["num_components"] = len(component_scores)
        else:
            aggregate["text_score"] = None
            aggregate["num_components"] = 0

        return aggregate

    def run(self, iteration_value: Any = None, iteration_index: int = 0) -> dict[str, Any]:
        """Run the Text Pipeline.

        Args:
            iteration_value: Unused, for compatibility with base class.
            iteration_index: Unused, for compatibility with base class.

        Returns:
            Dictionary containing all results including aggregate score.

        Raises:
            PipelineError: If any task fails (re-raised after saving results).
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

            if aggregate.get("text_score") is not None:
                logger.info("Text Pipeline complete. Aggregate text_score: %.4f", aggregate["text_score"])
            else:
                logger.warning("Text Pipeline complete but aggregate text_score could not be computed")

        if pipeline_error:
            raise pipeline_error

        return results
