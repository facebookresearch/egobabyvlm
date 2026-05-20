# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DevBench pipeline module for running the full evaluation suite."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from hydra.utils import instantiate
from stopes.core import Launcher, Requirements
from submitit.helpers import clean_env

from core.utils import LauncherConfig, setup_logging, to_yaml
from evaluation.base import PipelineError, process_task_results, to_path
from evaluation.base.eval_module import EvalConfig, EvalModule

logger = logging.getLogger(__name__)


@dataclass
class DevBenchPipelineConfig(EvalConfig):
    """Configuration for DevBench evaluation pipeline."""

    #: Required. Must have a 'name' field for result organization.
    model: dict[str, Any] = field(default_factory=dict)

    #: Maps task names to their evaluation configurations.
    tasks: dict[str, Any] = field(default_factory=dict)

    launcher: LauncherConfig = field(default_factory=lambda: LauncherConfig(cluster="slurm"))

    #: Whether to aggregate results across tasks.
    aggregate_results: bool = True


class DevBenchPipeline(EvalModule):
    """Pipeline for running all DevBench evaluation tasks."""

    def __init__(self, config: DevBenchPipelineConfig) -> None:
        super().__init__(config, DevBenchPipelineConfig)

        if not self.config.model or "name" not in self.config.model:
            raise ValueError("DevBenchPipeline requires a model configuration with a 'name' field.")

        model_name = self.config.model["name"]
        self.results_dir = (
            to_path(self.config.output_dir) / "devbench_results" / model_name / f"seed{self.config.seed}"
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
        return self.config.name

    async def run_tasks(self) -> dict[str, Any]:
        """Run all DevBench tasks asynchronously.

        Returns:
            Dictionary containing results from all successful evaluations.

        Raises:
            PipelineError: If any task fails (after all tasks complete).
        """
        logger.info("Starting DevBench evaluation pipeline")

        task_modules = {}
        for task_name, task_config in self.config.tasks.items():
            logger.info("Building module for task: %s", task_name)
            task_modules[task_name] = EvalModule.build(task_config)

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

            for task_name, result in successful_results.items():
                logger.info("Task %s completed: %s", task_name, result)

            if task_errors:
                raise PipelineError(self.config.name, task_errors)

        return successful_results

    def save_aggregate_results(self, results: dict[str, Any]) -> dict[str, float | None]:
        """Saves aggregate results across all tasks.

        Args:
            results: Dictionary mapping task names to their results
        """
        logger.info("Aggregating DevBench results")

        accuracy_data = []
        similarity_data = []
        agg_acc, acc_total = 0, 0
        sim_agg, sim_total = 0, 0
        for task_name, task_result in results.items():
            if "accuracy" in task_result:
                accuracy_data.append((task_name, task_result["accuracy"]))
                agg_acc += task_result["accuracy"]
                acc_total += 1
            if "kl_divergence" in task_result:
                similarity_data.append((f"{task_name}_kl", task_result["kl_divergence"]))
                sim_agg += 1 / (1 + task_result["kl_divergence"])
                sim_total += 1
            elif "spearman_correlation" in task_result:
                similarity_data.append((f"{task_name}_r", task_result["spearman_correlation"]))
                sim_agg += task_result["spearman_correlation"]
                sim_total += 1

        agg_acc_result: float | None = agg_acc / acc_total if acc_total > 0 else None
        agg_sim: float | None = sim_agg / sim_total if sim_total > 0 else None

        accuracy_data.append(("aggregate", agg_acc_result))
        similarity_data.append(("aggregate", agg_sim))

        acc_results_path = str(self.results_dir / "accuracy_results.csv")
        acc_df = pd.DataFrame(accuracy_data, columns=["task", "accuracy"])
        with Path(acc_results_path).open("w") as f:
            acc_df.to_csv(f, index=False)
        logger.info("Aggregated accuracy results saved to %s", acc_results_path)

        sim_results_path = str(self.results_dir / "similarity_results.csv")
        sim_df = pd.DataFrame(similarity_data, columns=["task", "similarity"])
        with Path(sim_results_path).open("w") as f:
            sim_df.to_csv(f, index=False)
        logger.info("Aggregated similarity results saved to %s", sim_results_path)

        return {"accuracy": agg_acc_result, "similarity": agg_sim}

    def run(self, iteration_value: int = 0, iteration_index: int = 0) -> dict[str, Any]:
        """Run the DevBench pipeline.

        Returns:
            Aggregated results across all tasks

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
            results["aggregate"] = self.save_aggregate_results(results)

        if pipeline_error:
            raise pipeline_error

        return results
