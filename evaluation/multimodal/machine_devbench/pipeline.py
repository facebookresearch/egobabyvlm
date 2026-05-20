# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MachineDevBench multi-style pipeline."""

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from hydra.utils import instantiate
from omegaconf import OmegaConf
from stopes.core import Launcher, Requirements
from submitit.helpers import clean_env

from core.utils import LauncherConfig, setup_logging, to_yaml
from evaluation.base import PipelineError, process_task_results, to_path
from evaluation.base.eval_module import EvalConfig, EvalModule

from .metrics import ResultAggregator, build_summary, merge_style_results

logger = logging.getLogger(__name__)


@dataclass
class MachineDevBenchPipelineConfig(EvalConfig):
    """Configuration for the multi-style MachineDevBench pipeline."""

    #: Required. Must have a ``name`` field for result organization.
    model: dict[str, Any] = field(default_factory=dict)

    #: One per-style task is launched per entry.
    styles: list[str] = field(default_factory=lambda: ["realistic", "cartoon"])

    #: Template for the per-style :class:`MachineDevBenchEvalModule` config. Must
    #: contain at least ``_target_`` and ``data_root``; the pipeline fills in
    #: ``style``, ``name``, ``model``, ``output_dir`` and ``seed`` per style.
    task_eval_config: dict[str, Any] = field(default_factory=dict)

    launcher: LauncherConfig = field(default_factory=lambda: LauncherConfig(cluster="slurm"))

    #: Whether to aggregate results across styles into ``summary``/``total``.
    aggregate_results: bool = True


class MachineDevBenchPipeline(EvalModule):
    """Coordinator job for running a model over all MachineDevBench styles."""

    def __init__(self, config: MachineDevBenchPipelineConfig) -> None:
        super().__init__(config, MachineDevBenchPipelineConfig)

        if not self.config.model or "name" not in self.config.model:
            raise ValueError("MachineDevBenchPipeline requires a model configuration with a 'name' field.")

        if not self.config.task_eval_config:
            raise ValueError("MachineDevBenchPipeline requires a non-empty 'task_eval_config' template.")

        model_name = self.config.model["name"]
        self.results_dir = (
            to_path(self.config.output_dir) / "machine_devbench_results" / model_name / f"seed{self.config.seed}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        """Lightweight CPU-only coordinator job (mirrors devbench pipeline)."""
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

    def _build_style_config(self, style: str) -> dict[str, Any]:
        """Build a per-style :class:`MachineDevBenchEvalConfig` dict.

        Starts from the user-provided ``task_eval_config`` template and
        injects the style-specific fields plus shared pipeline-level fields
        (model, output_dir, seed).
        """
        # Resolve into a plain dict so we can mutate it freely.
        template = OmegaConf.to_container(OmegaConf.create(self.config.task_eval_config), resolve=True)
        if not isinstance(template, dict):
            raise TypeError(f"'task_eval_config' must be a mapping, got {type(template).__name__}")
        template = cast("dict[str, Any]", template)

        style_cfg: dict[str, Any] = dict(template)
        style_cfg["style"] = style
        # Use pipeline-shared name + style suffix so each per-style module is uniquely named.
        base_name = template.get("name", self.config.name)
        style_cfg["name"] = f"{base_name}_{style}"
        style_cfg.setdefault("seed", self.config.seed)
        style_cfg.setdefault("output_dir", self.config.output_dir)

        # Always force the active model so it stays in sync with pipeline-level model.
        style_cfg["model"] = OmegaConf.to_container(OmegaConf.create(self.config.model), resolve=True)
        return style_cfg

    async def run_tasks(self) -> dict[str, dict[str, Any]]:
        """Schedule one per-style task and gather results.

        Returns:
            Mapping ``{style: {"results": <per-style results>, "raw_records": [...]}}``.

        Raises:
            PipelineError: If any per-style task fails (after all complete).
        """
        logger.info("Starting MachineDevBench pipeline for styles=%s", list(self.config.styles))

        style_modules: dict[str, EvalModule] = {}
        for style in self.config.styles:
            style_cfg = self._build_style_config(style)
            logger.info("Building per-style module: style=%s name=%s", style, style_cfg.get("name"))
            style_modules[style] = EvalModule.build(OmegaConf.create(style_cfg))

        launcher: Launcher = instantiate(self.config.launcher)
        logger.info("Launcher type: %s", type(launcher).__name__)

        with clean_env():
            tasks_to_run = []
            for style, module in style_modules.items():
                logger.info("Scheduling style: %s", style)
                tasks_to_run.append((style, launcher.schedule(module)))

            logger.info("All %d style tasks scheduled, waiting for completion ...", len(tasks_to_run))

            style_names = [s for s, _ in tasks_to_run]
            futures = [f for _, f in tasks_to_run]

            task_results = await asyncio.gather(*futures, return_exceptions=True)

            successful_results, task_errors = process_task_results(style_names, task_results, self.config.name)

            for task_error in task_errors:
                logger.error(
                    "Style task %s failed: %s: %s\n%s",
                    task_error.task_name,
                    task_error.error_type,
                    task_error.message,
                    task_error.traceback,
                )

            for style, result in successful_results.items():
                overall = result.get("results", {}).get("overall", {}).get("accuracy")
                logger.info("Style %s done. Overall acc: %s", style, overall)

            if task_errors:
                raise PipelineError(self.config.name, task_errors)

        return successful_results

    def _aggregate(self, per_style_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build hierarchical results and the compact summary across styles."""
        per_style: dict[str, Any] = {}
        total_aggregator = ResultAggregator()

        for style, payload in per_style_payloads.items():
            per_style[style] = payload.get("results", {})
            for record in payload.get("raw_records", []):
                total_aggregator.add(
                    record["task_name"],
                    int(record["prediction"]),
                    int(record["target"]),
                    record.get("metadata", {}),
                )

        total_results = total_aggregator.compute()
        merged = merge_style_results(per_style, total_results)
        summary = build_summary(merged)
        return {
            "overall": merged["overall"],
            "summary": summary,
            "total": merged["total"],
            "by_style": merged["by_style"],
        }

    def save_aggregate_results(
        self,
        per_style_payloads: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute and persist aggregate results to ``self.results_dir``.

        Writes:
          * ``summary.json`` — compact accuracy summary.
          * ``raw_results.yaml`` — full hierarchical results (overall + total +
            by_style) plus the per-style raw records used for total pooling.
          * ``accuracy_results.csv`` — flat ``style,task,accuracy`` table for
            quick inspection.
        """
        logger.info("Aggregating MachineDevBench results across %d styles", len(per_style_payloads))
        aggregated = self._aggregate(per_style_payloads)
        summary = aggregated["summary"]

        summary_path = self.results_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved to %s", summary_path)

        raw_path = self.results_dir / "raw_results.yaml"
        raw_payload = {
            "overall": aggregated["overall"],
            "total": aggregated["total"],
            "by_style": aggregated["by_style"],
            "raw_records": {style: payload.get("raw_records", []) for style, payload in per_style_payloads.items()},
        }
        with raw_path.open("w") as f:
            f.write(to_yaml(raw_payload))
        logger.info("Raw results saved to %s", raw_path)

        csv_path = self.results_dir / "accuracy_results.csv"
        rows: list[tuple[str, str, float | None]] = []
        for style, style_results in sorted(aggregated["by_style"].items()):
            rows.append((style, "overall", style_results.get("overall", {}).get("accuracy")))
            for type_key, type_info in style_results.get("by_task_type", {}).items():
                rows.append((style, type_key, type_info.get("accuracy")))
            for task_name, task_info in sorted(style_results.get("by_task", {}).items()):
                rows.append((style, task_name, task_info.get("accuracy")))
        # Append pooled-total rows.
        total = aggregated["total"]
        rows.append(("__total__", "overall", total.get("overall", {}).get("accuracy")))
        for type_key, type_info in total.get("by_task_type", {}).items():
            rows.append(("__total__", type_key, type_info.get("accuracy")))
        for task_name, task_info in sorted(total.get("by_task", {}).items()):
            rows.append(("__total__", task_name, task_info.get("accuracy")))

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["style", "task", "accuracy"])
            writer.writerows(rows)
        logger.info("Per-style accuracy CSV saved to %s", csv_path)

        return aggregated

    def run(self, iteration_value: int = 0, iteration_index: int = 0) -> dict[str, Any]:
        """Run the multi-style MachineDevBench pipeline.

        Returns:
            Aggregated results across all styles.

        Raises:
            PipelineError: If any per-style task fails (re-raised after saving).
        """
        setup_logging()

        pipeline_error: PipelineError | None = None
        try:
            per_style_payloads = asyncio.run(self.run_tasks())
        except PipelineError as e:
            per_style_payloads = e.get_results_with_errors({})
            pipeline_error = e

        # Always persist whatever we got, even if some styles failed.
        # Skip failed-style entries (which are dicts of error info, not result payloads).
        valid_payloads: dict[str, dict[str, Any]] = {
            style: payload
            for style, payload in per_style_payloads.items()
            if isinstance(payload, dict) and "results" in payload
        }

        if self.config.aggregate_results and valid_payloads:
            aggregated = self.save_aggregate_results(valid_payloads)
        else:
            aggregated = {"by_style": {style: payload.get("results", {}) for style, payload in valid_payloads.items()}}

        if pipeline_error:
            raise pipeline_error

        return aggregated
