# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Zorro evaluation module for grammatical phenomena evaluation."""

import json
import logging
from dataclasses import dataclass
from typing import Any

import lm_eval
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements

from core.utils import setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.eval_module import EvalConfig, EvalModule

logger = logging.getLogger(__name__)

ZORRO_TASKS = [
    "anaphor_agreement",
    "argument_structure",
    "binding",
    "determiner_noun_agreement",
    "ellipsis",
    "filler_gap",
    "irregular_forms",
    "island_effects",
    "npi_licensing",
    "quantifiers",
    "subject_verb_agreement",
    "case_subjective_pronoun",
    "local_attractor",
]


@dataclass
class ZorroEvalConfig(EvalConfig):
    """Configuration for Zorro grammatical evaluation."""

    _target_: str = "evaluation.text.zorro.ZorroEvalModule"

    #: As dict to support interpolation from shared_model.
    model: dict[str, Any] = MISSING

    #: Path to Zorro data files.
    data_dir: str = MISSING

    #: Data format variant.
    input_str: str = "zorro"

    #: Number of few-shot examples to show the model for each test example.
    num_fewshot: int = 0


class ZorroEvalModule(EvalModule):
    """Evaluation module for Zorro grammatical phenomena.

    Uses the lm_eval library to evaluate on 13 grammatical phenomena from the
    Zorro benchmark, including anaphor agreement, argument structure, binding,
    determiner-noun agreement, ellipsis, filler-gap dependencies, irregular forms,
    island effects, NPI licensing, quantifiers, subject-verb agreement,
    case/subjective pronoun, and local attractors.
    """

    def __init__(self, config: ZorroEvalConfig) -> None:
        """Initialize the Zorro evaluation module.

        Args:
            config: Configuration for the evaluation task.
        """
        super().__init__(config, ZorroEvalConfig)
        self.output_dir = to_path(self.config.output_dir) / "zorro" / self.config.model.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        """Return resource requirements for the job."""
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=8,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        """Return the name of the evaluation job."""
        return f"{self.config.name}_{self.config.model.name}_seed{self.config.seed}"

    def _create_model(self) -> Any:
        """Create the lm_eval model via hydra instantiation.

        Returns:
            The lm_eval model instance.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Creating model %s on %s", self.config.model.name, device)

        return instantiate({"_target_": self.config.model._target_}, **self.config.model.kwargs, device=device)

    def _evaluate_task(self, task_name: str, eval_model: Any, template_name: str = "null_prompt") -> float:
        """Evaluate a single grammatical task.

        Args:
            task_name: Name of the task (e.g., 'anaphor_agreement').
            eval_model: The lm_eval model to evaluate.
            template_name: Template name for the task.

        Returns:
            Accuracy score for the task.
        """
        predictions_dir = self.output_dir / f"zeroshot_{self.config.input_str}" / task_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / "predictions.txt"

        task_file = f"blimp_from_file:{self.config.data_dir}/{task_name}.json"

        logger.info("Evaluating task: %s", task_name)
        eval_task = lm_eval.get_task_list(task_file, template_names=[template_name])
        results = lm_eval.evaluate(
            model=eval_model,
            tasks=eval_task,
            seed=self.config.seed,
            num_fewshot=self.config.num_fewshot,
            predictions_path=str(predictions_path),
        )
        accuracy = results["results"][0]["acc"]

        eval_results_path = predictions_dir / "eval_results.json"
        with eval_results_path.open("w") as f:
            json.dump({"eval_accuracy": accuracy}, f)

        return accuracy

    def run(self, iteration_value: Any = None, iteration_index: int = 0) -> dict[str, Any]:
        """Run the Zorro evaluation pipeline.

        Args:
            iteration_value: Unused, for compatibility with base class.
            iteration_index: Unused, for compatibility with base class.

        Returns:
            Dictionary containing per-task accuracy and aggregate score.
        """
        setup_logging()

        logger.info("Running Zorro evaluation for model: %s", self.config.model.name)

        eval_model = self._create_model()

        accuracies: dict[str, float | None] = {}
        for task in ZORRO_TASKS:
            try:
                accuracy = self._evaluate_task(task, eval_model)
                accuracies[task] = accuracy
                logger.info("%s: %.2f%%", task, accuracy * 100)
            except Exception as e:
                logger.error("Failed to evaluate task %s: %s", task, e)
                accuracies[task] = None

        valid_accuracies = [acc for acc in accuracies.values() if acc is not None]
        aggregate_score = (sum(valid_accuracies) / len(valid_accuracies) * 100) if valid_accuracies else 0.0

        results = {
            **accuracies,
            "aggregate_score": aggregate_score,
        }

        results_path = self.output_dir / "zorro_results.yaml"
        with results_path.open("w") as f:
            f.write(to_yaml(results))

        logger.info("Zorro evaluation complete. Aggregate score: %.2f%%", aggregate_score)
        logger.info("Results saved to %s", results_path)

        return results
