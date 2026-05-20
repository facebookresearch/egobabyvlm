# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MachineDevBench single-style evaluation module."""

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements

from core.protocols import MultiModalFeatureExtractor
from core.utils import set_seed, setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.data.machine_devbench import (
    BenchmarkData,
    MachineDevBenchGrammaticalDataset,
    MachineDevBenchLexicalDataset,
)

from .metrics import ResultAggregator

logger = logging.getLogger(__name__)


@dataclass
class MachineDevBenchEvalConfig(EvalConfig):
    """Configuration for a single-style MachineDevBench evaluation."""

    #: Path to the MachineDevBench data root (e.g. ``data/coco_20260422_123739``).
    data_root: str = MISSING

    #: Image style to evaluate (e.g. ``"realistic"`` or ``"cartoon"``).
    style: str = "realistic"

    #: Optional task filter. ``None`` means all tasks discovered for ``style``.
    tasks: list[str] | None = None

    #: As dict to support interpolation from shared_model.
    model: dict[str, Any] = MISSING

    #: Effectively ignored — per-trial multi-image / multi-text layout forces
    #: ``batch_size=1`` in the dataloader, matching the devbench convention.
    batch_size: int = 32

    num_workers: int = 8
    seed: int = 42


class MachineDevBenchEvalModule(EvalModule):
    """Evaluate a egobabyvlm model on MachineDevBench for a single style.

    Iterates the benchmark's lexical (``lex_*``) and grammatical (``gram_*``)
    tasks for ``config.style`` and produces a structured results dict matching
    the standalone CustomDevBench harness.
    """

    def __init__(self, config: MachineDevBenchEvalConfig) -> None:
        super().__init__(config, MachineDevBenchEvalConfig)

        self.output_dir = (
            to_path(self.config.output_dir) / self.config.name / self.config.style / self.config.model.name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        """Return default Stopes requirements (mirrors the devbench module)."""
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        """Return the unique name of the evaluation job."""
        return f"{self.config.name}_{self.config.style}_{self.config.model.name}_seed{self.config.seed}"

    def _create_model(self, device: str = "cuda") -> MultiModalFeatureExtractor:
        """Instantiate the multimodal feature extractor on ``device``."""
        model: MultiModalFeatureExtractor = instantiate(
            {"_target_": self.config.model._target_}, **self.config.model.kwargs
        )
        model.eval()
        model.to(device)
        return model

    def _build_dataloader(
        self,
        dataset: MachineDevBenchLexicalDataset | MachineDevBenchGrammaticalDataset,
    ) -> EvalDataLoader:
        """Wrap a per-task dataset in an :class:`EvalDataLoader`.

        Uses ``batch_size=1`` whenever the dataset has multiple images or texts
        per trial — matching the convention used by
        :class:`evaluation.multimodal.devbench.base.DevBenchTaskEvalModule`.
        """
        batch_size = 1 if dataset.num_image_cols > 1 or dataset.num_text_cols > 1 else self.config.batch_size
        return EvalDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
        """Move tensor entries in ``batch`` to ``device`` (lists are passed through)."""
        for k, v in batch.items():
            if not isinstance(v, list):
                batch[k] = v.to(device, non_blocking=True)
        return batch

    def _compute_similarity(
        self,
        model: MultiModalFeatureExtractor,
        batch: dict[str, Any],
    ) -> np.ndarray:
        """Run the model on a batch and return a similarity matrix as numpy.

        Args:
            model: Multimodal feature extractor.
            batch: Collated batch with ``image`` (list of PIL images) and
                ``text`` (list of strings) entries.

        Returns:
            Similarity matrix of shape ``(N_images, N_texts)``.
        """
        features = cast("dict[str, torch.Tensor]", model.extract_features(batch))
        similarity = model.compute_similarity(
            features["image_features"],
            features["text_features"],
            normalize=True,
        )
        return similarity.detach().cpu().numpy()

    def _predict_batch(
        self,
        model: MultiModalFeatureExtractor,
        task_name: str,
        batch: dict[str, Any],
    ) -> list[int]:
        """Score the batch and return ``0`` / ``1`` predictions per trial.

        For both task types, ``0`` means the model picked the correct answer.

        Lexical (``num_image_cols=2``, ``num_text_cols=1``)
            ``score(img_pos, caption) > score(img_neg, caption)`` ⇒ ``0``.
        Grammatical (``num_image_cols=2``, ``num_text_cols=2``)
            ``score(img_0, cap_a) + score(img_1, cap_b) >
            score(img_0, cap_b) + score(img_1, cap_a)`` ⇒ ``0``.

        With ``batch_size=1`` (the only mode used here), each call yields a
        single-element list.
        """
        sim = self._compute_similarity(model, batch)

        if task_name.startswith("lex_"):
            # Trial i: image_positive at row 2i, image_negative at 2i+1; caption at column i.
            n_trials = sim.shape[1]
            predictions: list[int] = []
            for i in range(n_trials):
                score_pos = sim[2 * i, i]
                score_neg = sim[2 * i + 1, i]
                predictions.append(0 if score_pos > score_neg else 1)
            return predictions

        # Grammatical: trial i has img_0 at row 2i, img_1 at row 2i+1; cap_a at col 2i, cap_b at col 2i+1.
        n_trials = sim.shape[1] // 2
        predictions = []
        for i in range(n_trials):
            r0, r1 = 2 * i, 2 * i + 1
            c_a, c_b = 2 * i, 2 * i + 1
            matched = sim[r0, c_a] + sim[r1, c_b]
            mismatched = sim[r0, c_b] + sim[r1, c_a]
            predictions.append(0 if matched > mismatched else 1)
        return predictions

    def save_results(
        self,
        results: dict[str, Any],
        raw_predictions: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Persist structured results and raw predictions to ``output_dir``."""
        results_path = self.output_dir / f"{self.config.name}_{self.config.style}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", results_path)

        raw_path = self.output_dir / f"{self.config.name}_{self.config.style}_raw.yaml"
        with raw_path.open("w") as f:
            f.write(to_yaml({"predictions": raw_predictions}))
        logger.info("Raw predictions saved to %s", raw_path)

    def run(self, iteration_value: int = 0, iteration_index: int = 0) -> dict[str, Any]:
        """Run the single-style MachineDevBench evaluation.

        Args:
            iteration_value: Unused, for compatibility with the base class.
            iteration_index: Unused, for compatibility with the base class.

        Returns:
            Dict with keys:
                - ``"results"``: structured results dict (``overall``,
                  ``by_task_type``, ``by_task``) for this style.
                - ``"raw_records"``: list of per-trial records used by the
                  pipeline to compute pooled cross-style statistics.
                  Each record is ``{"task_name", "prediction", "target", "metadata"}``.
        """
        setup_logging()
        logger.info("Running %s evaluation (style=%s)", self.config.name, self.config.style)
        set_seed(self.config.seed)

        benchmark = BenchmarkData(self.config.data_root, style=self.config.style)
        available_tasks = benchmark.get_tasks()
        if self.config.tasks is not None:
            unknown = set(self.config.tasks) - set(available_tasks)
            if unknown:
                raise ValueError(f"Unknown MachineDevBench tasks: {sorted(unknown)}. Available: {available_tasks}")
            task_names = list(self.config.tasks)
        else:
            task_names = available_tasks

        if not task_names:
            logger.warning(
                "No MachineDevBench tasks found under %s for style=%s",
                self.config.data_root,
                self.config.style,
            )
            return {
                "results": {"overall": {"accuracy": 0.0}, "by_task_type": {}, "by_task": {}},
                "raw_records": [],
            }

        logger.info("Tasks to evaluate (%d): %s", len(task_names), task_names)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._create_model(device=device)

        aggregator = ResultAggregator()
        raw_records: list[dict[str, Any]] = []
        raw_predictions_by_task: dict[str, list[dict[str, Any]]] = {}

        for task_name in task_names:
            logger.info("Evaluating task: %s", task_name)
            dataset = benchmark.build_dataset(task_name)
            dataloader = self._build_dataloader(dataset)

            task_records: list[dict[str, Any]] = []
            with torch.no_grad():
                for batch in dataloader:
                    metadata_list = batch.pop("metadata")
                    batch = self._move_batch_to_device(batch, device)
                    predictions = self._predict_batch(model, task_name, batch)
                    for meta, pred in zip(metadata_list, predictions, strict=True):
                        # Correct answer is always index 0 (matches MachineDevBench convention).
                        aggregator.add(task_name, int(pred), 0, meta)
                        record = {"prediction": int(pred), "target": 0, "metadata": meta}
                        task_records.append(record)
                        raw_records.append({"task_name": task_name, **record})
            raw_predictions_by_task[task_name] = task_records

        logger.info("Computing aggregated metrics ...")
        results = aggregator.compute()

        logger.info("Saving results ...")
        self.save_results(results, raw_predictions_by_task)

        logger.info("Evaluation complete. Overall accuracy: %.4f", results["overall"]["accuracy"])
        return {"results": results, "raw_records": raw_records}
