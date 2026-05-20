# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base classes for multimodal evaluation modules."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import MISSING
from rds2py import read_rds
from scipy import io
from scipy.stats import spearmanr
from stopes.core import Requirements

from core.protocols import MultiModalFeatureExtractor
from core.utils import set_seed, setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig
from evaluation.data.devbench import DevBenchDataset

if TYPE_CHECKING:
    import torch

from .metrics import (
    compute_accuracy,
    compute_rsm_correlation,
    get_optimal_kl_divergence,
)

logger = logging.getLogger(__name__)


@dataclass
class DevBenchTaskEvalConfig(EvalConfig):
    """Base configuration for multimodal evaluation modules."""

    dataset: EvalDatasetConfig = MISSING

    #: As dict to support interpolation from shared_model.
    model: dict[str, Any] = MISSING

    batch_size: int = 256
    num_workers: int = 8
    seed: int = 42

    #: Path to human performance data for KL divergence/RSM computation.
    human_data_path: str | None = None


class DevBenchTaskEvalModule(EvalModule, ABC):
    """Base class for multimodal evaluation modules following the DevBench pattern."""

    def __init__(self, config: DevBenchTaskEvalConfig) -> None:
        """Initialize the evaluation module.

        Args:
            config: Configuration for the evaluation task.
        """
        super().__init__(config, DevBenchTaskEvalConfig)

        self.output_dir = (
            to_path(self.config.output_dir) / self.config.name / self.config.dataset.name / self.config.model.name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        """Return default requirements for multimodal evaluation jobs.

        Returns:
            Requirements object with resource specifications.
        """
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        """Return the name of the evaluation job.

        Returns:
            Formatted name combining task, dataset, and model names.
        """
        return f"{self.config.name}_{self.config.dataset.name}_{self.config.model.name}_seed{self.config.seed}"

    def _create_model(self, device: str = "cuda") -> MultiModalFeatureExtractor:
        """Create and return the feature extractor, using the MultiModalFeatureExtractor interface.

        Args:
            device: Device to place the model on (default: "cuda").

        Returns:
            Feature extractor.
        """
        model: MultiModalFeatureExtractor = instantiate(
            {"_target_": self.config.model._target_}, **self.config.model.kwargs
        )
        model.eval()
        model.to(device)
        return model

    def _get_dataloader(self, *, shuffle: bool = False) -> EvalDataLoader:
        """Create dataloader for the dataset.

        Args:
            shuffle: Whether to shuffle the dataset.

        Returns:
            EvalDataLoader instance.
        """
        dataset: DevBenchDataset = instantiate(
            {"_target_": self.config.dataset._target_}, **self.config.dataset.kwargs
        )

        # Use batch size of 1 if dataset has multiple images or texts per trial
        batch_size = 1 if dataset.num_image_cols > 1 or dataset.num_text_cols > 1 else self.config.batch_size

        return EvalDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    @abstractmethod
    def compute_metrics(self, results: np.ndarray, dataset: DevBenchDataset | None = None) -> dict[str, Any]:
        """Compute task-specific metrics from results.

        Args:
            results: Task-specific results (e.g., similarity scores, embeddings).
            dataset: Optional dataset instance for accessing manifest data.

        Returns:
            Dictionary of computed metrics.
        """

    def save_results(self, results: np.ndarray, metrics: dict[str, Any]) -> None:
        """Save results and metrics to disk.

        Args:
            results: Raw results to save (e.g., embeddings, scores).
            metrics: Computed metrics to save.
        """
        results_path = str(self.output_dir / f"{self.config.name}_{self.config.model.name}.npy")
        with Path(results_path).open("wb") as f:
            np.save(f, results)

        metrics_path = str(self.output_dir / f"{self.config.name}_{self.config.model.name}_metrics.yaml")
        with Path(metrics_path).open("w") as f:
            f.write(to_yaml(metrics))

    def run(
        self,
        iteration_value: int = 0,
        iteration_index: int = 0,
    ) -> dict[str, Any]:
        """Run the evaluation pipeline.

        Args:
            iteration_value: Unused, for compatibility with base class.
            iteration_index: Unused, for compatibility with base class.

        Returns:
            Dictionary containing computed metrics.
        """
        setup_logging()

        logger.info("Running %s evaluation", self.config.name)

        set_seed(self.config.seed)

        device = "cuda"
        model = self._create_model(device=device)
        dataloader = self._get_dataloader(shuffle=False)

        logger.info("Extracting similarity scores...")

        all_similarity_scores = []
        for batch in dataloader:
            # Move non-list items to device
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device, non_blocking=True)

            batch_features = cast("dict[str, torch.Tensor]", model.extract_features(batch))
            similarity_scores = model.compute_similarity(
                batch_features["image_features"],
                batch_features["text_features"],
                normalize=True,
            )
            all_similarity_scores.append(similarity_scores.detach().cpu().numpy())

        try:
            similarity_scores_np: np.ndarray = np.stack(all_similarity_scores, axis=0)
        except ValueError:
            logger.warning("Falling back to concatenation for similarity scores")
            similarity_scores_np = np.concatenate(all_similarity_scores, axis=0)

        logger.info("Computing metrics...")
        metrics = self.compute_metrics(similarity_scores_np, dataset=cast("DevBenchDataset", dataloader.dataset))

        logger.info("Saving results...")
        self.save_results(similarity_scores_np, metrics)

        logger.info("Evaluation complete. Metrics: %s", metrics)
        return metrics


class DevBenchGramTROGEvalModule(DevBenchTaskEvalModule):
    """TROG evaluation module for grammatical task evaluation."""

    def _squeeze_similarity_scores(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Squeeze similarity scores if they have an extra dimension.

        Args:
            similarity_scores: Array that may have shape (n_trials, n_images, 1).

        Returns:
            Array with shape (n_trials, n_images).
        """
        if similarity_scores.ndim == 3 and similarity_scores.shape[2] == 1:
            return similarity_scores.squeeze(axis=2)
        return similarity_scores

    def _load_human_data(self, human_path: Path) -> np.ndarray:
        """Load and normalize human probability data from CSV.

        Args:
            human_path: Path to CSV file containing human data.

        Returns:
            Normalized probability array.
        """
        with Path(str(human_path)).open("r") as f:
            human_data_df = pd.read_csv(f)
        human_cols = [col for col in human_data_df.columns if col.startswith("image")]
        human_probs = human_data_df[human_cols].to_numpy()
        return human_probs / np.sum(human_probs, axis=1, keepdims=True)

    def compute_metrics(
        self,
        similarity_scores: np.ndarray,
        dataset: DevBenchDataset | None = None,
    ) -> dict[str, Any]:
        """Compute accuracy and KL divergence for grammatical or lexical tasks.

        Args:
            similarity_scores: Array of shape (n_trials, n_images, n_texts).
            dataset: Optional dataset instance (not used in this implementation).

        Returns:
            Dictionary containing accuracy and KL metrics.
        """
        scores = self._squeeze_similarity_scores(similarity_scores)
        accuracy = compute_accuracy(scores, correct_index=0)

        metrics = {
            "accuracy": accuracy,
            "num_trials": len(scores),
        }

        if self.config.human_data_path is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    human_probs = self._load_human_data(human_path)
                    kl_result = get_optimal_kl_divergence(human_probs, scores)
                    metrics.update(
                        {
                            "kl_divergence": kl_result["kl_divergence"],
                            "optimal_beta": kl_result["optimal_beta"],
                        }
                    )
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("Could not compute KL divergence: %s", e)

        return metrics


class DevBenchGramWinogroundEvalModule(DevBenchTaskEvalModule):
    """Winoground evaluation module."""

    def _compute_winoground_accuracy(self, similarity_scores: np.ndarray) -> float:
        """Compute accuracy for Winoground task.

        Args:
            similarity_scores: Array of shape (n_trials, 2, 2).

        Returns:
            Accuracy as a float between 0 and 1.
        """
        correct_count = 0
        for i in range(len(similarity_scores)):
            # Check if caption_0 matches image_0 better than image_1
            if similarity_scores[i, 0, 0] > similarity_scores[i, 0, 1]:
                correct_count += 1
            # Check if caption_1 matches image_1 better than image_0
            if similarity_scores[i, 1, 0] < similarity_scores[i, 1, 1]:
                correct_count += 1

        return correct_count / (len(similarity_scores) * 2)

    def _load_human_results(self, human_path: Path, model_trials: set[int]) -> np.ndarray:
        """Load and filter human results from JSONL file.

        Args:
            human_path: Path to JSONL file with human results.
            model_trials: Set of trial indices to filter for.

        Returns:
            Array of human probabilities.
        """
        with Path(str(human_path)).open("r") as f:
            human_results = [json.loads(line) for line in f]

        filtered_results = [res for res in human_results if int(res["label"].split("_")[0]) in model_trials]

        human_probs = []
        for i in range(0, len(filtered_results), 4):
            scores = np.array([item["score"] for item in filtered_results[i : i + 4]])
            human_probs.append(scores / np.sum(scores))

        return np.array(human_probs)

    def compute_metrics(self, similarity_scores: np.ndarray, dataset: DevBenchDataset | None = None) -> dict[str, Any]:
        """Compute accuracy and KL divergence for Winoground task.

        Args:
            similarity_scores: Array of shape (n_trials, 2, 2) where
                [i, 0, 0] = score(caption_0, image_0),
                [i, 0, 1] = score(caption_0, image_1),
                [i, 1, 0] = score(caption_1, image_0),
                [i, 1, 1] = score(caption_1, image_1).
            dataset: Dataset instance for accessing manifest trial information.

        Returns:
            Dictionary containing accuracy and KL metrics.
        """
        accuracy = self._compute_winoground_accuracy(similarity_scores)

        metrics = {
            "accuracy": accuracy,
            "num_trials": len(similarity_scores),
        }

        if self.config.human_data_path is not None and dataset is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    manifest_df = dataset.manifest
                    manifest_df["trial_idx"] = manifest_df["image1"].str.extract(r"ex_(\d+)_img")
                    model_trials = set(manifest_df["trial_idx"].astype(int))

                    human_probs = self._load_human_results(human_path, model_trials)
                    model_logits = similarity_scores.reshape(len(similarity_scores), -1)

                    kl_result = get_optimal_kl_divergence(human_probs, model_logits)
                    metrics.update(
                        {
                            "kl_divergence": kl_result["kl_divergence"],
                            "optimal_beta": kl_result["optimal_beta"],
                        }
                    )
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("Could not compute KL divergence: %s", e)

        return metrics


class DevBenchLexLWLEvalModule(DevBenchTaskEvalModule):
    """LWL (Looking While Listening) evaluation module.

    Uses proportion-based human data for KL divergence computation.
    Computes KL divergence separately per age bin, matching the reference DevBench implementation.
    """

    def _squeeze_similarity_scores(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Squeeze similarity scores if they have an extra dimension.

        Args:
            similarity_scores: Array that may have shape (n_trials, n_options, 1).

        Returns:
            Array with shape (n_trials, n_options).
        """
        if similarity_scores.ndim == 3 and similarity_scores.shape[2] == 1:
            return similarity_scores.squeeze(axis=2)
        return similarity_scores

    def compute_metrics(
        self,
        similarity_scores: np.ndarray,
        dataset: DevBenchDataset | None = None,
    ) -> dict[str, Any]:
        """Compute accuracy and KL divergence for LWL task.

        Aligns human data to model scores by trial number and computes
        KL divergence separately per age bin (each with its own optimal beta).

        Args:
            similarity_scores: Array of shape (n_trials, 2) where
                [:, 0] = score for correct option,
                [:, 1] = score for incorrect option.
            dataset: Optional dataset instance (not used in this implementation).

        Returns:
            Dictionary containing accuracy and KL metrics.
        """
        scores = self._squeeze_similarity_scores(similarity_scores)
        accuracy = compute_accuracy(scores, correct_index=0)

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "num_trials": len(scores),
        }

        if self.config.human_data_path is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    self._compute_kl_metrics(scores, human_path, metrics)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("Could not compute KL divergence: %s", e)

        return metrics

    def _compute_kl_metrics(
        self,
        scores: np.ndarray,
        human_path: Path,
        metrics: dict[str, Any],
    ) -> None:
        """Compute per-age-bin KL divergence, aligning human data by trial number.

        Args:
            scores: Model similarity scores, shape (n_trials, 2), ordered by manifest row.
            human_path: Path to human CSV with columns: age_bin, prop, trial.
            metrics: Dictionary to update with KL metrics.
        """
        with Path(str(human_path)).open("r") as f:
            human_data_df = pd.read_csv(f)

        model_df = pd.DataFrame(
            {"image1": scores[:, 0], "image2": scores[:, 1], "trial": np.arange(1, len(scores) + 1)},
        )

        age_bins = sorted(human_data_df["age_bin"].unique())
        age_kl_divs = []

        for age in age_bins:
            age_human = human_data_df[human_data_df["age_bin"] == age]
            # Join by trial number
            merged = age_human.merge(model_df, on="trial")
            if len(merged) == 0:
                continue

            human_probs = np.column_stack([merged["prop"].values, 1 - merged["prop"].values])
            model_logits = merged[["image1", "image2"]].to_numpy()

            kl_result = get_optimal_kl_divergence(human_probs, model_logits)
            age_kl_divs.append(kl_result["kl_divergence"])
            metrics[f"kl_divergence_age_{age}"] = kl_result["kl_divergence"]
            metrics[f"optimal_beta_age_{age}"] = kl_result["optimal_beta"]

        if age_kl_divs:
            metrics["kl_divergence"] = float(np.mean(age_kl_divs))


class DevBenchLexVizVocabEvalModule(DevBenchTaskEvalModule):
    """VizVocab evaluation module.

    Computes KL divergence separately per age bin, matching the reference DevBench implementation.
    """

    def _squeeze_similarity_scores(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Squeeze similarity scores if they have an extra dimension.

        Args:
            similarity_scores: Array that may have shape (n_trials, n_images, 1).

        Returns:
            Array with shape (n_trials, n_images).
        """
        if similarity_scores.ndim == 3 and similarity_scores.shape[2] == 1:
            return similarity_scores.squeeze(axis=2)
        return similarity_scores

    def compute_metrics(self, similarity_scores: np.ndarray, dataset: DevBenchDataset | None = None) -> dict[str, Any]:
        """Compute accuracy and KL divergence for VizVocab task.

        Args:
            similarity_scores: Array of shape (n_trials, 4) where each row contains
                scores for [image1, image2, image3, image4] with image1 being correct.
            dataset: Dataset instance for accessing manifest data.

        Returns:
            Dictionary containing accuracy and KL metrics.
        """
        scores = self._squeeze_similarity_scores(similarity_scores)
        accuracy = compute_accuracy(scores, correct_index=0)

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "num_trials": len(scores),
        }

        if self.config.human_data_path is not None and dataset is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    self._compute_kl_metrics(scores, human_path, dataset, metrics)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("Could not compute KL divergence: %s", e)

        return metrics

    def _get_trial_indices(self, all_labels: np.ndarray, target_words: list[str]) -> list[int]:
        """Find trial indices that match target words in the manifest.

        Args:
            all_labels: Array of text labels from manifest.
            target_words: List of words to match.

        Returns:
            List of trial indices.
        """
        trial_indices = []
        for word in target_words:
            matches = np.where(all_labels == word)[0]
            if len(matches) > 0:
                trial_indices.extend(matches.tolist())
        return trial_indices

    def _compute_kl_metrics(
        self,
        scores: np.ndarray,
        human_path: Path,
        dataset: DevBenchDataset,
        metrics: dict[str, Any],
    ) -> None:
        """Compute per-age-bin KL divergence metrics.

        Computes KL divergence separately for each age bin (each with its own optimal beta),
        then sets the primary kl_divergence to the mean across age bins.

        Args:
            scores: Model similarity scores.
            human_path: Path to human data file.
            dataset: Dataset instance for accessing manifest.
            metrics: Dictionary to update with metrics.
        """
        with Path(str(human_path)).open("r") as f:
            human_data_df = pd.read_csv(f)
        all_labels = cast("np.ndarray", dataset.manifest["text1"].values)

        age_bins = sorted(human_data_df["age_bin"].unique())
        age_kl_divs = []

        for age in age_bins:
            age_data = human_data_df[human_data_df["age_bin"] == age]
            age_trial_indices = self._get_trial_indices(all_labels, age_data["text1"].values.tolist())

            if len(age_trial_indices) == 0:
                continue

            age_filtered_scores = scores[age_trial_indices]
            age_cols = [col for col in age_data.columns if col.startswith("image")]
            age_probs = age_data[age_cols].to_numpy()
            age_probs = age_probs / np.sum(age_probs, axis=1, keepdims=True)

            if len(age_probs) == len(age_filtered_scores):
                age_kl = get_optimal_kl_divergence(age_probs, age_filtered_scores)
                age_kl_divs.append(age_kl["kl_divergence"])
                metrics[f"kl_divergence_age_{age}"] = age_kl["kl_divergence"]
                metrics[f"optimal_beta_age_{age}"] = age_kl["optimal_beta"]

        if age_kl_divs:
            metrics["kl_divergence"] = float(np.mean(age_kl_divs))


class DevBenchSemEvalModule(DevBenchTaskEvalModule, ABC):
    """Base class for semantic evaluation tasks.

    Uses embeddings and RSM (Representational Similarity Matrix) correlation.
    """

    def run(
        self,
        iteration_value: int = 0,
        iteration_index: int = 0,
    ) -> dict[str, Any]:
        """Run evaluation by extracting embeddings (not similarity scores).

        Args:
            iteration_value: Unused, for compatibility with base class.
            iteration_index: Unused, for compatibility with base class.

        Returns:
            Dictionary containing computed metrics.
        """
        setup_logging()

        logger.info("Running %s evaluation", self.config.name)

        set_seed(self.config.seed)

        device = "cuda"
        model = self._create_model(device=device)
        dataloader = self._get_dataloader(shuffle=False)

        logger.info("Extracting image embeddings...")

        all_embeddings = []
        for batch in dataloader:
            # Move non-list items to device
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device, non_blocking=True)

            batch_features = cast("dict[str, torch.Tensor]", model.extract_features(batch))
            image_embeddings = batch_features["image_features"]
            all_embeddings.append(image_embeddings.cpu().numpy())

        embeddings_np = np.concatenate(all_embeddings, axis=0)

        logger.info("Computing metrics...")
        metrics = self.compute_metrics(embeddings_np, dataset=cast("DevBenchDataset", dataloader.dataset))

        logger.info("Saving results...")
        self.save_results(embeddings_np, metrics)

        logger.info("Evaluation complete. Metrics: %s", metrics)
        return metrics

    @abstractmethod
    def _load_human_similarity(self, path: Path) -> np.ndarray | list[np.ndarray]:
        """Load human similarity data.

        To be overridden by subclasses based on data format.

        Args:
            path: Path to human similarity data file.

        Returns:
            Human similarity matrix or list of matrices.
        """


class DevBenchSemThingsEvalModule(DevBenchSemEvalModule):
    """THINGS Similarity evaluation module."""

    def compute_metrics(
        self,
        embeddings: np.ndarray,
        dataset: DevBenchDataset | None = None,
    ) -> dict[str, Any]:
        """Compute RSM correlation for semantic tasks.

        Args:
            embeddings: Array of shape (n_items, embedding_dim).
            dataset: Optional dataset instance (not used in this implementation).

        Returns:
            Dictionary containing RSM correlation metrics.
        """
        metrics = {
            "num_items": len(embeddings),
            "embedding_dim": embeddings.shape[1],
        }

        if self.config.human_data_path is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    human_similarity = self._load_human_similarity(human_path)
                    rsm_result = compute_rsm_correlation(embeddings, human_similarity)
                    metrics.update(rsm_result)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning("Could not compute RSM correlation: %s", e)

        return metrics

    def _load_human_similarity(self, path: Path) -> np.ndarray:
        """Load THINGS human similarity from .mat file.

        Args:
            path: Path to .mat file containing human similarity data.

        Returns:
            Human similarity matrix.
        """
        with Path(str(path)).open("rb") as f:
            mat_data = io.loadmat(f)
        return mat_data["spose_sim"]


class DevBenchSemVizObjCatEvalModule(DevBenchSemEvalModule):
    """Visual Object Categorisation evaluation module.

    Uses matrix reduction and multi-RSM comparison for evaluation.
    """

    def compute_metrics(
        self,
        embeddings: np.ndarray,
        dataset: DevBenchDataset | None = None,
    ) -> dict[str, Any]:
        """Compute RSM correlation for Visual Object Categorisation.

        VOC has special requirements:
        - Reduces model RSM from 72x72 to 8x8 by averaging 9x9 blocks.
        - Compares against 3 separate human 8x8 RSMs.
        - Returns average correlation across all 3 human RSMs.

        Args:
            embeddings: Array of shape (72, embedding_dim) for 72 images
                (8 categories x 9 exemplars).
            dataset: Optional dataset instance (not used in this implementation).

        Returns:
            Dictionary containing averaged RSM correlation metrics.
        """
        metrics = {
            "num_items": len(embeddings),
            "embedding_dim": embeddings.shape[1],
        }

        if self.config.human_data_path is not None:
            human_path = Path(self.config.human_data_path)
            if human_path.exists():
                try:
                    human_rsms = self._load_human_similarity(human_path)

                    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    model_rsm_full = np.dot(normalized, normalized.T)  # 72x72
                    model_rsm_reduced = self._reduce_rsm(model_rsm_full)

                    correlations = []
                    p_values = []

                    for human_rsm in human_rsms:
                        tril_indices = np.tril_indices(model_rsm_reduced.shape[0], k=-1)

                        model_values = model_rsm_reduced[tril_indices]
                        human_values = human_rsm[tril_indices]

                        corr, p_val = spearmanr(model_values, human_values)
                        correlations.append(corr)
                        p_values.append(p_val)

                    metrics.update(
                        {
                            "spearman_correlation": float(np.mean(correlations)),
                            "p_value": float(np.mean(p_values)),
                            "num_human_rsms": len(human_rsms),
                            "individual_correlations": [float(c) for c in correlations],
                            "individual_p_values": [float(p) for p in p_values],
                        }
                    )

                except (OSError, ValueError, KeyError, ImportError) as e:
                    logger.warning("Could not compute RSM correlation: %s", e)

        return metrics

    def _reduce_rsm(self, rsm: np.ndarray, block_size: int = 9) -> np.ndarray:
        """Reduce RSM by averaging blocks.

        Args:
            rsm: Full RSM of shape (72, 72).
            block_size: Size of blocks to average (default: 9).

        Returns:
            Reduced RSM of shape (8, 8).
        """
        n_blocks = rsm.shape[0] // block_size
        reduced = np.zeros((n_blocks, n_blocks))

        for i in range(n_blocks):
            for j in range(n_blocks):
                block = rsm[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]
                reduced[i, j] = np.mean(block)

        return reduced

    def _load_human_similarity(self, path: Path) -> list[np.ndarray]:
        """Load VOC human similarity data from .rds file.

        Args:
            path: Path to .rds file containing human similarity data.

        Returns:
            List of 3 human RSM matrices, each of shape (8, 8).
        """
        human = read_rds(str(path))
        data = human["data"]

        human_rsms = []
        for dat in data[1]["data"]:
            mat = dat["data"].reshape((dat["attributes"]["dim"]["data"][0], dat["attributes"]["dim"]["data"][1]))
            human_rsms.append(mat)

        return human_rsms
