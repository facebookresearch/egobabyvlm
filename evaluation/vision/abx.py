# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements
from tqdm import tqdm

if TYPE_CHECKING:
    from evaluation.data.base import ClassificationMediaDataset

from core.modeling import freeze
from core.protocols import ImageFeatureExtractor
from core.utils import set_seed, setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig

logger = logging.getLogger(__name__)


def _classification_collate_fn(
    samples: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for classification datasets."""
    images_tup, labels_tup = zip(*samples, strict=True)
    images = torch.stack(images_tup)
    labels = torch.tensor(labels_tup)
    return images, labels


@dataclass
class ABXEvalModuleConfig(EvalConfig):
    _target_: str = "evaluation.vision.abx.ABXEvalModule"

    name: str = "abx"
    dataset: EvalDatasetConfig = MISSING

    #: As dict to support interpolation from shared_backbone.
    model: dict[str, Any] = MISSING

    #: Pooling strategy (cls, mean_patch, etc.). If None, uses model default.
    #: Set independently of model config when using a shared backbone in pipelines.
    pooling: str | None = None

    #: Number of layers to use. If None, uses model default.
    last_n_layers: int | None = None

    #: Number of ABX triplets per confusion matrix cell.
    num_triplets: int = 50

    batch_size: int = 512
    num_workers: int = 4

    #: Maximum samples to process (for memory efficiency).
    max_samples: int | None = None

    #: Number of classes to randomly sample for evaluation.
    subset: int | None = None

    seed: int = 42


@dataclass
class PrecomputedFeatures:
    """Container for precomputed features with efficient indexing."""

    features: torch.Tensor  # Shape: (total_samples, feature_dim)
    class_to_indices: dict[int, torch.Tensor]  # class -> tensor of indices
    index_to_class: torch.Tensor  # index -> class label
    device: torch.device

    def get_class_features(self, class_idx: int) -> torch.Tensor:
        """Get all features for a specific class."""
        indices = self.class_to_indices[class_idx]
        return self.features[indices]

    def get_features_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get features by their indices."""
        return self.features[indices]


def sample_triplets_vectorized(
    precomputed: PrecomputedFeatures, class_x: int, class_b: int, num_triplets: int, *, symmetric: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized triplet sampling for a confusion matrix cell.
    Returns tensors of indices for X, A, B samples.
    """
    x_indices = precomputed.class_to_indices[class_x]
    b_indices = precomputed.class_to_indices[class_b]

    if len(x_indices) < 2:
        # Not enough samples for this class
        return torch.empty(0), torch.empty(0), torch.empty(0)

    # Sample triplets efficiently
    sampled_x = []
    sampled_a = []
    sampled_b = []

    for _ in range(num_triplets):
        # Sample X and A from same class
        if symmetric:
            # For symmetric case, ensure X != A
            if len(x_indices) >= 2:
                xa_pair = torch.randperm(len(x_indices))[:2]
                sampled_x.append(x_indices[xa_pair[0]])
                sampled_a.append(x_indices[xa_pair[1]])
            else:
                continue
        else:
            sampled_x.append(x_indices[torch.randint(len(x_indices), (1,))])
            sampled_a.append(x_indices[torch.randint(len(x_indices), (1,))])

        # Sample B from different class
        sampled_b.append(b_indices[torch.randint(len(b_indices), (1,))])

    if not sampled_x:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    return (torch.cat(sampled_x), torch.cat(sampled_a), torch.cat(sampled_b))


def evaluate_cell_vectorized(
    precomputed: PrecomputedFeatures, class_x: int, class_b: int, num_triplets: int, *, symmetric: bool = False
) -> tuple[float, int]:
    """
    Evaluate entire confusion matrix cell at once using vectorized operations.
    Returns accuracy and number of triplets evaluated.
    """
    # Sample triplets
    x_indices, a_indices, b_indices = sample_triplets_vectorized(
        precomputed, class_x, class_b, num_triplets, symmetric=symmetric
    )

    if len(x_indices) == 0:
        return 0.0, 0

    feat_x = precomputed.get_features_by_indices(x_indices)  # (N, D)
    feat_a = precomputed.get_features_by_indices(a_indices)  # (N, D)
    feat_b = precomputed.get_features_by_indices(b_indices)  # (N, D)

    dist_xa = 1 - torch.sum(feat_x * feat_a, dim=1)  # (N,)
    dist_xb = 1 - torch.sum(feat_x * feat_b, dim=1)  # (N,)

    correct = (dist_xa < dist_xb).sum().item()
    total = len(x_indices)

    return correct / total if total > 0 else 0.0, total


def plot_distance_distributions(results: dict, save_path: Path) -> None:
    """Plot results summary."""
    plt.figure(figsize=(12, 4))

    # Plot 1: Confusion matrix
    plt.subplot(1, 2, 1)
    confusion_matrix = results["confusion_matrix"]
    mask = confusion_matrix > 0
    plt.imshow(confusion_matrix, cmap="Blues", aspect="auto")
    plt.colorbar(label="ABX Accuracy")
    plt.title("ABX Confusion Matrix")
    plt.xlabel("Class B (Different)")
    plt.ylabel("Class X (Same as A)")

    # Plot 2: Accuracy distribution
    plt.subplot(1, 2, 2)
    valid_accuracies = confusion_matrix[mask].flatten()
    if len(valid_accuracies) > 0:
        plt.hist(valid_accuracies, bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(
            np.mean(valid_accuracies), color="red", linestyle="--", label=f"Mean: {np.mean(valid_accuracies):.3f}"
        )
        plt.xlabel("ABX Accuracy")
        plt.ylabel("Number of Cells")
        plt.title("Distribution of Cell Accuracies")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    results: dict,
    save_path: Path,
    dataset_name: str = "dataset",
    class_names: list[str] | None = None,
) -> None:
    """Plot confusion matrix heatmap."""
    confusion_matrix = results["confusion_matrix"]
    num_classes = results["num_classes"]

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Limit size for readability
    if num_classes > 50:
        logger.info("Too many classes (%d) to display clearly. Showing summary instead.", num_classes)
        plot_distance_distributions(results, save_path)
        return

    fig_size = (max(8, num_classes * 0.4), max(8, num_classes * 0.4))
    plt.figure(figsize=fig_size)

    mask = confusion_matrix > 0
    sns.heatmap(
        confusion_matrix,
        annot=num_classes <= 20,  # Only annotate if not too many classes
        fmt=".3f",
        annot_kws={"size": 7},
        cmap="Blues",
        cbar_kws={"label": "ABX Accuracy"},
        square=True,
        mask=~mask,  # Mask zero values
        xticklabels=class_names if num_classes <= 20 else False,
        yticklabels=class_names if num_classes <= 20 else False,
    )

    plt.title(f"ABX Confusion Matrix - {dataset_name}\n(Accuracy: {results['accuracy']:.3f})")
    plt.xlabel("Class B (Different Class)")
    plt.ylabel("Class X (Same Class as A)")

    if num_classes <= 20:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class ABXEvalModule(EvalModule):
    def __init__(self, config: ABXEvalModuleConfig) -> None:
        super().__init__(config, ABXEvalModuleConfig)

        self.output_dir = to_path(self.config.output_dir) / self.name().replace("_", "/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def hparam_save_name(self) -> str:
        hparam_save_name = f"triplets{self.config.num_triplets}"
        if self.config.subset is not None:
            hparam_save_name += f"_subset{self.config.subset}"
        if self.config.max_samples is not None:
            hparam_save_name += f"_maxsamples{self.config.max_samples}"
        return hparam_save_name

    def requirements(self) -> Requirements:
        """Requirements for each submitted job"""
        return Requirements(
            nodes=1,
            mem_gb=140,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        return f"{self.config.name}_{self.config.dataset.name}_{self.config.model.name}_{self.hparam_save_name}_seed{self.config.seed}"

    def _create_model(self, device: str = "cuda") -> ImageFeatureExtractor:
        logger.info("Creating model")

        model: ImageFeatureExtractor = instantiate(
            {"_target_": self.config.model._target_}, **self.config.model.kwargs
        )

        # Apply pooling strategy if specified in config
        # This allows overriding the model's default pooling when using a shared backbone
        if self.config.pooling is not None or self.config.last_n_layers is not None:
            if hasattr(model, "set_pooling_strategy"):
                model.set_pooling_strategy(
                    pooling=self.config.pooling,
                    last_n_layers=self.config.last_n_layers,
                )
                logger.info(
                    "Applied pooling strategy: pooling=%s, last_n_layers=%s",
                    self.config.pooling,
                    self.config.last_n_layers,
                )
            else:
                logger.warning("Model does not support set_pooling_strategy, ignoring pooling config")

        freeze(model)
        model.eval()
        model.to(device)

        return model

    def _get_dataloader(self) -> EvalDataLoader:
        dataset: ClassificationMediaDataset = instantiate(
            {"_target_": self.config.dataset._target_}, **self.config.dataset.kwargs
        )

        logger.info("Dataset loaded: %d samples", len(dataset))

        if self.config.max_samples is not None:
            dataset = dataset.subset_from_max_samples(self.config.max_samples, rng=random.Random(self.config.seed))
            logger.info("Subsampled dataset to %d samples", len(dataset))

        if self.config.subset is not None:
            dataset = dataset.subset_from_subset_classes(self.config.subset, rng=random.Random(self.config.seed))

        return EvalDataLoader(
            dataset,
            collate_fn=_classification_collate_fn,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def _precompute_all_features(
        self, model: ImageFeatureExtractor, dataloader: EvalDataLoader, device: str = "cuda"
    ) -> PrecomputedFeatures:
        """
        Precompute all features at once - this is the key optimization!
        Similar to fastabx's approach of loading all data into memory.
        """

        logger.info("Precomputing features...")

        all_features = []
        all_labels = []

        for batch in tqdm(dataloader):
            batch_images, batch_labels = batch
            if batch_images.shape[0] == 0:
                continue

            batch_images = batch_images.to(device)

            with torch.inference_mode():
                features = model.extract_features(batch_images)
                features = torch.nn.functional.normalize(features, dim=-1)

            all_features.append(features.cpu())
            all_labels.append(batch_labels)

        features = torch.cat(all_features, dim=0).to(device)
        labels = torch.cat(all_labels, dim=0).to(device)

        unique_classes = torch.unique(labels).cpu().tolist()
        class_to_indices = {}

        for class_idx in unique_classes:
            mask = labels == class_idx
            indices = torch.where(mask)[0]
            class_to_indices[class_idx] = indices

        logger.info("Precomputed %d features with %d classes", features.shape[0], len(unique_classes))
        for class_idx in sorted(unique_classes)[:5]:  # Show first 5 classes
            count = len(class_to_indices[class_idx])
            logger.info("  Class %d: %d samples", class_idx, count)
        if len(unique_classes) > 5:
            logger.info("  ... and %d more classes", len(unique_classes) - 5)

        return PrecomputedFeatures(
            features=features, class_to_indices=class_to_indices, index_to_class=labels, device=torch.device(device)
        )

    def plot_and_save_results(self, results: dict[str, Any], class_names: list[str], output_dir: Path) -> None:
        summary = {
            "accuracy": results["accuracy"],
            "confusion_matrix": results["confusion_matrix"],
            "selected_classes": results["selected_classes"],
            "precomputed_features_shape": tuple(results["precomputed_features_shape"]),
        }

        yaml_results_path = str(output_dir / "abx_results.yaml")
        with Path(yaml_results_path).open("w") as f:
            f.write(to_yaml(summary))

        pkl_results_path = str(output_dir / "abx_results_full.pkl")
        with Path(pkl_results_path).open("wb") as f:
            pickle.dump(results, f)

        plot_distance_distributions(results, save_path=output_dir / "abx_summary.png")
        plot_confusion_matrix(
            results,
            save_path=output_dir / "confusion_matrix.png",
            dataset_name=self.config.dataset.name,
            class_names=class_names,
        )

        logger.info("Results saved to %s", output_dir)

    def run(self, iteration_value: int, iteration_index: int) -> dict[str, Any]:
        setup_logging()

        logger.info("Output directory: %s", self.output_dir)

        set_seed(self.config.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        dataloader = self._get_dataloader()

        model = self._create_model(device)

        precomputed_features = self._precompute_all_features(model, dataloader, device)

        valid_classes = sorted(precomputed_features.class_to_indices.keys())
        num_classes = len(valid_classes)

        logger.info("Running ABX evaluation on %d classes...", num_classes)

        confusion_matrix = np.zeros((num_classes, num_classes))
        cell_counts = np.zeros((num_classes, num_classes))

        all_correct: float = 0.0
        all_total = 0

        for i, class_x in enumerate(tqdm(valid_classes)):
            for j, class_b in enumerate(valid_classes):
                if class_x == class_b:
                    continue

                accuracy, count = evaluate_cell_vectorized(
                    precomputed_features, class_x, class_b, self.config.num_triplets
                )

                if count > 0:
                    confusion_matrix[i, j] = accuracy
                    cell_counts[i, j] = count
                    all_correct += accuracy * count
                    all_total += count

        overall_accuracy = all_correct / all_total if all_total > 0 else 0.0

        results = {
            "accuracy": overall_accuracy,
            "confusion_matrix": confusion_matrix,
            "cell_counts": cell_counts,
            "num_classes": num_classes,
            "selected_classes": valid_classes,
            "total_triplets": all_total,
            "num_triplets_per_cell": self.config.num_triplets,
            "precomputed_features_shape": precomputed_features.features.shape,
        }

        logger.info("Processed %d samples", results["precomputed_features_shape"][0])
        if self.config.subset is not None:
            logger.info(
                "Evaluated on %d randomly selected classes",
                cast("ClassificationMediaDataset", dataloader.dataset).num_classes,
            )

        logger.info("\nOptimized ABX Results:")
        logger.info("Overall Accuracy: %.4f", results["accuracy"])
        logger.info("Total triplets: %d", results["total_triplets"])
        logger.info("Confusion matrix shape: %s", confusion_matrix.shape)

        valid_cells = confusion_matrix[confusion_matrix > 0]
        if len(valid_cells) > 0:
            logger.info("Mean cell accuracy: %.4f", np.mean(valid_cells))
            logger.info("Std cell accuracy: %.4f", np.std(valid_cells))

        dataset_for_names = cast("ClassificationMediaDataset", dataloader.dataset)
        self.plot_and_save_results(
            results,
            class_names=[
                n
                for c, n in zip(dataset_for_names.class_ids, dataset_for_names.class_names, strict=True)
                if c in valid_classes
            ],
            output_dir=self.output_dir,
        )

        return results
