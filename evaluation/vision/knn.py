# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

if TYPE_CHECKING:
    from evaluation.data.base import ClassificationMediaDataset

from core.modeling import freeze
from core.protocols import ImageFeatureExtractor
from core.utils import set_seed, setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig
from evaluation.utils import MetricLogger

logger = logging.getLogger(__name__)


def _classification_collate_fn(
    samples: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for classification datasets."""
    images_tuple, labels_tuple = zip(*samples, strict=True)
    return torch.stack(images_tuple), torch.tensor(labels_tuple)


class KnnModule(nn.Module):
    """
    GPU-optimized k-NN module inspired by DINOv2 implementation.

    Efficiently computes k nearest neighbors on GPU using matrix multiplication
    and topk operations. Uses probability-based voting with softmax weights.
    """

    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        nb_knn: list[int],
        temperature: float,
        device: str,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.device = device
        # Store training features transposed for efficient matrix multiplication
        self.train_features_T = train_features.T.to(self.device)
        self.train_labels = train_labels.view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.temperature = temperature
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(
        self, similarity: torch.Tensor, train_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k similarities and corresponding labels."""
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        # Expand train_labels to match batch size for gathering
        batch_size = indices.shape[0]
        train_labels_expanded = train_labels.expand(batch_size, -1)
        neighbors_labels = torch.gather(train_labels_expanded, 1, indices)
        return topk_sims, neighbors_labels

    def forward(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Compute k-NN predictions for all values of k.

        Args:
            features: Test features of shape (batch_size, feature_dim)

        Returns:
            Dictionary mapping k -> class probabilities of shape (batch_size, num_classes)
        """
        features = features.to(self.device)

        # Compute similarity with all training samples
        similarity = torch.mm(features, self.train_features_T)

        # Get top-k similarities and labels
        topk_sims, neighbors_labels = self._get_knn_sims_and_labels(similarity, self.train_labels)

        batch_size = neighbors_labels.shape[0]

        # Apply temperature and softmax to similarities
        topk_sims_transform = F.softmax(topk_sims / self.temperature, dim=1)

        # Convert labels to one-hot and weight by similarities
        matmul = torch.mul(
            F.one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )

        # Sum weighted votes for each k value
        return {k: torch.sum(matmul[:, :k, :], dim=1) for k in self.nb_knn}


@dataclass
class KNNEvalModuleConfig(EvalConfig):
    _target_: str = "evaluation.vision.knn.KNNEvalModule"

    name: str = "knn"

    train_dataset: EvalDatasetConfig = MISSING
    val_dataset: EvalDatasetConfig = MISSING
    test_dataset: EvalDatasetConfig | None = None

    #: As dict to support interpolation from shared_backbone.
    model: dict[str, Any] = MISSING

    #: Pooling strategy (cls, mean_patch, etc.). If None, uses model default.
    #: Set independently of model config when using a shared backbone in pipelines.
    pooling: str | None = None

    #: Number of layers to use. If None, uses model default.
    last_n_layers: int | None = None

    #: List of k values for k-NN.
    nb_knn: list[int] = field(default_factory=lambda: [10, 20, 100, 200])

    #: Temperature for similarity scaling.
    temperature: float = 0.07

    batch_size: int = 256
    num_workers: int = 8

    #: List of number of samples per class to use from training set (-1 for all).
    n_per_class_list: list[int] = field(default_factory=lambda: [-1])

    #: Number of tries for each n_per_class setting.
    n_tries: int = 1

    #: Maximum samples to process (for memory efficiency).
    max_samples: int | None = None

    #: Number of classes to randomly sample for evaluation.
    subset: int | None = None

    seed: int = 42


class KNNEvalModule(EvalModule):
    def __init__(self, config: KNNEvalModuleConfig) -> None:
        super().__init__(config, KNNEvalModuleConfig)
        self.output_dir = (
            to_path(self.config.output_dir)
            / self.config.name
            / self.config.train_dataset.name
            / self.config.model.name
            / self.hparam_save_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self.output_dir)

    @property
    def hparam_save_name(self) -> str:
        nbknn_str = "-".join(map(str, list(self.config.nb_knn)))
        npc_str = "-".join(map(str, list(self.config.n_per_class_list)))
        return f"temp{self.config.temperature}_nb{nbknn_str}_npc{npc_str}_tries{self.config.n_tries}"

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=140,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        base = f"{self.config.name}_{self.config.train_dataset.name}_{self.config.model.name}"
        return f"{base}_{self.hparam_save_name}_seed{self.config.seed}"

    def _create_model(self, device: str = "cuda") -> ImageFeatureExtractor:
        logger.info("Loading pretrained feature extractor")
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

    def _get_dataloader(self, dataset_config: EvalDatasetConfig, *, shuffle: bool = False) -> EvalDataLoader:
        dataset: ClassificationMediaDataset | torch.utils.data.Subset[Any] = instantiate(
            {"_target_": dataset_config._target_}, **dataset_config.kwargs
        )
        logger.info("Dataset %s loaded: %d samples", dataset_config.name, len(dataset))
        if self.config.max_samples is not None:
            dataset = torch.utils.data.Subset(dataset, range(min(self.config.max_samples, len(dataset))))
        if self.config.subset is not None:
            classes = np.unique([dataset[i][1] for i in range(len(dataset))])
            rng = np.random.default_rng(self.config.seed)
            chosen = rng.choice(classes, self.config.subset, replace=False)
            indices = [i for i in range(len(dataset)) if dataset[i][1] in chosen]
            dataset = torch.utils.data.Subset(dataset, indices)

        return EvalDataLoader(
            dataset,
            collate_fn=_classification_collate_fn,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    @torch.inference_mode()
    def _extract_features_and_labels(
        self, model: ImageFeatureExtractor, dataloader: EvalDataLoader, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and labels, keeping them on GPU for efficiency."""
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        metric_logger = MetricLogger(delimiter="  ")
        for batch in metric_logger.log_every(dataloader, 10, "Extracting features"):
            batch_images, batch_labels = batch
            batch_features = model.extract_features(batch_images.to(device, non_blocking=True))
            batch_features = F.normalize(batch_features, dim=-1)
            # Keep features on GPU for faster kNN computation
            features.append(batch_features)
            labels.append(batch_labels.to(device, non_blocking=True))

        features_tensor = torch.cat(features)
        labels_tensor = torch.cat(labels)
        logger.info("Extracted features shape: %s, labels shape: %s", features_tensor.shape, labels_tensor.shape)
        return features_tensor, labels_tensor

    @torch.inference_mode()
    def _evaluate_knn(
        self,
        knn_module: "KnnModule",
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        num_classes: int,
        device: str,
    ) -> dict[int, dict[str, float]]:
        """Evaluate kNN predictions using torchmetrics."""
        # Create metrics for each k value
        metrics = {
            k: {
                "top-1": MulticlassAccuracy(top_k=1, num_classes=num_classes, average="micro").to(device),
                "top-5": MulticlassAccuracy(top_k=5, num_classes=num_classes, average="micro").to(device),
            }
            for k in knn_module.nb_knn
        }

        # Process in batches to avoid OOM for large validation sets
        batch_size = 1024
        num_samples = val_features.shape[0]

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            features_batch = val_features[start_idx:end_idx]
            labels_batch = val_labels[start_idx:end_idx]

            # Get probability distributions for all k values
            probas_for_k = knn_module(features_batch)

            # Update metrics for each k
            for k in knn_module.nb_knn:
                probas = probas_for_k[k]
                metrics[k]["top-1"].update(probas, labels_batch)
                metrics[k]["top-5"].update(probas, labels_batch)

        # Compute final results
        results = {}
        for k in knn_module.nb_knn:
            results[k] = {
                "top-1": metrics[k]["top-1"].compute().item(),
                "top-5": metrics[k]["top-5"].compute().item(),
            }
            logger.info("k=%d: Top-1 Acc=%.4f, Top-5 Acc=%.4f", k, results[k]["top-1"], results[k]["top-5"])

        return results

    def run(self, iteration_value: int = 0, iteration_index: int = 0) -> dict:
        """Run k-NN evaluation with GPU optimization."""
        setup_logging()

        set_seed(self.config.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        model = self._create_model(device)
        train_loader = self._get_dataloader(self.config.train_dataset, shuffle=False)
        val_loader = self._get_dataloader(self.config.val_dataset, shuffle=False)

        logger.info("Extracting training features...")
        train_features, train_labels = self._extract_features_and_labels(model, train_loader, device)

        logger.info("Extracting validation features...")
        val_features, val_labels = self._extract_features_and_labels(model, val_loader, device)

        num_classes = int(train_labels.max().item()) + 1
        logger.info("Number of classes: %d", num_classes)

        results = {}

        # Evaluate for each n_per_class setting
        for n_per_class in self.config.n_per_class_list:
            logger.info("=" * 80)
            if n_per_class > 0:
                logger.info("Evaluating with %d samples per class", n_per_class)
                unique_labels = torch.unique(train_labels)
                indices: list[int] = []
                rng = np.random.default_rng(self.config.seed)
                for label in unique_labels:
                    label_indices = (train_labels == label).nonzero(as_tuple=True)[0]
                    label_indices_cpu = label_indices.cpu().numpy()
                    chosen = rng.choice(label_indices_cpu, min(n_per_class, len(label_indices_cpu)), replace=False)
                    indices.extend(chosen.tolist())
                indices_tensor = torch.tensor(indices, device=device)
                sub_train_features = train_features[indices_tensor]
                sub_train_labels = train_labels[indices_tensor]
                logger.info("Subsampled training set size: %d", len(indices_tensor))
            else:
                logger.info("Evaluating with full training set")
                sub_train_features = train_features
                sub_train_labels = train_labels

            knn_module = KnnModule(
                train_features=sub_train_features,
                train_labels=sub_train_labels,
                nb_knn=self.config.nb_knn,
                temperature=self.config.temperature,
                num_classes=num_classes,
                device=device,
            )

            knn_results = self._evaluate_knn(
                knn_module=knn_module,
                val_features=val_features,
                val_labels=val_labels,
                num_classes=num_classes,
                device=device,
            )

            for k, metrics in knn_results.items():
                key = f"npc{n_per_class}_k{k}" if n_per_class > 0 else f"full_k{k}"
                results[key] = {
                    "top-1": metrics["top-1"],
                    "top-5": metrics["top-5"],
                    "n_per_class": n_per_class,
                    "k": k,
                }

        logger.info("=" * 80)
        logger.info("Final Results Summary:")
        for key, metrics in results.items():
            logger.info("%s: Top-1=%.4f, Top-5=%.4f", key, metrics["top-1"], metrics["top-5"])

        out_path = str(self.output_dir / "knn_results.yaml")
        with Path(out_path).open("w") as f:
            f.write(to_yaml(results))
        logger.info("Saved k-NN results to: %s", out_path)

        return results
