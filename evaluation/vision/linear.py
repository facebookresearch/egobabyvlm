# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from stopes.core import Requirements
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy

if TYPE_CHECKING:
    from evaluation.data.base import ClassificationMediaDataset


from core.modeling import freeze
from core.protocols import ImageFeatureExtractor
from core.utils import (
    get_world_size,
    is_main_process,
    set_seed,
    setup_distributed,
    setup_logging,
    to_yaml,
)
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig
from evaluation.utils import MetricLogger, unwrap_model

logger = logging.getLogger(__name__)


def _classification_collate_fn(
    samples: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for classification datasets."""
    images_tuple, labels_tuple = zip(*samples, strict=True)
    images = torch.stack(images_tuple)
    labels = torch.tensor(labels_tuple)
    return images, labels


@dataclass
class LinearEvalModuleConfig(EvalConfig):
    _target_: str = "evaluation.vision.linear.LinearEvalModule"

    name: str = "linear"

    train_dataset: EvalDatasetConfig = MISSING
    val_dataset: EvalDatasetConfig = MISSING
    test_dataset: EvalDatasetConfig | None = None

    #: As dict to support interpolation from shared_backbone.
    model: dict[str, Any] = MISSING

    #: List of pooling strategies to try, e.g. ``[{"pooling": "cls", "last_n_layers": 1}, ...]``.
    #: If provided, creates model variants with different pooling strategies.
    pooling_strategies: list[dict[str, Any]] | None = None

    #: Learning rates to grid search.
    learning_rates: list[float] = field(default_factory=lambda: [1e-3, 3e-3, 1e-2, 3e-2, 1e-1])

    batch_size: int = 256
    epochs: int = 10

    #: Number of iterations per epoch.
    epoch_length: int = 100

    num_workers: int = 8

    #: Number of iterations between evaluations.
    eval_period_iterations: int = 100

    #: Maximum samples to process (for memory efficiency).
    max_samples: int | None = None

    #: Number of classes to randomly sample for evaluation.
    subset: int | None = None

    seed: int = 42

    #: Number of GPUs for distributed training.
    num_gpus: int = 1


def scale_lr(learning_rate: float, batch_size: int, base_size: int = 256) -> float:
    return learning_rate * (batch_size * get_world_size()) / base_size


class LinearClassifier(nn.Module):
    """Simple linear classifier that operates on extracted features."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class AllClassifiers(nn.Module):
    """Container for multiple linear classifiers with different hyperparameters."""

    def __init__(self, classifiers_dict: dict[str, LinearClassifier]) -> None:
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, features_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for all classifiers.

        Args:
            features_dict: Dict mapping model keys to feature tensors

        Returns:
            Dict mapping classifier keys to logits
        """
        outputs = {}
        for key, classifier in self.classifiers_dict.items():
            model_key = self.get_model_key(key)
            outputs[key] = classifier(features_dict[model_key])
        return outputs

    def get_model_key(self, classifier_key: str) -> str:
        """Extract model key from classifier key."""
        # Key format: "model_{model_key}_lr_{lr}"
        parts = classifier_key.split("_lr_")[0]
        return parts.replace("model_", "", 1)

    def __len__(self) -> int:
        return len(self.classifiers_dict)


class LinearEvalModule(EvalModule):
    def __init__(self, config: LinearEvalModuleConfig) -> None:
        super().__init__(config, LinearEvalModuleConfig)

        # Expand model configs from single model + pooling strategies if needed
        self._models = self._expand_model_configs()

        self.output_dir = (
            to_path(self.config.output_dir)
            / self.config.name
            / self.config.train_dataset.name
            / self.config.model.name
            / self.hparam_save_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self.output_dir)

    def _expand_model_configs(self) -> list[Any]:
        """Expand model configs from single model + pooling strategies if needed.

        Returns:
            List of model configs to evaluate.
        """
        # Use single model with pooling strategies
        if self.config.pooling_strategies is not None and len(self.config.pooling_strategies) > 0:
            models = []
            for strategy in self.config.pooling_strategies:
                pooling = strategy.get("pooling", "cls")
                last_n_layers = strategy.get("last_n_layers", 1)
                # Create a copy of the model config with updated pooling
                model_dict = cast("dict[str, Any]", OmegaConf.to_container(self.config.model, resolve=True))
                model_dict["name"] = f"{model_dict['name']}-{pooling}-L{last_n_layers}"
                model_dict["kwargs"]["pooling"] = pooling
                model_dict["kwargs"]["last_n_layers"] = last_n_layers
                models.append(OmegaConf.create(model_dict))
            logger.info("Expanded single model to %d configs using pooling_strategies", len(models))
            return models
        # Just use the single model as-is
        logger.info("Using single model config")
        return [self.config.model]

    @property
    def hparam_save_name(self) -> str:
        hparam_save_name = f"lr{min(self.config.learning_rates)}-{max(self.config.learning_rates)}"
        hparam_save_name += f"_ep{self.config.epochs}"
        hparam_save_name += f"_models{len(self._models)}"
        if self.config.subset is not None:
            hparam_save_name += f"_subset{self.config.subset}"
        if self.config.max_samples is not None:
            hparam_save_name += f"_maxsamples{self.config.max_samples}"
        return hparam_save_name

    def requirements(self) -> Requirements:
        """Requirements for each submitted job"""
        return Requirements(
            nodes=1,
            mem_gb=140 * self.config.num_gpus,
            tasks_per_node=self.config.num_gpus,
            gpus_per_node=self.config.num_gpus,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        base = f"{self.config.name}_{self.config.train_dataset.name}_{self.config.model.name}"
        return f"{base}_{self.hparam_save_name}_seed{self.config.seed}"

    def _create_models(self, device: str = "cuda") -> dict[str, ImageFeatureExtractor]:
        """Create all feature extraction models."""
        logger.info("Loading %d pretrained feature extractor(s)", len(self._models))

        models = {}
        for model_config in self._models:
            model: ImageFeatureExtractor = instantiate({"_target_": model_config._target_}, **model_config.kwargs)
            model.eval()
            freeze(model)
            model.to(device)
            models[model_config.name] = model
            logger.info("Loaded model '%s' with feature_dim=%d", model_config.name, model.feature_dim)

        return models

    def _get_dataloader(self, dataset_config: EvalDatasetConfig, *, shuffle: bool = False) -> EvalDataLoader:
        dataset: ClassificationMediaDataset = instantiate(
            {"_target_": dataset_config._target_}, **dataset_config.kwargs
        )

        logger.info("Dataset %s loaded: %d samples", dataset_config.name, len(dataset))

        if self.config.max_samples is not None:
            dataset = dataset.subset_from_max_samples(self.config.max_samples, rng=random.Random(self.config.seed))
            logger.info("Subsampled dataset to %d samples", len(dataset))

        if self.config.subset is not None:
            dataset = dataset.subset_from_subset_classes(self.config.subset, rng=random.Random(self.config.seed))

        sampler: DistributedSampler[Any] | None = None
        if get_world_size() > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # sampler handles shuffling

        return EvalDataLoader(
            dataset,
            collate_fn=_classification_collate_fn,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=shuffle or (sampler is not None and sampler.shuffle),
        )

    def _create_linear_classifiers(
        self,
        feature_models: dict[str, ImageFeatureExtractor],
        num_classes: int,
        device: str = "cuda",
    ) -> tuple[AllClassifiers, list[dict]]:
        """Create linear classifiers for grid search over models and learning rates."""
        logger.info("Creating linear classifiers for grid search...")
        linear_classifiers_dict = {}
        optim_param_groups = []

        for model_key, model in feature_models.items():
            for base_lr in self.config.learning_rates:
                scaled_lr = scale_lr(base_lr, self.config.batch_size)

                classifier = LinearClassifier(
                    in_dim=model.feature_dim,
                    num_classes=num_classes,
                )
                classifier = classifier.to(device)

                key = f"model_{model_key}_lr_{scaled_lr:.5f}".replace(".", "_")
                linear_classifiers_dict[key] = classifier

                optim_param_groups.append({"params": classifier.parameters(), "lr": scaled_lr})

        all_classifiers = AllClassifiers(linear_classifiers_dict)

        logger.info("Created %d linear classifiers for grid search", len(linear_classifiers_dict))
        logger.info(
            "Grid search over %d model(s) x %d learning rate(s)",
            len(feature_models),
            len(self.config.learning_rates),
        )

        return all_classifiers, optim_param_groups

    def _create_optim_and_scheduler(
        self, optim_param_groups: list[dict], max_iter: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        return optimizer, scheduler

    def _extract_features(
        self,
        feature_models: dict[str, ImageFeatureExtractor],
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract features from all models."""
        features_dict = {}
        for model_key, model in feature_models.items():
            # Sanitize model key to match classifier key format (replace dots with underscores)
            sanitized_key = model_key.replace(".", "_")
            features_dict[sanitized_key] = model.extract_features(images)
        return features_dict

    @torch.inference_mode()
    def _evaluate_classifiers(
        self,
        feature_models: dict[str, ImageFeatureExtractor],
        classifiers: AllClassifiers | DistributedDataParallel | dict[str, LinearClassifier],
        data_loader: EvalDataLoader,
        num_classes: int,
        device: str = "cuda",
    ) -> dict[str, dict[str, float]]:
        if isinstance(classifiers, dict):
            classifiers_unwrapped = None
            classifiers_dict: dict[str, LinearClassifier] = classifiers
        else:
            classifiers_unwrapped = cast("AllClassifiers", unwrap_model(classifiers))
            classifiers_dict = cast("dict[str, LinearClassifier]", classifiers_unwrapped.classifiers_dict)

        metrics = {
            key: {
                "top-1": MulticlassAccuracy(top_k=1, num_classes=num_classes, average="micro").to(device),
                "top-5": MulticlassAccuracy(top_k=5, num_classes=num_classes, average="micro").to(device),
            }
            for key in classifiers_dict
        }

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.set_header("Evaluation:")

        for batch in metric_logger.log_every(data_loader, 10):
            images_batch, labels_batch = batch
            images_batch = images_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            features_dict = self._extract_features(feature_models, images_batch)

            for key, classifier in classifiers_dict.items():
                model_key = (
                    classifiers_unwrapped.get_model_key(key)
                    if classifiers_unwrapped is not None
                    else self._infer_model_key(key)
                )
                logits = classifier(features_dict[model_key])
                metrics[key]["top-1"].update(logits, labels_batch)
                metrics[key]["top-5"].update(logits, labels_batch)

        logger.info("Averaged stats: %s", metric_logger)

        results = {}
        for key in classifiers_dict:
            results[key] = {
                "top-1": metrics[key]["top-1"].compute().item(),
                "top-5": metrics[key]["top-5"].compute().item(),
            }

        return results

    def _infer_model_key(self, classifier_key: str) -> str:
        """Infer model key from classifier key for dict classifiers."""
        parts = classifier_key.split("_lr_")[0]
        return parts.replace("model_", "", 1)

    def _train_step(
        self,
        feature_models: dict[str, ImageFeatureExtractor],
        all_classifiers: AllClassifiers | DistributedDataParallel,
        images_batch: torch.Tensor,
        labels_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: str = "cuda",
    ) -> tuple[float, dict[str, float]]:
        all_classifiers.train()

        images_batch = images_batch.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        features_dict = self._extract_features(feature_models, images_batch)

        outputs = all_classifiers(features_dict)

        losses = {f"loss_{k}": F.cross_entropy(v, labels_batch) for k, v in outputs.items()}
        loss = cast("torch.Tensor", sum(losses.values()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        individual_losses = {key.replace("loss_", ""): loss_val.item() for key, loss_val in losses.items()}

        return loss.item(), individual_losses

    def plot_results(self, results: dict[str, Any], output_dir: Path) -> None:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        classifier_names = list(results["val_results"].keys())
        accuracies = [results["val_results"][k]["top-1"] for k in classifier_names]

        sorted_indices = np.argsort(accuracies)[::-1]
        top_n = min(10, len(classifier_names))

        ax.barh(range(top_n), [accuracies[i] for i in sorted_indices[:top_n]])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([classifier_names[i] for i in sorted_indices[:top_n]], fontsize=8)
        ax.set_xlabel("Validation Accuracy")
        ax.set_title(f"Top {top_n} Classifiers")
        ax.grid(axis="x", alpha=0.3)

        ax = axes[1]
        metrics = ["best_val_accuracy", "avg_val_accuracy"]
        values = [results["summary"]["best_val_accuracy"], results["summary"]["avg_val_accuracy"]]

        if results["summary"].get("best_test_accuracy") is not None:
            metrics.append("best_test_accuracy")
            values.append(results["summary"]["best_test_accuracy"])

        ax.bar(metrics, values, color=["green", "blue", "orange"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Summary Metrics")
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "results_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _process_results(
        self,
        *,
        val_results: dict[str, dict[str, float]],
        test_results: dict[str, dict[str, float]] | None,
        training_history: list[dict[str, Any]],
        best_classifier_key: str,
        best_val_accuracy: float,
        best_test_accuracy: float | None,
        all_classifiers: AllClassifiers,
        num_classes: int,
    ) -> dict[str, Any]:
        """Process and save evaluation results."""
        results: dict[str, Any] = {
            "config": {
                "models": {m.name: m.kwargs for m in self._models},
                "train_dataset": self.config.train_dataset.name,
                "val_dataset": self.config.val_dataset.name,
                "test_dataset": self.config.test_dataset.name if self.config.test_dataset else None,
                "num_classes": num_classes,
                "epochs": self.config.epochs,
                "learning_rates": list(self.config.learning_rates),
                "batch_size": self.config.batch_size,
            },
            "training_history": training_history,
            "val_results": val_results,
            "test_results": test_results if test_results else None,
            "best_classifier": {
                "name": best_classifier_key,
                "val_accuracy": float(best_val_accuracy),
                "test_accuracy": float(best_test_accuracy) if best_test_accuracy is not None else None,
            },
            "summary": {
                "best_val_accuracy": float(best_val_accuracy),
                "avg_val_accuracy": float(np.mean([v["top-1"] for v in val_results.values()])),
                "std_val_accuracy": float(np.std([v["top-1"] for v in val_results.values()])),
                "best_test_accuracy": float(best_test_accuracy) if best_test_accuracy is not None else None,
                "num_classifiers": len(all_classifiers),
            },
        }

        logger.info("Saving results...")

        yaml_results_path = str(self.output_dir / "linear_results.yaml")
        with Path(yaml_results_path).open("w") as f:
            f.write(to_yaml(results))

        pkl_results_path = str(self.output_dir / "linear_results_full.pkl")
        with Path(pkl_results_path).open("wb") as f:
            pickle.dump(results, f)

        best_classifier_module = all_classifiers.classifiers_dict[best_classifier_key]
        classifier_path = str(self.output_dir / f"best_classifier_{best_classifier_key}.pth")
        with Path(classifier_path).open("wb") as f:
            torch.save(best_classifier_module.state_dict(), f)

        self.plot_results(results, self.output_dir)

        logger.info("Results saved to %s", self.output_dir)
        logger.info("Results summary: %s", to_yaml(results["summary"]))

        return results

    def run(self, iteration_value: int, iteration_index: int) -> dict[str, Any] | None:
        setup_logging()
        setup_distributed()

        set_seed(self.config.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s (world_size=%d)", device, get_world_size())

        train_loader = self._get_dataloader(self.config.train_dataset, shuffle=True)
        val_loader = self._get_dataloader(self.config.val_dataset, shuffle=False)
        test_loader = (
            self._get_dataloader(self.config.test_dataset, shuffle=False)
            if self.config.test_dataset is not None
            else None
        )

        num_classes = len(cast("ClassificationMediaDataset", train_loader.dataset).class_ids)
        logger.info("Number of classes: %d", num_classes)

        feature_models = self._create_models(device)
        base_classifiers, optim_param_groups = self._create_linear_classifiers(feature_models, num_classes, device)

        # Wrap classifiers in DDP if distributed
        all_classifiers: AllClassifiers | DistributedDataParallel = base_classifiers
        if get_world_size() > 1:
            all_classifiers = DistributedDataParallel(base_classifiers)

        max_iter = self.config.epochs * self.config.epoch_length
        optimizer, scheduler = self._create_optim_and_scheduler(optim_param_groups, max_iter)

        logger.info("Training for %d epochs (%d iterations)...", self.config.epochs, max_iter)
        training_history = []
        metric_logger = MetricLogger(delimiter="  ")
        train_iter = iter(train_loader)
        iteration = 0
        epoch = 0
        best_val_accuracy = 0.0
        best_classifier_key: str = ""
        while iteration < max_iter:
            try:
                images_batch, labels_batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                images_batch, labels_batch = next(train_iter)

            loss, _ = self._train_step(
                feature_models, all_classifiers, images_batch, labels_batch, optimizer, scheduler, device
            )

            if iteration % 10 == 0:
                torch.cuda.synchronize()
                metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss)
                logger.info(
                    "[%s/%d] loss: %.4f, lr: %.6f",
                    iteration,
                    max_iter,
                    metric_logger.meters["loss"].value,
                    metric_logger.meters["lr"].value,
                )

            is_periodic_eval = (
                self.config.eval_period_iterations > 0 and (iteration + 1) % self.config.eval_period_iterations == 0
            )
            is_final_iteration = iteration == max_iter - 1

            if is_periodic_eval or is_final_iteration:
                logger.info("\nEvaluating at iteration %d/%d...", iteration + 1, max_iter)
                val_results = self._evaluate_classifiers(
                    feature_models, all_classifiers, val_loader, num_classes, device
                )

                best_key = max(val_results.keys(), key=lambda k: val_results[k]["top-1"])
                best_acc = val_results[best_key]["top-1"]
                logger.info("Best validation accuracy: %.4f (%s)", best_acc, best_key)

                # Save checkpoint if this is the best so far (main process only)
                if best_acc > best_val_accuracy:
                    best_val_accuracy = best_acc
                    best_classifier_key = best_key

                    if is_main_process():
                        checkpoint_path = str(self.output_dir / "best_checkpoint.pth")
                        checkpoint_data = {
                            "iteration": iteration + 1,
                            "all_classifiers_state_dict": unwrap_model(all_classifiers).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_accuracy": best_val_accuracy,
                            "best_classifier_key": best_classifier_key,
                            "val_results": val_results,
                        }
                        with Path(checkpoint_path).open("wb") as f:
                            torch.save(checkpoint_data, f)
                        logger.info("Saved best checkpoint with val accuracy: %.4f", best_val_accuracy)

                training_history.append(
                    {
                        "iteration": iteration + 1,
                        "epoch": (iteration + 1) / self.config.epoch_length,
                        "loss": metric_logger.meters["loss"].global_avg,
                        "lr": metric_logger.meters["lr"].value,
                        "val_results": val_results,
                        "best_classifier": best_key,
                        "best_val_accuracy": best_acc,
                    }
                )

                torch.cuda.synchronize()

            iteration += 1

        logger.info("\nFinal evaluation - loading best checkpoint...")

        checkpoint_path = str(self.output_dir / "best_checkpoint.pth")

        if is_main_process() and Path(checkpoint_path).exists():
            with Path(checkpoint_path).open("rb") as f:
                checkpoint = torch.load(f, map_location=device, weights_only=False)
            unwrap_model(all_classifiers).load_state_dict(checkpoint["all_classifiers_state_dict"])
            best_val_accuracy = checkpoint["best_val_accuracy"]
            best_classifier_key = checkpoint["best_classifier_key"]
            val_results = checkpoint["val_results"]
            logger.info("Loaded best checkpoint from iteration %d", checkpoint["iteration"])
            logger.info("Best validation accuracy from checkpoint: %.4f (%s)", best_val_accuracy, best_classifier_key)
        elif is_main_process():
            # Fallback to final state if no checkpoint exists (shouldn't happen in normal operation)
            logger.warning("No checkpoint found, using final training state")
            val_results = self._evaluate_classifiers(feature_models, all_classifiers, val_loader, num_classes, device)
            best_classifier_key = max(val_results.keys(), key=lambda k: val_results[k]["top-1"])
            best_val_accuracy = val_results[best_classifier_key]["top-1"]
            logger.info("Final state validation accuracy: %.4f (%s)", best_val_accuracy, best_classifier_key)

        test_results = None
        best_test_accuracy = None
        if test_loader is not None:
            logger.info("\nEvaluating best classifier on test set...")
            classifiers_unwrapped = cast("AllClassifiers", unwrap_model(all_classifiers))
            best_classifier = cast("LinearClassifier", classifiers_unwrapped.classifiers_dict[best_classifier_key])
            test_results = self._evaluate_classifiers(
                feature_models, {best_classifier_key: best_classifier}, test_loader, num_classes, device
            )
            best_test_accuracy = test_results[best_classifier_key]["top-1"]
            logger.info("Test accuracy: %.4f", best_test_accuracy)

        if is_main_process():
            result = self._process_results(
                val_results=val_results,
                test_results=test_results,
                training_history=training_history,
                best_classifier_key=best_classifier_key,
                best_val_accuracy=best_val_accuracy,
                best_test_accuracy=best_test_accuracy,
                all_classifiers=cast("AllClassifiers", unwrap_model(all_classifiers)),
                num_classes=num_classes,
            )
        else:
            result = None

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return result
