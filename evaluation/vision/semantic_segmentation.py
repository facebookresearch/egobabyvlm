# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic Segmentation evaluation module with nested run/sweep architecture.

This module provides:
- SemanticSegmentationRunModule: Handles sweeping via Stopes array functionality
- SemanticSegmentationSweepModule: Orchestrates the sweep and aggregates results

Architecture:
                SemanticSegmentationSweepModule (coordinator)
                              │
                   launcher.schedule(RunModule)
                              │
                SemanticSegmentationRunModule.array()
                        returns configs
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
    run(config1, 0)    run(config2, 1)    run(config3, 2)
     (model1, lr1)      (model1, lr2)      (model2, lr1)
                              │
                    (all run in single SLURM array job)
                              │
                SemanticSegmentationSweepModule
                    aggregates results
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from sklearn.metrics import confusion_matrix
from stopes.core import Launcher, Requirements
from submitit.helpers import clean_env
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

if TYPE_CHECKING:
    from evaluation.data.base import SemanticSegmentationDataset


from core.modeling import freeze
from core.protocols import ImageFeatureExtractor
from core.utils import LauncherConfig, set_seed, setup_logging, to_yaml
from evaluation.base import PipelineError, TaskError, to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig, EvalModelConfig, PoolingStrategy

logger = logging.getLogger(__name__)


class LinearSegmentationHead(nn.Module):
    """Linear probe head for semantic segmentation (bare Conv2d, matching NeCo)."""

    def __init__(self, in_channels: int, num_classes: int, *, sync_bn: bool = False) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.classifier.weight)
        assert self.classifier.bias is not None
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SegmentationModel(nn.Module):
    """Full semantic segmentation model combining feature extractor and segmentation head."""

    # Intermediate resolution for features and masks (matching NeCo).
    MASK_SIZE: int = 100

    def __init__(
        self,
        feature_extractor: ImageFeatureExtractor,
        num_classes: int,
        *,
        sync_bn: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.segmentation_head = LinearSegmentationHead(
            in_channels=self.feature_extractor.feature_dim, num_classes=num_classes, sync_bn=sync_bn
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        with torch.no_grad():
            features = self.feature_extractor.extract_features(x)
        _expected_ndim = 3
        assert features.ndim == _expected_ndim, "Expected features to have shape (B, N_patches, D)"
        assert math.sqrt(features.shape[1]).is_integer(), "Expected square number of patches"
        n_patches_per_side = int(math.sqrt(features.shape[1]))
        features = features.permute(0, 2, 1)
        features = features.reshape(
            batch_size, self.feature_extractor.feature_dim, n_patches_per_side, n_patches_per_side
        )
        logits = self.segmentation_head(features)
        return F.interpolate(logits, size=(self.MASK_SIZE, self.MASK_SIZE), mode="bilinear", align_corners=False)


class SegmentationMetrics:
    """Metrics tracker for semantic segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_arr = pred.cpu().numpy().flatten()
        target_arr = target.cpu().numpy().flatten()
        valid_mask = target_arr != self.ignore_index
        pred_arr = pred_arr[valid_mask]
        target_arr = target_arr[valid_mask]
        if len(pred_arr) > 0:
            self.confusion_matrix += confusion_matrix(
                target_arr, pred_arr, labels=list(range(self.num_classes))
            ).astype(np.int64)

    def calculate_metrics(self) -> dict[str, Any]:
        cm = self.confusion_matrix.astype(np.float64)
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-10)
        valid_classes = ~np.isnan(iou)
        mean_iou = np.mean(iou[valid_classes]) if valid_classes.any() else 0.0
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
        class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        mean_acc = np.mean(class_acc[valid_classes]) if valid_classes.any() else 0.0
        precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
        recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        mean_f1 = np.mean(f1[valid_classes]) if valid_classes.any() else 0.0
        return {
            "iou_per_class": iou,
            "miou": float(mean_iou),
            "pixel_acc": float(pixel_acc),
            "class_acc": class_acc,
            "mean_acc": float(mean_acc),
            "f1_per_class": f1,
            "mean_f1": float(mean_f1),
            "confusion_matrix": cm,
        }


@dataclass
class SemanticSegmentationRunConfig(EvalConfig):
    """Config for semantic segmentation training with array support for sweeps."""

    _target_: str = "evaluation.vision.semantic_segmentation.SemanticSegmentationRunModule"

    name: str = "semantic_segmentation_run"

    train_dataset: EvalDatasetConfig = MISSING
    val_dataset: EvalDatasetConfig = MISSING
    test_dataset: EvalDatasetConfig | None = None

    # For single runs
    model: EvalModelConfig | None = None
    learning_rate: float | None = None

    # For sweeps (list of {model, learning_rate} dicts)
    sweep_configs: list[dict[str, Any]] | None = None

    batch_size: int = 16
    epochs: int = 20
    drop_at: int = 20
    num_workers: int = 8
    eval_period_epochs: int = 5
    sync_bn: bool = False
    ignore_index: int = 255
    seed: int = 42


class SemanticSegmentationRunModule(EvalModule):
    """Semantic segmentation training module with array support for sweeps."""

    def __init__(self, config: SemanticSegmentationRunConfig) -> None:
        super().__init__(config, SemanticSegmentationRunConfig)
        self._base_output_dir = to_path(self.config.output_dir) / self.config.name / self.config.train_dataset.name

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
        if self.config.model is not None and self.config.learning_rate is not None:
            return f"{self.config.name}_{self.config.model.name}_lr{self.config.learning_rate}"
        return f"{self.config.name}_sweep"

    def array(self) -> list[dict[str, Any]] | None:
        """Return sweep configurations for array job execution."""
        if self.config.sweep_configs is not None:
            logger.info("Array mode: returning %d sweep configurations", len(self.config.sweep_configs))
            return self.config.sweep_configs
        return None

    def _get_output_dir(self, model_name: str, learning_rate: float) -> Path:
        output_dir = self._base_output_dir / model_name / f"lr{learning_rate}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _create_model_from_config(self, model_config: dict[str, Any], device: str = "cuda") -> ImageFeatureExtractor:
        model_name = model_config.get("name", "model")
        logger.info("Loading pretrained feature extractor: %s", model_name)
        model: ImageFeatureExtractor = instantiate(
            {"_target_": model_config["_target_"]}, **model_config.get("kwargs", {})
        )
        model.eval()
        freeze(model)
        model.to(device)
        logger.info("Loaded model '%s' with feature_dim=%d", model_name, model.feature_dim)
        return model

    def _create_model(self, device: str = "cuda") -> ImageFeatureExtractor:
        logger.info("Loading pretrained feature extractor: %s", self.config.model.name)
        model: ImageFeatureExtractor = instantiate(
            {"_target_": self.config.model._target_}, **self.config.model.kwargs
        )
        model.eval()
        freeze(model)
        model.to(device)
        logger.info("Loaded model '%s' with feature_dim=%d", self.config.model.name, model.feature_dim)
        return model

    def _get_dataloader(
        self,
        dataset_config: EvalDatasetConfig,
        feature_extractor: ImageFeatureExtractor | None = None,
        *,
        shuffle: bool = False,
    ) -> EvalDataLoader:
        logger.info("Creating dataloader for dataset: %s", dataset_config.name)
        dataset: SemanticSegmentationDataset = instantiate(
            {"_target_": dataset_config._target_, **dataset_config.kwargs}
        )

        # Override normalization in the dataset preprocessor to match the
        # backbone, so that the model YAML is the single source of truth.
        if feature_extractor is not None and hasattr(feature_extractor, "normalize_params"):
            norm_params = feature_extractor.normalize_params
            mean = list(norm_params["mean"])
            std = list(norm_params["std"])
            assert dataset.preprocessor is not None
            for t in dataset.preprocessor.transforms:
                if isinstance(t, transforms.Normalize):
                    t.mean = mean
                    t.std = std
                    break
            logger.info("Overrode dataset normalization with backbone params: mean=%s, std=%s", mean, std)

        return EvalDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _train_epoch(
        self,
        model: SegmentationModel,
        train_loader: EvalDataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        epoch: int,
        epochs: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        mask_size = SegmentationModel.MASK_SIZE
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]") as pbar:
            for batch_images, batch_masks, _ in pbar:
                images_gpu = batch_images.to(device)
                # Downsample masks to match feature/logit resolution
                masks_gpu = batch_masks.to(device).unsqueeze(1).float()
                masks_gpu = F.interpolate(masks_gpu, size=(mask_size, mask_size), mode="nearest")
                masks_gpu = masks_gpu.squeeze(1).long()
                optimizer.zero_grad()
                outputs = model(images_gpu)
                loss = criterion(outputs, masks_gpu)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        return total_loss / len(train_loader)

    @torch.inference_mode()
    def _evaluate(
        self,
        model: SegmentationModel,
        val_loader: EvalDataLoader,
        criterion: nn.Module,
        device: str,
        num_classes: int,
    ) -> tuple[float, dict[str, Any]]:
        model.eval()
        metrics = SegmentationMetrics(num_classes, ignore_index=self.config.ignore_index)
        total_loss = 0.0
        mask_size = SegmentationModel.MASK_SIZE
        with tqdm(val_loader, desc="Validation") as pbar:
            for batch_images, batch_masks, _ in pbar:
                images_gpu = batch_images.to(device)
                # Downsample masks to match feature/logit resolution
                masks_gpu = batch_masks.to(device).unsqueeze(1).float()
                masks_gpu = F.interpolate(masks_gpu, size=(mask_size, mask_size), mode="nearest")
                masks_gpu = masks_gpu.squeeze(1).long()
                outputs = model(images_gpu)
                loss = criterion(outputs, masks_gpu)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                metrics.update(preds, masks_gpu)
                pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        results = metrics.calculate_metrics()
        return total_loss / len(val_loader), results

    def run(self, iteration_value: dict[str, Any] | None = None, iteration_index: int = 0) -> dict:
        """Run semantic segmentation training."""
        setup_logging()

        set_seed(self.config.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        # Determine model config and learning rate based on mode
        if iteration_value is not None:
            model_config = iteration_value["model"]
            learning_rate = iteration_value["learning_rate"]
            model_name = model_config.get("name", f"model_{iteration_index}")
            feature_extractor = self._create_model_from_config(model_config, device)
            output_dir = self._get_output_dir(model_name, learning_rate)
            logger.info("Array task %d: Model=%s, LR=%f", iteration_index, model_name, learning_rate)
        else:
            if self.config.model is None or self.config.learning_rate is None:
                raise ValueError("Either provide model and learning_rate, or use sweep_configs for array mode")
            model_name = self.config.model.name
            learning_rate = self.config.learning_rate
            feature_extractor = self._create_model(device)
            output_dir = self._get_output_dir(model_name, learning_rate)
            logger.info("Single mode: Model=%s, LR=%f", model_name, learning_rate)

        logger.info("Output directory: %s", output_dir)

        train_loader = self._get_dataloader(self.config.train_dataset, feature_extractor, shuffle=True)
        val_loader = self._get_dataloader(self.config.val_dataset, feature_extractor, shuffle=False)
        test_loader = (
            self._get_dataloader(self.config.test_dataset, feature_extractor, shuffle=False)
            if self.config.test_dataset
            else None
        )

        train_dataset = cast("SemanticSegmentationDataset", train_loader.dataset)
        num_classes = train_dataset.num_classes
        class_names = train_dataset.class_names

        logger.info(
            "Dataset: %s, Classes: %d, Samples: %d", self.config.train_dataset.name, num_classes, len(train_dataset)
        )

        model = SegmentationModel(feature_extractor, num_classes, sync_bn=self.config.sync_bn)
        model.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        optimizer = torch.optim.SGD(
            model.segmentation_head.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
        scheduler = StepLR(optimizer, step_size=self.config.drop_at, gamma=0.1)

        best_miou = 0.0
        best_metrics = None
        training_history = []

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, self.config.epochs
            )
            scheduler.step()

            should_eval = (epoch + 1) % self.config.eval_period_epochs == 0 or (epoch == self.config.epochs - 1)
            if should_eval:
                val_loss, val_metrics = self._evaluate(model, val_loader, criterion, device, num_classes)
                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, mIoU: %.4f",
                    epoch + 1,
                    self.config.epochs,
                    train_loss,
                    val_loss,
                    val_metrics["miou"],
                )

                training_history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_miou": val_metrics["miou"],
                        "val_pixel_acc": val_metrics["pixel_acc"],
                        "val_mean_acc": val_metrics["mean_acc"],
                    }
                )

                if val_metrics["miou"] > best_miou:
                    best_miou = val_metrics["miou"]
                    best_metrics = val_metrics
                    checkpoint_path = str(output_dir / "best_model.pth")
                    with Path(checkpoint_path).open("wb") as f:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "metrics": val_metrics,
                            },
                            f,
                        )
                    logger.info("Saved new best model with mIoU: %.4f", best_miou)

        checkpoint_path = str(output_dir / "best_model.pth")
        with Path(checkpoint_path).open("rb") as f:
            checkpoint = torch.load(f, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics = checkpoint["metrics"]
        logger.info("Loaded best model from epoch %d with mIoU: %.4f", checkpoint["epoch"] + 1, best_metrics["miou"])

        test_metrics = None
        if test_loader is not None:
            _, test_metrics = self._evaluate(model, test_loader, criterion, device, num_classes)
            logger.info("Test mIoU: %.4f", test_metrics["miou"])

        result = {
            "model": model_name,
            "learning_rate": learning_rate,
            "iteration_index": iteration_index,
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "training_history": training_history,
            "num_classes": num_classes,
            "class_names": class_names,
            "output_dir": str(output_dir),
        }

        results_file = str(output_dir / "results.yaml")
        with Path(results_file).open("w") as f:
            f.write(to_yaml(result))
        logger.info("Saved results to %s", results_file)

        return result


# =============================================================================
# Outer Module: Sweep Coordinator
# =============================================================================


@dataclass
class SemanticSegmentationSweepConfig(EvalConfig):
    """Config for semantic segmentation hyperparameter sweep."""

    _target_: str = "evaluation.vision.semantic_segmentation.SemanticSegmentationSweepModule"

    name: str = "semantic_segmentation"

    train_dataset: EvalDatasetConfig = MISSING
    val_dataset: EvalDatasetConfig = MISSING
    test_dataset: EvalDatasetConfig | None = None

    models: list[EvalModelConfig] | None = None
    backbone: dict[str, Any] = MISSING
    pooling_strategies: list[PoolingStrategy] | None = field(
        default_factory=lambda: [
            PoolingStrategy(pooling="semantic_segmentation", last_n_layers=1),
            PoolingStrategy(pooling="semantic_segmentation", last_n_layers=4),
        ]
    )

    learning_rates: list[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    batch_size: int = 16
    epochs: int = 20
    drop_at: int = 20
    num_workers: int = 8
    eval_period_epochs: int = 5
    sync_bn: bool = False
    ignore_index: int = 255
    seed: int = 42

    launcher: LauncherConfig = field(default_factory=lambda: LauncherConfig(cluster="slurm"))


class SemanticSegmentationSweepModule(EvalModule):
    """Semantic segmentation sweep coordinator using Stopes array functionality."""

    def __init__(self, config: SemanticSegmentationSweepConfig) -> None:
        super().__init__(config, SemanticSegmentationSweepConfig)
        self.output_dir = (
            to_path(self.config.output_dir)
            / self.config.name
            / self.config.train_dataset.name
            / self._get_models_name()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Sweep output directory: %s", self.output_dir)

    def _get_models_name(self) -> str:
        if not OmegaConf.is_missing(self.config, "backbone"):
            return self.config.backbone.get("name", "backbone")
        if self.config.models is not None and len(self.config.models) > 0:
            # Extract base name from first model (strip pooling suffix if present)
            first_name = self.config.models[0].name
            # Remove common suffixes like -semantic_segmentation-L1, -L4, etc.
            for suffix in ["-semantic_segmentation", "-L1", "-L2", "-L4", "-L8"]:
                if suffix in first_name:
                    first_name = first_name.split(suffix)[0]
            return first_name
        return "models"

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1, mem_gb=16, tasks_per_node=1, gpus_per_node=0, cpus_per_task=4, timeout_min=60 * 72
        )

    def name(self) -> str:
        return f"{self.config.name}_{self.config.train_dataset.name}_{self._get_models_name()}_seed{self.config.seed}"

    def _expand_model_configs(self) -> list[EvalModelConfig]:
        if self.config.models is not None:
            return list(self.config.models)
        if OmegaConf.is_missing(self.config, "backbone"):
            raise ValueError("Either 'models' or 'backbone' must be provided")
        backbone = self.config.backbone
        backbone_target = backbone.get("_target_")
        backbone_name = backbone.get("name", "backbone")
        backbone_kwargs = backbone.get("kwargs", {})
        model_configs = []
        for strategy in self.config.pooling_strategies or []:
            suffix = strategy.name_suffix or f"{strategy.pooling}-L{strategy.last_n_layers}"
            model_config = EvalModelConfig(
                _target_=backbone_target,
                name=f"{backbone_name}-{suffix}",
                kwargs={**backbone_kwargs, "pooling": strategy.pooling, "last_n_layers": strategy.last_n_layers},
            )
            model_configs.append(model_config)
        return model_configs

    def _build_sweep_configs(self) -> list[dict[str, Any]]:
        model_configs = self._expand_model_configs()
        sweep_configs = [
            {
                "model": OmegaConf.to_container(OmegaConf.structured(model_config)),
                "learning_rate": lr,
            }
            for model_config in model_configs
            for lr in self.config.learning_rates
        ]
        logger.info(
            "Built %d sweep configs (%d models x %d LRs)",
            len(sweep_configs),
            len(model_configs),
            len(self.config.learning_rates),
        )
        return sweep_configs

    def _create_run_module(self) -> SemanticSegmentationRunModule:
        sweep_configs = self._build_sweep_configs()
        run_config = SemanticSegmentationRunConfig(
            output_dir=self.config.output_dir,
            name=self.config.name,
            train_dataset=self.config.train_dataset,
            val_dataset=self.config.val_dataset,
            test_dataset=self.config.test_dataset,
            sweep_configs=sweep_configs,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            drop_at=self.config.drop_at,
            num_workers=self.config.num_workers,
            eval_period_epochs=self.config.eval_period_epochs,
            sync_bn=self.config.sync_bn,
            ignore_index=self.config.ignore_index,
            seed=self.config.seed,
        )
        return SemanticSegmentationRunModule(run_config)

    async def _run_sweep(self) -> list[dict[str, Any]]:
        run_module = self._create_run_module()
        launcher: Launcher = instantiate(self.config.launcher)
        logger.info(
            "Launcher: %s, Scheduling %d configs", type(launcher).__name__, len(run_module.config.sweep_configs)
        )
        with clean_env():
            results = await launcher.schedule(run_module)
        logger.info("All %d array tasks completed", len(results))
        return results

    def _aggregate_results(self, all_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results from all sweep configurations.

        Args:
            all_results: List of results from each configuration (may include exceptions).

        Returns:
            Dictionary containing best configuration and all valid results.

        Raises:
            PipelineError: If any sweep configurations failed.
        """
        all_configs: list[dict[str, Any]] = []
        task_errors: list[TaskError] = []
        best_val_miou = 0.0
        best_config = None

        for idx, result in enumerate(all_results):
            if isinstance(result, Exception):
                task_name = f"sweep_config_{idx}"
                logger.warning("Sweep config %d failed: %s", idx, result)
                task_errors.append(TaskError.from_exception(task_name, result))
                continue

            all_configs.append(result)
            val_miou = result.get("best_val_metrics", {}).get("miou", 0.0)
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_config = result

        if best_config:
            logger.info(
                "Best: Model=%s, LR=%f, mIoU=%.4f", best_config["model"], best_config["learning_rate"], best_val_miou
            )

        aggregate = {
            "best_config": best_config,
            "all_configs": all_configs,
            "num_configs": len(all_configs),
            "num_classes": best_config.get("num_classes") if best_config else None,
            "class_names": best_config.get("class_names") if best_config else None,
        }

        if task_errors:
            aggregate["num_failed"] = len(task_errors)
            aggregate["failed_configs"] = [e.to_dict() for e in task_errors]
            raise PipelineError(self.config.name, task_errors, partial_results=aggregate)

        return aggregate

    def _plot_results(self, results: dict[str, Any], output_dir: Path) -> None:
        best_config = results.get("best_config")
        if not best_config:
            return
        best_metrics = best_config["best_val_metrics"]
        class_names = results.get("class_names", [])

        _fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        iou_per_class = best_metrics["iou_per_class"]
        sorted_indices = np.argsort(iou_per_class)[::-1][:20]
        axes[0].barh(
            [class_names[i][:30] if i < len(class_names) else str(i) for i in sorted_indices],
            [iou_per_class[i] for i in sorted_indices],
        )
        axes[0].set_xlabel("IoU")
        axes[0].set_title(f"Top 20 Classes by IoU (mIoU: {best_metrics['miou']:.3f})")
        axes[0].invert_yaxis()

        if best_config.get("training_history"):
            history = best_config["training_history"]
            epochs = [h["epoch"] for h in history]
            val_mious = [h["val_miou"] for h in history]
            axes[1].plot(epochs, val_mious, "b-", label="Val mIoU", marker="o")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("mIoU")
            axes[1].set_title("Training Progress (Best Config)")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / "results.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved results plot to %s", output_dir / "results.png")

    def run(
        self,
        iteration_value: object = None,
        iteration_index: int = 0,
    ) -> dict:
        """Run the semantic segmentation sweep.

        Returns:
            Aggregated results across all configurations.

        Raises:
            PipelineError: If any sweep configurations failed (after saving partial results).
        """
        setup_logging()

        logger.info("Starting semantic segmentation sweep")
        all_results = asyncio.run(self._run_sweep())

        pipeline_error = None
        try:
            results = self._aggregate_results(all_results)
        except PipelineError as e:
            results = e.get_results_with_errors({})
            pipeline_error = e

        results_file = str(self.output_dir / "results.yaml")
        with Path(results_file).open("w") as f:
            f.write(to_yaml(results))
        logger.info("Saved aggregated results to %s", results_file)

        if results.get("class_names"):
            self._plot_results(results, self.output_dir)

        if results.get("best_config"):
            best = results["best_config"]
            logger.info("=" * 60)
            logger.info(
                "FINAL: Model=%s, LR=%f, Val mIoU=%.4f",
                best["model"],
                best["learning_rate"],
                best["best_val_metrics"]["miou"],
            )
            if best.get("test_metrics"):
                logger.info("Test mIoU=%.4f", best["test_metrics"]["miou"])
            logger.info("=" * 60)

        if pipeline_error:
            raise pipeline_error

        return results


SemanticSegmentationEvalModuleConfig = SemanticSegmentationSweepConfig
SemanticSegmentationEvalModule = SemanticSegmentationSweepModule
