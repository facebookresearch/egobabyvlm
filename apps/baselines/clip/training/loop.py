# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ContrastiveTrainer: pure-PyTorch DDP trainer for the contrastive baselines.

A single mode-agnostic loop dispatches each step to the loss head named
by :class:`InterleaveScheduler`:

* ``contrastive`` — InfoNCE on a (image, text) batch.
* ``mlm``         — BERT MLM on a raw-text batch.
* ``dinov2``      — DINOv2 SSL forward/backward + teacher EMA.

After a ``dinov2`` block exits, the teacher backbone is copied into the
CLIP vision encoder if ``mode.sync_vision_from_dinov2`` is set.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from apps.baselines.clip.modeling import MLMHead
from apps.baselines.clip.training.checkpoint import load_checkpoint, save_checkpoint
from core.utils.checkpoints import prune_checkpoints
from core.utils.distributed import (
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    unwrap_model,
)
from core.utils.wandb import finish_wandb, wandb_log

if TYPE_CHECKING:
    from collections.abc import Iterator

    from omegaconf import DictConfig
    from torch.nn.parallel import DistributedDataParallel
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from apps.baselines.clip.modeling import DINOv2SSL, MultiModalModel
    from apps.baselines.clip.training.interleave import InterleaveScheduler

logger = logging.getLogger(__name__)


def _cycle_iter(loader: DataLoader) -> Iterator[Any]:
    """Infinite iterator that restarts on StopIteration. Honors DistributedSampler.set_epoch."""
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


class ContrastiveTrainer:
    """Mode-agnostic DDP trainer.

    Args:
        model: The :class:`MultiModalModel` (already DDP-wrapped if applicable).
        contrastive_optimizer: Optimizer for the contrastive loss.
        scheduler: :class:`InterleaveScheduler` deciding which mode runs each step.
        train_loader: Image-text DataLoader for the contrastive loss.
        config: Full training config.
        device: Training device.
        val_loader: Optional image-text DataLoader for validation.
        mlm_head: Optional MLM head (required when ``mode.interleave["mlm"] > 0``).
        mlm_optimizer: Required iff ``mlm_head`` is set.
        mlm_loader: Required iff ``mlm_head`` is set.
        ssl: Optional :class:`DINOv2SSL` (required when ``mode.interleave["dinov2"] > 0``).
        sync_vision_from_ssl: Copy SSL teacher backbone into CLIP vision encoder
            after every ``dinov2`` block.

    Note:
        W&B init (including ``define_metric`` for the ``train/*`` / ``epoch/*``
        / ``val/*`` x-axes) is set up by the caller before the trainer is
        constructed; logging here goes through :func:`core.utils.wandb.wandb_log`.
    """

    def __init__(
        self,
        *,
        model: MultiModalModel | DistributedDataParallel,
        contrastive_optimizer: Optimizer,
        scheduler: InterleaveScheduler,
        train_loader: DataLoader,
        config: DictConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
        mlm_head: MLMHead | DistributedDataParallel | None = None,
        mlm_optimizer: Optimizer | None = None,
        mlm_loader: DataLoader | None = None,
        ssl: DINOv2SSL | None = None,
        sync_vision_from_ssl: bool = False,
    ) -> None:
        self.model = model
        self.contrastive_optimizer = contrastive_optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.mlm_head = mlm_head
        self.mlm_optimizer = mlm_optimizer
        self.mlm_loader = mlm_loader
        self.mlm_iter: Iterator[Any] | None = _cycle_iter(mlm_loader) if mlm_loader is not None else None

        self.ssl = ssl
        self.sync_vision_from_ssl = sync_vision_from_ssl
        self.ssl_iteration = 0

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self._validate_mode()

    def _validate_mode(self) -> None:
        modes = set(self.scheduler.modes)
        if "mlm" in modes and (self.mlm_head is None or self.mlm_optimizer is None or self.mlm_iter is None):
            raise ValueError("mode contains 'mlm' but mlm_head/mlm_optimizer/mlm_loader is not set")
        if "dinov2" in modes and self.ssl is None:
            raise ValueError("mode contains 'dinov2' but ssl is not set")
        if self.sync_vision_from_ssl:
            self._validate_sync_compatibility()

    def _validate_sync_compatibility(self) -> None:
        """Fail at construction if cross-tower sync would silently lose weights.

        The DINOv2 SSL teacher and the contrastive vision encoder must share an
        architecture and image_size for ``load_state_dict(strict=True)`` to copy
        the teacher's weights cleanly. The common breakage is a Hub-loaded
        contrastive encoder (trained at 518x518 -> 1370 patches) paired with an
        SSL student trained at 224x224 (257 patches) -- pos_embed shapes differ
        and silently dropping it leaves the contrastive encoder half-stale.

        Also require the SSL model to have been resumed from a pretrained
        checkpoint (``dinov2.pretrained_dir``).
        """
        assert self.ssl is not None
        if self.ssl._pretrained_iteration <= 0:
            raise ValueError(
                "sync_vision_from_dinov2=True requires cfg.dinov2.pretrained_dir to be set so the SSL "
                "teacher is initialized from pretrained weights before the first cross-tower sync.",
            )
        vision = self._model().image_embed
        ssl_arch = self.ssl.arch
        ssl_image_size = self.ssl.image_size
        vision_arch = getattr(vision, "arch", None)
        vision_image_size = getattr(vision, "image_size", None)
        if vision_arch != ssl_arch or vision_image_size != ssl_image_size:
            raise ValueError(
                "sync_vision_from_dinov2=True but the contrastive vision encoder is incompatible with the SSL "
                f"student: vision_encoder=(arch={vision_arch!r}, image_size={vision_image_size}) vs "
                f"DINOv2 SSL=(arch={ssl_arch!r}, image_size={ssl_image_size}). Either disable the sync or use "
                "matching architectures (e.g. CustomDINOv2VisionEncoder loading the same SSL config).",
            )

    def _model(self) -> MultiModalModel:
        return cast("MultiModalModel", unwrap_model(self.model))

    def _mlm(self) -> MLMHead:
        assert self.mlm_head is not None
        return cast("MLMHead", unwrap_model(self.mlm_head))

    # ------------------------------------------------------------------
    # Per-mode steps
    # ------------------------------------------------------------------

    def contrastive_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        captions = batch["captions"]
        out = self._model().compute_contrastive_loss(images, captions)
        self.contrastive_optimizer.zero_grad()
        out.loss.backward()
        clip_grad = self.config.optim.grad_clip
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
        self.contrastive_optimizer.step()
        return {
            "loss": out.loss.item(),
            "image_accuracy": out.image_accuracy.item(),
            "text_accuracy": out.text_accuracy.item(),
            "image_entropy": out.image_entropy.item(),
            "text_entropy": out.text_entropy.item(),
        }

    def mlm_step(self) -> dict[str, float]:
        assert self.mlm_iter is not None
        assert self.mlm_optimizer is not None
        batch = next(self.mlm_iter)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        text_encoder = self._model().text_embed
        outputs = text_encoder.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        prediction_scores = self._mlm()(outputs.last_hidden_state)
        loss = MLMHead.loss(prediction_scores, batch["labels"])
        accuracy = MLMHead.accuracy(prediction_scores, batch["labels"])

        self.mlm_optimizer.zero_grad()
        loss.backward()
        clip_grad = self.config.optim.grad_clip
        if clip_grad:
            params = list(text_encoder.parameters()) + list(self._mlm().parameters())
            torch.nn.utils.clip_grad_norm_(params, clip_grad)
        self.mlm_optimizer.step()
        return {"mlm_loss": loss.item(), "mlm_accuracy": accuracy.item()}

    def dinov2_step(self, images: torch.Tensor) -> dict[str, float]:
        assert self.ssl is not None
        ssl_batch = self.ssl.prepare_batch(images)
        results = self.ssl.step(ssl_batch, self.ssl_iteration)
        self.ssl_iteration += 1
        return results

    def _on_dinov2_block_exit(self) -> None:
        if not self.sync_vision_from_ssl or self.ssl is None:
            return
        teacher_state = self.ssl.teacher_backbone_state_dict()
        # Strict load: incompatibility was already caught at construction by
        # _validate_sync_compatibility(). If we reach here with mismatched
        # shapes / keys it's a bug and we should fail loudly.
        backbone = cast("torch.nn.Module", self._model().image_embed.backbone)
        backbone.load_state_dict(teacher_state, strict=True)
        logger.info("Synced contrastive vision encoder from DINOv2 teacher backbone (%d keys)", len(teacher_state))

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        if self.mlm_head is not None:
            self.mlm_head.train()
        if self.ssl is not None:
            self.ssl.model.train()

        # Honor DistributedSampler shuffling per epoch.
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

        running: dict[str, list[float]] = {}
        for batch in self.train_loader:
            mode, advanced = self.scheduler.step()
            metrics = self._dispatch_step(mode, batch)
            for k, v in metrics.items():
                running.setdefault(f"{mode}/{k}", []).append(v)

            if advanced and mode == "dinov2":
                self._on_dinov2_block_exit()

            self.global_step += 1
            if self.global_step % self.config.log_interval == 0:
                self._log_step(mode, metrics)

        return {k: sum(v) / len(v) for k, v in running.items() if v}

    def _dispatch_step(self, mode: str, batch: dict[str, Any]) -> dict[str, float]:
        if mode == "contrastive":
            return self.contrastive_step(batch)
        if mode == "mlm":
            return self.mlm_step()
        if mode == "dinov2":
            return self.dinov2_step(batch["images"].to(self.device, non_blocking=True))
        raise ValueError(f"Unknown training mode: {mode!r}")

    def _log_step(self, mode: str, metrics: dict[str, float]) -> None:
        prefix = f"train/{mode}"
        line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("[step %d, mode=%s] %s", self.global_step, mode, line)
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
        payload["step"] = self.global_step
        wandb_log(payload, step=self.global_step, disable_format=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        total_img_acc = 0.0
        total_txt_acc = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            captions = batch["captions"]
            out = self._model().compute_contrastive_loss(images, captions)
            total_loss += out.loss.item()
            total_img_acc += out.image_accuracy.item()
            total_txt_acc += out.text_accuracy.item()
            n += 1
        if n == 0:
            return None
        return {
            "val/loss": total_loss / n,
            "val/image_accuracy": total_img_acc / n,
            "val/text_accuracy": total_txt_acc / n,
        }

    # ------------------------------------------------------------------
    # Checkpoint + driver
    # ------------------------------------------------------------------

    def _save(self, tag: str = "latest") -> Path | None:
        if not is_main_process():
            return None
        save_dir = Path(self.config.checkpoint.save_dir)
        path = save_dir / f"{tag}.pt"
        save_checkpoint(
            path,
            model=self._model(),
            optimizers=self._all_optimizers(),
            scheduler=self.scheduler,
            config=self.config,
            epoch=self.epoch,
            step=self.global_step,
            best_val_loss=self.best_val_loss,
            mlm_head=self._mlm() if self.mlm_head is not None else None,
            ssl=self.ssl,
        )
        logger.info("Saved checkpoint to %s", path)
        return path

    def _prune_old_epoch_checkpoints(self) -> None:
        """Keep only the most recent ``checkpoint.keep_last`` ``epoch_*.pt`` files."""
        if not is_main_process():
            return
        prune_checkpoints(
            self.config.checkpoint.save_dir,
            "epoch_*.pt",
            int(self.config.checkpoint.keep_last),
        )

    def _all_optimizers(self) -> dict[str, Optimizer]:
        opts = {"contrastive": self.contrastive_optimizer}
        if self.mlm_optimizer is not None:
            opts["mlm"] = self.mlm_optimizer
        return opts

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(
            path,
            model=self._model(),
            optimizers=self._all_optimizers(),
            scheduler=self.scheduler,
            mlm_head=self._mlm() if self.mlm_head is not None else None,
            ssl=self.ssl,
            map_location=self.device,
        )
        self.epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("step", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        logger.info("Resumed from %s at epoch %d, step %d", path, self.epoch, self.global_step)

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: object) -> None:
            logger.warning("Received signal %d — saving checkpoint before exit", signum)
            self._save(tag="interrupted")
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def fit(self) -> None:
        if self.config.checkpoint.resume_from is not None:
            self.resume(self.config.checkpoint.resume_from)

        self._install_signal_handlers()

        logger.info(
            "Starting training: world_size=%d, modes=%s, epochs=%d",
            get_world_size(),
            self.scheduler.modes,
            self.config.epochs,
        )
        start = time.time()
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                epoch_metrics = self.train_epoch()
                logger.info("Epoch %d train: %s", epoch, epoch_metrics)
                wandb_log(
                    {**{f"epoch/{k}": v for k, v in epoch_metrics.items()}, "epoch": epoch},
                    step=self.global_step,
                    disable_format=True,
                )

                val_metrics = self.validate()
                if val_metrics is not None:
                    logger.info("Epoch %d val: %s", epoch, val_metrics)
                    wandb_log(
                        {**val_metrics, "epoch": epoch},
                        step=self.global_step,
                        disable_format=True,
                    )
                    val_loss = val_metrics["val/loss"]
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save(tag="best")

                if (epoch + 1) % self.config.checkpoint.save_every == 0:
                    self._save(tag=f"epoch_{epoch:04d}")
                    self._prune_old_epoch_checkpoints()
                self._save(tag="latest")

                if is_dist_avail_and_initialized():
                    torch.distributed.barrier()
        finally:
            elapsed = time.time() - start
            logger.info("Training finished in %.1fs", elapsed)
            finish_wandb()
