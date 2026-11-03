# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""FlamingoTrainer: pure-PyTorch DDP trainer in the CLIP-baseline style.

* The optimizer is built by the caller with the weight-decay param-group split
  in :mod:`apps.baselines.flamingo.training.optim`.
* :meth:`validate` computes a held-out caption cross-entropy; ``best.pt`` is
  selected on **min ``val/loss``** (plus a grounding-accuracy proxy logged for
  observability). The paper evaluates the final epoch, not ``best.pt``.
* Gate openness (``tanh(gate)``) is logged to W&B each epoch.

The loss itself (caption LL + optional aux terms) is computed inside the
DDP-wrapped :class:`FlamingoTrainingModule`, so a single wrapper forward → single
backward keeps DDP correct even with the multi-forward aux objectives.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from apps.baselines.flamingo.training.checkpoint import load_checkpoint, save_checkpoint
from apps.baselines.flamingo.training.losses import sequence_ll
from core.utils.checkpoints import prune_checkpoints
from core.utils.distributed import (
    is_dist_avail_and_initialized,
    is_main_process,
    unwrap_model,
)
from core.utils.wandb import finish_wandb, wandb_log

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.nn.parallel import DistributedDataParallel
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from apps.baselines.flamingo.modeling.flamingo_model import FlamingoVLM
    from apps.baselines.flamingo.training.losses import FlamingoTrainingModule

logger = logging.getLogger(__name__)


class FlamingoTrainer:
    """DDP trainer for the native Flamingo VLM."""

    def __init__(
        self,
        *,
        training_module: FlamingoTrainingModule | DistributedDataParallel,
        model: FlamingoVLM,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: DictConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.training_module = training_module
        self.model = model  # unwrapped FlamingoVLM (for save + gate logging)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------ helpers

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }

    def _max_steps_reached(self) -> bool:
        """True once the optional ``max_steps`` optimizer-step cap is hit.

        ``config.max_steps is None`` (the default) means unlimited, so the full
        ``epochs`` schedule runs unchanged.
        """
        max_steps = getattr(self.config, "max_steps", None)
        return max_steps is not None and self.global_step >= int(max_steps)

    # ------------------------------------------------------------------ epoch

    def train_epoch(self) -> dict[str, float]:
        self.training_module.train()
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

        running: dict[str, list[float]] = {}
        for raw_batch in self.train_loader:
            batch = self._move_batch(raw_batch)
            loss, metrics = self.training_module(batch)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad = self.config.optim.grad_clip
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.training_module.parameters(), clip_grad)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in metrics.items():
                running.setdefault(k, []).append(v)

            self.global_step += 1
            if self.global_step % self.config.log_interval == 0:
                self._log_step(metrics)

            if self._max_steps_reached():
                logger.info(
                    "[step %d] reached max_steps=%s — stopping epoch early",
                    self.global_step,
                    self.config.max_steps,
                )
                break

        return {k: sum(v) / len(v) for k, v in running.items() if v}

    def _log_step(self, metrics: dict[str, float]) -> None:
        line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("[step %d] %s", self.global_step, line)
        payload = {f"train/{k}": v for k, v in metrics.items()}
        payload["step"] = self.global_step
        if self.scheduler is not None:
            payload["train/lr"] = self.optimizer.param_groups[0]["lr"]
        wandb_log(payload, step=self.global_step, disable_format=True)

    # ------------------------------------------------------------------ validation

    @torch.no_grad()
    def validate(self) -> dict[str, float] | None:
        """Held-out caption cross-entropy (+ a correct-vs-foil grounding proxy)."""
        if self.val_loader is None:
            return None
        self.training_module.eval()
        vlm = cast("FlamingoVLM", unwrap_model(self.model))
        total_loss = 0.0
        total_correct = 0
        total = 0
        n = 0
        for raw_batch in self.val_loader:
            batch = self._move_batch(raw_batch)
            out = vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
            )
            total_loss += float(out.loss.item())
            n += 1

            # Grounding proxy: correct caption should out-score the rolled foil
            # for the same image (the MDB correct-vs-foil comparison).
            image_embeds = vlm.encode_images(batch["pixel_values"])
            ll_pos = sequence_ll(vlm, batch["input_ids"], batch["attention_mask"], batch["labels"], image_embeds)
            ll_neg = sequence_ll(
                vlm, batch["neg_input_ids"], batch["neg_attention_mask"], batch["neg_labels"], image_embeds
            )
            total_correct += int((ll_pos > ll_neg).sum().item())
            total += int(ll_pos.shape[0])

        if is_dist_avail_and_initialized():
            # Each rank saw only its DistributedSampler shard; sum across ranks.
            stats = torch.tensor([total_loss, n, total_correct, total], dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(stats)
            total_loss, n, total_correct, total = (
                stats[0].item(),
                int(stats[1].item()),
                int(stats[2].item()),
                int(stats[3].item()),
            )
        if n == 0:
            return None
        return {
            "val/loss": total_loss / n,
            "val/grounding_acc": (total_correct / total) if total else 0.0,
        }

    # ------------------------------------------------------------------ checkpoint

    def _save(self, tag: str = "latest", *, epoch_complete: bool = True) -> Path | None:
        if not is_main_process():
            return None
        save_dir = Path(self.config.checkpoint.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{tag}.pt"
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            epoch=self.epoch,
            step=self.global_step,
            best_val_loss=self.best_val_loss,
            epoch_complete=epoch_complete,
        )
        logger.info("Saved checkpoint to %s", path)
        return path

    def _prune_old_epoch_checkpoints(self) -> None:
        if not is_main_process():
            return
        prune_checkpoints(self.config.checkpoint.save_dir, "epoch_*.pt", int(self.config.checkpoint.keep_last))

    def resume(self, path: str | Path) -> None:
        """Resume after the last *completed* epoch stored in ``path``.

        Mid-epoch saves (``interrupted.pt``) are rejected: restarting after them
        would skip the rest of that epoch. Resume from ``latest.pt`` instead,
        which is written at the end of every epoch.
        """
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        if not payload.get("epoch_complete", True):
            msg = f"{path} was saved mid-epoch; resume from latest.pt (end of the last completed epoch) instead."
            raise ValueError(msg)
        self.epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("step", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        logger.info("Resumed from %s at epoch %d, step %d", path, self.epoch, self.global_step)

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: object) -> None:
            logger.warning("Received signal %d — saving checkpoint before exit", signum)
            self._save(tag="interrupted", epoch_complete=False)
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # ------------------------------------------------------------------ driver

    def fit(self) -> None:
        if self.config.checkpoint.resume_from is not None:
            self.resume(self.config.checkpoint.resume_from)
        self._install_signal_handlers()

        logger.info("Starting Flamingo training: epochs=%d", self.config.epochs)
        start = time.time()
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                epoch_metrics = self.train_epoch()
                logger.info("Epoch %d train: %s", epoch, epoch_metrics)

                gate = cast("FlamingoVLM", unwrap_model(self.model)).gate_summary()
                wandb_log(
                    {**{f"epoch/{k}": v for k, v in epoch_metrics.items()}, **gate, "epoch": epoch},
                    step=self.global_step,
                    disable_format=True,
                )

                # Honor the optional optimizer-step cap. Do this BEFORE the
                # (epoch-oriented, potentially very large) validation pass: for a
                # step-capped smoke run we just save a checkpoint via the normal
                # save path and stop, rather than validating over the full set.
                if self._max_steps_reached():
                    self._save(tag="latest")
                    logger.info(
                        "Reached max_steps=%s at step %d — saved checkpoint and stopping training",
                        self.config.max_steps,
                        self.global_step,
                    )
                    break

                val_metrics = self.validate()
                if val_metrics is not None:
                    logger.info("Epoch %d val: %s", epoch, val_metrics)
                    wandb_log({**val_metrics, "epoch": epoch}, step=self.global_step, disable_format=True)
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
