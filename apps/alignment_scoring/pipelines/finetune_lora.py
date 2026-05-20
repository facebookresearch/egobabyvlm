# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA finetuning of CLIP / Perception Encoder backbones.

Distributed (DDP) finetune loop using PEFT for the LoRA wrapping. Each epoch:

1. Trains the LoRA-wrapped model on the train manifest with cosine LR schedule.
2. Periodically validates mid-epoch and saves a "best" checkpoint when val loss
   improves — important so SLURM preemption can't lose hours of work.
3. At epoch end, evaluates again, optionally early-stops on plateau.
4. On exit, reloads the best checkpoint and writes it in open_clip's
   safetensors format so it's a drop-in replacement for any open_clip
   checkpoint elsewhere in the pipeline (including ``alignment-clip-scoring``).

Checkpointing is full-state (model + optimizer + scaler + epoch + step) so
``resume=`` (or just rerunning) picks up exactly where it left off, even
mid-epoch. Signal handlers (SIGTERM/SIGINT) save before exit.

Run with::

    alignment-finetune-lora --config-path apps/alignment_scoring/configs \\
        --config-name pipeline/finetune_lora \\
        name=coco_lora \\
        ++data_train.dataset.manifest_path=/data/coco/captions_train2017.json \\
        ++data_train.dataset.dataset_dir=/data/coco/train2017 \\
        ++data_val.dataset.manifest_path=/data/coco/captions_val2017.json \\
        ++data_val.dataset.dataset_dir=/data/coco/val2017

For multi-GPU, launch via torchrun (or srun on SLURM) — the trainer reads
``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` from the env. The trained adapter
weights merge cleanly into the base model so the saved checkpoint can be
loaded via ``open_clip.create_model_and_transforms``.
"""

from __future__ import annotations

import itertools
import logging
import math
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from open_clip.loss import ClipLoss
from open_clip_train.train import get_clip_metrics
from torch.nn.parallel import DistributedDataParallel

from apps.alignment_scoring.configs import FinetuneLoraConfig
from apps.alignment_scoring.utils import (
    build_optimizer,
    clip_forward,
    create_alignment_dataloader,
    create_model,
    post_collate_fn,
    save_openclip_checkpoint,
)
from core.utils import (
    MetricLogger,
    SmoothedValue,
    all_gather_tensor,
    all_reduce_mean,
    get_world_size,
    init_wandb,
    is_dist_avail_and_initialized,
    is_main_process,
    load_checkpoint,
    resolve_and_print_config,
    save_checkpoint,
    set_seed,
    setup_distributed,
    setup_logging,
    to_yaml,
    unwrap_model,
    wandb_log,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import open_clip
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def _logit_scale(model: torch.nn.Module) -> torch.Tensor:
    """Typed accessor for ``model.logit_scale`` (mypy can't see through nn.Module getattr)."""
    return cast("torch.Tensor", model.logit_scale)


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    config: FinetuneLoraConfig,
) -> None:
    """Cosine LR schedule with linear warmup (per-step, not per-epoch)."""
    if epoch < config.optim.warmup_epochs:
        lr = config.optim.lr * epoch / max(config.optim.warmup_epochs, 1e-6)
    else:
        progress = (epoch - config.optim.warmup_epochs) / max(
            config.optim.epochs - config.optim.warmup_epochs,
            1e-6,
        )
        lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        scale = group.get("lr_scale", 1.0)
        group["lr"] = lr * scale


def _evaluate(
    model: torch.nn.Module,
    tokenizer: open_clip.SimpleTokenizer,
    dataloader: DataLoader,
    device: torch.device | str,
    dtype: torch.dtype,
    *,
    is_video_dataset: bool = False,
) -> dict[str, float]:
    """Run one validation pass under DDP; returns CLIP retrieval metrics + mean loss."""
    model.eval()
    model_no_ddp = unwrap_model(model)
    loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, rank=0, world_size=1)
    metric_logger = MetricLogger(delimiter="  ")

    cumulative_loss = 0.0
    num_samples = 0
    all_image_features: list[torch.Tensor] = []
    all_text_features: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in metric_logger.log_every(dataloader, print_freq=50, header="Val:"):
            image_features, text_features = clip_forward(
                model_no_ddp,
                post_collate_fn(batch, tokenizer, device, dtype),
                dtype,
                is_video_dataset=is_video_dataset,
                normalize=True,
            )
            logit_scale = _logit_scale(model_no_ddp).exp().mean()
            loss = loss_fn(image_features, text_features, logit_scale, output_dict=False)
            cumulative_loss += loss.item() * image_features.size(0)
            num_samples += image_features.size(0)
            all_image_features.append(image_features)
            all_text_features.append(text_features)

    metric_logger.synchronize_between_processes()
    image_features_all = all_gather_tensor(torch.cat(all_image_features)).cpu()
    text_features_all = all_gather_tensor(torch.cat(all_text_features)).cpu()

    val_metrics = get_clip_metrics(
        image_features=image_features_all,
        text_features=text_features_all,
        logit_scale=logit_scale.cpu(),
    )
    val_metrics["loss"] = all_reduce_mean(cumulative_loss / max(num_samples, 1), device=device)
    val_metrics["logit_scale"] = float(logit_scale.item())
    logger.info(
        "Validation: %s",
        "  ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()),
    )
    return val_metrics


def _train_one_epoch(
    config: FinetuneLoraConfig,
    epoch: int,
    model: torch.nn.Module,
    tokenizer: open_clip.SimpleTokenizer,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler,
    device: torch.device | str,
    *,
    best_loss: float = float("inf"),
    is_video_dataset: bool = False,
) -> tuple[dict[str, float], float]:
    """Train one epoch with mid-epoch validation + best-checkpoint saves."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dtype = getattr(torch, config.optim.dtype, torch.bfloat16)
    model_no_ddp = unwrap_model(model)
    loss_fn = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=get_world_size(),
    )

    num_batches = len(dataloader)
    epoch_offset = config.optim.epoch_offset or 0
    train_batches: Iterable
    if epoch_offset:
        logger.info("Skipping the first %d batches in epoch %d", epoch_offset, epoch)
        train_batches = itertools.islice(dataloader, epoch_offset, None)
    else:
        train_batches = dataloader

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.set_header(f"Epoch: [{epoch}]")

    for batch_idx, batch in enumerate(metric_logger.log_every(train_batches, print_freq=config.log_interval)):
        epoch_step = batch_idx + epoch_offset
        epoch_progress = epoch_step / num_batches + epoch
        epoch_1000x = int(epoch_progress * 1000)

        _adjust_learning_rate(optimizer, epoch_progress, config)

        image_features, text_features = clip_forward(
            model_no_ddp,
            post_collate_fn(batch, tokenizer, device, dtype),
            dtype,
            is_video_dataset=is_video_dataset,
            normalize=True,
        )
        loss = loss_fn(image_features, text_features, _logit_scale(model_no_ddp).exp(), output_dict=False)

        grad_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=True,
            clip_grad=config.optim.clip_grad_norm,
        )
        optimizer.zero_grad()

        # CLIP paper appendix B: clamp logit_scale to [0, log(100)].
        with torch.no_grad():
            _logit_scale(model_no_ddp).clamp_(0, math.log(100))

        metric_logger.update(loss=loss.item())
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if (epoch_step + 1) % config.log_interval == 0 and config.wandb.enabled:
            log_stats = {f"train/{k}": meter.value for k, meter in metric_logger.meters.items()}
            log_stats.update(
                {
                    "train/logit_scale": _logit_scale(model_no_ddp).exp().item(),
                    "train/logit_scale_raw": _logit_scale(model_no_ddp).item(),
                    "train/loss": all_reduce_mean(loss.item(), device=device),
                    "train/step": epoch_1000x,
                }
            )
            wandb_log(log_stats)

        if (epoch_step + 1) % config.eval_interval == 0:
            val_stats = _evaluate(
                model=model,
                tokenizer=tokenizer,
                dataloader=val_dataloader,
                device=device,
                dtype=dtype,
                is_video_dataset=is_video_dataset,
            )
            if val_stats["loss"] < best_loss:
                best_loss = val_stats["loss"]
                save_checkpoint(
                    output_dir=config.output_dir,
                    epoch=epoch,
                    container={
                        "model": model_no_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                    },
                    config=config,
                    max_checkpoints=config.max_checkpoints,
                    epoch_offset=epoch_step,
                    is_best=True,
                    lora_only=True,
                )
            val_stats.update({"best_loss": best_loss, "step": epoch_1000x})
            if config.wandb.enabled:
                wandb_log({f"val_inner/{k}": v for k, v in val_stats.items()}, disable_format=True)
            model.train()

    # Reset epoch_offset after a successful epoch (only the resumed epoch is partial).
    if config.optim.epoch_offset is not None:
        config.optim.epoch_offset = None

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", str(metric_logger))
    return {k: meter.value for k, meter in metric_logger.meters.items()}, best_loss


def finetune(config: FinetuneLoraConfig) -> str:
    """Run the full DDP LoRA finetune loop; returns the path to the best checkpoint."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    setup_distributed()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    if config.model.lora is None:
        raise ValueError("config.model.lora must be set — use the lora group on the CLI")

    model, preprocess, tokenizer, open_clip_config = create_model(config.model, device=device)

    if config.wandb.enabled:
        init_wandb(
            project=config.wandb.project,
            entity=config.wandb.entity,
            output_dir=config.output_dir,
            log_dir=config.output_dir,
            config=hydra.utils.instantiate(config) if False else None,
            run_id=config.wandb.run_id,
            resume=config.resume is not None,
            log_code=config.wandb.log_code,
            model=model,
        )

    ddp_model: torch.nn.Module = (
        DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
        if is_dist_avail_and_initialized()
        else model
    )

    train_loader, is_video_dataset = create_alignment_dataloader(
        config.data_train,
        indices=None,
        preprocessor=preprocess,
        mode="train",
        distributed=is_dist_avail_and_initialized(),
    )
    val_loader, _ = create_alignment_dataloader(
        config.data_val,
        indices=None,
        preprocessor=preprocess,
        mode="eval",
        distributed=is_dist_avail_and_initialized(),
    )
    optimizer, scaler = build_optimizer(config.optim, model)

    start_epoch, epoch_offset = load_checkpoint(
        container={"model": model, "optimizer": optimizer, "scaler": scaler},
        output_dir=config.output_dir,
        resume=config.resume,
    )
    if start_epoch:
        config.optim.start_epoch = start_epoch
    if epoch_offset:
        config.optim.epoch_offset = epoch_offset

    def _signal_handler(signum: int, _frame: object) -> None:
        logger.warning("Received signal %d — saving current model in OpenCLIP format", signum)
        save_openclip_checkpoint(unwrap_model(ddp_model), open_clip_config, config.output_dir, tag="latest")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Configuration:\n%s", to_yaml(cast("DictConfig", config)))
    logger.info("Training %d epochs", config.optim.epochs)
    start_time = time.time()
    best_loss = float("inf")
    patience = config.optim.patience

    try:
        for epoch in range(config.optim.start_epoch, config.optim.epochs):
            if is_dist_avail_and_initialized():
                cast("DistributedSampler", train_loader.sampler).set_epoch(epoch)

            train_stats, best_loss_inner = _train_one_epoch(
                config=config,
                epoch=epoch,
                model=ddp_model,
                tokenizer=tokenizer,
                dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                loss_scaler=scaler,
                device=device,
                best_loss=best_loss,
                is_video_dataset=is_video_dataset,
            )

            dtype = getattr(torch, config.optim.dtype, torch.bfloat16)
            val_stats = _evaluate(
                model=ddp_model,
                tokenizer=tokenizer,
                dataloader=val_loader,
                device=device,
                dtype=dtype,
                is_video_dataset=is_video_dataset,
            )

            inner_improved = best_loss_inner < best_loss
            end_of_epoch_improved = val_stats["loss"] < best_loss

            if end_of_epoch_improved:
                patience = config.optim.patience
                is_best = True
                best_loss = val_stats["loss"]
            else:
                patience -= 1
                is_best = False

            if inner_improved:
                best_loss = best_loss_inner

            if end_of_epoch_improved or not inner_improved:
                save_checkpoint(
                    output_dir=config.output_dir,
                    epoch=epoch,
                    container={
                        "model": unwrap_model(ddp_model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    },
                    config=config,
                    max_checkpoints=config.max_checkpoints,
                    is_best=is_best,
                    lora_only=True,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "val_best_loss": best_loss,
                "epoch": epoch,
            }
            if config.wandb.enabled:
                wandb_log(log_stats)

            if config.optim.early_stopping and patience == 0:
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    config.optim.patience,
                )
                break

        logger.info("Training time: %.1fs", time.time() - start_time)

    finally:
        # Reload the best checkpoint and emit it in open_clip format so the
        # exported "best" alias actually points at the best epoch (not the last).
        if is_main_process():
            best_path = Path(config.output_dir) / "best_model.pth"
            if best_path.exists():
                logger.info("Reloading %s before final OpenCLIP export", best_path)
                load_checkpoint(
                    container={"model": unwrap_model(ddp_model)},
                    output_dir=config.output_dir,
                    resume=str(best_path),
                )
            save_openclip_checkpoint(unwrap_model(ddp_model), open_clip_config, config.output_dir, tag="best")

    return str(Path(config.output_dir) / "openclip_checkpoint_best")


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/finetune_lora")
def main(config: DictConfig) -> None:
    """CLI entrypoint."""
    import apps.alignment_scoring  # noqa: F401

    setup_logging()
    resolve_and_print_config(config)
    try:
        finetune(cast("FinetuneLoraConfig", config))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
