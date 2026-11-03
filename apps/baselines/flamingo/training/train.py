# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""@hydra.main entrypoint for the native Flamingo trainer."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import apps.baselines.flamingo  # registers Hydra ConfigStore nodes  # noqa: F401
from apps.baselines.clip.data import build_eval_transform, build_train_transform
from apps.baselines.flamingo.data import flamingo_collate
from apps.baselines.flamingo.modeling.flamingo_model import build_flamingo_model_and_tokenizer
from apps.baselines.flamingo.training.loop import FlamingoTrainer
from apps.baselines.flamingo.training.losses import FlamingoTrainingModule, ObjectiveWeights
from apps.baselines.flamingo.training.optim import build_adamw, build_cosine_scheduler
from core.utils.distributed import (
    distributed_environment,
    is_dist_avail_and_initialized,
    setup_distributed,
)
from core.utils.logging import resolve_and_print_config, setup_logging
from core.utils.seeding import set_seed
from core.utils.wandb import init_wandb

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _build_flamingo_config_dict(cfg: DictConfig, *, use_aux_heads: bool) -> dict[str, Any]:
    """Turn ``cfg.model`` into a plain dict of FlamingoConfig fields."""
    model_dict = cast("dict[str, Any]", OmegaConf.to_container(cfg.model, resolve=True))
    model_dict["use_aux_heads"] = use_aux_heads
    return model_dict


def _build_loaders(
    cfg: DictConfig,
    tokenizer: Any,  # noqa: ANN401
) -> tuple[DataLoader, DataLoader | None]:
    train_tf = build_train_transform(image_size=cfg.data.image_size, augment=cfg.data.augment)
    val_tf = build_eval_transform(image_size=cfg.data.image_size)

    # The dataset builder returns (train_ds, val_ds); tokenizer + transforms are
    # injected at call time (mirrors CLIP's transform injection).
    train_ds, val_ds = instantiate(
        cfg.data.dataset,
        tokenizer=tokenizer,
        transform=train_tf,
        val_transform=val_tf,
    )

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    collate = partial(flamingo_collate, pad_token_id=pad_id)

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        if val_ds is not None:
            val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.data.val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=collate,
        )
        if val_ds is not None
        else None
    )
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    setup_distributed()
    env = distributed_environment()
    set_seed(cfg.seed + env.global_rank)

    device = torch.device(f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu")
    resolve_and_print_config(cfg)
    Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)

    weights = ObjectiveWeights(
        caption_loss_weight=cfg.objective.caption_loss_weight,
        hard_negative_loss_weight=cfg.objective.hard_negative_loss_weight,
        hard_negative_margin=cfg.objective.hard_negative_margin,
        itm_loss_weight=cfg.objective.itm_loss_weight,
        contrastive_loss_weight=cfg.objective.contrastive_loss_weight,
    )

    model_cfg_dict = _build_flamingo_config_dict(cfg, use_aux_heads=weights.uses_aux_heads)
    model, tokenizer = build_flamingo_model_and_tokenizer(model_cfg_dict)
    model.to(device)

    train_loader, val_loader = _build_loaders(cfg, tokenizer)

    training_module: FlamingoTrainingModule | DistributedDataParallel = FlamingoTrainingModule(model, weights)
    optimizer = build_adamw(
        model,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
        split_weight_decay=cfg.optim.split_weight_decay,
    )
    total_steps = max(len(train_loader) * cfg.epochs, 1)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, min_lr_ratio=cfg.optim.min_lr_ratio)

    if is_dist_avail_and_initialized():
        # Single-objective aux runs leave one aux sub-head unused per step
        # (contrastive-only -> itm_head unused; itm-only -> img/txt_proj+log_temp
        # unused), which trips DDP's default reducer. Enable find_unused_parameters
        # only when aux heads are active; the caption-only recipe (the paper's)
        # uses every trainable param, so it keeps the default (False).
        training_module = DistributedDataParallel(
            training_module,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=weights.uses_aux_heads,
        )

    init_wandb(
        project=cfg.wandb.project if cfg.wandb.enabled else None,
        run_name=cfg.wandb.run_name,
        config=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.wandb.mode,
        metric_axes={
            "step": None,
            "epoch": None,
            "train/*": "step",
            "epoch/*": "step",
            "val/*": "epoch",
            "gate/*": "step",
        },
    )

    trainer = FlamingoTrainer(
        training_module=training_module,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        scheduler=scheduler,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
