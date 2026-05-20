# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""@hydra.main entrypoint for the contrastive trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import apps.baselines.clip  # registers Hydra ConfigStore nodes  # noqa: F401
from apps.baselines.clip.data import (
    MLMCollator,
    TextOnlyDataset,
    build_eval_transform,
    build_train_transform,
    contrastive_collate,
)
from apps.baselines.clip.modeling import DINOv2SSL, MLMHead, MultiModalModel
from apps.baselines.clip.training import (
    ContrastiveTrainer,
    InterleaveScheduler,
    build_adamw,
)
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

    from apps.baselines.clip.modeling import TextEncoder

logger = logging.getLogger(__name__)


def _build_loaders(
    cfg: DictConfig,
    *,
    augment: bool,
) -> tuple[DataLoader, DataLoader | None]:
    train_ds = instantiate(cfg.data.train_dataset, transform=build_train_transform(augment=augment))
    val_ds = (
        instantiate(cfg.data.val_dataset, transform=build_eval_transform())
        if cfg.data.val_dataset is not None
        else None
    )

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
        collate_fn=contrastive_collate,
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
            collate_fn=contrastive_collate,
        )
        if val_ds is not None
        else None
    )
    return train_loader, val_loader


def _build_mlm(
    cfg: DictConfig,
    text_encoder: TextEncoder,
    device: torch.device,
) -> tuple[MLMHead | DistributedDataParallel, torch.optim.Optimizer, DataLoader]:
    head: MLMHead | DistributedDataParallel = MLMHead(text_encoder).to(device)

    text_only = cfg.text_only_data
    dataset = TextOnlyDataset(text_only.train_file, text_encoder.tokenizer, max_seq_len=text_only.max_seq_len)
    collator = MLMCollator(text_encoder.tokenizer, mlm_probability=text_only.mlm_probability)
    sampler: DistributedSampler | None = (
        DistributedSampler(dataset, shuffle=True) if is_dist_avail_and_initialized() else None
    )
    loader = DataLoader(
        dataset,
        batch_size=text_only.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=text_only.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    # Dedupe by id: ``MLMHead`` ties decoder weight to the encoder word
    # embedding, so the same tensor appears in both groups.
    seen: set[int] = set()
    mlm_params: list[torch.nn.Parameter] = []
    for group in (
        text_encoder.model.embeddings.parameters(),
        text_encoder.model.encoder.parameters(),
        head.parameters(),
    ):
        for p in group:
            if id(p) in seen:
                continue
            seen.add(id(p))
            mlm_params.append(p)
    optimizer = build_adamw(
        mlm_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
    )

    if is_dist_avail_and_initialized():
        head = DistributedDataParallel(head, device_ids=[device.index] if device.type == "cuda" else None)
    return head, optimizer, loader


def _build_ssl(cfg: DictConfig, device: torch.device) -> DINOv2SSL:
    overrides = dict(cfg.dinov2.overrides) if cfg.dinov2.overrides else None
    return DINOv2SSL(
        cfg.dinov2.config_path,
        device=device,
        overrides=overrides,
        pretrained_dir=cfg.dinov2.pretrained_dir,
    )


def _maybe_ddp(module: torch.nn.Module, device: torch.device) -> torch.nn.Module | DistributedDataParallel:
    if not is_dist_avail_and_initialized():
        return module
    if device.type == "cuda":
        return DistributedDataParallel(module, device_ids=[device.index])
    return DistributedDataParallel(module)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    setup_distributed()
    env = distributed_environment()
    set_seed(cfg.seed + env.global_rank)

    device = torch.device(f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu")

    resolve_and_print_config(cfg)

    Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)

    text_encoder = instantiate(cfg.model.text_encoder).to(device)
    vision_encoder = instantiate(cfg.model.vision_encoder).to(device)
    model = MultiModalModel(
        vision_encoder,
        text_encoder,
        normalize_features=cfg.model.normalize_features,
        temperature=cfg.model.temperature,
        fix_temperature=cfg.model.fix_temperature,
    ).to(device)
    model_for_trainer = _maybe_ddp(model, device)

    contrastive_optimizer = build_adamw(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
    )

    train_loader, val_loader = _build_loaders(cfg, augment=cfg.data.augment)
    scheduler = InterleaveScheduler(dict(cfg.mode.interleave))

    mlm_head: Any = None
    mlm_optimizer = None
    mlm_loader = None
    if "mlm" in scheduler.modes:
        if cfg.text_only_data is None:
            raise ValueError("mode includes 'mlm' but cfg.text_only_data is not set")
        mlm_head, mlm_optimizer, mlm_loader = _build_mlm(cfg, text_encoder, device)

    ssl: DINOv2SSL | None = None
    if "dinov2" in scheduler.modes:
        if cfg.dinov2 is None:
            raise ValueError("mode includes 'dinov2' but cfg.dinov2 is not set")
        ssl = _build_ssl(cfg, device)

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
        },
    )

    trainer = ContrastiveTrainer(
        model=cast("MultiModalModel | DistributedDataParallel", model_for_trainer),
        contrastive_optimizer=contrastive_optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        mlm_head=mlm_head,
        mlm_optimizer=mlm_optimizer,
        mlm_loader=mlm_loader,
        ssl=ssl,
        sync_vision_from_ssl=cfg.mode.sync_vision_from_dinov2,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
