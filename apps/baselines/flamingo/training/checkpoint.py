# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained checkpoint save / load for the Flamingo trainer.

Embeds ``flamingo_config`` (the full :class:`FlamingoConfig`, including
``media_token_id`` + ``vocab_size`` + all arch knobs) so the checkpoint reloads
standalone via ``apps.baselines.flamingo.model.builder.load_pretrained_model``.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime by callers' type hints
from typing import TYPE_CHECKING, Any

import torch

from apps.baselines.flamingo.modeling.flamingo_model import config_to_dict
from core.utils.checkpoints import atomic_torch_save
from core.utils.distributed import is_main_process, unwrap_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from apps.baselines.flamingo.modeling.flamingo_model import FlamingoVLM


def save_checkpoint(
    path: str | Path,
    *,
    model: FlamingoVLM,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    config: DictConfig | dict[str, Any],
    epoch: int,
    step: int,
    best_val_loss: float = float("inf"),
    epoch_complete: bool = True,
) -> None:
    """Save a self-contained Flamingo checkpoint (rank 0 only).

    ``epoch_complete=False`` marks a mid-epoch save (``interrupted.pt``).
    """
    if not is_main_process():
        return

    from omegaconf import DictConfig, OmegaConf

    cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    vlm = unwrap_model(model)
    payload: dict[str, Any] = {
        "model_state_dict": vlm.state_dict(),
        "flamingo_config": config_to_dict(vlm.config),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "epoch_complete": epoch_complete,
        "config": cfg_dict,
    }
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: FlamingoVLM,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into the provided modules in place; return the payload."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
