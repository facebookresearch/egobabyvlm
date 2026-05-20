# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained checkpoint save / load for the contrastive trainer.

Format::

    {
        "model_state_dict": ...,                # MultiModalModel state
        "mlm_head_state_dict": ... or None,     # if interleaved_lm/triple
        "ssl_state_dict": ... or None,          # DINOv2SSL.state_dict()
        "optimizer_state_dicts": {              # one per active loss head
            "contrastive": ...,
            "mlm": ... or None,
        },
        "scheduler_state": ...,                 # InterleaveScheduler.state_dict()
        "epoch": int, "step": int,
        "best_val_loss": float,
        "config": dict,                         # full OmegaConf, including HF model names
    }

Loading does NOT need any additional config files: ``config`` is embedded
as-is, so a feature extractor reads ``torch.load(...)["config"]`` and
re-instantiates encoders/MLM/SSL accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from core.utils.checkpoints import atomic_torch_save
from core.utils.distributed import is_main_process, unwrap_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.optim import Optimizer

    from apps.baselines.clip.modeling import DINOv2SSL, MLMHead, MultiModalModel
    from apps.baselines.clip.training.interleave import InterleaveScheduler


def _embed_vision_encoder_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    """Inline external file references in the vision_encoder config.

    Replaces ``config_path: <path>`` with ``dinov2_config: <full yaml dict>``
    and drops ``checkpoint_path`` (the SSL teacher seed weights are dead
    after contrastive training overwrites them anyway). Result: the saved
    ``.pt`` is fully self-contained — no SSL workspace needs to be
    reachable to load the checkpoint later.

    No-op if the encoder is anything other than ``CustomDINOv2VisionEncoder``
    (e.g. ``HubDINOv2VisionEncoder`` already has no on-disk dependencies).
    """
    from omegaconf import OmegaConf

    ve = config.get("model", {}).get("vision_encoder", {})
    if "config_path" not in ve:
        return config
    cfg_path = Path(ve["config_path"])
    if not cfg_path.is_file():
        return config  # leave path-based reference; loader will error clearly later

    embedded = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    new_ve = {k: v for k, v in ve.items() if k not in ("config_path", "checkpoint_path")}
    new_ve["dinov2_config"] = embedded
    out = {**config, "model": {**config["model"], "vision_encoder": new_ve}}
    return out


def save_checkpoint(
    path: str | Path,
    *,
    model: MultiModalModel,
    optimizers: dict[str, Optimizer],
    scheduler: InterleaveScheduler,
    config: DictConfig | dict[str, Any],
    epoch: int,
    step: int,
    best_val_loss: float = float("inf"),
    mlm_head: MLMHead | None = None,
    ssl: DINOv2SSL | None = None,
) -> None:
    """Save a self-contained checkpoint. Only rank 0 writes."""
    if not is_main_process():
        return

    from omegaconf import DictConfig, OmegaConf

    cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    cfg_dict = _embed_vision_encoder_dependencies(cast("dict[str, Any]", cfg_dict))

    payload: dict[str, Any] = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "mlm_head_state_dict": unwrap_model(mlm_head).state_dict() if mlm_head is not None else None,
        "ssl_state_dict": ssl.state_dict() if ssl is not None else None,
        "optimizer_state_dicts": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": cfg_dict,
    }

    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: MultiModalModel,
    optimizers: dict[str, Optimizer] | None = None,
    scheduler: InterleaveScheduler | None = None,
    mlm_head: MLMHead | None = None,
    ssl: DINOv2SSL | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into the provided modules in place.

    Returns the raw payload so callers can inspect ``epoch``, ``step``,
    ``best_val_loss``, ``config``.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(payload["model_state_dict"])

    if mlm_head is not None and payload.get("mlm_head_state_dict") is not None:
        unwrap_model(mlm_head).load_state_dict(payload["mlm_head_state_dict"])

    if ssl is not None and payload.get("ssl_state_dict") is not None:
        ssl.load_state_dict(payload["ssl_state_dict"])

    if optimizers is not None:
        for name, opt in optimizers.items():
            saved = payload.get("optimizer_state_dicts", {}).get(name)
            if saved is not None:
                opt.load_state_dict(saved)

    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    return payload
