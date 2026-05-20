# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper around the DINOv2 ``SSLMetaArch`` for SSL co-training.

Encapsulates the DINOv2 SSL setup: model + optimizer + cosine schedulers +
masking + augmentation + collate. The trainer calls :meth:`prepare_batch` to
build a DINOv2 batch from raw image tensors, :meth:`step` to do one
forward/backward pass + teacher EMA update, and
:meth:`teacher_backbone_state_dict` to grab the synchronized backbone
weights for cross-tower copy into the CLIP vision encoder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from apps.baselines.clip.data.transforms import denormalize_imagenet

# DINOv2 stores its FSDP mixed-precision dtype as a short string in the
# ``compute_precision`` config; the data collate must produce tensors in the
# same dtype the model expects, so we read this off the config rather than
# letting the user double-specify it on the trainer side.
_DTYPE_BY_NAME = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _resolve_collate_dtype(cfg: DictConfig) -> torch.dtype:
    """Return the dtype for the DINOv2 collate, derived from compute_precision.

    The dtype must agree with the FSDP mixed-precision setting on the student
    backbone (the module that consumes the input batch). If the config is
    missing this field or names a dtype we don't recognize, raise — silently
    falling back to fp16 hides bugs that surface as opaque conv errors later.
    """
    name = str(cfg.compute_precision.student.backbone.mixed_precision.param_dtype)
    if name not in _DTYPE_BY_NAME:
        raise ValueError(
            f"Unsupported compute_precision.student.backbone.mixed_precision.param_dtype={name!r}; "
            f"expected one of {sorted(_DTYPE_BY_NAME)}",
        )
    return _DTYPE_BY_NAME[name]


def load_dinov2_config(config_path: str | Path) -> DictConfig:
    """Load a DINOv2 SSL training config and merge it with the SSL defaults.

    The DINOv2 package ships ``ssl_default_config.yaml`` as the base; per-run
    YAMLs (e.g. ``vitb14_coco.yaml``) override a subset. This helper composes
    them in the standard DINOv2 order and validates the file exists.

    Args:
        config_path: Absolute or relative path to a DINOv2 SSL training YAML.
    """
    from dinov2.configs import dinov2_default_config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 config not found: {path}")
    return cast("DictConfig", OmegaConf.merge(dinov2_default_config, OmegaConf.load(path)))


class DINOv2SSL:
    """DINOv2 self-supervised co-training wrapper.

    Owns the model, optimizer, schedulers, augmentation, and masking. One
    instance per training run.

    Args:
        config_path: Path to a DINOv2 SSL training YAML (composed with
            ``ssl_default_config.yaml`` via :func:`load_dinov2_config`).
        device: CUDA device for the SSL model.
        overrides: Optional nested dict applied on top of the loaded config.
            Use this to bump ``train.OFFICIAL_EPOCH_LENGTH`` for short smoke
            runs without editing the on-disk YAMLs.
        pretrained_dir: Optional directory containing a DINOv2 SSL FSDP
            checkpoint (``last_checkpoint.rank_0`` + matching
            ``model_*.rank_0.pth``) to resume the SSL model state from.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        device: torch.device | str = "cuda",
        overrides: dict[str, Any] | DictConfig | None = None,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        from dinov2 import distributed as dinov2_distributed
        from dinov2.data import (
            DataAugmentationDINO,
            MaskingGenerator,
            collate_data_and_cast,
        )
        from dinov2.train.ssl_meta_arch import SSLMetaArch
        from dinov2.utils.utils import CosineScheduler

        self._collate_data_and_cast = collate_data_and_cast
        self._CosineScheduler = CosineScheduler

        cfg = load_dinov2_config(config_path)
        if overrides:
            cfg = cast("DictConfig", OmegaConf.merge(cfg, OmegaConf.create(dict(overrides))))
        self.config_path = Path(config_path)
        self.cfg = cfg
        self.device = torch.device(device)
        self.arch = f"{cfg.student.arch}{cfg.student.patch_size}"
        self.image_size = int(cfg.crops.global_crops_size)

        # SSLMetaArch.prepare_for_distributed_training wraps modules in FSDP, which
        # requires both an initialized torch process group AND dinov2's own
        # _LOCAL_RANK / _LOCAL_WORLD_SIZE module globals. core.utils.setup_distributed
        # skips init at world_size=1 (no DDP needed for that case), so we bootstrap
        # both here. We set the dinov2 module globals directly because
        # ``dinov2_distributed.enable()`` insists on detecting the launcher
        # (slurm/torchrun/local) from environment variables, which is brittle
        # when running inside an srun --overlap step that inherits SLURM_JOB_ID
        # but not SLURM_JOB_NUM_NODES.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=1,
                rank=0,
                init_method="env://",
            )
        dinov2_distributed._LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))  # type: ignore[attr-defined]
        dinov2_distributed._LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))  # type: ignore[attr-defined]

        self.model = SSLMetaArch(cfg).to(self.device)
        self.model.prepare_for_distributed_training()

        if pretrained_dir is not None:
            from dinov2.fsdp import FSDPCheckpointer

            pretrained_dir = Path(pretrained_dir)
            if not pretrained_dir.is_dir():
                raise FileNotFoundError(f"DINOv2 pretrained_dir not found: {pretrained_dir}")
            checkpointer = FSDPCheckpointer(self.model, str(pretrained_dir))
            ckpt_info = checkpointer.resume_or_load("", resume=True)
            self._pretrained_iteration = int(ckpt_info.get("iteration", -1)) + 1
        else:
            self._pretrained_iteration = 0

        self.optimizer = torch.optim.AdamW(
            self.model.get_params_groups(),
            betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
        )
        self.schedulers = self._build_schedulers()

        patch_size = cfg.student.patch_size
        n_tokens_per_axis = self.image_size // patch_size
        self._n_tokens = n_tokens_per_axis**2
        self._collate_dtype = _resolve_collate_dtype(cfg)
        self.mask_generator = MaskingGenerator(
            input_size=(n_tokens_per_axis, n_tokens_per_axis),
            max_num_patches=int(0.5 * n_tokens_per_axis**2),
        )
        self.augmentation = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    def _build_schedulers(self) -> dict[str, Any]:
        cfg = self.cfg
        epoch_len = cfg.train.OFFICIAL_EPOCH_LENGTH
        total = cfg.optim.epochs * epoch_len
        warmup = cfg.optim.warmup_epochs * epoch_len
        cosine = self._CosineScheduler

        schedulers = {
            "lr": cosine(
                base_value=cfg.optim.lr,
                final_value=cfg.optim.min_lr,
                total_iters=total,
                warmup_iters=warmup,
                start_warmup_value=0,
            ),
            "wd": cosine(
                base_value=cfg.optim.weight_decay,
                final_value=cfg.optim.weight_decay_end,
                total_iters=total,
            ),
            "momentum": cosine(
                base_value=cfg.teacher.momentum_teacher,
                final_value=cfg.teacher.final_momentum_teacher,
                total_iters=total,
            ),
            "teacher_temp": cosine(
                base_value=cfg.teacher.teacher_temp,
                final_value=cfg.teacher.teacher_temp,
                total_iters=cfg.teacher.warmup_teacher_temp_epochs * epoch_len,
                warmup_iters=cfg.teacher.warmup_teacher_temp_epochs * epoch_len,
                start_warmup_value=cfg.teacher.warmup_teacher_temp,
            ),
            "last_layer_lr": cosine(
                base_value=cfg.optim.lr,
                final_value=cfg.optim.min_lr,
                total_iters=total,
                warmup_iters=warmup,
                start_warmup_value=0,
            ),
        }
        # Freeze the head's last-layer LR for the configured warmup window.
        freeze_iters = cfg.optim.freeze_last_layer_epochs * epoch_len
        schedulers["last_layer_lr"].schedule[:freeze_iters] = 0
        return schedulers

    def _apply_schedulers(self, iteration: int) -> dict[str, float]:
        lr = self.schedulers["lr"][iteration]
        wd = self.schedulers["wd"][iteration]
        last_layer_lr = self.schedulers["last_layer_lr"][iteration]
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd * group["wd_multiplier"]
            group["lr"] = (last_layer_lr if group["is_last_layer"] else lr) * group["lr_multiplier"]
        return {
            "lr": lr,
            "wd": wd,
            "momentum": self.schedulers["momentum"][iteration],
            "teacher_temp": self.schedulers["teacher_temp"][iteration],
            "last_layer_lr": last_layer_lr,
        }

    def prepare_batch(self, images: torch.Tensor) -> dict[str, Any]:
        """Build a DINOv2 batch from a tensor of CLIP-normalized images.

        Denormalizes (using the shared :func:`denormalize_imagenet` helper),
        converts to PIL, applies DINOv2's augmentation pipeline (global+local
        crops + masking), and collates into the dict shape
        :meth:`SSLMetaArch.forward_backward` expects.
        """
        augmented = []
        for img in images:
            pil_img = transforms.ToPILImage()(denormalize_imagenet(img.cpu()))
            augmented.append((self.augmentation(pil_img), None))
        return self._collate_data_and_cast(
            augmented,
            mask_ratio_tuple=self.cfg.ibot.mask_ratio_min_max,
            mask_probability=self.cfg.ibot.mask_sample_probability,
            n_tokens=self._n_tokens,
            mask_generator=self.mask_generator,
            dtype=self._collate_dtype,
        )

    def step(self, batch: dict[str, Any], iteration: int) -> dict[str, float]:
        """One SSL forward/backward + teacher EMA update."""
        sched = self._apply_schedulers(iteration)
        self.optimizer.zero_grad()
        loss_dict = self.model.forward_backward(batch, teacher_temp=sched["teacher_temp"])

        clip_grad = self.cfg.optim.clip_grad
        if self.model.fp16_scaler is not None:
            if clip_grad:
                self.model.fp16_scaler.unscale_(self.optimizer)
                for v in self.model.student.values():
                    v.clip_grad_norm_(clip_grad)
            self.model.fp16_scaler.step(self.optimizer)
            self.model.fp16_scaler.update()
        else:
            if clip_grad:
                for v in self.model.student.values():
                    v.clip_grad_norm_(clip_grad)
            self.optimizer.step()

        self.model.update_teacher(sched["momentum"])

        results = {"total_loss": float(sum(v.item() for v in loss_dict.values()))}
        results.update({k: v.item() for k, v in loss_dict.items()})
        results.update(sched)
        return results

    def teacher_backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict of the teacher's backbone, for cross-tower copy."""
        return self.model.teacher.backbone.state_dict()

    def state_dict(self) -> dict[str, Any]:
        """Serialize the whole SSL state (student + teacher + optimizer + scaler)."""
        return {
            "student": self.model.student.state_dict(),
            "teacher": self.model.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.model.fp16_scaler.state_dict() if self.model.fp16_scaler is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.student.load_state_dict(state["student"])
        self.model.teacher.load_state_dict(state["teacher"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None and self.model.fp16_scaler is not None:
            self.model.fp16_scaler.load_state_dict(state["scaler"])
