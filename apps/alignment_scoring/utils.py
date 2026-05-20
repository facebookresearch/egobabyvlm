# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the alignment-scoring pipelines."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import open_clip
import peft
import torch
import torch.nn.functional as F
from safetensors.torch import save as save_safetensors
from scipy.spatial.distance import jensenshannon
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, Subset

from core.utils import get_rank, get_world_size, is_main_process, to_yaml

if TYPE_CHECKING:
    from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def flattened(x: Iterable[Iterable[Any]]) -> chain:
    """Flatten an iterable of iterables — :func:`itertools.chain.from_iterable`."""
    return chain.from_iterable(x)


def create_model(
    config: Any,
    device: torch.device | str = "cuda",
) -> tuple[
    open_clip.CLIP | open_clip.CustomTextCLIP | peft.PeftModel,
    Compose,
    open_clip.SimpleTokenizer,
    dict[str, Any],
]:
    """Build an open_clip model from a Hydra ``ModelConfig`` and optionally wrap it in LoRA."""
    model_name = config.model_name
    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=config.pretrained,
    )
    open_clip_config = {
        "model_cfg": open_clip.get_model_config(model_name) or {},
        "preprocess_cfg": open_clip.get_model_preprocess_cfg(model),
    }
    logger.info("Created model %s with config: %s", model_name, to_yaml(open_clip_config))

    if config.gradient_checkpointing:
        model.set_grad_checkpointing(enable=True)
        logger.info("Enabled gradient checkpointing")

    tokenizer = open_clip.get_tokenizer(model_name)

    if config.lora is not None:
        lora_config = hydra.utils.instantiate(config.lora)
        model = peft.get_peft_model(model, lora_config)
        model.logit_scale.requires_grad_(config.mode == "train")
        model.print_trainable_parameters()

    model = model.to(device)
    if config.mode == "eval":
        model.eval()
        preprocess = val_preprocess
    else:
        model.train()
        preprocess = train_preprocess

    return model, preprocess, tokenizer, open_clip_config


def create_alignment_dataloader(
    config: Any,
    indices: tuple[int, int] | None,
    preprocessor: Compose | None,
    mode: str = "eval",
    *,
    distributed: bool = False,
) -> tuple[DataLoader, bool]:
    """Instantiate a caption dataset and wrap it in a DataLoader.

    Args:
        config: A Hydra-style ``DataConfig`` with ``dataset``, ``batch_size``,
            ``num_workers``, ``pin_memory`` fields.
        indices: Optional ``(start, end)`` range — used by Stopes job arrays
            so each task owns a slice of the dataset.
        preprocessor: Image preprocessor passed to the dataset constructor.
        mode: ``"train"`` shuffles + drops the last batch; ``"eval"`` does neither.
        distributed: Use a DistributedSampler instead of Random/Sequential.
    """
    from .data.collate import image_captions_collate_fn

    dataset: torch.utils.data.Dataset = hydra.utils.instantiate(config.dataset, preprocessor=preprocessor)
    is_video_dataset = getattr(dataset, "is_video_dataset", False)

    if indices is not None:
        start_idx, end_idx = indices
        end_idx = min(end_idx, len(dataset))  # type: ignore[arg-type]
        logger.info("Dataset size: %d, using indices %d-%d", len(dataset), start_idx, end_idx)  # type: ignore[arg-type]
        dataset = Subset(dataset, range(start_idx, end_idx))

    sampler: torch.utils.data.Sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
    else:
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)  # type: ignore[arg-type]

    return (
        DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=image_captions_collate_fn,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=mode == "train",
            persistent_workers=False,
            multiprocessing_context="spawn" if config.num_workers > 0 else None,
        ),
        is_video_dataset,
    )


def post_collate_fn(
    batch: tuple[Any, Any],
    tokenizer: open_clip.SimpleTokenizer,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move an image-caption batch to ``device`` and tokenize the captions.

    Image batches arrive as either tensors (image datasets) or list-of-tensors
    (video datasets); the latter are stacked + permuted to ``(B, N_frames, C, H, W)``.
    """
    batch_images, batch_texts = batch
    if isinstance(batch_images, torch.Tensor):
        batch_images = batch_images.to(device, dtype)
    elif isinstance(batch_images, (list, tuple)):
        batch_images = [x.to(device, dtype) for x in batch_images]
        batch_images = torch.stack(batch_images, dim=0).permute(1, 0, 2, 3, 4).contiguous()
    else:
        raise NotImplementedError(type(batch_images))

    if isinstance(batch_texts[0], (list, tuple)):
        batch_texts_tok = tokenizer(list(flattened(batch_texts))).to(device)
    else:
        batch_texts_tok = tokenizer(batch_texts).to(device)
    return batch_images, batch_texts_tok


def clip_forward(
    model: open_clip.CustomTextCLIP | open_clip.CLIP | peft.PeftModel,
    batch: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
    *,
    is_video_dataset: bool = False,
    is_video_model: bool = True,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode images and texts with autocast; mean-pool video frames if needed."""
    batch_images, batch_texts_tok = batch
    n: int | None = None
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        if is_video_dataset:
            b, n, c, h, w = batch_images.shape
            batch_images_flat = batch_images.reshape(b * n, c, h, w)
            image_features = model.encode_image(batch_images_flat)
            if is_video_model:
                image_features = image_features.reshape(b, n, -1).mean(dim=1)
        else:
            image_features = model.encode_image(batch_images)

        text_features = model.encode_text(batch_texts_tok)

        if is_video_dataset and not is_video_model and n is not None:
            text_features = text_features.repeat_interleave(n, dim=0)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

    return image_features, text_features


def add_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    skip_list: tuple[str, ...] = (),
    *,
    bias_wd: bool = False,
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups (biases + 1-D shapes opt out)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name in skip_list:
            continue
        if ((not bias_wd) and len(param.shape) == 1) or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class NativeScalerWithGradNormCount:
    """GradScaler that performs gradient scaling (fp16) and clipping."""

    state_dict_key = "amp_scaler"

    def __init__(self, *, fp16: bool = True) -> None:
        self._scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters: Iterable | None = None,
        *,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> torch.Tensor | None:
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not update_grad:
            return None
        assert parameters is not None
        if clip_grad is not None:
            self._scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            self._scaler.unscale_(optimizer)
            norm = _get_grad_norm(parameters)
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self) -> dict[str, Any]:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scaler.load_state_dict(state_dict)


def _get_grad_norm(parameters: Iterable, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )


def build_optimizer(
    config: Any,
    model: open_clip.CLIP | open_clip.CustomTextCLIP | peft.PeftModel,
) -> tuple[AdamW, NativeScalerWithGradNormCount]:
    """Build an AdamW optimizer + grad scaler, with logit_scale on its own no-decay group."""
    if config.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(config.dtype)
    if config.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        raise ValueError("Cannot use bfloat16 on this GPU (V100?), try again with float16")

    if hasattr(model, "logit_scale"):
        param_groups: list[dict[str, Any]] = [
            {
                "params": [model.logit_scale],
                "weight_decay": 0.0,
                "lr_scale": config.logit_scale_lr_scale,
            },
        ]
        skip_list = tuple(name for name, _ in model.named_parameters() if name.endswith("logit_scale"))
    else:
        param_groups = []
        skip_list = ()

    param_groups.extend(
        add_weight_decay(model, config.weight_decay, bias_wd=config.bias_wd, skip_list=skip_list),
    )
    logger.info("Created %d parameter groups for the optimizer", len(param_groups))

    optimizer = AdamW(
        param_groups,
        lr=config.lr,
        weight_decay=0.0,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.eps,
        fused=True,
    )
    scaler = NativeScalerWithGradNormCount(fp16=config.dtype == "float16")
    return optimizer, scaler


def save_openclip_checkpoint(
    model: peft.PeftModel | open_clip.CLIP | open_clip.CustomTextCLIP,
    open_clip_config: dict,
    output_dir: Path | str,
    tag: str = "latest",
) -> None:
    """Write an open_clip-format checkpoint (safetensors + config json) under ``output_dir``."""
    if not is_main_process():
        return

    openclip_dir = Path(output_dir) / f"openclip_checkpoint_{tag}"
    openclip_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving OpenCLIP-compatible checkpoint to %s", openclip_dir)

    if isinstance(model, peft.PeftModel):
        logger.info("Detected PeftModel, merging and unloading LoRA weights")
        model = model.merge_and_unload()

    state_dict = model.state_dict()
    safetensors_path = openclip_dir / "open_clip_model.safetensors"
    safetensors_path.write_bytes(save_safetensors(state_dict))
    logger.info("Saved model weights to %s", safetensors_path)

    config_path = openclip_dir / "open_clip_config.json"
    config_path.write_text(json.dumps(open_clip_config, indent=2))
    logger.info("Saved OpenCLIP config to %s", config_path)


def calculate_kl_divergence(
    cos_sims: list[float] | np.ndarray,
    cos_sims_shuffled: list[float] | np.ndarray,
    bins: int = 50,
) -> dict[str, float]:
    """Compute KL(matched || shuffled) and KL(shuffled || matched) on histogram-binned data."""
    cos_sims = np.asarray(cos_sims)
    cos_sims_shuffled = np.asarray(cos_sims_shuffled)
    edges = np.linspace(
        min(cos_sims.min(), cos_sims_shuffled.min()),
        max(cos_sims.max(), cos_sims_shuffled.max()),
        bins + 1,
    )
    p, _ = np.histogram(cos_sims, bins=edges, density=True)
    q, _ = np.histogram(cos_sims_shuffled, bins=edges, density=True)
    p, q = p + 1e-12, q + 1e-12
    p, q = p / p.sum(), q / q.sum()
    kl_pq = float(np.sum(p * np.log2(p / q)))
    kl_qp = float(np.sum(q * np.log2(q / p)))
    return {
        "kl_matched_shuffled": kl_pq,
        "kl_shuffled_matched": kl_qp,
        "kl_symmetric": (kl_pq + kl_qp) / 2.0,
    }


def bootstrap_js(
    cos_sims: list[float] | np.ndarray,
    cos_sims_shuffled: list[float] | np.ndarray,
    n_bootstrap: int = 1000,
    max_samples: int = 1_000_000,
    bins: int = 50,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap-estimate JS divergence between matched and shuffled cosine-sim distributions."""
    rng = np.random.default_rng(seed)
    cos_sims = np.asarray(cos_sims)
    cos_sims_shuffled = np.asarray(cos_sims_shuffled)
    n = min(len(cos_sims), len(cos_sims_shuffled), max_samples)

    edges = np.linspace(
        min(cos_sims.min(), cos_sims_shuffled.min()),
        max(cos_sims.max(), cos_sims_shuffled.max()),
        bins + 1,
    )

    distribution = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        a = cos_sims[rng.integers(0, len(cos_sims), size=n)]
        b = cos_sims_shuffled[rng.integers(0, len(cos_sims_shuffled), size=n)]
        p, _ = np.histogram(a, bins=edges, density=True)
        q, _ = np.histogram(b, bins=edges, density=True)
        p, q = p + 1e-12, q + 1e-12
        p, q = p / p.sum(), q / q.sum()
        distribution[i] = jensenshannon(p, q, base=2) ** 2  # JS divergence in bits, bounded [0, 1]

    alpha = (1 - confidence) / 2
    return {
        "bootstrap_distribution": distribution,
        "standard_error": float(distribution.std()),
        "ci": (float(np.quantile(distribution, alpha)), float(np.quantile(distribution, 1 - alpha))),
    }
