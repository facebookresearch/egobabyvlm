# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoVLP model wrapper, checkpoint save and load.

The shared source of truth for the EgoVLP dual encoder used by both the
fine-tuning entrypoint (``training/train.py``) and the DevBench feature
extractor (``trained_extractor.py``). Training and eval build the model through
the same :func:`build_egovlp_model` / :func:`load_egovlp_model`, so the embedding
space (projection dim, pooling, normalization) matches.

:class:`EgoVLPModel` wraps the vendored EgoVLP ``FrozenInTime``
(``model/video_transformer.py``), whose parameter names match the released
EgoClip ``egovlp.pth`` exactly. It exposes ``encode_image(clip) -> (B, embed_dim)``
and ``encode_text(list[str]) -> (B, embed_dim)``, both returning **un-normalized**
projected features (callers L2-normalize).

Checkpoint format (EgoVLP-native ``.pth``)
------------------------------------------
Fine-tuning saves an EgoVLP-native ``.pth``, not the CLIP+ ``.pt`` schema::

    {
      "state_dict":   model.state_dict(),   # EgoVLPModel weights
      "egovlp_config": {...},               # kwargs to rebuild the model
      "epoch": int, "step": int,
      "args": {...},                        # training CLI args (provenance)
    }

:func:`load_egovlp_model` rebuilds the model from ``egovlp_config`` and loads
``state_dict`` (``strict=False``, logging missing/unexpected keys).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: HF id (or local dir) of the DistilBERT text encoder. Point this at a local
#: copy if the machine has no internet access.
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"

#: EgoVLP's shared projection dim.
DEFAULT_EMBED_DIM = 256


@dataclass
class EgoVLPConfig:
    """Rebuildable description of an :class:`EgoVLPModel`.

    Serialized into the checkpoint under ``egovlp_config`` so eval can rebuild
    the exact architecture the fine-tune produced.
    """

    embed_dim: int = DEFAULT_EMBED_DIM
    num_frames: int = 1
    image_size: int = 224
    text_model: str = DEFAULT_TEXT_MODEL
    max_text_len: int = 64
    #: Released EgoClip checkpoint (``egovlp.pth``) loaded at construction time.
    #: ``None`` => random video encoder + HF DistilBERT (tests only).
    pretrained_checkpoint: str | None = None
    #: Load HF pretrained DistilBERT weights before ``pretrained_checkpoint``
    #: overwrites them.
    pretrained_text: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (stored in checkpoints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EgoVLPConfig:
        """Rebuild from a checkpoint dict, ignoring unknown keys.

        Checkpoints from the paper runs also carry fields of an earlier
        development backbone (``use_official``, ``egovlp_src``, ...), which
        are dropped here.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


_IMAGE_NDIM = 4  # (B, C, H, W)
_CLIP_NDIM = 5  # (B, T, C, H, W)


def _as_clip(x: torch.Tensor) -> torch.Tensor:
    """Coerce an image/clip tensor to ``(B, T, C, H, W)``.

    Accepts ``(B, C, H, W)`` (single frame -> T=1) or an already-5D clip.
    """
    if x.dim() == _IMAGE_NDIM:
        return x.unsqueeze(1)
    if x.dim() == _CLIP_NDIM:
        return x
    msg = f"Expected image tensor of rank 4 (B,C,H,W) or 5 (B,T,C,H,W); got shape {tuple(x.shape)}"
    raise ValueError(msg)


class EgoVLPModel(nn.Module):
    """EgoVLP ``FrozenInTime`` dual encoder with a built-in tokenizer.

    Args:
        config: Architecture + init description.
    """

    def __init__(self, config: EgoVLPConfig) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        from apps.baselines.egovlp.model.video_transformer import FrozenInTime

        self.config = config
        self.embed_dim = config.embed_dim
        self.num_frames = config.num_frames
        self.max_text_len = config.max_text_len
        self.net = FrozenInTime(
            projection_dim=config.embed_dim,
            text_model_name=config.text_model,
            load_checkpoint=config.pretrained_checkpoint,
            pretrained_text=config.pretrained_text,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model)

    def encode_image(self, clip: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, H, W)`` images or ``(B, T, C, H, W)`` clips."""
        return self.net.compute_video(_as_clip(clip))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Tokenize (DistilBERT, max ``max_text_len`` tokens) and encode."""
        device = next(self.net.parameters()).device
        tok = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(device)
        return self.net.compute_text({"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]})


def build_egovlp_model(config: EgoVLPConfig) -> EgoVLPModel:
    """Build the EgoVLP dual encoder described by ``config``."""
    if config.pretrained_checkpoint is None:
        logger.warning("Building EgoVLP without the EgoClip checkpoint (random video encoder).")
    return EgoVLPModel(config)


def save_egovlp_checkpoint(
    model: nn.Module,
    config: EgoVLPConfig,
    path: str | Path,
    *,
    epoch: int = 0,
    step: int = 0,
    args: dict[str, Any] | None = None,
) -> None:
    """Save an EgoVLP-native ``.pth`` (see module docstring for the schema)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "egovlp_config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "args": args or {},
    }
    torch.save(payload, path)
    logger.info("Saved EgoVLP checkpoint to %s (epoch=%d, step=%d)", path, epoch, step)


def load_egovlp_model(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[EgoVLPModel, EgoVLPConfig]:
    """Rebuild + load an EgoVLP model from an EgoVLP-native ``.pth``.

    Args:
        path: Checkpoint saved by :func:`save_egovlp_checkpoint`.
        device: Target device (defaults to cuda if available).

    Returns:
        The model in eval mode on ``device`` and its config.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = EgoVLPConfig.from_dict(payload["egovlp_config"])
    # The fine-tuned state_dict is authoritative; don't also load the EgoClip
    # or HF pretrained weights underneath it.
    config.pretrained_checkpoint = None
    config.pretrained_text = False
    model = EgoVLPModel(config)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        logger.warning(
            "Loaded EgoVLP checkpoint with %d missing / %d unexpected keys (first missing=%s, first unexpected=%s)",
            len(missing),
            len(unexpected),
            missing[:3],
            unexpected[:3],
        )
    model.eval().to(device)
    return model, config
