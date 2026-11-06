# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Save / load for Flamingo checkpoints.

``load_pretrained_model`` mirrors the LLaVA baseline's signature/return
``(tokenizer, model, image_processor, context_len)`` so the eval adapter depends
only on this local module.

Checkpoint format (a single ``.pt``), written by the trainer or
:func:`save_flamingo_model`::

    {
        "model_state_dict": <FlamingoVLM.state_dict()>,
        "flamingo_config":  <asdict(FlamingoConfig)>,   # incl. media_token_id, vocab_size
        ...                                             # trainer may add optim/epoch/...
    }

``flamingo_config`` stores the architecture, media-token id and vocab size. The
tokenizer is loaded from ``decoder_init_path`` and the frozen DINOv2 tower from
``vision_tower_path`` (both recorded in the config; override the tower path at
load time if it moved).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from apps.baselines.flamingo.modeling.flamingo_model import (
    FlamingoConfig,
    FlamingoVLM,
    build_tokenizer,
    config_to_dict,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

#: Checkpoint filenames searched (in order) when ``model_path`` is a directory.
#: ``latest.pt`` (the final epoch after a full run) comes first because the paper
#: evaluates the final checkpoint, not ``best.pt``.
_CKPT_CANDIDATES = ("latest.pt", "model.pt", "best.pt", "checkpoint.pt")

_DEFAULT_CONTEXT_LEN = 1024


def _tokenizer_source(config: FlamingoConfig) -> str:
    """Return the tokenizer to load for ``config``.

    Phase-0 GPT-2 checkpoints ship their OWN COCO-MC BPE vocab (size 22358), so
    when ``decoder_init == "phase0_ckpt"`` the tokenizer must come from
    ``decoder_init_path`` to stay aligned with the decoder embeddings. Otherwise
    (``scratch`` / ``hf_gpt2``) we use the standard ``tokenizer_name`` (gpt2).
    """
    if config.decoder_init == "phase0_ckpt" and config.decoder_init_path:
        return str(config.decoder_init_path)
    return config.tokenizer_name


def _resolve_ckpt_file(model_path: str) -> Path:
    """Return the checkpoint ``.pt`` (accepts a dir or the file itself)."""
    p = Path(model_path)
    if p.is_file():
        return p
    if p.is_dir():
        for name in _CKPT_CANDIDATES:
            cand = p / name
            if cand.is_file():
                return cand
        msg = f"No checkpoint found in {p} (looked for {_CKPT_CANDIDATES})."
        raise FileNotFoundError(msg)
    msg = f"model_path does not exist: {model_path}"
    raise FileNotFoundError(msg)


def _config_from_payload(payload: dict[str, Any]) -> FlamingoConfig:
    """Reconstruct a :class:`FlamingoConfig` from a checkpoint payload."""
    raw = payload.get("flamingo_config")
    if raw is None:
        msg = (
            "Checkpoint is missing 'flamingo_config'. Native Flamingo checkpoints must embed the "
            "FlamingoConfig (media_token_id + vocab_size + arch knobs) so they reload standalone."
        )
        raise KeyError(msg)
    valid = set(FlamingoConfig().__dict__)
    return FlamingoConfig(**{k: v for k, v in dict(raw).items() if k in valid})


def save_flamingo_model(
    save_dir: str | Path,
    model: FlamingoVLM,
    tokenizer: PreTrainedTokenizerBase | None = None,
    *,
    filename: str = "model.pt",
) -> Path:
    """Write a standalone ``model.pt`` (+ tokenizer dir) for the given model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "flamingo_config": config_to_dict(model.config),
        },
        path,
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(str(save_dir / "tokenizer"))
    return path


def load_pretrained_model(
    model_path: str,
    model_name: str | None = None,  # noqa: ARG001 -- LLaVA-signature parity
    device_map: str | dict[str, str] = "auto",  # noqa: ARG001 -- LLaVA-signature parity
    device: str = "cuda",
    vision_tower_path: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[PreTrainedTokenizerBase, FlamingoVLM, Any, int]:
    """Load a native Flamingo checkpoint.

    Returns ``(tokenizer, model, image_processor, context_len)`` (LLaVA parity).

    Args:
        model_path: A ``.pt`` file or a dir containing one of
            :data:`_CKPT_CANDIDATES`.
        device: Target device string (falls back to CPU if CUDA unavailable).
        vision_tower_path: Optional override for the DINOv2 tower checkpoint dir
            (else uses the value stored in the checkpoint config / torch.hub).
    """
    torch_dtype: torch.dtype = kwargs.pop("torch_dtype", torch.float32)
    resolved_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    ckpt_file = _resolve_ckpt_file(model_path)
    payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    config = _config_from_payload(payload)
    if vision_tower_path is not None:
        config = replace(config, vision_tower_path=vision_tower_path)

    tokenizer, media_token_id = build_tokenizer(
        _tokenizer_source(config),
        media_token=config.media_token,
    )
    if config.media_token_id is not None and int(config.media_token_id) != media_token_id:
        # The tokenizer at decoder_init_path changed since training: <image> would
        # map to a different embedding row and break cross-attention masking.
        msg = (
            f"Tokenizer from {_tokenizer_source(config)!r} gives media_token_id={media_token_id}, but the "
            f"checkpoint was trained with {config.media_token_id}. Point decoder_init_path at the training tokenizer."
        )
        raise ValueError(msg)

    # Rebuild with a scratch GPT-2 (identical module structure) then load weights —
    # avoids a redundant HF download for hf_gpt2 / phase0 checkpoints.
    build_cfg = replace(config, decoder_init="scratch", decoder_init_path=None)
    model = FlamingoVLM(build_cfg)
    model.config = config  # restore provenance (real decoder_init etc.)
    # Strict: the checkpoint stores the vision tower, Perceiver and decoder, so any
    # mismatch means a wrong architecture (e.g. a torch.hub DINOv2 fallback, or a
    # decoder whose sizes differ from FlamingoConfig's hidden_size / n_layer /
    # n_head / max_position_embeddings) and would otherwise score garbage.
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        msg = (
            f"Flamingo checkpoint {ckpt_file} does not match the rebuilt model: {len(missing)} missing keys "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected keys (e.g. {unexpected[:3]}). Check "
            "vision_tower_path and that the decoder sizes in flamingo_config match the trained GPT-2."
        )
        raise RuntimeError(msg)

    model.to(device=resolved_device, dtype=torch_dtype)
    model.eval()

    image_processor = model.vision_tower.image_processor
    context_len = int(config.max_position_embeddings or _DEFAULT_CONTEXT_LEN)
    return tokenizer, model, image_processor, context_len
