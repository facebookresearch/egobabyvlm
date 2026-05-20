# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training-checkpoint save / load helpers shared by trainers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from core.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def atomic_torch_save(payload: Any, path: str | Path) -> None:  # noqa: ANN401 -- torch.save accepts any pickleable
    """``torch.save(payload, path)`` with a tmp+rename so SIGTERM mid-save can't leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def prune_checkpoints(directory: str | Path, pattern: str, keep_last: int) -> None:
    """Delete all but the ``keep_last`` most recently-named matches of ``pattern`` in ``directory``.

    Matches are sorted by filename, so use zero-padded epoch suffixes for
    correct ordering. No-op when ``keep_last <= 0``. Caller is responsible
    for restricting to rank 0 if needed.
    """
    if keep_last <= 0:
        return
    existing = sorted(Path(directory).glob(pattern))
    for stale in existing[:-keep_last]:
        stale.unlink(missing_ok=True)
        logger.info("Pruned old checkpoint %s", stale)


def get_last_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the most recent ``checkpoint-NNNNN.pth`` in ``output_dir``, or None."""
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        logger.info("No checkpoint directory at '%s'", output_dir_path)
        return None
    existing = sorted(output_dir_path.glob("checkpoint-*.pth"), key=lambda p: p.name)
    if not existing:
        logger.info("No checkpoints found in '%s'", output_dir_path)
        return None
    return existing[-1]


def save_checkpoint(  # noqa: PLR0913
    output_dir: str | Path,
    epoch: int,
    container: dict[str, Any],
    *,
    config: Any | None = None,  # noqa: ANN401 — accepts plain dicts or OmegaConf DictConfig
    max_checkpoints: int | None = None,
    epoch_offset: int | None = None,
    is_best: bool = False,
    lora_only: bool = False,
) -> str:
    """Save state-dicts to ``<output_dir>/checkpoint-<epoch:05d>.pth``.

    ``container`` is a dict of ``{"model": ..., "optimizer": ..., "scaler": ...}``
    state_dicts. Returns the path written. No-op on non-main ranks.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir_path / f"checkpoint-{epoch:05d}.pth"

    payload = dict(container)
    if lora_only and "model" in payload:
        original_size = len(payload["model"])
        lora_state = {
            name: param for name, param in payload["model"].items() if "lora_" in name or "logit_scale" in name
        }
        payload["model"] = lora_state
        payload["lora_only"] = True
        logger.info(
            "Saving LoRA-only checkpoint: %d params (reduced from %d)",
            len(lora_state),
            original_size,
        )

    if config is not None:
        payload["config"] = OmegaConf.to_container(config) if OmegaConf.is_config(config) else config
    payload["epoch"] = epoch
    if epoch_offset is not None:
        payload["epoch_offset"] = epoch_offset

    if not is_main_process():
        return str(checkpoint_path)

    atomic_torch_save(payload, checkpoint_path)
    if is_best:
        shutil.copy2(checkpoint_path, output_dir_path / "best_model.pth")

    if max_checkpoints is not None:
        prune_checkpoints(output_dir_path, "checkpoint-*.pth", max_checkpoints)

    logger.info("Saved checkpoint '%s'", checkpoint_path)
    return str(checkpoint_path)


def _resolve_resume_path(
    output_dir: str | Path,
    resume: str | Path | None,
    basename: str | None,
) -> Path | None:
    """Resolve a resume path: prefer ``resume`` (with optional ``basename`` if it's a dir)."""
    resolved = Path(resume) if resume is not None else get_last_checkpoint(output_dir)
    if resolved is None:
        return None
    if resolved.is_dir() and basename:
        resolved = resolved / f"{basename}.pth"
    return resolved


def _load_state_dicts(
    container: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    is_lora_only: bool,
) -> list[str]:
    """Load each ``container`` value's state_dict from ``checkpoint`` in place."""
    matched: list[str] = []
    for key, target in container.items():
        if key not in checkpoint:
            continue
        state = checkpoint.pop(key)
        if key == "model" and is_lora_only:
            msg = target.load_state_dict(state, strict=False)
            logger.info("Loaded LoRA weights into model: %s", msg)
        else:
            target.load_state_dict(state)
        matched.append(key)
    return matched


def _infer_start_epoch(
    checkpoint: dict[str, Any],
    epoch_offset: int | None,
    fallback_path: Path,
) -> int:
    """Pop ``epoch`` from the checkpoint or fall back to the filename suffix."""
    if "epoch" in checkpoint:
        start_epoch = checkpoint.pop("epoch") + 1
        if epoch_offset is not None:
            start_epoch -= 1  # current epoch is unfinished
        return start_epoch
    suffix = fallback_path.stem.split("-")[-1].split(".")[0]
    if suffix.isnumeric():
        return int(suffix) + 1
    logger.warning("Could not infer epoch from %s; starting at 0", fallback_path)
    return 0


def load_checkpoint(
    container: dict[str, Any],
    *,
    output_dir: str | Path,
    resume: str | Path | None = None,
    basename: str | None = None,
) -> tuple[int, int | None]:
    """Load a checkpoint into ``container`` (a dict of state-dict-bearing objects).

    Args:
        container: ``{"model": model, "optimizer": optim, ...}`` — each value must
            implement ``load_state_dict``. Keys missing from the checkpoint are
            silently skipped.
        output_dir: Where to scan for ``checkpoint-*.pth`` if ``resume`` is None.
        resume: Explicit checkpoint path; if a directory, ``basename + ".pth"`` is
            appended; if None, the latest ``checkpoint-*.pth`` in ``output_dir`` wins.
        basename: When ``resume`` points at a directory, the file to load
            (e.g. ``"best_model"`` → ``"best_model.pth"``).

    Returns:
        ``(start_epoch, epoch_offset)`` to resume training from. ``start_epoch=0``
        and ``epoch_offset=None`` mean nothing was loaded.
    """
    resolved = _resolve_resume_path(output_dir, resume, basename)
    if resolved is None:
        return 0, None

    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    is_lora_only = checkpoint.pop("lora_only", False)
    if is_lora_only:
        logger.info("Detected LoRA-only checkpoint; using non-strict model loading")

    matched = _load_state_dicts(container, checkpoint, is_lora_only=is_lora_only)

    epoch_offset = checkpoint.pop("epoch_offset", None)
    if epoch_offset is not None:
        epoch_offset += 1
    start_epoch = _infer_start_epoch(checkpoint, epoch_offset, resolved)

    checkpoint.pop("config", None)
    if checkpoint:
        logger.info("Extra keys in checkpoint (ignored): %s", list(checkpoint.keys()))

    logger.info(
        "Resumed from %s at epoch=%d (offset=%s, matched=%s)",
        resolved,
        start_epoch,
        epoch_offset,
        matched,
    )
    return start_epoch, epoch_offset
