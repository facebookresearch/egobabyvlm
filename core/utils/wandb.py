# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Weights & Biases helpers — thin wrappers that no-op gracefully when wandb is unavailable.

These take plain arguments rather than a particular config schema so any
trainer can use them without coupling to a top-level ``Config`` type.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from core.utils.distributed import is_main_process

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

try:
    import wandb

    _is_wandb_available = True
except ImportError:
    logger.warning("wandb is not installed; wandb_log / init_wandb will no-op")
    _is_wandb_available = False


def wandb_run_name(output_dir: str) -> str:
    """Build a stable wandb run name from SLURM env + the experiment's output dir."""
    slurm_str = (
        f"{os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID', '0')}_{os.getenv('SLURM_ARRAY_TASK_ID', '0')}"
    )
    run_str = str(output_dir).replace("/", "__")
    return f"{slurm_str}_{run_str}"


def _stable_run_id(run_name: str) -> str:
    """Deterministic 16-hex-char wandb run id derived from ``run_name``.

    Used as a fallback when neither an explicit ``run_id`` nor a SLURM job id
    is available, so a re-launched job with the same ``run_name`` re-attaches
    to the same wandb run via ``resume="allow"``.
    """
    return hashlib.blake2b(run_name.encode(), digest_size=8).hexdigest()


def init_wandb(  # noqa: PLR0913
    *,
    project: str | None,
    entity: str | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    log_dir: str | None = None,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    resume: bool = False,
    log_code: bool = False,
    model: torch.nn.Module | None = None,
    mode: str | None = None,
    metric_axes: dict[str, str | None] | None = None,
) -> str | None:
    """Initialize wandb on the main process only; returns the resolved run_id (or None).

    Args:
        project: W&B project; ``None`` disables logging entirely.
        run_name: Explicit run name. If ``None``, derived from ``output_dir``
            via :func:`wandb_run_name` (which encodes the SLURM job id).
        run_id: Explicit run id. Precedence when not given: ``SLURM_JOB_ID``
            env var > ``_stable_run_id(run_name)`` (hashed) > ``time.time()``.
            A stable id lets a re-launched job re-attach to the same wandb
            run via ``resume="allow"``.
        mode: ``"online"`` (default), ``"offline"``, or ``"disabled"``.
            The ``WANDB_MODE`` env var overrides this when set.
        metric_axes: Optional ``{glob_pattern: step_metric_name}`` mapping
            passed to ``wandb.define_metric``. ``None`` as the value registers
            the metric itself as an x-axis. Example:
            ``{"step": None, "train/*": "step", "val/*": "epoch"}``.
    """
    if not _is_wandb_available or project is None or not is_main_process():
        return None

    resolved_name = run_name or (wandb_run_name(output_dir) if output_dir else None)
    slurm_id = os.getenv("SLURM_JOB_ID")
    if run_id:
        resolved_run_id = run_id
    elif slurm_id:
        resolved_run_id = slurm_id
    elif resolved_name:
        resolved_run_id = _stable_run_id(resolved_name)
    else:
        resolved_run_id = str(int(time.time()))

    env_mode = os.environ.get("WANDB_MODE")
    resolved_mode = cast("Literal['online', 'offline', 'disabled', 'shared']", env_mode or mode or "online")
    # Resume if explicitly requested, an explicit run_id was passed, or we
    # derived a stable id (from SLURM or from run_name); skip when only a
    # time.time() fallback was used (a new run each launch).
    resume_kw: Literal["allow"] | None = (
        "allow" if (resume or run_id or slurm_id or (resolved_name and not run_id and not slurm_id)) else None
    )

    wandb.init(
        name=resolved_name,
        project=project,
        entity=entity,
        dir=log_dir,
        config=config,
        id=resolved_run_id,
        resume=resume_kw,
        mode=resolved_mode,
        save_code=False,
    )
    assert wandb.run is not None
    if metric_axes:
        for pattern, step_metric in metric_axes.items():
            if step_metric is None:
                wandb.run.define_metric(pattern)
            else:
                wandb.run.define_metric(pattern, step_metric=step_metric)
    if log_code:
        wandb.run.log_code(
            str(Path.cwd().parent),
            include_fn=lambda path: path.endswith((".py", ".yaml")),
        )
    if model is not None:
        wandb.watch(model, log="all", log_freq=1000)
    logger.info(
        "Initialized W&B run: project=%s, name=%s, id=%s, mode=%s",
        project,
        resolved_name,
        resolved_run_id,
        resolved_mode,
    )
    return resolved_run_id


def wandb_log(data: dict[str, Any], *, step: int | None = None, disable_format: bool = False) -> None:
    """Log a dict to wandb on the main process; no-op when wandb is unavailable.

    Args:
        data: Flat dict of metrics.
        step: Optional global step.
        disable_format: If False (the default), rewrites legacy ``train_*`` /
            ``val_*`` key prefixes to ``train/*`` / ``val/*`` so the
            grouped-metric x-axes set up via ``metric_axes`` apply.
    """
    if not _is_wandb_available or not is_main_process():
        return
    formatted = (
        data if disable_format else {k.replace("val_", "val/").replace("train_", "train/"): v for k, v in data.items()}
    )
    wandb.log(formatted, step=step)


def finish_wandb() -> None:
    """Close the active wandb run on the main process; no-op otherwise."""
    if not _is_wandb_available or not is_main_process():
        return
    if wandb.run is not None:
        wandb.finish()
