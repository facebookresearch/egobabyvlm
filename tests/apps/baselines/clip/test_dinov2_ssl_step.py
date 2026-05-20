# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: DINOv2SSL builds, prepares a batch, and runs a single training step.

Exercises the full path that the dinov2 packaging refactor touched:
- bare ``import dinov2.distributed`` resolves to the packaged tree
- ``DINOv2SSL.__init__`` builds ``SSLMetaArch`` + FSDP + schedulers
- the previously-needed sys.modules gymnastics are gone (we no longer set
  ``_LOCAL_RANK`` on two distinct dinov2 module entries)
- ``prepare_batch`` + ``step`` run end-to-end on a dummy image batch

Skipped when CUDA isn't available (FSDP requires NCCL).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DINOV2_CONFIG = _REPO_ROOT / "apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_coco.yaml"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DINOv2 SSL requires CUDA + NCCL")
def test_dinov2_ssl_step() -> None:
    """One forward/backward step on a dummy batch should complete without error."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

    from apps.baselines.clip.modeling.dinov2_ssl import DINOv2SSL

    ssl = DINOv2SSL(
        config_path=str(_DINOV2_CONFIG),
        device="cuda",
        overrides={
            "train": {"OFFICIAL_EPOCH_LENGTH": 2, "batch_size_per_gpu": 2},
            "optim": {"epochs": 2, "warmup_epochs": 0},
            "teacher": {"warmup_teacher_temp_epochs": 0},
        },
    )

    images = torch.randn(2, 3, 224, 224)
    batch = ssl.prepare_batch(images)
    losses = ssl.step(batch, iteration=0)
    assert isinstance(losses, dict), f"step() returned {type(losses).__name__}, expected dict"
    assert losses, "step() returned an empty loss dict"
    for k, v in losses.items():
        assert torch.isfinite(torch.tensor(v)).item(), f"loss {k} is not finite: {v}"
