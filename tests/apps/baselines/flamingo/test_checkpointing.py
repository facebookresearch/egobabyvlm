# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Flamingo resume and eval-time checkpoint loading.

The loading tests need a trained Flamingo checkpoint (``$FLAMINGO_CHECKPOINT``,
a ``.pt`` whose recorded vision tower and tokenizer paths exist) and are marked
``integration``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from apps.baselines.flamingo.training.loop import FlamingoTrainer

FLAMINGO_CHECKPOINT = os.environ.get("FLAMINGO_CHECKPOINT", "")
_have_ckpt = pytest.mark.skipif(
    not (FLAMINGO_CHECKPOINT and Path(FLAMINGO_CHECKPOINT).exists()), reason="$FLAMINGO_CHECKPOINT not set"
)


@pytest.mark.parametrize(("epoch_complete", "expected_epoch"), [(True, 4), (False, None)])
def test_resume_rejects_mid_epoch_checkpoints(epoch_complete: bool, expected_epoch: int | None) -> None:
    payload = {"epoch": 3, "step": 100, "best_val_loss": 1.0, "epoch_complete": epoch_complete}
    trainer = SimpleNamespace(model=None, optimizer=None, scheduler=None, device="cpu")
    with patch("apps.baselines.flamingo.training.loop.load_checkpoint", return_value=payload):
        if expected_epoch is None:
            with pytest.raises(ValueError, match="mid-epoch"):
                FlamingoTrainer.resume(trainer, "interrupted.pt")  # type: ignore[arg-type]
        else:
            FlamingoTrainer.resume(trainer, "latest.pt")  # type: ignore[arg-type]
            assert trainer.epoch == expected_epoch
            assert trainer.global_step == 100


@pytest.mark.integration
@_have_ckpt
def test_load_pretrained_model_strict(tmp_path: Path) -> None:
    from apps.baselines.flamingo.model.builder import load_pretrained_model

    _, model, _, _ = load_pretrained_model(FLAMINGO_CHECKPOINT, device="cpu")
    assert not model.training

    payload = torch.load(FLAMINGO_CHECKPOINT, map_location="cpu", weights_only=False)
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(RuntimeError, match="does not match the rebuilt model"):
        load_pretrained_model(str(tampered), device="cpu")


@pytest.mark.integration
@_have_ckpt
def test_missing_vision_tower_path_raises() -> None:
    from apps.baselines.flamingo.model.builder import load_pretrained_model

    with pytest.raises(FileNotFoundError, match="vision_tower_path"):
        load_pretrained_model(
            FLAMINGO_CHECKPOINT, device="cpu", vision_tower_path="/nonexistent/teacher_checkpoint.pth"
        )
