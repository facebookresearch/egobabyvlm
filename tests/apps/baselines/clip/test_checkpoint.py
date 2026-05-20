# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the self-contained checkpoint format.

The ``test_save_load_round_trip`` test is marked ``integration`` because it
instantiates a HuggingFace BERT backbone (777 MB checkpoint) — too heavy for
the default ``pytest -q`` run. Other tests in this file are pure file-system
manipulation and run unconditionally.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.integration
def test_save_load_round_trip(tmp_path: Path) -> None:
    from apps.baselines.clip.modeling import (
        MultiModalModel,
        RandomViTVisionEncoder,
        TextEncoder,
    )
    from apps.baselines.clip.training import (
        InterleaveScheduler,
        build_adamw,
        load_checkpoint,
        save_checkpoint,
    )

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm = MultiModalModel(ve, te)
    opt = build_adamw(mm.parameters(), lr=1e-3)
    sched = InterleaveScheduler({"contrastive": 1})
    cfg = OmegaConf.create({"foo": "bar", "model": {"embedding_dim": 128}})

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=mm,
        optimizers={"contrastive": opt},
        scheduler=sched,
        config=cfg,
        epoch=3,
        step=42,
        best_val_loss=0.5,
    )

    te2 = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve2 = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm2 = MultiModalModel(ve2, te2)
    opt2 = build_adamw(mm2.parameters(), lr=1e-3)
    sched2 = InterleaveScheduler({"contrastive": 1})

    payload = load_checkpoint(
        ckpt_path,
        model=mm2,
        optimizers={"contrastive": opt2},
        scheduler=sched2,
    )
    assert payload["epoch"] == 3
    assert payload["step"] == 42
    assert payload["best_val_loss"] == 0.5
    assert payload["config"] == {"foo": "bar", "model": {"embedding_dim": 128}}

    for (n1, p1), (_, p2) in zip(mm.named_parameters(), mm2.named_parameters(), strict=True):
        assert torch.allclose(p1, p2), f"{n1} mismatch"


def _make_trainer_for_prune(tmp_path: Path, keep_last: int) -> object:
    """Construct a minimal ContrastiveTrainer instance without doing any training.

    We bypass ``__init__`` so we don't need a real model / optimizer / loaders;
    only the attributes that ``_prune_old_epoch_checkpoints`` reads are set.
    """
    from apps.baselines.clip.training.loop import ContrastiveTrainer

    trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
    trainer.config = OmegaConf.create({"checkpoint": {"save_dir": str(tmp_path), "keep_last": keep_last}})
    return trainer


def _touch_epoch_files(save_dir: Path, epochs: list[int]) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = [save_dir / f"epoch_{e:04d}.pt" for e in epochs]
    for p in paths:
        p.write_bytes(b"x")
    return paths


def _run_prune(trainer: object, *, is_main: bool = True) -> None:
    from apps.baselines.clip.training.loop import ContrastiveTrainer

    with patch("apps.baselines.clip.training.loop.is_main_process", return_value=is_main):
        ContrastiveTrainer._prune_old_epoch_checkpoints(trainer)


def test_prune_keeps_last_n_epoch_checkpoints(tmp_path: Path) -> None:
    _touch_epoch_files(tmp_path, [0, 1, 2, 3, 4])
    trainer = _make_trainer_for_prune(tmp_path, keep_last=2)
    _run_prune(trainer)

    remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
    assert remaining == ["epoch_0003.pt", "epoch_0004.pt"]


def test_prune_preserves_special_checkpoints(tmp_path: Path) -> None:
    _touch_epoch_files(tmp_path, [0, 1, 2, 3])
    for special in ("best.pt", "latest.pt", "interrupted.pt"):
        (tmp_path / special).write_bytes(b"x")

    trainer = _make_trainer_for_prune(tmp_path, keep_last=1)
    _run_prune(trainer)

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "best.pt",
        "epoch_0003.pt",
        "interrupted.pt",
        "latest.pt",
    ]


def test_prune_disabled_when_keep_last_non_positive(tmp_path: Path) -> None:
    _touch_epoch_files(tmp_path, [0, 1, 2])
    trainer = _make_trainer_for_prune(tmp_path, keep_last=0)
    _run_prune(trainer)

    assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == [
        "epoch_0000.pt",
        "epoch_0001.pt",
        "epoch_0002.pt",
    ]


def test_prune_when_fewer_than_keep_last(tmp_path: Path) -> None:
    _touch_epoch_files(tmp_path, [0, 1])
    trainer = _make_trainer_for_prune(tmp_path, keep_last=5)
    _run_prune(trainer)

    assert sorted(p.name for p in tmp_path.glob("epoch_*.pt")) == ["epoch_0000.pt", "epoch_0001.pt"]


def test_prune_noop_on_non_main_rank(tmp_path: Path) -> None:
    _touch_epoch_files(tmp_path, [0, 1, 2, 3, 4])
    trainer = _make_trainer_for_prune(tmp_path, keep_last=1)
    _run_prune(trainer, is_main=False)

    assert len(list(tmp_path.glob("epoch_*.pt"))) == 5
