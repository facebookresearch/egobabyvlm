# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Verify the contrastive trainer ConfigStore registers every schema."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

# Resolve relative to this file so the test runs the same on local workstations
# and in CI checkouts (e.g. /home/runner/work/egobabyvlm/egobabyvlm/...).
CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "apps" / "baselines" / "clip" / "configs")


def test_config_store_registers_all_trainer_schemas() -> None:
    import apps.baselines.clip  # noqa: F401  — import for ConfigStore side-effect

    repo = ConfigStore.instance().repo
    expected = {
        "base_contrastive_trainer.yaml",
    }
    assert expected.issubset(repo.keys()), f"Missing schemas: {expected - set(repo.keys())}"


def test_dataclass_schemas_importable() -> None:
    from apps.baselines.clip import configs

    for name in (
        "ContrastiveTrainerConfig",
        "ModeConfig",
        "ModelConfig",
        "DataConfig",
        "OptimConfig",
        "TextEncoderConfig",
        "TextOnlyDataConfig",
        "DINOv2Config",
        "CheckpointConfig",
        "WandbConfig",
    ):
        assert hasattr(configs, name), f"Missing dataclass {name}"


def _compose(mode: str, *extras: str) -> object:
    import apps.baselines.clip  # noqa: F401

    overrides = [
        "name=test",
        f"mode={mode}",
        "data.train_dataset.manifest_path=/tmp/x",
        "data.train_dataset.image_root=/tmp/y",
        "data.val_dataset.manifest_path=/tmp/x",
        "checkpoint.save_dir=/tmp/ckpt",
        *extras,
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_compose_contrastive() -> None:
    cfg = _compose("contrastive")
    assert cfg.mode.name == "contrastive"
    assert dict(cfg.mode.interleave) == {"contrastive": 1}
    assert cfg.mode.sync_vision_from_dinov2 is False


def test_compose_interleaved_lm() -> None:
    cfg = _compose("interleaved_lm", "+text_only_data=default", "text_only_data.train_file=/tmp/text.txt")
    assert cfg.mode.name == "interleaved_lm"
    assert "mlm" in cfg.mode.interleave
    assert cfg.text_only_data.train_file == "/tmp/text.txt"


def test_compose_interleaved_dino() -> None:
    cfg = _compose("interleaved_dino", "+dinov2=vitb14_coco")
    assert cfg.mode.name == "interleaved_dino"
    assert "dinov2" in cfg.mode.interleave
    assert cfg.mode.sync_vision_from_dinov2 is True
    assert cfg.dinov2.config_path.endswith("vitb14_coco.yaml")


def test_compose_triple() -> None:
    cfg = _compose(
        "triple",
        "+text_only_data=default",
        "text_only_data.train_file=/tmp/text.txt",
        "+dinov2=vitb14_coco",
    )
    assert cfg.mode.name == "triple"
    assert set(cfg.mode.interleave.keys()) == {"contrastive", "mlm", "dinov2"}
