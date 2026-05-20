# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Verify the alignment_scoring ConfigStore registers every pipeline schema."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore


def test_config_store_registers_all_pipeline_schemas() -> None:
    """Importing apps.alignment_scoring must register every base pipeline node."""
    import apps.alignment_scoring  # noqa: F401  — import for side-effect

    repo = ConfigStore.instance().repo
    expected = {
        "base_clip_scoring_pipeline.yaml",
        "base_clip_processor.yaml",
        "base_sts_scoring_pipeline.yaml",
        "base_sts_processor.yaml",
        "base_captioning_pipeline.yaml",
        "base_vqa_scoring_pipeline.yaml",
        "base_finetune_lora.yaml",
    }
    assert expected.issubset(repo.keys()), (
        f"Missing schemas: {expected - set(repo.keys())}; have {sorted(repo.keys())[:30]}"
    )


def test_dataclass_schemas_have_no_circular_imports() -> None:
    """Just instantiating the module should not crash and should expose the public dataclasses."""
    from apps.alignment_scoring import configs

    for name in (
        "DatasetConfig",
        "DataConfig",
        "ModelConfig",
        "LoraConfig",
        "CLIPOptimConfig",
        "CLIPProcessorConfig",
        "CLIPScoringPipelineConfig",
        "SonarSTSScoringPipelineConfig",
        "PLMGenerationConfig",
        "CaptioningPipelineConfig",
        "VQAScoringPipelineConfig",
        "FinetuneLoraConfig",
    ):
        assert hasattr(configs, name), f"Missing dataclass {name}"
