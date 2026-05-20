# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the stopes-driven preprocessing modules.

These pipelines depend on heavy deps (ffmpeg, WhisperX, CUDA torch) at
``run()`` time but the dataclass + ConfigStore + Hydra entry plumbing
should import cleanly on any host.
"""

from __future__ import annotations

from apps.data_preprocessing.frames.extract_frames import (
    FrameExtractionPipelineConfig,
    FrameExtractorConfig,
)
from apps.data_preprocessing.transcription.whisperx_transcribe import (
    WhisperXConfig,
    WhisperXPipelineConfig,
)


def test_frame_extractor_config_defaults() -> None:
    cfg = FrameExtractorConfig()
    assert cfg.fps == 1
    assert cfg.videos_per_chunk == 100
    assert cfg.video_extensions == ("mp4", "avi", "mov", "mkv")


def test_whisperx_config_defaults() -> None:
    cfg = WhisperXConfig()
    assert cfg.whisperx_model == "large-v2"
    assert cfg.compute_type == "float16"
    assert cfg.batch_size == 16
    assert cfg.language == "en"


def test_pipeline_configs_compose() -> None:
    """Both pipeline configs should instantiate with their default processor + launcher."""
    fcfg = FrameExtractionPipelineConfig()
    assert fcfg.processor.fps == 1
    assert fcfg.launcher is not None

    wcfg = WhisperXPipelineConfig()
    assert wcfg.processor.batch_size == 16
    assert wcfg.launcher is not None
