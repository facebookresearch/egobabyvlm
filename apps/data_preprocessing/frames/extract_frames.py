# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frame extraction from videos using stopes job arrays for parallel processing.

Frames land directly under ``output_dir/frames/<video_name>/`` as JPEGs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore
from stopes.core import Requirements, StopesModule

from core.utils import LauncherConfig
from core.utils.logging import setup_logging

if TYPE_CHECKING:
    from stopes.core import Launcher

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractorConfig:
    """Configuration for the frame extraction processor."""

    #: Local directory containing video files (recursively searched).
    data_dir: str = "???"

    #: Local directory where extracted frames + summary will be written.
    output_dir: str = "???"

    #: Number of videos processed by each array job (smaller = better load
    #: balance, more job overhead; larger = the opposite).
    videos_per_chunk: int = 100

    #: Frames per second to sample.
    fps: int = 1

    #: Video file extensions to discover under ``data_dir`` (without leading dot).
    video_extensions: tuple[str, ...] = ("mp4", "avi", "mov", "mkv")


@dataclass
class FrameExtractionPipelineConfig:
    """Top-level Hydra config: processor + launcher."""

    processor: FrameExtractorConfig = field(default_factory=FrameExtractorConfig)

    launcher: LauncherConfig = field(default_factory=LauncherConfig)


cs = ConfigStore.instance()
cs.store(name="frame_extraction_pipeline", node=FrameExtractionPipelineConfig)


def _video_name(path: str | Path) -> str:
    """Return the video name (filename stem) for a video path."""
    return Path(path).stem


def _extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps: int,
    video_name: str,
) -> int:
    """Extract frames from ``video_path`` into ``output_dir`` at ``fps`` FPS.

    Returns the number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / f"{video_name}_%d.jpg")

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        output_pattern,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        msg = f"ffmpeg failed for {video_path}: {completed.stderr.strip()}"
        raise RuntimeError(msg)

    return sum(1 for _ in output_dir.glob(f"{video_name}_*.jpg"))


class FrameExtractionModule(StopesModule):
    """Stopes module for extracting frames from videos in parallel using job arrays."""

    def __init__(self, config: FrameExtractorConfig) -> None:
        super().__init__(config, FrameExtractorConfig)
        self._video_paths: list[Path] | None = None

    def requirements(self) -> Requirements:
        """CPU-only ffmpeg work; one task per array element."""
        return Requirements(
            nodes=1,
            mem_gb=32,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=24,
            timeout_min=360,
        )

    def name(self) -> str:
        return "frame_extraction"

    @property
    def video_paths(self) -> list[Path]:
        """Discover video files under ``config.data_dir`` (cached for the run)."""
        if self._video_paths is None:
            data_root = Path(self.config.data_dir)
            paths: list[Path] = []
            for ext in self.config.video_extensions:
                paths.extend(data_root.rglob(f"*.{ext}"))
            self._video_paths = sorted(paths)
            logger.info("Found %d video files under %s", len(self._video_paths), data_root)
        return self._video_paths

    @property
    def num_chunks(self) -> int:
        return math.ceil(len(self.video_paths) / self.config.videos_per_chunk)

    def array(self) -> list[tuple[int, int]]:
        """Array job indices as ``(start_idx, end_idx)`` slices over ``video_paths``."""
        return [
            (
                self.config.videos_per_chunk * i,
                min(self.config.videos_per_chunk * (i + 1), len(self.video_paths)),
            )
            for i in range(self.num_chunks)
        ]

    def run(self, iteration_value: tuple[int, int], iteration_index: int) -> list[dict]:
        """Process one chunk: extract frames into ``output_dir/frames/<video_name>/``."""
        start_idx, end_idx = iteration_value
        logger.info("Processing chunk %d: videos %d-%d", iteration_index, start_idx, end_idx)

        chunk_paths = self.video_paths[start_idx:end_idx]
        frames_root = Path(self.config.output_dir) / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)

        chunk_results: list[dict] = []
        for i, video_path in enumerate(chunk_paths):
            name = _video_name(video_path)
            logger.info("Processing video %d/%d: %s", i + 1, len(chunk_paths), name)

            video_frames_dir = frames_root / name
            try:
                frame_count = _extract_frames_from_video(
                    video_path=video_path,
                    output_dir=video_frames_dir,
                    fps=self.config.fps,
                    video_name=name,
                )
                chunk_results.append(
                    {
                        "video_path": str(video_path),
                        "video_name": name,
                        "frames_dir": str(video_frames_dir),
                        "frame_count": frame_count,
                        "status": "success",
                    }
                )
            except Exception as e:
                logger.exception("Error processing %s", video_path)
                # Best-effort cleanup of partial output.
                if video_frames_dir.exists():
                    shutil.rmtree(video_frames_dir, ignore_errors=True)
                chunk_results.append(
                    {
                        "video_path": str(video_path),
                        "video_name": name,
                        "frames_dir": None,
                        "frame_count": 0,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        logger.info(
            "Chunk %d completed: %d/%d videos succeeded",
            iteration_index,
            sum(1 for r in chunk_results if r["status"] == "success"),
            len(chunk_paths),
        )
        return chunk_results


def _aggregate_results(all_results: list[list[dict]]) -> dict:
    flat = [r for chunk in all_results for r in chunk]
    return {
        "summary": {
            "total_videos": len(flat),
            "successful": sum(1 for r in flat if r["status"] == "success"),
            "failed": sum(1 for r in flat if r["status"] == "failed"),
            "total_frames": sum(r["frame_count"] for r in flat),
        },
        "videos": flat,
    }


async def pipeline(config: FrameExtractionPipelineConfig) -> dict:
    """Schedule the frame-extraction array job and write a JSON summary."""
    launcher: Launcher = hydra.utils.instantiate(config.launcher)
    module = FrameExtractionModule(config.processor)
    logger.info(
        "Launching %d jobs for %d videos",
        module.num_chunks,
        len(module.video_paths),
    )

    all_results = await launcher.schedule(module)
    results = _aggregate_results(all_results)

    output_dir = Path(config.processor.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"frame_extraction_results_{config.processor.fps}fps.json"
    logger.info("Writing summary to %s", summary_path)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    summary = results["summary"]
    logger.info(
        "Done! %d videos processed (%d succeeded), %d total frames.",
        summary["total_videos"],
        summary["successful"],
        summary["total_frames"],
    )
    return results


@hydra.main(version_base=None, config_name="frame_extraction_pipeline")
def main(config: FrameExtractionPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    asyncio.run(pipeline(config))


if __name__ == "__main__":
    main()
