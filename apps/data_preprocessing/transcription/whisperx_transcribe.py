# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""WhisperX transcription with word-level timestamps, scheduled via stopes job arrays."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.core.config_store import ConfigStore
from stopes.core import Requirements, StopesModule

from core.utils import LauncherConfig
from core.utils.logging import setup_logging

if TYPE_CHECKING:
    from stopes.core import Launcher

logger = logging.getLogger(__name__)

#: Sample rate WhisperX expects.
_AUDIO_SAMPLE_RATE = 16_000


@dataclass
class WhisperXConfig:
    """Configuration for the WhisperX transcription processor."""

    #: Local directory containing video files (recursively searched).
    data_dir: str = "???"

    #: Local directory where transcription JSONs + summary will be written.
    output_dir: str = "???"

    #: Number of videos processed by each array job (smaller chunks reduce
    #: tail-latency since transcription dominates).
    videos_per_chunk: int = 10

    #: WhisperX model name (large-v2, medium, small, base, tiny).
    whisperx_model: str = "large-v2"

    #: Batch size for WhisperX inference.
    batch_size: int = 16

    #: Compute type for WhisperX (float16, int8, float32).
    compute_type: str = "float16"

    #: Language code (e.g. ``en``); empty string = auto-detect.
    language: str = "en"

    #: Output format suffix on per-video files (``json``, ``srt``, ``vtt``,
    #: ``txt``, ``tsv``). The pipeline always writes JSON; this only affects
    #: the on-disk extension.
    output_format: str = "json"

    #: Video file extensions to discover under ``data_dir`` (without leading dot).
    video_extensions: tuple[str, ...] = ("mp4", "avi", "mov", "mkv")


@dataclass
class WhisperXPipelineConfig:
    """Top-level Hydra config: processor + launcher."""

    processor: WhisperXConfig = field(default_factory=WhisperXConfig)

    launcher: LauncherConfig = field(default_factory=LauncherConfig)


cs = ConfigStore.instance()
cs.store(name="whisperx_pipeline", node=WhisperXPipelineConfig)


def _video_name(path: str | Path) -> str:
    return Path(path).stem


def _extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract a 16 kHz mono PCM-16 WAV from ``video_path``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{_video_name(video_path)}.wav"
    if audio_path.exists():
        return audio_path

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(_AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-y",
        "-f",
        "wav",
        str(audio_path),
        "-loglevel",
        "error",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        msg = f"ffmpeg failed for {video_path}: {completed.stderr.strip()}"
        raise RuntimeError(msg)
    return audio_path


@dataclass
class _WhisperXModels:
    """WhisperX + alignment model bundle, passed as a single argument."""

    whisper_model: Any
    align_model: Any
    align_metadata: Any
    device: str


def _transcribe_audio(
    audio_path: Path,
    models: _WhisperXModels,
    *,
    batch_size: int = 16,
    language: str | None = "en",
) -> dict | None:
    """Transcribe ``audio_path`` and align word timestamps; ``None`` on failure."""
    import whisperx

    try:
        audio = whisperx.load_audio(str(audio_path))
        result = models.whisper_model.transcribe(audio, batch_size=batch_size)
        detected_language = result.get("language", "unknown")

        if language and detected_language != language:
            logger.warning(
                "Detected language %s doesn't match target %s; skipping word alignment.",
                detected_language,
                language,
            )
            return {
                "segments": result.get("segments", []),
                "language": detected_language,
                "skipped_alignment": True,
            }

        result = whisperx.align(
            result["segments"],
            models.align_model,
            models.align_metadata,
            audio,
            models.device,
            return_char_alignments=False,
        )
    except Exception:
        logger.exception("Error transcribing %s", audio_path)
        return None
    else:
        result["language"] = detected_language
        return result


class WhisperXModule(StopesModule):
    """Stopes module for transcribing videos in parallel using job arrays."""

    def __init__(self, config: WhisperXConfig) -> None:
        super().__init__(config, WhisperXConfig)
        self._video_paths: list[Path] | None = None

    def _load_models(self) -> _WhisperXModels:
        """Load WhisperX + alignment models bundled into a ``_WhisperXModels``."""
        import torch

        # whisperx pulls in pyannote which calls torch.load with weights_only=True
        # under newer torch — patch so older pyannote checkpoints still load.
        original_load = torch.load

        def patched_load(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- torch.load takes arbitrary args.
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load  # type: ignore[assignment]

        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        language = self.config.language or None

        logger.info("Loading WhisperX model '%s' on %s...", self.config.whisperx_model, device)
        whisper_model = whisperx.load_model(
            self.config.whisperx_model,
            device,
            compute_type=self.config.compute_type,
            language=language,
        )
        logger.info("Loading alignment model for language '%s'...", language or "en")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language or "en",
            device=device,
        )
        logger.info("Models loaded.")
        return _WhisperXModels(
            whisper_model=whisper_model,
            align_model=align_model,
            align_metadata=align_metadata,
            device=device,
        )

    def requirements(self) -> Requirements:
        """One GPU per array job for WhisperX inference."""
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=8,
            timeout_min=480,
        )

    def name(self) -> str:
        return "whisperx_transcription"

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
        return [
            (
                self.config.videos_per_chunk * i,
                min(self.config.videos_per_chunk * (i + 1), len(self.video_paths)),
            )
            for i in range(self.num_chunks)
        ]

    def run(self, iteration_value: tuple[int, int], iteration_index: int) -> list[dict]:
        """Process one chunk: extract audio → WhisperX → write per-video JSON."""
        start_idx, end_idx = iteration_value
        logger.info("Processing chunk %d: videos %d-%d", iteration_index, start_idx, end_idx)

        models = self._load_models()
        chunk_paths = self.video_paths[start_idx:end_idx]

        transcriptions_root = Path(self.config.output_dir) / "transcriptions"
        transcriptions_root.mkdir(parents=True, exist_ok=True)

        chunk_results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="whisperx_audio_") as audio_scratch:
            audio_dir = Path(audio_scratch)
            for i, video_path in enumerate(chunk_paths):
                name = _video_name(video_path)
                logger.info("Processing video %d/%d in chunk: %s", i + 1, len(chunk_paths), name)
                result: dict = {
                    "video_path": str(video_path),
                    "video_name": name,
                    "status": "success",
                    "error": None,
                    "language": None,
                    "num_segments": 0,
                    "num_words": 0,
                    "transcription_path": None,
                }

                try:
                    audio_path = _extract_audio(video_path, audio_dir)
                    transcription = _transcribe_audio(
                        audio_path,
                        models,
                        batch_size=self.config.batch_size,
                        language=self.config.language or None,
                    )

                    if transcription is None:
                        result["status"] = "failed"
                        result["error"] = "Transcription returned None"
                    else:
                        segments = transcription.get("segments", [])
                        num_words = sum(len(s.get("words", [])) for s in segments)
                        result["language"] = transcription.get("language", "unknown")
                        result["num_segments"] = len(segments)
                        result["num_words"] = num_words

                        out_path = transcriptions_root / f"{name}.{self.config.output_format}"
                        with out_path.open("w", encoding="utf-8") as f:
                            json.dump(transcription, f, indent=2, ensure_ascii=False)
                        result["transcription_path"] = str(out_path)
                        logger.info("  Saved %d segments, %d words", len(segments), num_words)

                    if audio_path.exists():
                        audio_path.unlink()
                except Exception as e:
                    logger.exception("Error processing %s", video_path)
                    result["status"] = "failed"
                    result["error"] = str(e)

                chunk_results.append(result)

        logger.info(
            "Chunk %d completed: %d/%d videos succeeded",
            iteration_index,
            sum(1 for r in chunk_results if r["status"] == "success"),
            len(chunk_paths),
        )
        return chunk_results


def _aggregate_results(all_results: list[list[dict]]) -> dict:
    flat = [r for chunk in all_results for r in chunk]
    successful = sum(1 for r in flat if r["status"] == "success")
    failed = sum(1 for r in flat if r["status"] == "failed")
    total_segments = sum(r.get("num_segments", 0) for r in flat)
    total_words = sum(r.get("num_words", 0) for r in flat)

    language_counts: dict[str, int] = {}
    for r in flat:
        lang = r.get("language") or "unknown"
        language_counts[lang] = language_counts.get(lang, 0) + 1

    return {
        "summary": {
            "total_videos": len(flat),
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / len(flat) * 100, 2) if flat else 0,
            "total_segments": total_segments,
            "total_words": total_words,
            "avg_segments_per_video": round(total_segments / successful, 2) if successful else 0,
            "avg_words_per_video": round(total_words / successful, 2) if successful else 0,
        },
        "language_distribution": language_counts,
        "videos": flat,
    }


async def pipeline(config: WhisperXPipelineConfig) -> dict:
    """Schedule the WhisperX array job and write a JSON summary."""
    launcher: Launcher = hydra.utils.instantiate(config.launcher)
    module = WhisperXModule(config.processor)
    logger.info("Launching %d jobs for %d videos", module.num_chunks, len(module.video_paths))

    all_results = await launcher.schedule(module)
    results = _aggregate_results(all_results)

    output_dir = Path(config.processor.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "transcription_summary.json"
    logger.info("Writing summary to %s", summary_path)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    summary = results["summary"]
    logger.info(
        "Done! %d videos processed (%d succeeded, %.2f%%), %d total words.",
        summary["total_videos"],
        summary["successful"],
        summary["success_rate"],
        summary["total_words"],
    )
    return results


@hydra.main(version_base=None, config_name="whisperx_pipeline")
def main(config: WhisperXPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    asyncio.run(pipeline(config))


if __name__ == "__main__":
    main()
