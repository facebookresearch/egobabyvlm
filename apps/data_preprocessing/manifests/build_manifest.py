# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build train/val/test caption-frame manifests for the contrastive trainer.

Pairs WhisperX-format transcripts (one JSON per source video) with the JPEG
frames produced by ``egobabyvlm-extract-frames`` and emits the
``[{"utterance", "frame_filenames", ...}]`` JSON consumed by
``apps.baselines.clip.data.HowToCaptionsDataset`` /
``Ego4DCaptionsDataset``.

Frame naming convention assumed (matches ``apps/data_preprocessing/frames/``):

    <frames_dir>/<video_name>/<video_name>_<idx>.jpg

where ``idx`` is 1-indexed (ffmpeg's default for ``-vf fps=N``). With a
known ``frames_fps``, frame ``idx`` maps to time
``t = (idx - 0.5) / frames_fps`` (midpoint of the i-th sample window).

The output layout matches ``apps/baselines/clip/configs/data/howto.yaml``
defaults: pass ``train.json`` as ``manifest_path`` and ``frames_dir`` as
``image_root``.
"""

from __future__ import annotations

import bisect
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class BuildManifestConfig:
    """Configuration for the manifest builder."""

    #: Directory of per-video WhisperX transcript JSONs (or filtered VTC outputs).
    transcripts_dir: str = "???"

    #: Directory of per-video frame subdirectories produced by extract_frames.py.
    #: Each ``<frames_dir>/<video_name>/`` holds JPEGs named ``<video_name>_<idx>.jpg``.
    frames_dir: str = "???"

    #: Where to write ``train.json``, ``val.json``, ``test.json``.
    output_dir: str = "???"

    #: FPS the frames were extracted at (must match ``processor.fps`` used for
    #: ``egobabyvlm-extract-frames``). Determines how frame indices map to time.
    frames_fps: float = 1.0

    #: Train/val split fractions (test gets the remainder).
    train_frac: float = 0.85
    val_frac: float = 0.10

    #: Skip utterances with fewer than this many matching frames.
    min_frames_per_utterance: int = 1

    #: Sample shuffling seed.
    seed: int = 42

    #: Filename suffix for input transcripts (typically ``json`` for WhisperX
    #: outputs and the VTC-filtered outputs that share the schema).
    transcript_suffix: str = "json"

    #: Filename pattern for frame files (used to discover frames per video).
    frame_suffix: str = "jpg"


@dataclass
class BuildManifestPipelineConfig:
    """Top-level Hydra config; processor only — no launcher needed."""

    processor: BuildManifestConfig = field(default_factory=BuildManifestConfig)


cs = ConfigStore.instance()
cs.store(name="build_manifest_pipeline", node=BuildManifestPipelineConfig)


def _frame_index_to_time_s(frame_idx: int, frames_fps: float) -> float:
    """Midpoint of the i-th sample window when sampling at ``frames_fps``.

    ffmpeg's ``-vf fps=N`` writes frames at the start of each 1/N-second window
    (idx=1 at t=0). Treating each frame as the midpoint of its window (t = (idx - 0.5)/N)
    makes a frame represent the time interval most likely captured by that JPEG,
    which is what we want for assigning frames to transcript segments.
    """
    return (frame_idx - 0.5) / frames_fps


def _list_frames_for_video(
    video_frames_dir: Path,
    suffix: str,
    frames_fps: float,
) -> list[tuple[str, float]]:
    """List ``(filename, time_s)`` pairs sorted by time, for one video.

    Filenames are returned relative to ``video_frames_dir.parent`` so they can
    be passed straight into the trainer's ``image_root``-rooted lookup.
    """
    if not video_frames_dir.is_dir():
        return []
    name = video_frames_dir.name
    prefix = f"{name}_"
    triples: list[tuple[int, str, float]] = []
    for p in video_frames_dir.glob(f"{name}_*.{suffix}"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        try:
            idx = int(stem[len(prefix) :])
        except ValueError:
            continue
        triples.append((idx, f"{name}/{p.name}", _frame_index_to_time_s(idx, frames_fps)))
    triples.sort(key=lambda t: t[0])
    return [(fname, t) for _, fname, t in triples]


def _segments_for_transcript(transcript_path: Path) -> list[dict[str, Any]]:
    """Load a WhisperX-shaped transcript and return its ``segments`` list."""
    try:
        with transcript_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read transcript %s", transcript_path)
        return []
    segments = data.get("segments")
    if not isinstance(segments, list):
        return []
    return segments


def build_records_for_video(
    transcript_path: Path,
    video_frames_dir: Path,
    *,
    frames_fps: float,
    min_frames_per_utterance: int,
    frame_suffix: str,
) -> list[dict[str, Any]]:
    """Build per-utterance records for one video.

    Returns one record per usable transcript segment, in source order. Each
    record has the schema consumed by ``HowToCaptionsDataset`` /
    ``Ego4DCaptionsDataset``: ``utterance`` (str) and ``frame_filenames``
    (list of paths relative to the trainer's ``image_root``), plus
    ``utterance_num`` / ``video_filename`` / ``transcript_filename`` for
    debugging and downstream filtering.
    """
    segments = _segments_for_transcript(transcript_path)
    if not segments:
        return []

    frames = _list_frames_for_video(video_frames_dir, frame_suffix, frames_fps)
    if not frames:
        return []

    frame_times = [t for _, t in frames]
    records: list[dict[str, Any]] = []

    video_filename = f"{video_frames_dir.name}.mp4"
    transcript_filename = transcript_path.name

    for utterance_num, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip() if isinstance(seg, dict) else ""
        if not text:
            continue
        start_s = float(seg.get("start", 0.0))
        end_s = float(seg.get("end", start_s))
        if end_s < start_s:
            continue

        # Both frame_times and segments are time-sorted; bisect to slice the
        # matching frame window in O(log N) instead of scanning all frames.
        lo = bisect.bisect_left(frame_times, start_s)
        hi = bisect.bisect_right(frame_times, end_s)
        matched = frames[lo:hi]
        if len(matched) < min_frames_per_utterance:
            continue

        records.append(
            {
                "utterance": text,
                "frame_filenames": [fname for fname, _ in matched],
                "utterance_num": utterance_num,
                "video_filename": video_filename,
                "transcript_filename": transcript_filename,
            }
        )

    return records


def split_records(
    records: list[dict[str, Any]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle ``records`` deterministically and slice into train/val/test."""
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac > 1:
        msg = f"Invalid splits: train_frac={train_frac}, val_frac={val_frac}"
        raise ValueError(msg)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n = len(records)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }


def _summarize(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"split": name, "utterances": 0, "frames": 0, "videos": 0}
    videos = Counter(r["video_filename"] for r in records)
    frames = sum(len(r["frame_filenames"]) for r in records)
    return {
        "split": name,
        "utterances": len(records),
        "frames": frames,
        "videos": len(videos),
        "frames_per_utterance_mean": round(frames / len(records), 2),
    }


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(records, f)
    logger.info("Wrote %d records to %s", len(records), path)


@hydra.main(version_base=None, config_name="build_manifest_pipeline")
def main(config: BuildManifestPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    cfg = config.processor

    transcripts_dir = Path(cfg.transcripts_dir)
    frames_dir = Path(cfg.frames_dir)
    output_dir = Path(cfg.output_dir)

    transcript_paths = sorted(transcripts_dir.rglob(f"*.{cfg.transcript_suffix}"))
    logger.info("Found %d transcripts under %s", len(transcript_paths), transcripts_dir)
    if not transcript_paths:
        logger.error("No transcripts found; nothing to do.")
        return

    all_records: list[dict[str, Any]] = []
    videos_with_records = 0
    videos_without_frames = 0
    videos_without_segments = 0
    for tp in transcript_paths:
        video_name = tp.stem
        video_frames_dir = frames_dir / video_name
        records = build_records_for_video(
            tp,
            video_frames_dir,
            frames_fps=cfg.frames_fps,
            min_frames_per_utterance=cfg.min_frames_per_utterance,
            frame_suffix=cfg.frame_suffix,
        )
        if not records:
            if not video_frames_dir.is_dir():
                videos_without_frames += 1
            else:
                videos_without_segments += 1
            continue
        videos_with_records += 1
        all_records.extend(records)

    logger.info(
        "Built %d records from %d/%d videos (%d missing frames, %d no usable segments)",
        len(all_records),
        videos_with_records,
        len(transcript_paths),
        videos_without_frames,
        videos_without_segments,
    )

    if not all_records:
        logger.error("No records built; check that transcripts and frames are aligned.")
        return

    splits = split_records(
        all_records,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        _write_manifest(output_dir / f"{name}.json", records)

    splits_summary = [_summarize(name, recs) for name, recs in splits.items()]
    summary = {
        "total_videos_seen": len(transcript_paths),
        "videos_with_records": videos_with_records,
        "videos_missing_frames": videos_without_frames,
        "videos_no_segments": videos_without_segments,
        "frames_fps": cfg.frames_fps,
        "splits": splits_summary,
    }
    summary_path = output_dir / "manifest_build_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)
    for s in splits_summary:
        logger.info("  %s", s)


if __name__ == "__main__":
    main()
