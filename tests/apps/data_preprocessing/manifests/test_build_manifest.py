# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the manifest builder pure helpers + round-trip with the trainer dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from apps.baselines.clip.data import HowToCaptionsDataset
from apps.data_preprocessing.manifests.build_manifest import (
    _frame_index_to_time_s,
    _list_frames_for_video,
    build_records_for_video,
    split_records,
)

if TYPE_CHECKING:
    from pathlib import Path


def _whisperx_transcript(segments: list[dict]) -> dict:
    """Wrap segment dicts in a WhisperX-shaped envelope."""
    return {"language": "en", "segments": segments}


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text, "words": []}


def _make_video_dir(frames_dir: Path, video_name: str, num_frames: int) -> Path:
    """Create ``frames_dir/<video_name>/<video_name>_<i>.jpg`` for i=1..num_frames."""
    vdir = frames_dir / video_name
    vdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_frames + 1):
        Image.new("RGB", (8, 8), color=(i, 0, 0)).save(vdir / f"{video_name}_{i}.jpg")
    return vdir


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("idx", "fps", "expected"),
    [
        (1, 1.0, 0.5),
        (2, 1.0, 1.5),
        (1, 2.0, 0.25),
        (4, 2.0, 1.75),
    ],
)
def test_frame_index_to_time_s(idx: int, fps: float, expected: float) -> None:
    assert _frame_index_to_time_s(idx, fps) == pytest.approx(expected)


def test_list_frames_for_video_returns_sorted_pairs(tmp_path: Path) -> None:
    vdir = _make_video_dir(tmp_path, "vid_a", 3)
    pairs = _list_frames_for_video(vdir, "jpg", frames_fps=1.0)
    assert pairs == [
        ("vid_a/vid_a_1.jpg", 0.5),
        ("vid_a/vid_a_2.jpg", 1.5),
        ("vid_a/vid_a_3.jpg", 2.5),
    ]


def test_list_frames_for_video_skips_unrelated_files(tmp_path: Path) -> None:
    vdir = _make_video_dir(tmp_path, "vid_a", 2)
    (vdir / "summary.txt").write_text("hi")
    (vdir / "vid_a_x.jpg").write_text("not an int")
    (vdir / "other_5.jpg").write_text("wrong prefix")
    pairs = _list_frames_for_video(vdir, "jpg", frames_fps=1.0)
    assert [fname for fname, _ in pairs] == ["vid_a/vid_a_1.jpg", "vid_a/vid_a_2.jpg"]


def test_list_frames_for_video_missing_dir(tmp_path: Path) -> None:
    assert _list_frames_for_video(tmp_path / "nope", "jpg", frames_fps=1.0) == []


# ---------------------------------------------------------------------------
# build_records_for_video
# ---------------------------------------------------------------------------


def test_build_records_pairs_segments_with_overlapping_frames(tmp_path: Path) -> None:
    transcript = tmp_path / "vid_a.json"
    transcript.write_text(
        json.dumps(_whisperx_transcript([_seg(0.0, 2.0, "hello"), _seg(2.0, 4.0, "world")])),
    )
    frames_root = tmp_path / "frames"
    _make_video_dir(frames_root, "vid_a", 4)  # midpoints: 0.5, 1.5, 2.5, 3.5

    records = build_records_for_video(
        transcript,
        frames_root / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )

    assert len(records) == 2
    assert records[0]["utterance"] == "hello"
    assert records[0]["frame_filenames"] == ["vid_a/vid_a_1.jpg", "vid_a/vid_a_2.jpg"]
    assert records[0]["video_filename"] == "vid_a.mp4"
    assert records[0]["transcript_filename"] == "vid_a.json"
    assert records[0]["utterance_num"] == 1
    assert len(records[0]["frame_filenames"]) == 2

    assert records[1]["utterance"] == "world"
    assert records[1]["frame_filenames"] == ["vid_a/vid_a_3.jpg", "vid_a/vid_a_4.jpg"]
    assert records[1]["utterance_num"] == 2


def test_build_records_skips_segments_below_min_frames(tmp_path: Path) -> None:
    transcript = tmp_path / "vid_a.json"
    transcript.write_text(
        json.dumps(
            _whisperx_transcript(
                [
                    _seg(0.0, 0.4, "tiny"),  # midpoint 0.5 falls outside [0, 0.4] -> 0 frames
                    _seg(0.0, 2.0, "kept"),  # 2 frames
                ],
            ),
        ),
    )
    frames_root = tmp_path / "frames"
    _make_video_dir(frames_root, "vid_a", 2)

    records = build_records_for_video(
        transcript,
        frames_root / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )
    assert len(records) == 1
    assert records[0]["utterance"] == "kept"


def test_build_records_skips_empty_text_and_inverted_intervals(tmp_path: Path) -> None:
    transcript = tmp_path / "vid_a.json"
    transcript.write_text(
        json.dumps(
            _whisperx_transcript(
                [
                    _seg(0.0, 2.0, "   "),  # blank text
                    _seg(2.0, 1.0, "inverted"),  # end < start
                    _seg(0.0, 2.0, "ok"),
                ],
            ),
        ),
    )
    frames_root = tmp_path / "frames"
    _make_video_dir(frames_root, "vid_a", 2)

    records = build_records_for_video(
        transcript,
        frames_root / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )
    assert [r["utterance"] for r in records] == ["ok"]
    # utterance_num should come from the *original* segment position so downstream
    # tooling can map back to the source transcript.
    assert records[0]["utterance_num"] == 3


def test_build_records_returns_empty_when_no_frames(tmp_path: Path) -> None:
    transcript = tmp_path / "vid_a.json"
    transcript.write_text(json.dumps(_whisperx_transcript([_seg(0.0, 2.0, "hi")])))
    records = build_records_for_video(
        transcript,
        tmp_path / "frames" / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )
    assert records == []


def test_build_records_returns_empty_for_unparsable_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "vid_a.json"
    transcript.write_text("not json")
    frames_root = tmp_path / "frames"
    _make_video_dir(frames_root, "vid_a", 2)
    records = build_records_for_video(
        transcript,
        frames_root / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )
    assert records == []


# ---------------------------------------------------------------------------
# split_records
# ---------------------------------------------------------------------------


def test_split_records_partitions_disjointly() -> None:
    records = [{"utterance": f"u{i}"} for i in range(100)]
    splits = split_records(records, train_frac=0.8, val_frac=0.1, seed=42)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    seen = [r["utterance"] for r in splits["train"] + splits["val"] + splits["test"]]
    assert sorted(seen) == sorted(r["utterance"] for r in records)


def test_split_records_seed_is_deterministic() -> None:
    records = [{"utterance": f"u{i}"} for i in range(50)]
    a = split_records(records, train_frac=0.7, val_frac=0.2, seed=7)
    b = split_records(records, train_frac=0.7, val_frac=0.2, seed=7)
    assert a == b


def test_split_records_rejects_invalid_fractions() -> None:
    records = [{}]
    with pytest.raises(ValueError, match="Invalid splits"):
        split_records(records, train_frac=0.0, val_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="Invalid splits"):
        split_records(records, train_frac=0.6, val_frac=0.5, seed=0)


# ---------------------------------------------------------------------------
# Round-trip with the trainer dataset
# ---------------------------------------------------------------------------


def test_manifest_builder_output_loads_in_howto_dataset(tmp_path: Path) -> None:
    """End-to-end: builder output → HowToCaptionsDataset → first item is loadable."""
    transcript = tmp_path / "vid_a.json"
    transcript.write_text(
        json.dumps(_whisperx_transcript([_seg(0.0, 2.0, "the cat sat on the mat")])),
    )
    frames_root = tmp_path / "frames"
    _make_video_dir(frames_root, "vid_a", 2)

    records = build_records_for_video(
        transcript,
        frames_root / "vid_a",
        frames_fps=1.0,
        min_frames_per_utterance=1,
        frame_suffix="jpg",
    )

    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(json.dumps(records))

    ds = HowToCaptionsDataset(
        manifest_path=manifest_path,
        image_root=frames_root,
        transform=None,
        multiple_frames=False,
    )
    assert len(ds) == 1
    image, caption, sample_id = ds[0]
    assert isinstance(image, Image.Image)
    assert image.size == (8, 8)
    assert caption == "the cat sat on the mat"
    assert sample_id == "0"
