# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Round-trip tests for the alignment-scoring caption datasets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from PIL import Image
from torchvision.transforms import Compose, ToTensor

from apps.alignment_scoring.data import (
    CocoCaptionsDataset,
    CocoCaptionsPathDataset,
    KarpathyCocoCaptionsPathDataset,
    TextPairDataset,
    VideoCaptionsPathDataset,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def coco_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny COCO-format dataset with three images and four captions on disk."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_id, color in zip([1, 2, 3], ["red", "green", "blue"], strict=True):
        Image.new("RGB", (32, 32), color).save(image_dir / f"img_{image_id}.jpg")

    manifest = {
        "images": [
            {"id": 1, "file_name": "img_1.jpg"},
            {"id": 2, "file_name": "img_2.jpg"},
            {"id": 3, "file_name": "img_3.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "caption": "a red square"},
            {"image_id": 2, "caption": "a green square"},
            {"image_id": 2, "caption": "a verdant square"},
            {"image_id": 3, "caption": "a blue square"},
        ],
    }
    manifest_path = tmp_path / "captions.json"
    manifest_path.write_text(json.dumps(manifest))
    return image_dir, manifest_path


@pytest.fixture
def karpathy_coco_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny Karpathy-format COCO manifest."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_id in [10, 20]:
        Image.new("RGB", (32, 32), "white").save(image_dir / f"k_{image_id}.jpg")

    manifest = {
        "images": [
            {
                "imgid": 10,
                "filename": "k_10.jpg",
                "sentences": [{"raw": "first caption"}, {"raw": "alternate"}],
            },
            {
                "imgid": 20,
                "filename": "k_20.jpg",
                "sentences": [{"raw": "second caption"}],
            },
        ],
    }
    manifest_path = tmp_path / "karpathy.json"
    manifest_path.write_text(json.dumps(manifest))
    return image_dir, manifest_path


def test_coco_path_dataset_yields_paths_and_captions(coco_fixture: tuple[Path, Path]) -> None:
    image_dir, manifest_path = coco_fixture
    ds = CocoCaptionsPathDataset(str(image_dir), str(manifest_path))
    assert len(ds) == 3

    sample = ds[0]
    assert sample.media_id == 1
    assert sample.media_path.endswith("img_1.jpg")
    assert sample.text == "a red square"


def test_coco_media_dataset_returns_preprocessed_tensor(coco_fixture: tuple[Path, Path]) -> None:
    image_dir, manifest_path = coco_fixture
    ds = CocoCaptionsDataset(
        str(image_dir),
        str(manifest_path),
        preprocessor=Compose([ToTensor()]),
    )
    sample = ds[1]
    assert sample.media.shape == (3, 32, 32)
    assert sample.text in {"a green square", "a verdant square"}
    assert sample.media_id == 2


def test_coco_use_first_picks_first_caption_deterministically(coco_fixture: tuple[Path, Path]) -> None:
    image_dir, manifest_path = coco_fixture
    ds = CocoCaptionsPathDataset(str(image_dir), str(manifest_path), use_first=True)
    # image_id=2 has two captions; use_first=True must pick the first one.
    sample = next(s for s in ds if s.media_id == 2)
    assert sample.text == "a green square"


def test_karpathy_path_dataset(karpathy_coco_fixture: tuple[Path, Path]) -> None:
    image_dir, manifest_path = karpathy_coco_fixture
    ds = KarpathyCocoCaptionsPathDataset(str(image_dir), str(manifest_path))
    assert len(ds) == 2
    assert ds[0].media_id == 10
    assert ds[0].text == "first caption"
    assert ds[1].text == "second caption"


def test_video_path_dataset_reads_csv_manifest(tmp_path: Path) -> None:
    """VideoCaptionsPathDataset reads the CSV manifest and skips NaN utterances."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "clip_filename": ["a.mp4", "b.mp4", "c.mp4"],
            "utterance": ["hello", "world", None],
        }
    ).to_csv(manifest_path, index=False)

    ds = VideoCaptionsPathDataset(str(manifest_path), str(tmp_path / "videos"))
    assert len(ds) == 2  # NaN utterance row dropped.
    assert ds[0].text == "hello"
    assert ds[0].media_id == "a.mp4"
    assert ds[0].media_path.endswith("/videos/a.mp4")


def test_text_pair_dataset_joins_on_media_id(coco_fixture: tuple[Path, Path]) -> None:
    """Two manifests over the same images get joined into (text_a, text_b, media_id) triples."""
    image_dir, manifest_path = coco_fixture
    ds_a = CocoCaptionsPathDataset(str(image_dir), str(manifest_path))
    ds_b = CocoCaptionsPathDataset(str(image_dir), str(manifest_path))

    pair = TextPairDataset(ds_a, ds_b)
    assert len(pair) == 3
    text_a, text_b, media_id = pair[0]
    assert text_a == text_b  # same manifest both sides
    assert media_id == 1
