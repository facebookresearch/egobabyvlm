# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for manifest manipulation scripts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from apps.alignment_scoring.scripts.create_shuffled_manifest import (
    shuffle_csv_manifest,
    shuffle_json_manifest,
    shuffle_karpathy_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_shuffle_csv_preserves_row_count_and_clip_filenames(tmp_path: Path) -> None:
    src = tmp_path / "manifest.csv"
    out = tmp_path / "shuffled.csv"
    pd.DataFrame(
        {
            "clip_filename": ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
            "utterance": ["one", "two", "three", "four"],
        }
    ).to_csv(src, index=False)

    shuffle_csv_manifest(str(src), str(out))
    shuffled = pd.read_csv(out)
    assert list(shuffled["clip_filename"]) == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    assert sorted(shuffled["utterance"]) == ["four", "one", "three", "two"]


def test_shuffle_json_preserves_image_count_and_caption_set(tmp_path: Path) -> None:
    src = tmp_path / "manifest.json"
    out = tmp_path / "shuffled.json"
    src.write_text(
        json.dumps(
            {
                "images": [{"id": i, "file_name": f"img_{i}.jpg"} for i in range(4)],
                "annotations": [
                    {"image_id": 0, "caption": "first"},
                    {"image_id": 1, "caption": "second"},
                    {"image_id": 2, "caption": "third"},
                    {"image_id": 3, "caption": "fourth"},
                ],
            }
        )
    )

    shuffle_json_manifest(str(src), str(out))
    shuffled = json.loads(out.read_text())
    assert len(shuffled["images"]) == 4
    assert len(shuffled["annotations"]) == 4
    captions = sorted(ann["caption"] for ann in shuffled["annotations"])
    assert captions == ["first", "fourth", "second", "third"]


def test_shuffle_karpathy_json_swaps_sentence_lists(tmp_path: Path) -> None:
    src = tmp_path / "karpathy.json"
    out = tmp_path / "shuffled.json"
    src.write_text(
        json.dumps(
            {
                "images": [
                    {"imgid": 0, "filename": "a.jpg", "sentences": [{"raw": "alpha"}]},
                    {"imgid": 1, "filename": "b.jpg", "sentences": [{"raw": "beta"}]},
                    {"imgid": 2, "filename": "c.jpg", "sentences": [{"raw": "gamma"}]},
                ],
            }
        )
    )

    shuffle_karpathy_json(str(src), str(out))
    shuffled = json.loads(out.read_text())
    # imgid order is preserved; sentence lists are shuffled across images.
    assert [img["imgid"] for img in shuffled["images"]] == [0, 1, 2]
    sentences = sorted(img["sentences"][0]["raw"] for img in shuffled["images"])
    assert sentences == ["alpha", "beta", "gamma"]
