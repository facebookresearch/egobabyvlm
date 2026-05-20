# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the data layer (datasets + collators + transforms)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

from apps.baselines.clip.data import (
    CocoCaptionsDataset,
    Ego4DCaptionsDataset,
    HowToCaptionsDataset,
    build_train_transform,
    contrastive_collate,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_image_root(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (100, 100), (i * 50, 0, 0)).save(img_dir / f"img{i}.jpg")
    return img_dir


def test_coco_dataset_loads_records(fake_image_root: Path) -> None:
    manifest = fake_image_root.parent / "coco.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {"imgid": i, "filename": f"img{i}.jpg", "sentences": [{"raw": f"caption {i}"}]} for i in range(4)
                ],
            },
        ),
    )
    ds = CocoCaptionsDataset(manifest, fake_image_root, transform=build_train_transform())
    assert len(ds) == 4
    img, caption, sample_id = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert caption == "caption 0"
    assert sample_id == "0"


def test_coco_multiple_captions_samples_one(fake_image_root: Path) -> None:
    manifest = fake_image_root.parent / "coco.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "imgid": 0,
                        "filename": "img0.jpg",
                        "sentences": [{"raw": "first"}, {"raw": "second"}, {"raw": "third"}],
                    },
                ],
            },
        ),
    )
    ds = CocoCaptionsDataset(manifest, fake_image_root, multiple_captions=True)
    captions_seen = {ds[0][1] for _ in range(20)}
    assert captions_seen.issubset({"first", "second", "third"})
    assert len(captions_seen) > 1


def test_howto_dataset_samples_one_frame(fake_image_root: Path) -> None:
    manifest = fake_image_root.parent / "howto.json"
    manifest.write_text(
        json.dumps([{"utterance": "narration", "frame_filenames": ["img0.jpg", "img1.jpg", "img2.jpg"]}]),
    )
    ds = HowToCaptionsDataset(manifest, fake_image_root, transform=build_train_transform())
    img, caption, _ = ds[0]
    assert img.shape == (3, 224, 224)
    assert caption == "narration"


def test_howto_dataset_accepts_data_envelope(fake_image_root: Path) -> None:
    """Pre-built manifests may wrap the records in ``{"data": [...]}``; both load identically."""
    flat = [{"utterance": "u", "frame_filenames": ["img0.jpg"]}]
    flat_path = fake_image_root.parent / "howto_flat.json"
    flat_path.write_text(json.dumps(flat))
    envelope_path = fake_image_root.parent / "howto_envelope.json"
    envelope_path.write_text(json.dumps({"data": flat}))

    ds_flat = HowToCaptionsDataset(flat_path, fake_image_root)
    ds_env = HowToCaptionsDataset(envelope_path, fake_image_root)
    assert len(ds_flat) == len(ds_env) == 1
    assert ds_flat.records == ds_env.records


def test_ego4d_is_compatible_with_howto(fake_image_root: Path) -> None:
    manifest = fake_image_root.parent / "ego4d.json"
    manifest.write_text(json.dumps([{"utterance": "u", "frame_filenames": ["img0.jpg"]}]))
    ds = Ego4DCaptionsDataset(manifest, fake_image_root, transform=build_train_transform())
    assert len(ds) == 1
    assert ds[0][1] == "u"


def test_contrastive_collate_stacks_images_and_keeps_strings() -> None:
    batch = [(torch.zeros(3, 224, 224), f"caption {i}", str(i)) for i in range(4)]
    out = contrastive_collate(batch)
    assert out["images"].shape == (4, 3, 224, 224)
    assert out["captions"] == ["caption 0", "caption 1", "caption 2", "caption 3"]
    assert out["ids"] == ["0", "1", "2", "3"]
