# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``COCOStuffDataset`` after the 171-class simplification.

The dataset class no longer accepts ``coarse_labels``, ``instances_json``, or
``subset_fraction``; these tests pin the simplified behaviour and verify the
removed kwargs are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation.data.semantic_segmentation import COCOStuffDataset


def _make_fixture(root: Path, mode: str = "val", num_categories: int = 5, num_images: int = 3) -> None:
    """Write a minimal COCO-Stuff-shaped JSON + matching RGB images.

    Mirrors what ``download_cocostuff.py`` produces for the 171-class layout:
    one ``stuff_<mode>2017.json`` with images / annotations / categories, plus
    a ``<mode>2017/`` directory of RGB images. RLE polygons are intentionally
    simple (full-image rectangles) so the test stays fast.
    """
    img_dir = root / f"{mode}2017"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    for i in range(num_images):
        img_id = i + 1
        fname = f"img_{i:04d}.jpg"
        Image.new("RGB", (16, 16), color=(i * 30, 100, 150)).save(img_dir / fname)
        images.append({"id": img_id, "file_name": fname, "height": 16, "width": 16})
        # One annotation per image, painting the full image with category (i % num_categories) + 92.
        annotations.append(
            {
                "id": i + 1,
                "image_id": img_id,
                "category_id": (i % num_categories) + 92,
                "segmentation": [[0, 0, 16, 0, 16, 16, 0, 16]],
                "iscrowd": 0,
            }
        )

    categories = [{"id": 92 + i, "name": f"cat{i}", "supercategory": "stuff"} for i in range(num_categories)]
    # Add the excluded "other" supercategory entry to mirror the real JSON.
    categories.append({"id": 183, "name": "other", "supercategory": "other"})

    data = {"images": images, "annotations": annotations, "categories": categories}
    json_name = "stuff_train2017.json" if mode == "train" else f"stuff_{mode}2017.json"
    (root / json_name).write_text(json.dumps(data))


def test_loads_minimal_fixture(tmp_path: Path) -> None:
    """Default kwargs against a synthesized fixture builds and serves samples."""
    _make_fixture(tmp_path, mode="val", num_categories=5, num_images=3)

    ds = COCOStuffDataset(dataset_root=str(tmp_path), mode="val", image_size=32)

    assert len(ds) == 3
    assert ds.num_classes == 5  # 5 stuff categories; "other" is excluded.
    assert ds.class_names == [f"cat{i}" for i in range(5)]

    sample = ds[0]
    assert sample.media.shape == (3, 32, 32)
    assert sample.mask.shape == (32, 32)
    assert sample.mask.dtype == np.dtype("int64") or str(sample.mask.dtype) == "torch.int64"


def test_class_ids_match_continuous_range(tmp_path: Path) -> None:
    """``class_ids`` is the contiguous ``range(num_classes)`` after remap."""
    _make_fixture(tmp_path, num_categories=4)

    ds = COCOStuffDataset(dataset_root=str(tmp_path), mode="val")

    assert ds.class_ids == [0, 1, 2, 3]


@pytest.mark.parametrize("kwarg", ["coarse_labels", "instances_json", "subset_fraction"])
def test_removed_kwargs_rejected(tmp_path: Path, kwarg: str) -> None:
    """Kwargs that were dropped in the simplification raise ``TypeError``."""
    _make_fixture(tmp_path)

    kwargs: dict = {kwarg: True}
    with pytest.raises(TypeError, match=kwarg):
        COCOStuffDataset(dataset_root=str(tmp_path), mode="val", **kwargs)


def test_missing_annotation_file_raises(tmp_path: Path) -> None:
    """A clear ``FileNotFoundError`` if the JSON is absent."""
    (tmp_path / "val2017").mkdir()
    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        COCOStuffDataset(dataset_root=str(tmp_path), mode="val")
