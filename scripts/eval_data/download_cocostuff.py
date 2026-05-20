#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download COCO-Stuff (images + stuff/thing JSON annotations) into the cache.

Layout written matches what :class:`evaluation.data.semantic_segmentation.COCOStuffDataset`
expects:

.. code-block::

    {cache_dir}/
        train2017/                          # RGB images
        val2017/
        annotations/
            stuff_train2017.json            # 91 stuff categories (RLE polygons)
            stuff_val2017.json
            stuffthings_train2017.json      # 171 cats: stuff + COCO thing instances merged
            stuffthings_val2017.json

Total download size is ~20 GB.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

DOWNLOADS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "stuff_trainval2017.zip": "https://calvin-vision.net/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with progress logging."""
    logger.info("Downloading %s -> %s", url, dest)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Unzipping %s -> %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _build_stuffthings(stuff_json: Path, instances_json: Path, out_json: Path) -> None:
    """Merge stuff (91 cats, IDs 92-183) + COCO instances (80 thing cats, IDs 1-90) into one JSON.

    Mirrors the runtime merge in ``COCOStuffDataset.__init__`` so the 171-class
    config can load a single pre-built file.
    """
    logger.info("Building %s from %s + %s", out_json.name, stuff_json.name, instances_json.name)
    with stuff_json.open() as f:
        stuff = json.load(f)
    with instances_json.open() as f:
        things = json.load(f)

    stuff_cat_ids = {c["id"] for c in stuff["categories"]}
    thing_cat_ids = {c["id"] for c in things["categories"]}
    overlap = stuff_cat_ids & thing_cat_ids
    if overlap:
        raise RuntimeError(f"Category ID overlap between stuff/things: {sorted(overlap)}")

    max_stuff_ann_id = max((a["id"] for a in stuff["annotations"]), default=0)
    for ann in things["annotations"]:
        ann["id"] = max_stuff_ann_id + ann["id"] + 1

    merged = dict(stuff)
    merged["categories"] = stuff["categories"] + things["categories"]
    merged["annotations"] = stuff["annotations"] + things["annotations"]

    with out_json.open("w") as f:
        json.dump(merged, f)
    logger.info(
        "  wrote %d cats (%d stuff + %d things), %d annotations",
        len(merged["categories"]),
        len(stuff["categories"]),
        len(things["categories"]),
        len(merged["annotations"]),
    )


def main() -> None:
    parser: argparse.ArgumentParser = cache_argparser("cocostuff")
    args = parser.parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("COCO-Stuff already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "COCO-Stuff", env_var="COCOSTUFF_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = args.cache_dir / "_downloads"
    downloads_dir.mkdir(exist_ok=True)

    logger.warning("COCO-Stuff is ~20 GB; this will take a while.")

    for fname, url in DOWNLOADS.items():
        zip_path = downloads_dir / fname
        if not zip_path.exists() or args.force:
            try:
                _download(url, zip_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Download failed for %s: %s", url, e)
                sys.exit(1)

    # Images: extracted into top-level cache dir.
    _unzip(downloads_dir / "train2017.zip", args.cache_dir)
    _unzip(downloads_dir / "val2017.zip", args.cache_dir)

    # Annotations: stuff_{train,val}2017.json land directly under cache_dir/annotations.
    annotations_dir = args.cache_dir / "annotations"
    _unzip(downloads_dir / "stuff_trainval2017.zip", annotations_dir)
    # COCO official zip extracts to annotations/{instances,captions,person_keypoints}_{train,val}2017.json.
    _unzip(downloads_dir / "annotations_trainval2017.zip", args.cache_dir)

    # Pre-merge stuff + things for the 171-class config.
    for split in ("train", "val"):
        _build_stuffthings(
            stuff_json=annotations_dir / f"stuff_{split}2017.json",
            instances_json=annotations_dir / f"instances_{split}2017.json",
            out_json=annotations_dir / f"stuffthings_{split}2017.json",
        )

    logger.info("Cleaning up zip downloads in %s", downloads_dir)
    shutil.rmtree(downloads_dir)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "COCO-Stuff", env_var="COCOSTUFF_ROOT")


if __name__ == "__main__":
    main()
