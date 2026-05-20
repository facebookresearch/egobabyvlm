# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""COCO-MC image dataset (local-FS only).

COCO-MC is a separate dataset built around COCO: caption-conditioned
retrievals from MetaCLIP (one or more nearest images per COCO caption)
plus, for DINO SSL, additional images sampled uniformly at random from
MetaCLIP. The on-disk layout is a flat or nested directory of JPEGs;
this dataset class enumerates them recursively at first access and
caches the resulting entries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_TRAIN_SPLIT_FRACTION = 0.9
_LOG_EVERY_N_ENTRIES = 10_000


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"


class CocoMc(ExtendedVisionDataset):
    """COCO-MC image dataset reading images from a local directory tree.

    Layout::

        root/
            <any nesting>/
                <image>.{jpg,jpeg,png}
            ...

    On first access, the dataset enumerates ``root/**/*.{jpg,jpeg,png}``,
    splits 90/10 into train/val, and writes an ``entries-{TRAIN,VAL}.npy``
    cache under ``extra``. Re-uses the cache on subsequent runs.
    """

    Target = int  # No labels; we use frame indices as targets.
    Split = _Split

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        split: _Split = _Split.TRAIN,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = Path(extra)
        self._split = split
        self._entries: np.ndarray | None = None

    @property
    def split(self) -> _Split:
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> Path:
        return self._extra_root / extra_path

    def _load_extra(self, extra_path: str) -> np.ndarray:
        return np.load(self._get_extra_full_path(extra_path), mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        full_path = self._get_extra_full_path(extra_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            try:
                self._entries = self._load_extra(self._entries_path)
            except (FileNotFoundError, OSError):
                logger.info("Entries file not found. Creating it now.")
                self.dump_extra()
                self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = entries[index]["image_path"]
        return (Path(self.root) / image_relpath).read_bytes()

    def get_target(self, index: int) -> int:  # noqa: ARG002 -- COCO-MC images have no labels.
        return index

    def get_targets(self) -> np.ndarray:
        return np.arange(len(self._get_entries()))

    def __len__(self) -> int:
        return len(self._get_entries())

    def _find_all_images(self) -> list[str]:
        """Find all image files under ``self.root``."""
        root = Path(self.root)
        image_paths: list[str] = []
        for ext in _IMAGE_EXTENSIONS:
            for path in root.glob(f"**/*{ext}"):
                image_paths.append(str(path.relative_to(root)))
        return sorted(image_paths)

    def dump_extra(self) -> None:
        """Enumerate images under ``root`` and persist the train/val entries cache."""
        logger.info("Creating entries for COCO-MC %s split", self.split.value)

        all_images = self._find_all_images()
        total_images = len(all_images)
        logger.info("Found %d images in total", total_images)

        if total_images == 0:
            msg = f"No images found in {self.root}"
            raise RuntimeError(msg)

        split_size = int(total_images * _TRAIN_SPLIT_FRACTION)
        if self.split == _Split.TRAIN:
            image_paths = all_images[:split_size]
        else:
            image_paths = all_images[split_size:]
        logger.info("Using %d images for %s split", len(image_paths), self.split.value)

        max_path_len = max(len(p) for p in image_paths)
        dtype = np.dtype([("image_path", f"U{max_path_len}")])
        entries_array = np.empty(len(image_paths), dtype=dtype)
        for i, image_path in enumerate(image_paths):
            entries_array[i] = (image_path,)
            if i % _LOG_EVERY_N_ENTRIES == 0:
                logger.info("Processing entry %d/%d", i, len(image_paths))

        logger.info('Saving entries to "%s"', self._entries_path)
        self._save_extra(entries_array, self._entries_path)
