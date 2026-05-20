# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        # Placeholder values - update these with your actual split sizes
        split_lengths = {
            _Split.TRAIN: 743_198,  # ~90% of the dataset for training
            _Split.VAL: 82_577,     # ~10% for validation
        }
        return split_lengths[self]


class Ego4D(ExtendedVisionDataset):
    Target = Union[int]  # No labels, so we'll use frame indices as targets
    Split = Union[_Split]

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        split: "Ego4D.Split" = _Split.TRAIN,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._entries = None

    @property
    def split(self) -> "Ego4D.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(os.path.dirname(extra_full_path), exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            try:
                self._entries = self._load_extra(self._entries_path)
            except (FileNotFoundError, IOError):
                logger.info(f"Entries file not found. Creating it now.")
                self.dump_extra()
                self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = entries[index]["image_path"]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> int:
        # Using frame index as target since Ego4D frames don't have labels
        return index

    def get_targets(self) -> np.ndarray:
        entries = self._get_entries()
        return np.arange(len(entries))

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _find_all_images(self) -> List[str]:
        """Find all image files in the root directory."""
        image_paths = []
        for dirpath, _, filenames in os.walk(self.root, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(dirpath, filename), self.root)
                    image_paths.append(rel_path)
        return sorted(image_paths)  # Sort to ensure deterministic ordering

    def dump_extra(self) -> None:
        """Create and save the entries file."""
        logger.info(f"Creating entries for Ego4D {self.split.value} split")
        
        # Find all image files
        all_images = self._find_all_images()
        total_images = len(all_images)
        logger.info(f"Found {total_images} images in total")
        
        # Determine split sizes
        if total_images == 0:
            raise RuntimeError(f"No images found in {self.root}")
        
        # Create split based on total count
        if self.split == _Split.TRAIN:
            # Use first 90% for training
            split_size = int(total_images * 0.9)
            image_paths = all_images[:split_size]
        else:  # VAL split
            # Use last 10% for validation
            split_size = int(total_images * 0.9)
            image_paths = all_images[split_size:]
        
        logger.info(f"Using {len(image_paths)} images for {self.split.value} split")
        
        # Create entries array
        dtype = np.dtype([
            ("image_path", f"U{max([len(p) for p in image_paths])}")
        ])
        entries_array = np.empty(len(image_paths), dtype=dtype)
        
        for i, image_path in enumerate(image_paths):
            entries_array[i] = (image_path,)
            if i % 10000 == 0:
                logger.info(f"Processing entry {i}/{len(image_paths)}")
        
        logger.info(f'Saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)