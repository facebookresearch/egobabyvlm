# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MSCOCO captions datasets."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .base import (
    CaptionsMediaDataset,
    CaptionsMediaSample,
    CaptionsPathDataset,
    CaptionsPathSample,
    MultiCaptionPathSample,
)

if TYPE_CHECKING:
    from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def load_coco_samples(dataset_dir: str, manifest_path: str) -> list[MultiCaptionPathSample]:
    """Read a COCO-format JSON manifest and return one sample per image."""
    with Path(manifest_path).open() as f:
        data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    image_captions: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        image_captions[ann["image_id"]].append(ann["caption"])

    samples: list[MultiCaptionPathSample] = []
    dataset_dir_str = str(dataset_dir).rstrip("/")
    for image_id, captions in image_captions.items():
        file_name = image_id_to_file[image_id]
        image_path = f"{dataset_dir_str}/{file_name}"
        samples.append(MultiCaptionPathSample(image_path, captions, image_id))

    samples.sort(key=lambda x: x.media_id)
    return samples


def load_preprocessed_karpathy_coco_samples(
    dataset_dir: str,
    manifest_path: str,
    permutation_metadata_path: str | None = None,
) -> list[MultiCaptionPathSample]:
    """Read a Karpathy-format JSON manifest with optional caption-permutation overrides.

    The ``permutation_metadata_path``, when given, swaps in captions from a
    different image index to support partial-shuffle ablations.
    """
    with Path(manifest_path).open() as f:
        data = json.load(f)
    all_images = data["images"]

    permutation_map: dict[int, int] | None = None
    if permutation_metadata_path is not None:
        with Path(permutation_metadata_path).open() as f:
            metadata = json.load(f)
            permutation_map = {int(k): v for k, v in metadata["permutation_map"].items()}

    samples: list[MultiCaptionPathSample] = []
    counter: Counter[str] = Counter()
    dataset_dir_str = str(dataset_dir).rstrip("/")
    for idx, image in enumerate(data["images"]):
        captions = image["sentences"]

        counter["total"] += 1
        if permutation_map is not None and idx in permutation_map:
            captions = all_images[permutation_map[idx]]["sentences"]
            counter["permuted"] += 1

        captions_text = [c["raw"] for c in captions]
        image_path = f"{dataset_dir_str}/{image['filename']}"
        samples.append(MultiCaptionPathSample(image_path, captions_text, image["imgid"]))

    samples.sort(key=lambda x: x.media_id)

    pct = 100.0 * counter["permuted"] / counter["total"] if counter["total"] else 0.0
    logger.info(
        "Loaded Karpathy COCO samples: %d total, %d permuted (%.1f%%)",
        counter["total"],
        counter["permuted"],
        pct,
    )
    return samples


class CocoCaptionsDataset(CaptionsMediaDataset):
    """COCO captions: yields ``(image_tensor, caption, image_id)``."""

    def __init__(
        self,
        dataset_dir: str,
        manifest_path: str,
        preprocessor: Compose | None = None,
        random_seed: int = 42,
        *,
        use_first: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.preprocessor = preprocessor
        self.samples = load_coco_samples(dataset_dir, manifest_path)
        self.rng = random.Random(random_seed)
        self.use_first = use_first

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CaptionsMediaSample:
        image_path, captions, image_id = self.samples[idx]
        caption = captions[0] if self.use_first else self.rng.choice(captions)
        with Path(image_path).open("rb") as f:
            image = Image.open(f).convert("RGB")
        if self.preprocessor is not None:
            image = self.preprocessor(image)
        return CaptionsMediaSample(image, caption, image_id)


class CocoCaptionsPathDataset(CaptionsPathDataset):
    """COCO captions: yields ``(image_path, caption, image_id)`` without loading pixels."""

    def __init__(
        self,
        dataset_dir: str,
        manifest_path: str,
        random_seed: int = 42,
        *,
        use_first: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.samples = load_coco_samples(dataset_dir, manifest_path)
        self.rng = random.Random(random_seed)
        self.use_first = use_first

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CaptionsPathSample:
        image_path, captions, image_id = self.samples[idx]
        caption = captions[0] if self.use_first else self.rng.choice(captions)
        return CaptionsPathSample(image_path, caption, image_id)


class KarpathyCocoCaptionsDataset(CocoCaptionsDataset):
    """Karpathy-split variant; supports caption permutation for ablations."""

    def __init__(
        self,
        dataset_dir: str,
        manifest_path: str,
        permutation_metadata_path: str | None = None,
        preprocessor: Compose | None = None,
        random_seed: int = 42,
        *,
        use_first: bool = True,
    ) -> None:
        # Skip CocoCaptionsDataset.__init__ — we load samples differently.
        CaptionsMediaDataset.__init__(self)
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.permutation_metadata_path = permutation_metadata_path
        self.preprocessor = preprocessor
        self.samples = load_preprocessed_karpathy_coco_samples(
            dataset_dir,
            manifest_path,
            permutation_metadata_path,
        )
        self.rng = random.Random(random_seed)
        self.use_first = use_first


class KarpathyCocoCaptionsPathDataset(CocoCaptionsPathDataset):
    """Path-only Karpathy variant; mirrors :class:`KarpathyCocoCaptionsDataset`."""

    def __init__(
        self,
        dataset_dir: str,
        manifest_path: str,
        permutation_metadata_path: str | None = None,
        random_seed: int = 42,
        *,
        use_first: bool = True,
    ) -> None:
        CaptionsPathDataset.__init__(self)
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path
        self.permutation_metadata_path = permutation_metadata_path
        self.samples = load_preprocessed_karpathy_coco_samples(
            dataset_dir,
            manifest_path,
            permutation_metadata_path,
        )
        self.rng = random.Random(random_seed)
        self.use_first = use_first
