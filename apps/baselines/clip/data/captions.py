# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image-caption datasets for COCO, HowTo100M, and Ego4D.

Plain ``Dataset`` subclasses that emit ``(image, caption, sample_id)``.
Each dataset only differs in how it parses its manifest; the trainer
stays dataset-agnostic via ``hydra.utils.instantiate``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch
    from torchvision import transforms


class CaptionsDataset(Dataset):
    """Base class for image-caption datasets.

    Subclasses provide ``_load_records()`` returning a list of dicts with at
    least ``image_path`` (str, relative or absolute) and ``caption`` (str)
    keys, plus an optional ``id`` key. ``__getitem__`` returns
    ``(image_tensor, caption, sample_id)``.

    Args:
        manifest_path: Path to the dataset's manifest file.
        image_root: Root directory containing the images (joined with each
            record's ``image_path``). May be ``None`` if records hold absolute paths.
        transform: Image transform pipeline; if ``None``, returns raw PIL.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.records: list[dict[str, Any]] = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, str, str]:
        record = self.records[idx]
        image = Image.open(self._resolve_image_path(record["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record["caption"], str(record.get("id", idx))


class CocoCaptionsDataset(CaptionsDataset):
    """COCO Karpathy-format manifest: ``{"images": [{"filename", "sentences": [{"raw"|"tokens"}]}]}``.

    Each image becomes one record; if the image has multiple captions, one is
    sampled per ``__getitem__`` when ``multiple_captions=True`` (the default).
    When ``multiple_captions=False`` the first caption is used deterministically.

    Args:
        manifest_path: Path to a Karpathy-style JSON file (e.g. ``dataset_coco.json``
            or ``preprocessed_captions_train.json``).
        image_root: Directory holding the COCO images (e.g. ``coco/all_images``).
        transform: Image transform.
        multiple_captions: If ``True``, sample a random caption per epoch.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        transform: transforms.Compose | None = None,
        *,
        multiple_captions: bool = True,
    ) -> None:
        self.multiple_captions = multiple_captions
        super().__init__(manifest_path, image_root, transform)

    def _load_records(self) -> list[dict[str, Any]]:
        with self.manifest_path.open() as f:
            data = json.load(f)
        records: list[dict[str, Any]] = []
        for image in data["images"]:
            captions = [self._caption_text(s) for s in image["sentences"]]
            records.append(
                {
                    "image_path": image["filename"],
                    "captions": captions,
                    "id": image.get("imgid", image.get("cocoid", image["filename"])),
                }
            )
        return records

    @staticmethod
    def _caption_text(sentence: dict[str, Any]) -> str:
        if "raw" in sentence:
            return sentence["raw"]
        return " ".join(sentence["tokens"])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, str, str]:
        record = self.records[idx]
        image = Image.open(self._resolve_image_path(record["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        captions = record["captions"]
        caption = random.choice(captions) if (self.multiple_captions and len(captions) > 1) else captions[0]
        return image, caption, str(record["id"])


class HowToCaptionsDataset(CaptionsDataset):
    """HowTo100M-format manifest: list of ``{"utterance", "frame_filenames"}`` dicts.

    Each utterance becomes one record; one frame is sampled at random from
    ``frame_filenames`` per ``__getitem__`` when ``multiple_frames=True``
    (the default); otherwise the first frame is used deterministically.

    The manifest may be either a flat JSON list ``[...]`` or a
    ``{"data": [...]}`` envelope; both are accepted.

    Args:
        manifest_path: Path to a HowTo-style JSON list (e.g. ``train.json``).
        image_root: Directory holding the extracted frames.
        transform: Image transform.
        multiple_frames: If ``True``, sample a random frame per ``__getitem__``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        transform: transforms.Compose | None = None,
        *,
        multiple_frames: bool = True,
    ) -> None:
        self.multiple_frames = multiple_frames
        super().__init__(manifest_path, image_root, transform)

    def _load_records(self) -> list[dict[str, Any]]:
        with self.manifest_path.open() as f:
            data = json.load(f)
        # Accept either a flat list or a ``{"data": [...]}`` envelope.
        if isinstance(data, dict):
            data = data["data"]
        records: list[dict[str, Any]] = []
        for i, entry in enumerate(data):
            records.append(
                {
                    "image_path": None,  # resolved per __getitem__ from frame_filenames
                    "frame_filenames": entry["frame_filenames"],
                    "caption": entry["utterance"],
                    "id": entry.get("utterance_id", i),
                }
            )
        return records

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, str, str]:
        record = self.records[idx]
        frames = record["frame_filenames"]
        frame = random.choice(frames) if (self.multiple_frames and len(frames) > 1) else frames[0]
        image = Image.open(self._resolve_image_path(frame)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record["caption"], str(record["id"])


class Ego4DCaptionsDataset(HowToCaptionsDataset):
    """Ego4D narration-format manifest. Identical schema to HowTo100M.

    Kept as a separate class so configs can target it semantically and so
    Ego4D-specific tweaks (e.g. timestamp-based filtering) can be added later
    without forking ``HowToCaptionsDataset``.
    """
