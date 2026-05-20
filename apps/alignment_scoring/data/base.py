# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base classes and sample tuples for caption-style datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torchvision.transforms import Compose


class CaptionsMediaSample(NamedTuple):
    """A loaded media object (image tensor or list of frames) with its caption."""

    media: Any
    text: str
    media_id: int | str


class CaptionsPathSample(NamedTuple):
    """A media path plus its caption — for pipelines that load media themselves."""

    media_path: str
    text: str
    media_id: int | str


class MultiCaptionPathSample(NamedTuple):
    """Used internally by COCO loaders that index over (image, list-of-captions)."""

    media_path: str
    texts: list[str]
    media_id: int | str


class CaptionsMediaDataset(Dataset):
    """Abstract dataset that yields (preprocessed_media, caption, media_id)."""

    is_video_dataset: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Compose | None = None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> CaptionsMediaSample:
        raise NotImplementedError

    def set_preprocessor(self, preprocessor: Compose | None) -> None:
        """Replace the preprocessor (used by pipelines that build their own transform)."""
        self.preprocessor = preprocessor


class CaptionsPathDataset(Dataset):
    """Abstract dataset that yields (media_path, caption, media_id)."""

    is_video_dataset: bool = False

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> CaptionsPathSample:
        raise NotImplementedError
