# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Video captions datasets backed by a CSV manifest."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import cv2
import decord
import pandas as pd
from PIL import Image

from .base import (
    CaptionsMediaDataset,
    CaptionsMediaSample,
    CaptionsPathDataset,
    CaptionsPathSample,
)

if TYPE_CHECKING:
    from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def _load_csv_manifest(manifest_path: str) -> pd.DataFrame:
    """Read a CSV manifest, drop NaN utterances, coerce to str."""
    data = pd.read_csv(manifest_path)
    if data["utterance"].isna().any():
        nan_count = int(data["utterance"].isna().sum())
        logger.warning("%d utterances are NaN, dropping those rows.", nan_count)
        data = data.dropna(subset=["utterance"])
    data["utterance"] = data["utterance"].astype(str)
    return data


def _load_frames_decord(media_path: str, num_frames: int) -> list[Image.Image]:
    """Sample ``num_frames`` evenly across the clip using decord; return PIL images."""
    try:
        vr = decord.VideoReader(media_path)
        total = len(vr)
        if total == 0:
            logger.warning("Empty video: %s", media_path)
            return []
        if num_frames == 1:
            indices = [total // 2]
        else:
            indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
        return [Image.fromarray(f) for f in vr.get_batch(indices).asnumpy()]
    except (decord.DECORDError, RuntimeError) as e:
        logger.warning("Decord failed for %s: %s", media_path, str(e))
        return []


def _load_frames_cv2(media_path: str, num_frames: int) -> list[Image.Image]:
    """Fallback frame loader using OpenCV; same sampling as :func:`_load_frames_decord`."""
    cap = cv2.VideoCapture(media_path)
    images: list[Image.Image] = []
    try:
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", media_path)
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            logger.warning("Empty video: %s", media_path)
            return []
        if num_frames == 1:
            indices = [total // 2]
        else:
            indices = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
        for pos in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return images


class VideoCaptionsDataset(CaptionsMediaDataset):
    """CSV-manifest video dataset: yields ``(list_of_frames, caption, clip_filename)``."""

    is_video_dataset = True

    def __init__(
        self,
        manifest_path: str,
        dataset_dir: str,
        preprocessor: Compose,
        num_frames: int = 8,
    ) -> None:
        super().__init__()
        self.data = _load_csv_manifest(manifest_path)
        self.dataset_dir = str(dataset_dir).rstrip("/")
        self.preprocessor = preprocessor
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> CaptionsMediaSample:
        # Retry up to 10 times on different rows (some clips can be corrupt or truncated).
        for _ in range(10):
            video_filename = self.data["clip_filename"].values[index]
            caption = self.data["utterance"].values[index]
            video_path = f"{self.dataset_dir}/{video_filename}"

            images = _load_frames_decord(video_path, self.num_frames)
            if not images:
                logger.warning("Trying cv2 for %s...", video_path)
                images = _load_frames_cv2(video_path, self.num_frames)

            if images:
                break

            logger.warning("Retrying with a different sample (was %s)", video_path)
            index = random.randint(0, len(self.data) - 1)
        else:
            raise RuntimeError(f"Failed to load video after 10 attempts: {video_path}")

        assert self.preprocessor is not None  # set in __init__, parent type is Optional
        frames = [self.preprocessor(img.convert("RGB") if img.mode == "L" else img) for img in images]
        return CaptionsMediaSample(frames, caption, video_filename)


class VideoCaptionsPathDataset(CaptionsPathDataset):
    """CSV-manifest video dataset: yields ``(clip_path, caption, clip_filename)``."""

    is_video_dataset = True

    def __init__(self, manifest_path: str, dataset_dir: str) -> None:
        super().__init__()
        self.data = _load_csv_manifest(manifest_path)
        self.dataset_dir = str(dataset_dir).rstrip("/")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> CaptionsPathSample:
        video_filename = self.data["clip_filename"].values[idx]
        caption = self.data["utterance"].values[idx]
        video_path = f"{self.dataset_dir}/{video_filename}"
        return CaptionsPathSample(video_path, caption, video_filename)
