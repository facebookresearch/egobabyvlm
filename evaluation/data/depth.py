# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NYUv2 depth estimation dataset."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from evaluation.data.base import DepthPathSample, DepthSample

# OpenCV's TBB-backed parallelism doesn't survive ``fork()``: when DataLoader
# workers call into a cv2 op after being forked, they deadlock waiting on a
# thread pool that no longer exists in the child. Single-threading cv2 is the
# canonical workaround, and is harmless because we already parallelize across
# DataLoader workers.
cv2.setNumThreads(0)

logger = logging.getLogger(__name__)


class DepthEstimationDataset(Dataset):
    """Abstract base for depth estimation datasets."""

    def __getitem__(self, index: int) -> DepthSample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class NYUv2DepthEstimationDataset(DepthEstimationDataset):
    """NYUv2 dataset that loads RGB and depth images from disk."""

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        depth_image_size: tuple[int, int] = (480, 640),
        backbone_input_size: int = 518,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_root: Root directory of the NYUv2 dataset.
            mode: Split, ``"train"`` or ``"test"``.
            depth_image_size: Target ``(H, W)`` for depth GT maps; must match the model's output resolution.
            backbone_input_size: Input size for the backbone (e.g. 518 for DINOv2).
            normalize_mean: Per-channel mean for image normalization. Defaults to ImageNet.
            normalize_std: Per-channel std for image normalization. Defaults to ImageNet.
        """
        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.depth_image_size = depth_image_size

        mean = list(normalize_mean) if normalize_mean is not None else [0.485, 0.456, 0.406]
        std = list(normalize_std) if normalize_std is not None else [0.229, 0.224, 0.225]

        if mode == "train":
            self.preprocessor = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((backbone_input_size, backbone_input_size)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.preprocessor = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((backbone_input_size, backbone_input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        split_file = self.dataset_root / f"nyu_{mode}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found")

        self.samples: list[DepthPathSample] = []
        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    rgb_path = parts[0]
                    depth_path = parts[1]
                    focal_length = float(parts[2])
                    self.samples.append(
                        DepthPathSample(media_path=rgb_path, depth_path=depth_path, focal_length=focal_length)
                    )

        logger.info("Loaded %s %s samples from NYUv2 dataset", len(self), mode)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DepthSample:
        rgb_rel_path, depth_rel_path, focal_length = self.samples[index]

        rgb_path = self.dataset_root / rgb_rel_path.removeprefix("/")
        depth_path = self.dataset_root / depth_rel_path.removeprefix("/")

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found: {depth_path}")

        image = cv2.imread(str(rgb_path))
        if image is None:
            raise RuntimeError(f"cv2 failed to decode RGB image: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"cv2 failed to decode depth image: {depth_path}")
        depth_arr: np.ndarray = (
            depth_raw.astype(np.float32) / 1000.0
            if depth_raw.dtype == np.uint16
            else self._normalize_unexpected_depth(depth_raw)
        )

        target_h, target_w = self.depth_image_size
        depth_arr = cv2.resize(depth_arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if self.preprocessor:
            image = self.preprocessor(image)

        # Keep zeros as-is (mark invalid Kinect-border pixels). SigLoss masks via target > 0.
        depth = torch.from_numpy(depth_arr)
        valid = depth > 0
        depth[valid] = torch.clamp(depth[valid], min=0.001, max=10.0)

        return DepthSample(media=image, depth=depth, focal_length=focal_length)

    @staticmethod
    def _normalize_unexpected_depth(depth_raw: np.ndarray) -> np.ndarray:
        logger.warning("Unexpected depth image format: %s - trying to adapt", depth_raw.dtype)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        arr = depth_raw.astype(np.float32)
        if arr.max() > 1000:
            arr = arr / 1000.0
        elif arr.max() > 10:
            arr = arr / 100.0
        return arr
