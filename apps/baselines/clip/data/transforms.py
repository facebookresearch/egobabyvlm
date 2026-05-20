# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image transform builders for the contrastive trainer.

Resize → optional augmentation → ToTensor → ImageNet normalization. Uses
the standard ImageNet stats so DINOv2 + BERT runs feed numerically
consistent pixels to the encoders.
"""

from __future__ import annotations

import torch
from torchvision import transforms

# Standard ImageNet normalization stats (matches torchvision).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default image size for CLIP-style training.
IMAGE_SIZE = 224

# SimCLR-style train-time blur: applied to half the images.
GAUSSIAN_BLUR_PROB = 0.5
GAUSSIAN_BLUR_SIGMA: tuple[float, float] = (0.1, 2.0)
GAUSSIAN_BLUR_KERNEL = 5


def build_train_transform(*, image_size: int = IMAGE_SIZE, augment: bool = False) -> transforms.Compose:
    """Train-time transform: resize → (optional augmentations) → tensor + normalize."""
    pipeline: list = [transforms.Resize((image_size, image_size))]
    if augment:
        pipeline += [
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=GAUSSIAN_BLUR_KERNEL, sigma=GAUSSIAN_BLUR_SIGMA)],
                p=GAUSSIAN_BLUR_PROB,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def build_eval_transform(*, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Eval/val transform: deterministic resize + normalize, no augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization on a single ``(C, H, W)`` tensor.

    Returns a tensor with values in [0, 1] suitable for ``ToPILImage``.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std + mean, 0, 1)
