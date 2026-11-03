# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoVLP image transforms.

EgoVLP / Frozen-in-Time preprocess frames with a 224x224 resize and standard
ImageNet normalization. We reuse the exact same stats as the CLIP baseline so
BabyView and COCO-MC pixels are numerically consistent across baselines. The
identical transform is used by the fine-tuning data loaders AND the DevBench
feature extractor, so train/eval pixels match.
"""

from __future__ import annotations

from torchvision import transforms

# EgoVLP uses standard ImageNet normalization (matches torchvision / CLIP baseline).
EGOVLP_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
EGOVLP_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
EGOVLP_IMAGE_SIZE = 224


def build_egovlp_transform(*, image_size: int = EGOVLP_IMAGE_SIZE, augment: bool = False) -> transforms.Compose:
    """Resize -> (optional hflip) -> tensor -> ImageNet normalize."""
    pipeline: list = [transforms.Resize((image_size, image_size))]
    if augment:
        pipeline.append(transforms.RandomHorizontalFlip())
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(EGOVLP_MEAN, EGOVLP_STD),
    ]
    return transforms.Compose(pipeline)
