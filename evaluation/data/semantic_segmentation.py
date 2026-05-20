# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic segmentation datasets."""

import json
import logging
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms

from evaluation.data.base import SemanticSegmentationDataset, SemanticSegmentationSample

logger = logging.getLogger(__name__)


class COCOStuffDataset(SemanticSegmentationDataset):
    """COCO-Stuff semantic segmentation dataset (171-class stuff+thing setup).

    Loads RLE polygon segmentations from a pre-merged ``stuffthings_*.json``
    (built once by ``scripts/eval_data/download_cocostuff.py``) covering 171
    categories (80 things + 91 stuff). Category 183 ("other") is excluded and
    mapped to ``ignore_index``.
    """

    is_video_dataset = False

    def __init__(
        self,
        dataset_root: str,
        mode: str = "val",
        image_size: int = 224,
        *,
        image_dir: str | None = None,
        annotation_file: str | None = None,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_root: Directory holding the COCO-Stuff annotation JSONs.
            mode: ``"train"`` or ``"val"``.
            image_size: Image side length after resize.
            image_dir: Directory with the COCO RGB images. Defaults to ``{dataset_root}/{mode}2017``.
            annotation_file: Custom annotation JSON path. Overrides default lookup.
            normalize_mean: Per-channel mean. Defaults to ImageNet.
            normalize_std: Per-channel std. Defaults to ImageNet.
        """
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.image_size = image_size

        mean = list(normalize_mean) if normalize_mean is not None else [0.485, 0.456, 0.406]
        std = list(normalize_std) if normalize_std is not None else [0.229, 0.224, 0.225]

        # v2 transforms apply geometric ops jointly to image + mask, color ops to image only.
        if mode == "train":
            self.preprocessor = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.preprocessor = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        if annotation_file is not None:
            json_file = Path(annotation_file)
        elif mode == "train":
            json_file = self.dataset_root / "stuff_train2017.json"
        else:
            json_file = self.dataset_root / "stuff_val2017.json"

        if image_dir is not None:
            self.img_dir = Path(image_dir)
        else:
            self.img_dir = self.dataset_root / f"{mode}2017"

        if not json_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_file}")

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        logger.info("Loading COCO-Stuff %s annotations from %s", mode, json_file)
        with json_file.open() as f:
            self.data = json.load(f)

        self.images = {img["id"]: img for img in self.data["images"]}
        annotations = self.data["annotations"]

        self._annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in annotations:
            self._annotations_by_image[ann["image_id"]].append(ann)

        self.img_ids = list(self._annotations_by_image.keys())

        # Category 183 ("other") is excluded in both modes (mapped to ignore_index).
        self.categories = {cat["id"]: cat for cat in self.data["categories"] if cat["supercategory"] != "other"}
        self.id_to_name = {cat_id: cat["name"] for cat_id, cat in self.categories.items()}

        self._num_classes = len(self.categories)
        self.id_to_continuous = {cat_id: i for i, cat_id in enumerate(sorted(self.categories.keys()))}
        self.continuous_to_id = dict(enumerate(sorted(self.categories.keys())))

        valid_img_ids = []
        for img_id in self.img_ids:
            img_info = self.images[img_id]
            img_path = self.img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_path.with_suffix(".png")

            try:
                with Image.open(img_path):
                    pass
                valid_img_ids.append(img_id)
            except (OSError, FileNotFoundError) as e:
                logger.warning("Removing invalid/missing image %s: %s", img_path, e)

        self.img_ids = valid_img_ids

        logger.info("Loaded %d valid %s images with %d classes", len(self), mode, self._num_classes)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> SemanticSegmentationSample:
        try:
            img_id = self.img_ids[idx]
            img_info = self.images[img_id]

            img_path = self.img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = img_path.with_suffix(".png")

            image = Image.open(img_path).convert("RGB")
            img_anns = self._annotations_by_image[img_id]

            h, w = img_info["height"], img_info["width"]
            ignore_index = 255
            mask = np.full((h, w), ignore_index, dtype=np.int64)

            for ann in img_anns:
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], dict):
                        rle = ann["segmentation"]
                        if isinstance(rle["counts"], list):
                            # Uncompressed RLE (e.g. crowd annotations) — compress first.
                            rle = mask_utils.frPyObjects(rle, h, w)
                        binary_mask = mask_utils.decode(rle)
                        if binary_mask.shape != (h, w):
                            binary_mask = np.resize(binary_mask, (h, w))
                    else:
                        rles = mask_utils.frPyObjects(ann["segmentation"], h, w)
                        rle = mask_utils.merge(rles)
                        binary_mask = mask_utils.decode(rle)

                    cat_id = ann["category_id"]
                    continuous_id = self.id_to_continuous.get(cat_id)
                    if continuous_id is not None:
                        mask[binary_mask > 0] = continuous_id

            image = tv_tensors.Image(image)
            mask_tensor = tv_tensors.Mask(torch.from_numpy(mask))

            assert self.preprocessor is not None, "preprocessor is required at __getitem__ time"
            image, mask_tensor = self.preprocessor(image, mask_tensor)

            return SemanticSegmentationSample(media=image, mask=mask_tensor.long(), sample_id=img_id)

        except (OSError, FileNotFoundError, KeyError) as e:
            logger.warning("Error loading image at index %d, img_id %d: %s", idx, self.img_ids[idx], e)
            return self.__getitem__((idx + 1) % len(self))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @cached_property
    def class_ids(self) -> list[int]:
        return list(range(self._num_classes))

    @cached_property
    def class_names(self) -> list[str]:
        return [self.id_to_name[self.continuous_to_id[i]] for i in range(self._num_classes)]
