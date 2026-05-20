# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Classification datasets used by ABX, KNN, and linear-probe eval tasks."""

import logging
from functools import cached_property
from typing import Any, cast

from torchvision import datasets, transforms

from evaluation.data.base import ClassificationMediaDataset, ClassificationSample
from evaluation.utils import numpy_to_py_types

logger = logging.getLogger(__name__)


class DinoImagenetClassificationDataset(ClassificationMediaDataset):
    """ImageNet-style dataset built via DINOv2's data/transforms helpers."""

    is_video_dataset = False

    def __init__(self, dataset_str: str, mode: str = "test") -> None:
        super().__init__()

        # Imported lazily so eval modules that don't touch ImageNet don't pull in dinov2.
        from dinov2.data import make_dataset
        from dinov2.data.transforms import (
            make_classification_eval_transform,
            make_classification_train_transform,
        )

        self.preprocessor = (
            make_classification_train_transform() if mode == "train" else make_classification_eval_transform()
        )
        self.dataset = make_dataset(dataset_str=dataset_str, transform=self.preprocessor)

        self._indices: list[int] | None = None
        self._subset_classes: set[int] | None = None

    def __len__(self) -> int:
        return len(cast("Any", self.dataset))

    def __getitem__(self, index: int) -> ClassificationSample:
        return cast("ClassificationSample", cast("Any", self.dataset)[index])

    @property
    def class_ids(self) -> list[int]:
        all_class_ids = list(range(len(cast("Any", self.base_dataset)._get_class_ids())))  # noqa: SLF001
        if self._subset_classes is not None:
            return sorted(self._subset_classes & set(all_class_ids))
        return all_class_ids

    @property
    def class_names(self) -> list[str]:
        all_class_names: list[str] = numpy_to_py_types(
            cast("Any", self.base_dataset)._get_class_names(),  # noqa: SLF001
        )  # type: ignore[assignment]
        if self._subset_classes is not None:
            return [all_class_names[i] for i in self.class_ids]
        return all_class_names

    @cached_property
    def all_labels(self) -> list[int]:
        return cast("list[int]", numpy_to_py_types(cast("Any", self.base_dataset).get_targets()))


class CIFAR10ClassificationDataset(ClassificationMediaDataset):
    """torchvision CIFAR-10 wrapper with the standard ImageNet-norm preprocessor."""

    is_video_dataset = False

    def __init__(self, dataset_root: str, mode: str = "test") -> None:
        super().__init__()

        self.preprocessor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        self.dataset = datasets.CIFAR10(
            root=dataset_root,
            train=mode == "train",
            download=True,
            transform=self.preprocessor,
        )

        self._subset_classes: set[int] | None = None
        self._indices: list[int] | None = None

    def __len__(self) -> int:
        return len(cast("Any", self.dataset))

    def __getitem__(self, index: int) -> ClassificationSample:
        return cast("ClassificationSample", cast("Any", self.dataset)[index])

    @property
    def class_ids(self) -> list[int]:
        all_class_ids = list(range(len(cast("Any", self.base_dataset).classes)))
        if self._subset_classes is not None:
            return sorted(self._subset_classes & set(all_class_ids))
        return all_class_ids

    @property
    def class_names(self) -> list[str]:
        all_class_names: list[str] = cast("Any", self.base_dataset).classes
        if self._subset_classes is not None:
            return [all_class_names[i] for i in self.class_ids]
        return all_class_names

    @cached_property
    def all_labels(self) -> list[int]:
        return cast("list[int]", numpy_to_py_types(cast("Any", self.base_dataset).targets))


class MNISTClassificationDataset(ClassificationMediaDataset):
    """torchvision MNIST wrapper that upcasts to 3 channels at ``224x224``."""

    is_video_dataset = False

    def __init__(self, dataset_root: str, mode: str = "test") -> None:
        super().__init__()

        self.preprocessor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        self.dataset = datasets.MNIST(
            root=dataset_root,
            train=mode == "train",
            download=True,
            transform=self.preprocessor,
        )

        self._subset_classes: set[int] | None = None
        self._indices: list[int] | None = None

    def __len__(self) -> int:
        return len(cast("Any", self.dataset))

    def __getitem__(self, index: int) -> ClassificationSample:
        return cast("ClassificationSample", cast("Any", self.dataset)[index])

    @property
    def class_ids(self) -> list[int]:
        all_class_ids = list(range(len(cast("Any", self.base_dataset).classes)))
        if self._subset_classes is not None:
            return sorted(self._subset_classes & set(all_class_ids))
        return all_class_ids

    @property
    def class_names(self) -> list[str]:
        all_class_names: list[str] = cast("Any", self.base_dataset).classes
        if self._subset_classes is not None:
            return [all_class_names[i] for i in self.class_ids]
        return all_class_names

    @cached_property
    def all_labels(self) -> list[int]:
        return cast("list[int]", numpy_to_py_types(cast("Any", self.base_dataset).targets))


class CountBenchClassificationDataset(ClassificationMediaDataset):
    """CountBench (10-class object-counting) dataset loaded via :mod:`datasets`."""

    def __init__(self, mode: str = "test", seed: int = 42) -> None:
        super().__init__()

        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("pip install datasets to use CountBench") from e

        if mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode {mode} for CountBenchClassificationDataset")

        self.dataset = (
            load_dataset("nielsr/countbench")["train"]
            .train_test_split(test_size=0.2, seed=seed)[mode]
            .filter(lambda x: x.get("image", None) is not None)
        )

        self.preprocessor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

        self._subset_classes: set[int] | None = None
        self._indices: list[int] | None = None

    def __len__(self) -> int:
        return len(cast("Any", self.dataset))

    def __getitem__(self, index: int) -> ClassificationSample:
        sample = cast("Any", self.dataset)[index]
        image = sample["image"].convert("RGB")
        label = self._class_name_to_id(sample["number"])

        if self.preprocessor is not None:
            image = self.preprocessor(image)

        return ClassificationSample(media=image, label=label)

    def _class_name_to_id(self, class_name: str | int) -> int:
        return int(class_name) - 1

    def _class_id_to_name(self, class_id: int) -> str:
        return str(class_id + 1)

    @property
    def class_ids(self) -> list[int]:
        all_class_ids = list(range(10))
        if self._subset_classes is not None:
            return sorted(self._subset_classes & set(all_class_ids))
        return all_class_ids

    @property
    def class_names(self) -> list[str]:
        return [self._class_id_to_name(i) for i in self.class_ids]

    @cached_property
    def all_labels(self) -> list[int]:
        return [self._class_name_to_id(n) for n in cast("Any", self.dataset)["number"]]
