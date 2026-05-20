# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base dataset classes for evaluation tasks."""

import logging
import random
from typing import Any, NamedTuple

from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def unwrap_subset(dataset: Dataset) -> Dataset:
    """Recursively unwrap :class:`torch.utils.data.Subset` to the underlying dataset."""
    if isinstance(dataset, Subset):
        return unwrap_subset(dataset.dataset)
    return dataset


class ClassificationSample(NamedTuple):
    """Single classification sample."""

    media: Any
    label: int


class DepthPathSample(NamedTuple):
    """Path-only depth sample (paths resolved at ``__getitem__`` time)."""

    media_path: str
    depth_path: str
    focal_length: float


class DepthSample(NamedTuple):
    """Loaded depth sample."""

    media: Any
    depth: Any
    focal_length: float


class SemanticSegmentationSample(NamedTuple):
    """Single semantic-segmentation sample."""

    media: Any
    mask: Any
    sample_id: int | None = None


class BaseDataset(Dataset):
    """Common base for evaluation datasets."""

    is_video_dataset: bool = False

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError


class ClassificationMediaDataset(BaseDataset):
    """Classification dataset that yields loaded media plus an integer label."""

    is_video_dataset: bool = False

    def __init__(self) -> None:
        super().__init__()

        self.dataset: Dataset | None = None
        self.preprocessor: Compose | None = None

        self._subset_classes: set[int] | None = None
        self._indices: list[int] | None = None

    def __getitem__(self, index: int) -> ClassificationSample:
        raise NotImplementedError

    def set_preprocessor(self, preprocessor: Compose | None) -> None:
        """Install a transform applied to each sample's media."""
        self.preprocessor = preprocessor

    @property
    def class_ids(self) -> list[int]:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def all_labels(self) -> list[int]:
        raise NotImplementedError

    @property
    def base_dataset(self) -> Dataset:
        if self.dataset is None:
            raise RuntimeError(f"{type(self).__name__}.dataset has not been initialized")
        return unwrap_subset(self.dataset)

    def subset_from_subset_classes(
        self, max_subset_classes: int, rng: random.Random | None = None
    ) -> "ClassificationMediaDataset":
        """Restrict the dataset to a random subset of ``max_subset_classes`` classes."""
        logger.info("Randomly sampling %d classes from dataset...", max_subset_classes)

        rng = rng or random.Random(42)

        if self.num_classes <= max_subset_classes:
            logger.warning(
                "Number of classes in dataset (%d) is less than or equal to max_subset_classes (%d). "
                "Returned subset is original dataset.",
                self.num_classes,
                max_subset_classes,
            )
            return self

        self._subset_classes = set(rng.sample(self.class_ids, max_subset_classes))
        logger.info("Selected classes: %s", sorted(self._subset_classes))

        dataset = self.base_dataset

        indices = self._indices or range(len(self))
        sampled_indices = [i for i in indices if self.all_labels[i] in self._subset_classes]
        self._indices = sampled_indices

        self.dataset = Subset(dataset, sampled_indices)

        return self

    def subset_from_max_samples(
        self, max_samples: int, rng: random.Random | None = None
    ) -> "ClassificationMediaDataset":
        """Restrict the dataset to a random subset of ``max_samples`` indices."""
        logger.info("Randomly sampling %d samples from dataset...", max_samples)

        rng = rng or random.Random(42)

        dataset = self.base_dataset

        indices = self._indices or list(range(len(self)))

        if len(indices) <= max_samples:
            logger.warning(
                "Number of samples in dataset (%d) is less than or equal to max_samples (%d). "
                "Returned subset is original dataset.",
                len(indices),
                max_samples,
            )
            return self

        sampled_indices = rng.sample(indices, max_samples)

        self._subset_classes = {self.all_labels[i] for i in sampled_indices}
        self._indices = sampled_indices
        self.dataset = Subset(dataset, sampled_indices)

        return self


class SemanticSegmentationDataset(BaseDataset):
    """Base class for semantic segmentation datasets."""

    is_video_dataset: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Compose | None = None

    def __getitem__(self, index: int) -> SemanticSegmentationSample:
        raise NotImplementedError

    def set_preprocessor(self, preprocessor: Compose | None) -> None:
        """Install a transform applied to each sample's media."""
        self.preprocessor = preprocessor

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def class_ids(self) -> list[int]:
        raise NotImplementedError
