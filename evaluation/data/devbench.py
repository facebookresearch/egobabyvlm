# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DevBench dataset wrappers compatible with the evaluation framework."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DevBenchDataset(Dataset):
    """Loads a DevBench task from a folder containing a ``manifest.csv``.

    The manifest must have columns ``image1, image2, ...`` and ``text1, text2, ...``
    with relative paths to images and text content for each trial.
    """

    def __init__(self, dataset_folder: str | Path, manifest_file: str = "manifest.csv") -> None:
        """Initialize the dataset.

        Args:
            dataset_folder: Path to the folder containing the dataset and manifest.
            manifest_file: Name of the manifest CSV within the dataset folder.
        """
        self.dataset_folder = Path(dataset_folder)
        manifest_path = self.dataset_folder / manifest_file

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.manifest = pd.read_csv(manifest_path)
        self.num_image_cols = len([c for c in self.manifest.columns if re.compile(r"image\d+").match(c)])
        self.num_text_cols = len([c for c in self.manifest.columns if re.compile(r"text\d+").match(c)])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """Return a single trial as ``{"image": [...], "text": [...]}``."""
        row = self.manifest.iloc[idx]

        images = []
        for i in range(1, self.num_image_cols + 1):
            image_path = self.dataset_folder / row[f"image{i}"]
            with Image.open(image_path).convert("RGB") as img:
                images.append(img.copy())

        texts = [row[f"text{i}"] for i in range(1, self.num_text_cols + 1)]
        return {"image": images, "text": texts}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Flatten lists from multiple trials into a single dict of lists."""
        return {key: [item for sample in batch for item in sample[key]] for key in batch[0]}
