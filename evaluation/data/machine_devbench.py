# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MachineDevBench dataset wrappers (lexical + grammatical, per-style)."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

_LEXICAL_POS = ("nouns", "verbs", "adjectives")
_GRAMMATICAL_PREFIX = "gram_"


def _open_image(path: str | Path) -> Image.Image:
    """Open an image from disk and return an in-memory RGB copy."""
    with Image.open(path).convert("RGB") as img:
        return img.copy()


class MachineDevBenchLexicalDataset(Dataset):
    """Lexical MachineDevBench task (two images, one positive caption).

    Each item is::

        {
            "image": [PIL_positive, PIL_negative],
            "text": [caption_positive],
            "metadata": {... full manifest item ...},
        }
    """

    num_image_cols = 2
    num_text_cols = 1

    def __init__(self, data_root: str | Path, style: str, pos: str) -> None:
        """Initialize the dataset.

        Args:
            data_root: Root path containing the ``Lexical/`` subtree.
            style: Image style (e.g. ``"realistic"`` or ``"cartoon"``).
            pos: Part-of-speech key — ``"nouns"``, ``"verbs"`` or ``"adjectives"``.
        """
        self.data_root = Path(data_root)
        self.style = style
        self.pos = pos

        subdir = pos.capitalize()  # Nouns / Verbs / Adjectives
        self.manifest_path = self.data_root / "Lexical" / subdir / f"manifest_{pos}_{style}.json"
        self._pos_dir = self.manifest_path.parent

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Lexical manifest not found: {self.manifest_path}")

        with self.manifest_path.open() as f:
            data = json.load(f)
        self.items: list[dict[str, Any]] = data.get("items", [])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        img_pos = _open_image(self._pos_dir / item["image_positive"])
        img_neg = _open_image(self._pos_dir / item["image_negative"])
        return {
            "image": [img_pos, img_neg],
            "text": [item["caption_positive"]],
            "metadata": dict(item),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Flatten ``image`` / ``text`` lists across samples and pass metadata through."""
        images = [img for sample in batch for img in sample["image"]]
        texts = [t for sample in batch for t in sample["text"]]
        metadata = [sample["metadata"] for sample in batch]
        return {"image": images, "text": texts, "metadata": metadata}


class MachineDevBenchGrammaticalDataset(Dataset):
    """Grammatical MachineDevBench task (two images, two captions).

    Each item is::

        {
            "image": [PIL_0, PIL_1],
            "text": [caption_a, caption_b],
            "metadata": {... full manifest item ...},
        }
    """

    num_image_cols = 2
    num_text_cols = 2

    def __init__(self, data_root: str | Path, style: str, category: str) -> None:
        """Initialize the dataset.

        Args:
            data_root: Root path containing the ``Grammatical/`` subtree.
            style: Image style (e.g. ``"realistic"`` or ``"cartoon"``).
            category: Grammatical subcategory (e.g. ``"negation"``), without the
                ``gram_`` prefix.
        """
        self.data_root = Path(data_root)
        self.style = style
        self.category = category

        self._category_dir = self.data_root / "Grammatical" / f"{_GRAMMATICAL_PREFIX}{category}"
        self.manifest_path = self._category_dir / f"manifest_grammatical_{category}_{style}.json"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Grammatical manifest not found: {self.manifest_path}")

        with self.manifest_path.open() as f:
            data = json.load(f)
        self.items: list[dict[str, Any]] = data.get("items", [])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        img_0 = _open_image(self._category_dir / item["image_0"])
        img_1 = _open_image(self._category_dir / item["image_1"])
        return {
            "image": [img_0, img_1],
            "text": [item["caption_a"], item["caption_b"]],
            "metadata": dict(item),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Flatten ``image`` / ``text`` lists across samples and pass metadata through."""
        images = [img for sample in batch for img in sample["image"]]
        texts = [t for sample in batch for t in sample["text"]]
        metadata = [sample["metadata"] for sample in batch]
        return {"image": images, "text": texts, "metadata": metadata}


class BenchmarkData:
    """Discovers tasks and builds the appropriate per-task ``Dataset``.

    Lifted (minus the in-memory item loading) from
    :class:`custom_devbench_eval.data_loader.BenchmarkData`. Used by the
    eval module to enumerate available tasks for a given (data_root, style)
    pair and to instantiate the right dataset for each task name.
    """

    LEXICAL_POS = _LEXICAL_POS
    GRAMMATICAL_PREFIX = _GRAMMATICAL_PREFIX

    def __init__(self, data_root: str | Path, style: str = "realistic") -> None:
        """Initialize the benchmark discovery helper.

        Args:
            data_root: Root path of the benchmark (containing ``Lexical/`` and
                ``Grammatical/`` subdirectories).
            style: Image style to discover manifests for.
        """
        self.data_root = Path(data_root)
        self.style = style

    def get_tasks(self) -> list[str]:
        """Return the list of available task names for this (data_root, style).

        Lexical tasks are named ``lex_{pos}`` (``lex_nouns``, ``lex_verbs``,
        ``lex_adjectives``). Grammatical tasks keep their full
        ``gram_{category}`` directory name.
        """
        tasks: list[str] = []
        for pos in self.LEXICAL_POS:
            subdir = pos.capitalize()
            manifest = self.data_root / "Lexical" / subdir / f"manifest_{pos}_{self.style}.json"
            if manifest.exists():
                tasks.append(f"lex_{pos}")

        gram_root = self.data_root / "Grammatical"
        if gram_root.exists():
            for entry in sorted(gram_root.iterdir()):
                if not entry.is_dir() or not entry.name.startswith(self.GRAMMATICAL_PREFIX):
                    continue
                category = entry.name.removeprefix(self.GRAMMATICAL_PREFIX)
                manifest = entry / f"manifest_grammatical_{category}_{self.style}.json"
                if manifest.exists():
                    tasks.append(entry.name)
        return tasks

    def build_dataset(self, task_name: str) -> "MachineDevBenchLexicalDataset | MachineDevBenchGrammaticalDataset":
        """Instantiate the per-task ``Dataset`` for ``task_name``.

        Args:
            task_name: Either ``lex_{pos}`` or ``gram_{category}``.

        Returns:
            A :class:`MachineDevBenchLexicalDataset` or
            :class:`MachineDevBenchGrammaticalDataset` instance.
        """
        if task_name.startswith("lex_"):
            pos = task_name[len("lex_") :]
            return MachineDevBenchLexicalDataset(self.data_root, self.style, pos)
        if task_name.startswith(self.GRAMMATICAL_PREFIX):
            category = task_name.removeprefix(self.GRAMMATICAL_PREFIX)
            return MachineDevBenchGrammaticalDataset(self.data_root, self.style, category)
        raise ValueError(f"Unknown MachineDevBench task: {task_name!r}")
