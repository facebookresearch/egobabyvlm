# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Task registry for benchmark_creation."""

from dataclasses import dataclass


@dataclass
class TaskInfo:
    """Metadata for a single benchmark task."""

    name: str

    #: ``"lexical"`` or ``"grammatical"``.
    task_type: str

    #: ``"2afc"``, ``"4afc"``, ``"2x2"``, or task-specific paradigm name.
    paradigm: str

    data_dir: str
    description: str = ""
    n_choices: int = 2
    manifest_file: str = "manifest.csv"
    word_file: str | None = None
    sentence_file: str | None = None


TASK_REGISTRY: dict[str, TaskInfo] = {}


def register_task(task: TaskInfo) -> TaskInfo:
    """Register a task in the global registry."""
    TASK_REGISTRY[task.name] = task
    return task


def get_task(name: str) -> TaskInfo:
    """Retrieve a registered task by name."""
    if name not in TASK_REGISTRY:
        msg = f"Task '{name}' not found. Available: {list(TASK_REGISTRY.keys())}"
        raise KeyError(msg)
    return TASK_REGISTRY[name]


def list_tasks() -> list[str]:
    """Return names of all registered tasks."""
    return list(TASK_REGISTRY.keys())


register_task(
    TaskInfo(
        name="lex_nouns",
        task_type="lexical",
        paradigm="word_recognition",
        data_dir="Lexical/Nouns",
        description="Lexical Nouns: word-to-image recognition across semantic categories",
        n_choices=4,
        word_file="word_list.json",
    )
)

register_task(
    TaskInfo(
        name="lex_adjectives",
        task_type="lexical",
        paradigm="pos_neg",
        data_dir="Lexical/Adjectives",
        description="Lexical Adjectives: property recognition via positive/negative image pairs",
        n_choices=2,
        word_file="word_list.json",
    )
)

register_task(
    TaskInfo(
        name="gram",
        task_type="grammatical",
        paradigm="2afc",
        data_dir="Grammatical",
        description="Grammatical: 2-AFC sentence-image matching across 8 categories",
        n_choices=2,
        sentence_file="sentence_list.json",
    )
)
