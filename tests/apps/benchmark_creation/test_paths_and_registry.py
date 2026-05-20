# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for path resolution and the task registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from apps.benchmark_creation import paths
from apps.benchmark_creation.task_registry import (
    TASK_REGISTRY,
    TaskInfo,
    get_task,
    list_tasks,
    register_task,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---- paths ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_paths_cache() -> None:
    """Reset the module-level cache between tests."""
    paths.reset_cache()
    yield
    paths.reset_cache()


def test_get_paths_loads_default_yaml() -> None:
    """Default config has the expected manifest keys."""
    cfg = paths.get_paths()
    assert "outputs_root" in cfg
    assert "coco_captions" in cfg
    assert "howto100m_manifest" in cfg


def test_get_paths_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``BENCHMARK_CREATION_PATHS`` overrides individual keys without affecting others."""
    override = tmp_path / "custom.yaml"
    override.write_text("outputs_root: /tmp/custom-out\n")
    monkeypatch.setenv("BENCHMARK_CREATION_PATHS", str(override))

    cfg = paths.get_paths()
    assert cfg["outputs_root"] == "/tmp/custom-out"
    # Non-overridden keys still come from the default file.
    assert "coco_captions" in cfg


def test_get_paths_is_cached() -> None:
    """Second call returns the cached object, not a fresh load."""
    first = paths.get_paths()
    second = paths.get_paths()
    assert first is second


def test_get_styles_has_expected_styles() -> None:
    styles = paths.get_styles()
    assert "realistic" in styles
    assert "cartoon" in styles


def test_get_style_prefixes_returns_strings() -> None:
    prefixes = paths.get_style_prefixes()
    assert all(isinstance(v, str) and v for v in prefixes.values())


# ---- task registry ----------------------------------------------------


def test_list_tasks_includes_shipped_tasks() -> None:
    tasks = list_tasks()
    assert "lex_nouns" in tasks
    assert "lex_adjectives" in tasks
    assert "gram" in tasks


def test_get_task_returns_lex_nouns_metadata() -> None:
    task = get_task("lex_nouns")
    assert task.task_type == "lexical"
    assert task.n_choices == 4
    assert task.word_file == "word_list.json"


def test_get_task_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="not found"):
        get_task("nonexistent_task")


def test_register_task_round_trip() -> None:
    """A registered task is retrievable by name and gets cleaned up."""
    new = TaskInfo(name="_test_task", task_type="lexical", paradigm="2afc", data_dir="Foo")
    try:
        register_task(new)
        assert get_task("_test_task") is new
    finally:
        # Don't pollute the global registry for sibling tests.
        TASK_REGISTRY.pop("_test_task", None)
