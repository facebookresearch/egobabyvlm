# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``LTSwapEvalModule._get_pair_files`` glob-based discovery."""

from __future__ import annotations

from pathlib import Path

from evaluation.text.ltswap import LTSwapEvalConfig, LTSwapEvalModule


def _make_module(tmp_path: Path, *, task_types: list[str], pair_files: dict | None = None) -> LTSwapEvalModule:
    config = LTSwapEvalConfig(
        _target_="evaluation.text.ltswap.LTSwapEvalModule",
        name="test",
        output_dir=str(tmp_path / "out"),
        model={"_target_": "lm_eval.models.huggingface.AutoMaskedLM", "name": "dummy", "kwargs": {}},
        task_types=task_types,
        data_dir=str(tmp_path),
        pair_files=pair_files or {},
    )
    return LTSwapEvalModule(config)


def test_discovers_fixed_filename(tmp_path: Path) -> None:
    """``wordswap`` resolves to a single ``wordswap_pairs.txt`` path."""
    (tmp_path / "wordswap_pairs.txt").write_text("dummy\n")

    module = _make_module(tmp_path, task_types=["wordswap"])
    pair_files = module._get_pair_files()

    assert pair_files == {"wordswap": str(tmp_path / "wordswap_pairs.txt")}


def test_glob_pattern_returns_sorted_list(tmp_path: Path) -> None:
    """``visual`` globs ``vp_swap_*_pairs.txt`` into a sorted list."""
    (tmp_path / "vp_swap_color_pairs.txt").write_text("a\n")
    (tmp_path / "vp_swap_shape_pairs.txt").write_text("b\n")
    (tmp_path / "vp_swap_material_pairs.txt").write_text("c\n")
    # A non-matching file in the same dir should not be picked up.
    (tmp_path / "wordswap_pairs.txt").write_text("ignore\n")

    module = _make_module(tmp_path, task_types=["visual"])
    pair_files = module._get_pair_files()

    assert "visual" in pair_files
    assert pair_files["visual"] == sorted(
        [
            str(tmp_path / "vp_swap_color_pairs.txt"),
            str(tmp_path / "vp_swap_material_pairs.txt"),
            str(tmp_path / "vp_swap_shape_pairs.txt"),
        ]
    )


def test_explicit_pair_files_win_over_discovery(tmp_path: Path) -> None:
    """Caller-supplied ``pair_files`` paths take precedence over auto-discovery."""
    (tmp_path / "wordswap_pairs.txt").write_text("auto\n")
    override = tmp_path / "custom_overridden.txt"
    override.write_text("explicit\n")

    module = _make_module(tmp_path, task_types=["wordswap"], pair_files={"wordswap": str(override)})
    pair_files = module._get_pair_files()

    assert pair_files == {"wordswap": str(override)}


def test_missing_files_are_skipped_not_raised(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    """Tasks with no matching files log a warning and are absent from the result."""
    # Empty data_dir: no pair files at all.
    module = _make_module(tmp_path, task_types=["wordswap", "visual"])

    with caplog.at_level("WARNING"):
        pair_files = module._get_pair_files()

    assert pair_files == {}
    assert any("wordswap" in r.message for r in caplog.records)
    assert any("visual" in r.message for r in caplog.records)
