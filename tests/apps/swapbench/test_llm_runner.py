# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for swapbench.utils.llm_runner pure helpers (no LLM calls)."""

from __future__ import annotations

from pathlib import Path

from apps.swapbench.utils import llm_runner


def test_read_done_indices_collects_completed_skips_errors(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    out.write_text(
        "0|metadata|hello world\n"
        "1|metadata|<ERROR after 3 retries>\n"
        "2|metadata|another response\n"
        "3-0|metadata|nested ok\n"
        "\n"  # blank line skipped
        "broken row no pipe\n",
    )
    done, errors = llm_runner._read_done_indices(out)
    assert done == {"0", "2", "3-0"}
    assert errors == 1


def test_read_done_indices_missing_file(tmp_path: Path) -> None:
    done, errors = llm_runner._read_done_indices(tmp_path / "absent.txt")
    assert done == set()
    assert errors == 0


def test_iter_jobs_single_prompt_per_line(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text(
        "meta_a|prompt one\nmeta_b|prompt two\nmeta_c|\n",  # empty prompt should be skipped
    )
    jobs = llm_runner._iter_jobs(src, done=set())
    assert jobs == [
        ("0", "meta_a", "prompt one", None),
        ("1", "meta_b", "prompt two", None),
    ]


def test_iter_jobs_multiple_prompts_with_answers(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    # Slash count must be odd (1 + 2k separators between (k+1) prompts and (k+1) answers).
    # Even slash counts (0, 2, 4, ...) are warnings/skips in upstream mp_main.
    # 7 slashes -> 4 prompts + 4 answers -> 4 jobs.
    # 4 slashes (even) -> skipped.
    src.write_text(
        "meta_skip|p1/p2/p3/p4/p5\n"  # 4 slashes -> skipped
        "meta_quad|p1/p2/p3/p4/A/B/A/B\n",  # 7 slashes -> 4 jobs
    )
    jobs = llm_runner._iter_jobs(src, done=set())
    assert len(jobs) == 4
    indices = [j[0] for j in jobs]
    assert indices == ["1-0", "1-1", "1-2", "1-3"]
    prompts_and_answers = [(j[2], j[3]) for j in jobs]
    assert prompts_and_answers == [("p1", "A"), ("p2", "B"), ("p3", "A"), ("p4", "B")]


def test_iter_jobs_skips_already_done(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text(
        "meta_a|prompt one\nmeta_b|prompt two\n",
    )
    jobs = llm_runner._iter_jobs(src, done={"0"})
    assert len(jobs) == 1
    assert jobs[0][0] == "1"
