# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the VP-Swap prompt + parsing helpers (no LLM calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.swapbench.visual_property_swap import generate as gen
from apps.swapbench.visual_property_swap.prompts import (
    SUPPORTED_PROPERTIES,
    filter_prompt,
    generation_prompt,
    physical_object_prompt,
)


def test_supported_properties_match_paper() -> None:
    assert SUPPORTED_PROPERTIES == ("color", "material", "relative_size", "shape")


@pytest.mark.parametrize("prop", SUPPORTED_PROPERTIES)
def test_generation_prompt_renders_word_pair(prop: str) -> None:
    text = generation_prompt(prop, "apple", "banana")
    assert "apple" in text
    assert "banana" in text
    assert "brackets" in text


@pytest.mark.parametrize("prop", SUPPORTED_PROPERTIES)
def test_filter_prompt_renders_sentence_pair(prop: str) -> None:
    text = filter_prompt(prop, "An apple is red.", "A car is red.")
    assert "An apple is red." in text
    assert "A car is red." in text
    assert "A or B" in text


def test_physical_object_prompt_includes_word() -> None:
    text = physical_object_prompt("apple")
    assert "apple" in text
    assert "yes or no" in text


def test_unsupported_property_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported property"):
        generation_prompt("size_in_meters", "a", "b")
    with pytest.raises(ValueError, match="Unsupported property"):
        filter_prompt("size_in_meters", "a", "b")


# ---------------------------------------------------------------------------
# Pure parsing helpers in generate.py
# ---------------------------------------------------------------------------


def test_bin_for_freq_assigns_log_bins() -> None:
    # _FREQ_BIN_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, inf]
    assert gen._bin_for_freq(1) == 0
    assert gen._bin_for_freq(3) == 1
    assert gen._bin_for_freq(8) == 3
    assert gen._bin_for_freq(1000) == 9


def test_read_visual_words_dedupes_and_skips_malformed(tmp_path: Path) -> None:
    src = tmp_path / "longtail_visualnouns"
    src.write_text(
        "apple,12\n"
        "banana,8\n"
        "apple,99\n"  # duplicate; first occurrence wins (legacy behaviour)
        "BAD ROW\n"
        "car,not_an_int\n",
    )
    words = gen._read_visual_words(src)
    assert words == {"apple": gen._bin_for_freq(12), "banana": gen._bin_for_freq(8)}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("[yes]", True),
        ("[no]", False),
        ("[YES]", True),
        ("yes", True),
        ("no", False),
        ("the answer is [yes].", True),
    ],
)
def test_parse_yes_no_response(response: str, expected: bool) -> None:
    assert gen._parse_yes_no_response(response) is expected


def test_split_two_sentences_handles_period() -> None:
    body = "[An apple is red. A car is also red.]"
    result = gen._split_two_sentences(body)
    assert result is not None
    s1, s2 = result
    assert s1.startswith("An apple")
    assert s2.startswith("A car")


def test_split_two_sentences_returns_none_without_brackets() -> None:
    assert gen._split_two_sentences("no brackets here") is None


def test_word_indices_finds_words() -> None:
    indices = gen._word_indices("An apple is red.", "A car is red.", "apple", "car")
    assert indices == (3, 2)


def test_word_indices_returns_none_when_missing() -> None:
    assert gen._word_indices("nothing here", "or here", "apple", "car") is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("[A]", "A"),
        ("[ B ]", "B"),
        ("A", "A"),
        ("B", "B"),
        ("[C]", None),
        ("garbage", None),
    ],
)
def test_format_answer_extracts_ab(response: str, expected: str | None) -> None:
    assert gen._format_answer(response) == expected


def test_write_generation_prompts_pairs_within_bins(tmp_path: Path) -> None:
    physical = {"apple": 3, "banana": 3, "car": 5}
    out = tmp_path / "prompts.txt"
    gen._write_generation_prompts(physical, "color", out, seed=0)
    rows = out.read_text().strip().splitlines()
    # apple/banana share bin 3; car has no same-bin partner -> 1 pair total
    assert len(rows) == 1
    bin_str, w1, w2, pos, rule, _prompt = rows[0].split("|", 5)
    assert bin_str == "3"
    assert sorted((w1, w2)) == ["apple", "banana"]
    assert pos == "NOUN"
    assert rule == "VISUAL"


def test_retrieve_correct_visualswap_pairs_keeps_only_unanimous(tmp_path: Path) -> None:
    """Only rows where all four returned answers match the expected ones should survive."""
    src = tmp_path / "filter_responses.txt"
    # Worker-pool output: one ``{i}-{j}|{metadata}|{ground_truth}|{response}`` row
    # per filter prompt, so each input row fans out to 4 rows here. We keep
    # an input only if every returned bracketed verdict matches its ground truth
    # AND w1 != w2 (the dataset's anti-self-pair invariant).
    rows = [
        # idx 0: apple vs car, all four verdicts match -> kept
        "0-0|3|VISUAL|apple|s1|0|car|s2|0|A|[A]",
        "0-1|3|VISUAL|apple|s1|0|car|s2|0|B|[B]",
        "0-2|3|VISUAL|apple|s1|0|car|s2|0|A|[A]",
        "0-3|3|VISUAL|apple|s1|0|car|s2|0|B|[B]",
        # idx 1: cat vs dog, one mismatch (B expected, [A] returned) -> dropped
        "1-0|3|VISUAL|cat|s1|0|dog|s2|0|A|[A]",
        "1-1|3|VISUAL|cat|s1|0|dog|s2|0|B|[A]",
        "1-2|3|VISUAL|cat|s1|0|dog|s2|0|A|[A]",
        "1-3|3|VISUAL|cat|s1|0|dog|s2|0|B|[B]",
        # idx 2: fish vs fish, verdicts match but w1==w2 -> dropped
        "2-0|3|VISUAL|fish|s1|0|fish|s2|0|A|[A]",
        "2-1|3|VISUAL|fish|s1|0|fish|s2|0|B|[B]",
        "2-2|3|VISUAL|fish|s1|0|fish|s2|0|A|[A]",
        "2-3|3|VISUAL|fish|s1|0|fish|s2|0|B|[B]",
    ]
    src.write_text("\n".join(rows) + "\n")
    out = tmp_path / "final_pairs.txt"
    accepted = gen._retrieve_correct_visualswap_pairs(src, out)
    assert accepted == 1
    survivors = out.read_text().strip().splitlines()
    assert len(survivors) == 1
    assert "apple" in survivors[0]
    assert "car" in survivors[0]
