# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MachineDevBench result aggregator."""

from __future__ import annotations

import pytest

from evaluation.multimodal.machine_devbench.metrics import (
    ResultAggregator,
    accuracy,
    accuracy_per_group,
)


def test_accuracy_basic() -> None:
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 1]) == 0.5
    assert accuracy([], []) == 0.0
    assert accuracy([0, 0, 0], [0, 0, 0]) == 1.0


def test_accuracy_per_group_sorts_keys() -> None:
    preds = [0, 1, 1, 0]
    tgts = [0, 1, 1, 0]
    groups = ["b", "a", "b", "a"]
    out = accuracy_per_group(preds, tgts, groups)
    assert list(out.keys()) == ["a", "b"]
    assert out["a"] == 1.0
    assert out["b"] == 1.0


def test_aggregator_lexical_only() -> None:
    """Lexical task accuracy is the mean of per-frequency-bin accuracies."""
    agg = ResultAggregator()
    # Two trials in [4,8) — 1 correct → bin acc 0.5
    agg.add("lex_nouns", 0, 0, {"frequency_bin": "[4,8)", "category": "animal"})
    agg.add("lex_nouns", 1, 0, {"frequency_bin": "[4,8)", "category": "animal"})
    # One trial in [8,16) — correct → bin acc 1.0
    agg.add("lex_nouns", 0, 0, {"frequency_bin": "[8,16)", "category": "animal"})

    out = agg.compute()
    nouns = out["by_task"]["lex_nouns"]
    # task_acc = mean(0.5, 1.0) = 0.75
    assert nouns["accuracy"] == pytest.approx(0.75)
    assert nouns["by_freq_bin"]["[4,8)"] == 0.5
    assert nouns["by_freq_bin"]["[8,16)"] == 1.0
    # by_category only on lex_nouns
    assert nouns["by_category"]["animal"] == pytest.approx(2 / 3)
    assert out["by_task_type"]["lexical"]["accuracy"] == pytest.approx(0.75)
    assert "grammatical" not in out["by_task_type"]
    assert out["overall"]["accuracy"] == pytest.approx(0.75)


def test_aggregator_grammatical_only() -> None:
    """Grammatical task accuracy is plain accuracy (not bin-averaged)."""
    agg = ResultAggregator()
    agg.add("gram_negation", 0, 0, {"freq_bin": "[16,32)"})
    agg.add("gram_negation", 0, 0, {"freq_bin": "[16,32)"})
    agg.add("gram_negation", 1, 0, {"freq_bin": "[16,32)"})

    out = agg.compute()
    assert out["by_task"]["gram_negation"]["accuracy"] == pytest.approx(2 / 3)
    assert out["by_task_type"]["grammatical"]["accuracy"] == pytest.approx(2 / 3)
    assert out["overall"]["accuracy"] == pytest.approx(2 / 3)


def test_aggregator_low_freq_bins_merged() -> None:
    """[1,2) and [2,4) bins are pooled into [1,4) before per-bin accuracy."""
    agg = ResultAggregator()
    # Two [1,2) trials, both wrong
    agg.add("lex_nouns", 1, 0, {"frequency_bin": "[1,2)", "category": "x"})
    agg.add("lex_nouns", 1, 0, {"frequency_bin": "[1,2)", "category": "x"})
    # Two [2,4) trials, both correct
    agg.add("lex_nouns", 0, 0, {"frequency_bin": "[2,4)", "category": "x"})
    agg.add("lex_nouns", 0, 0, {"frequency_bin": "[2,4)", "category": "x"})

    out = agg.compute()
    nouns = out["by_task"]["lex_nouns"]
    # All four trials pool into a single [1,4) bin → 2/4 = 0.5.
    assert "[1,2)" not in nouns["by_freq_bin"]
    assert "[2,4)" not in nouns["by_freq_bin"]
    assert nouns["by_freq_bin"]["[1,4)"] == 0.5
    # task_acc = mean of bin accuracies = 0.5 (only one bin after merge)
    assert nouns["accuracy"] == 0.5


def test_aggregator_overall_is_type_mean_not_pooled() -> None:
    """``overall`` averages lexical and grammatical type means, not all trials."""
    agg = ResultAggregator()
    # Lexical: 1 trial, correct → lex acc 1.0
    agg.add("lex_nouns", 0, 0, {"frequency_bin": "[8,16)", "category": "x"})
    # Grammatical: 4 trials, all wrong → gram acc 0.0
    for _ in range(4):
        agg.add("gram_x", 1, 0, {"freq_bin": "[8,16)"})

    out = agg.compute()
    assert out["by_task_type"]["lexical"]["accuracy"] == 1.0
    assert out["by_task_type"]["grammatical"]["accuracy"] == 0.0
    # overall = mean(1.0, 0.0) = 0.5, NOT pooled (which would be 1/5 = 0.2)
    assert out["overall"]["accuracy"] == 0.5


def test_aggregator_empty() -> None:
    agg = ResultAggregator()
    out = agg.compute()
    assert out["overall"]["accuracy"] == 0.0
    assert out["by_task"] == {}
    assert out["by_task_type"] == {}
