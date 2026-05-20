# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for vocabulary helpers (pure-Python; no NLTK/network)."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest

from apps.benchmark_creation.utils.vocabulary import (
    DEFAULT_BIN_EDGES,
    VocabEntry,
    assign_frequency_bins,
    deduplicate_entries,
    load_vocab_csv,
    stratified_sample,
    write_longtail_csv,
)

if TYPE_CHECKING:
    from pathlib import Path


def _entry(word: str, count: int, *, pos: str = "NOUN", is_valid: bool = True) -> VocabEntry:
    return VocabEntry(word=word, count=count, rank=0, pos=pos, is_valid=is_valid)


def _wordnet_available() -> bool:
    """``deduplicate_entries`` lemmatises via WordNet; skip if the corpus is missing."""
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
    except (ImportError, LookupError):
        return False
    return True


_REQUIRES_WORDNET = pytest.mark.skipif(
    not _wordnet_available(),
    reason="NLTK wordnet corpus not downloaded; run `python -m nltk.downloader wordnet`",
)


def test_load_vocab_csv_skips_short_rows(tmp_path: Path) -> None:
    """Rows with fewer than 3 columns are silently skipped."""
    csv_path = tmp_path / "vocab.csv"
    csv_path.write_text("rank,word,count\n1,dog,42\n2,cat,7\nbad,row\n")
    entries = load_vocab_csv(csv_path)
    assert [(e.word, e.count, e.rank) for e in entries] == [("dog", 42, 1), ("cat", 7, 2)]


def test_load_vocab_csv_strips_whitespace(tmp_path: Path) -> None:
    csv_path = tmp_path / "vocab.csv"
    csv_path.write_text("rank,word,count\n  1  , hello , 99 \n")
    [entry] = load_vocab_csv(csv_path)
    assert entry.word == "hello"
    assert entry.count == 99


def test_assign_frequency_bins_default_edges() -> None:
    """Counts are bucketed into half-open intervals [edge_i, edge_i+1)."""
    entries = [_entry("a", 1), _entry("b", 3), _entry("c", 100), _entry("d", 999)]
    assign_frequency_bins(entries)

    bin_for_count = {e.count: e.freq_bin for e in entries}
    # DEFAULT_BIN_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert bin_for_count[1] == 1  # count=1 lands in [1,2)
    assert bin_for_count[3] == 2  # count=3 lands in [2,4)
    assert bin_for_count[100] == 7  # count=100 lands in [64,128)
    assert bin_for_count[999] == 10  # overflow bin = len(edges) - 1


def test_assign_frequency_bins_overflow_label() -> None:
    """Counts above the last edge get the ``[edge_last,inf)`` label."""
    [entry] = assign_frequency_bins([_entry("over", 5_000)])
    assert entry.bin_label == "[512,inf)"


def test_assign_frequency_bins_custom_edges() -> None:
    edges = [0, 10, 100]
    entries = [_entry("low", 5), _entry("mid", 50), _entry("high", 500)]
    assign_frequency_bins(entries, bin_edges=edges)
    assert [e.freq_bin for e in entries] == [0, 1, 2]
    assert [e.bin_label for e in entries] == ["[0,10)", "[10,100)", "[100,inf)"]


def test_default_bin_edges_are_log_scale() -> None:
    """Past the leading 0, every interior edge is a power of 2 (half-decade scale)."""
    from itertools import pairwise

    interior = DEFAULT_BIN_EDGES[1:]  # skip the leading 0
    for prev, curr in pairwise(interior):
        assert curr == 2 * prev


@_REQUIRES_WORDNET
def test_deduplicate_entries_drops_invalid() -> None:
    """Invalid entries are dropped before merging."""
    entries = [
        _entry("dog", 5, is_valid=True),
        _entry("dog", 3, is_valid=False),  # dropped
    ]
    result = deduplicate_entries(entries)
    # WordNet lemmatizer leaves "dog" as "dog"; only the valid copy survives.
    assert [(e.word, e.count) for e in result] == [("dog", 5)]


@_REQUIRES_WORDNET
def test_deduplicate_entries_merges_same_lemma_pos() -> None:
    """Entries collapsing to the same (lemma, POS) have their counts summed."""
    # WordNet lemmatizer maps "dogs" → "dog" for NOUN.
    entries = [_entry("dogs", 7, pos="NOUN"), _entry("dog", 3, pos="NOUN")]
    result = deduplicate_entries(entries)
    assert len(result) == 1
    assert result[0].word == "dog"
    assert result[0].count == 10


@_REQUIRES_WORDNET
def test_deduplicate_entries_keeps_different_pos_separate() -> None:
    """Same surface form with different POS must NOT merge."""
    entries = [_entry("run", 5, pos="NOUN"), _entry("run", 3, pos="VERB")]
    result = deduplicate_entries(entries)
    assert len(result) == 2
    assert {(e.word, e.pos, e.count) for e in result} == {("run", "NOUN", 5), ("run", "VERB", 3)}


def test_stratified_sample_returns_all_when_n_exceeds_total() -> None:
    entries = [_entry(f"w{i}", i) for i in range(5)]
    result = stratified_sample(entries, n=100, seed=0)
    assert len(result) == 5


def test_stratified_sample_empty_input_returns_empty() -> None:
    assert stratified_sample([], n=10) == []


def test_stratified_sample_is_deterministic() -> None:
    """Same seed → same sample (within proportional-allocation noise)."""
    entries = [_entry(f"w{i}", i % 5) for i in range(50)]
    assign_frequency_bins(entries, bin_edges=[0, 1, 2, 4, 8])
    a = stratified_sample(entries, n=20, seed=123)
    b = stratified_sample(entries, n=20, seed=123)
    assert [e.word for e in a] == [e.word for e in b]


def test_write_longtail_csv_round_trip(tmp_path: Path) -> None:
    """Written CSV has the expected header + row count."""
    entries = [
        VocabEntry(word="cat", count=10, rank=2, freq_bin=3, bin_label="[8,16)", pos="NOUN", category="animal"),
        VocabEntry(word="run", count=5, rank=8, freq_bin=2, bin_label="[4,8)", pos="VERB", category=""),
    ]
    out = tmp_path / "longtail.csv"
    write_longtail_csv(entries, out)

    with out.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["word", "count", "freq_bin", "bin_label", "pos", "category"]
    assert len(rows) == 3  # header + 2 data rows
    # Ensure word column round-trips.
    assert {row[0] for row in rows[1:]} == {"cat", "run"}
