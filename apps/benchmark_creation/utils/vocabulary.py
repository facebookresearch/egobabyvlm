# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vocabulary processing pipeline shared across lexical/grammatical benchmark stages."""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nltk
from nltk.corpus import names, stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_BIN_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

PENN_TO_COARSE = {
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
}

_VOCAB_CSV_MIN_COLUMNS = 3
_LONGTAIL_CSV_MIN_COLUMNS = 6
_POS_TAG_PROGRESS_INTERVAL = 5000
_FILTER_PROGRESS_INTERVAL = 5000


@dataclass
class VocabEntry:
    """Single vocabulary item with frequency, POS, and category metadata."""

    word: str
    count: int
    rank: int
    freq_bin: int = -1
    bin_label: str = ""
    pos: str = "OTHER"
    category: str = ""
    is_valid: bool = True


def ensure_nltk_resources() -> None:
    """Download required NLTK data if not already present."""
    resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)


def load_vocab_csv(csv_path: str | Path) -> list[VocabEntry]:
    """Load vocab_sorted.csv into a list of :class:`VocabEntry`."""
    entries: list[VocabEntry] = []
    with Path(csv_path).open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < _VOCAB_CSV_MIN_COLUMNS:
                continue
            rank = int(row[0].strip())
            word = row[1].strip()
            count = int(row[2].strip())
            entries.append(VocabEntry(word=word, count=count, rank=rank))
    logger.info("Loaded %d vocabulary entries", len(entries))
    return entries


def assign_frequency_bins(
    entries: list[VocabEntry],
    bin_edges: list[int] | None = None,
) -> list[VocabEntry]:
    """Assign log-scale frequency bins to each entry."""
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    for entry in entries:
        entry.freq_bin = len(bin_edges) - 1
        for i in range(len(bin_edges) - 1):
            if entry.count < bin_edges[i + 1]:
                entry.freq_bin = i
                break
        if entry.freq_bin < len(bin_edges) - 1:
            entry.bin_label = f"[{bin_edges[entry.freq_bin]},{bin_edges[entry.freq_bin + 1]})"
        else:
            entry.bin_label = f"[{bin_edges[-1]},inf)"
    return entries


def _resolve_pos_tag(
    word: str,
    coarse: str,
    wn_pos_map: dict[str, str],
    has_synsets: Callable[[str, str], bool],
) -> str:
    if coarse not in wn_pos_map:
        return coarse
    if has_synsets(word, wn_pos_map[coarse]):
        return coarse
    for pos_name in ("NOUN", "VERB", "ADJ"):
        if pos_name != coarse and has_synsets(word, wn_pos_map[pos_name]):
            return pos_name
    return coarse


def pos_tag_words(entries: list[VocabEntry]) -> list[VocabEntry]:
    """POS-tag each word using NLTK + WordNet cross-reference."""
    words = [e.word for e in entries]
    tagged = pos_tag(words)
    wn_pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
    wn_cache: dict[tuple[str, str], bool] = {}

    def _has_synsets(word: str, wn_pos: str) -> bool:
        key = (word, wn_pos)
        if key not in wn_cache:
            wn_cache[key] = bool(wn.synsets(word, pos=wn_pos))
        return wn_cache[key]

    n = len(entries)
    for i, (entry, (_, penn_tag)) in enumerate(zip(entries, tagged, strict=False)):
        if i % _POS_TAG_PROGRESS_INTERVAL == 0 and i > 0:
            logger.info("  POS tagging progress: %d/%d", i, n)
        coarse = PENN_TO_COARSE.get(penn_tag, "OTHER")
        entry.pos = _resolve_pos_tag(entry.word, coarse, wn_pos_map, _has_synsets)

    pos_counts: dict[str, int] = {}
    for e in entries:
        pos_counts[e.pos] = pos_counts.get(e.pos, 0) + 1
    logger.info("POS distribution: %s", pos_counts)
    return entries


_NAME_SET: set[str] | None = None


def _is_primarily_a_name(word: str) -> bool:
    """Return True if *word* is a first/last name whose primary WordNet sense is a named entity."""
    global _NAME_SET  # noqa: PLW0603
    if _NAME_SET is None:
        try:
            name_words = names.words()
        except LookupError:
            nltk.download("names", quiet=True)
            name_words = names.words()
        _NAME_SET = {w.lower() for w in name_words}
    if word.lower() not in _NAME_SET:
        return False
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    first = synsets[0]
    if first.instance_hypernyms():
        return True
    return any(any(s.name().startswith("person.") for s in path) for path in first.hypernym_paths())


def _filter_reason(  # noqa: PLR0911
    entry: VocabEntry,
    *,
    min_length: int,
    min_freq: int,
    stop_words: set[str],
    synset_cache: dict[str, bool],
) -> str | None:
    if len(entry.word) < min_length:
        return "short"
    if entry.count < min_freq:
        return "low_freq"
    if entry.pos == "PROPN":
        return "propn"
    if _is_primarily_a_name(entry.word):
        return "name"
    if not all(c.isalpha() or c == "-" for c in entry.word):
        return "non_alpha"
    if entry.word not in synset_cache:
        synset_cache[entry.word] = bool(wn.synsets(entry.word))
    if not synset_cache[entry.word]:
        return "no_synsets"
    if entry.word in stop_words:
        return "stopword"
    return None


def filter_words(
    entries: list[VocabEntry],
    min_length: int = 2,
    min_freq: int = 1,
) -> list[VocabEntry]:
    """Mark invalid words (too short, stopwords, no synsets, names, etc.)."""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    synset_cache: dict[str, bool] = {}
    counts = {"short": 0, "low_freq": 0, "propn": 0, "name": 0, "non_alpha": 0, "no_synsets": 0, "stopword": 0}
    n = len(entries)
    for i, entry in enumerate(entries):
        if i % _FILTER_PROGRESS_INTERVAL == 0 and i > 0:
            logger.info("  Filtering progress: %d/%d", i, n)
        reason = _filter_reason(
            entry,
            min_length=min_length,
            min_freq=min_freq,
            stop_words=stop_words,
            synset_cache=synset_cache,
        )
        if reason is not None:
            entry.is_valid = False
            counts[reason] += 1

    valid = sum(1 for e in entries if e.is_valid)
    logger.info("Filtering: %d valid of %d total", valid, len(entries))
    logger.info("Filter reasons: %s", counts)
    return entries


def deduplicate_entries(entries: list[VocabEntry]) -> list[VocabEntry]:
    """Deduplicate entries by lowercasing and lemmatising to singular/base form.

    Entries that share the same (lemma, POS) are merged: counts are summed and
    the best frequency-bin metadata is kept from the entry with the highest
    original count. Only valid entries are deduplicated; invalid ones are dropped.
    """
    lemmatizer = WordNetLemmatizer()
    wn_pos = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
    merged: dict[tuple[str, str], VocabEntry] = {}

    for entry in entries:
        if not entry.is_valid:
            continue
        word_lower = entry.word.lower()
        wn_tag = wn_pos.get(entry.pos)
        lemma = lemmatizer.lemmatize(word_lower, pos=wn_tag) if wn_tag else word_lower

        key = (lemma, entry.pos)
        if key in merged:
            merged[key].count += entry.count
            merged[key].word = lemma
        else:
            merged[key] = VocabEntry(
                word=lemma,
                count=entry.count,
                rank=entry.rank,
                freq_bin=entry.freq_bin,
                bin_label=entry.bin_label,
                pos=entry.pos,
                category=entry.category,
                is_valid=True,
            )

    result = list(merged.values())
    n_before = sum(1 for e in entries if e.is_valid)
    logger.info(
        "Deduplication: %d valid entries -> %d unique (lemma, POS) entries",
        n_before,
        len(result),
    )
    return result


def stratified_sample(
    entries: list[VocabEntry],
    n: int,
    seed: int = 42,
) -> list[VocabEntry]:
    """Sample n entries with proportional representation across frequency bins."""
    rng = random.Random(seed)
    if len(entries) <= n:
        return list(entries)
    if not entries:
        return []

    by_bin: dict[int, list[VocabEntry]] = {}
    for e in entries:
        by_bin.setdefault(e.freq_bin, []).append(e)

    total = len(entries)
    sampled: list[VocabEntry] = []
    remainder_entries: list[tuple[float, int]] = []

    for bin_idx, bin_entries in sorted(by_bin.items()):
        proportion = len(bin_entries) / total
        exact = proportion * n
        floor_n = int(exact)
        remainder_entries.append((exact - floor_n, bin_idx))
        chosen = rng.sample(bin_entries, min(floor_n, len(bin_entries)))
        sampled.extend(chosen)

    remaining = n - len(sampled)
    remainder_entries.sort(key=lambda x: x[0], reverse=True)
    for _, bin_idx in remainder_entries:
        if remaining <= 0:
            break
        already = [e for e in sampled if e.freq_bin == bin_idx]
        pool = [e for e in by_bin.get(bin_idx, []) if e not in already]
        if pool:
            sampled.append(rng.choice(pool))
            remaining -= 1

    return sampled


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    f_size = path.stat().st_size
    logger.info("Wrote %s (%.1f KB)", path, f_size / 1024)


def load_longtail_csv(csv_path: str | Path) -> list[VocabEntry]:
    """Read longtail_wordlist.csv back into list[VocabEntry].

    Expects columns: word, count, freq_bin, bin_label, pos, category.
    All loaded entries have ``is_valid=True`` (only valid entries are written).
    """
    entries: list[VocabEntry] = []
    with Path(csv_path).open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < _LONGTAIL_CSV_MIN_COLUMNS:
                continue
            entries.append(
                VocabEntry(
                    word=row[0].strip(),
                    count=int(row[1].strip()),
                    rank=0,
                    freq_bin=int(row[2].strip()),
                    bin_label=row[3].strip(),
                    pos=row[4].strip(),
                    category=row[5].strip(),
                    is_valid=True,
                )
            )
    logger.info("Loaded %d entries from %s", len(entries), csv_path)
    return entries


def write_longtail_csv(entries: list[VocabEntry], output_path: Path) -> None:
    """Write all valid words with metadata to CSV."""
    valid = [e for e in entries if e.is_valid]
    valid.sort(key=lambda e: (e.pos, e.category, -e.count))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count", "freq_bin", "bin_label", "pos", "category"])
        for e in valid:
            writer.writerow([e.word, e.count, e.freq_bin, e.bin_label, e.pos, e.category])
    logger.info("Wrote %d entries to %s", len(valid), output_path)


def write_frequency_report(
    entries: list[VocabEntry],
    output_path: Path,
    bin_edges: list[int],
) -> None:
    """Write summary statistics report."""
    valid = [e for e in entries if e.is_valid]
    total = len(entries)
    n_valid = len(valid)

    lines = [
        "=" * 60,
        "FREQUENCY-STRATIFIED BENCHMARK REPORT",
        "=" * 60,
        "",
        f"Total vocabulary entries: {total}",
        f"Valid entries after filtering: {n_valid} ({100 * n_valid / max(total, 1):.1f}%)",
        f"Filtered out: {total - n_valid}",
        "",
        "--- Words per POS ---",
    ]

    pos_counts: dict[str, int] = {}
    for e in valid:
        pos_counts[e.pos] = pos_counts.get(e.pos, 0) + 1
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {pos:8s}: {cnt:5d}")

    lines.append("")
    lines.append("--- Words per frequency bin ---")
    bin_counts: dict[int, int] = {}
    for e in valid:
        bin_counts[e.freq_bin] = bin_counts.get(e.freq_bin, 0) + 1
    for bi in range(len(bin_edges)):
        label = f"[{bin_edges[bi]},{bin_edges[bi + 1]})" if bi < len(bin_edges) - 1 else f"[{bin_edges[-1]},inf)"
        cnt = bin_counts.get(bi, 0)
        lines.append(f"  bin {bi:2d} {label:>12s}: {cnt:5d}")

    lines.append("")
    lines.append("--- Nouns per semantic category ---")
    cat_counts: dict[str, int] = {}
    for e in valid:
        if e.pos == "NOUN" and e.category:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:20s}: {cnt:5d}")

    lines.append("")
    lines.append("--- Sample words per bin (first 5) ---")
    by_bin: dict[int, list[str]] = {}
    for e in valid:
        by_bin.setdefault(e.freq_bin, []).append(e.word)
    for bi in sorted(by_bin.keys()):
        sample = by_bin[bi][:5]
        lines.append(f"  bin {bi:2d}: {', '.join(sample)}")

    lines.append("")
    lines.append("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote frequency report to %s", output_path)
