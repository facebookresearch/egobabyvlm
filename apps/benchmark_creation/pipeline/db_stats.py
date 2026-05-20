#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
db_stats.py

Compute comprehensive statistics from a benchmark_creation dataset directory.

Usage:
    python -m benchmark_creation.pipeline.db_stats --data-dir /path/to/dataset
    python -m benchmark_creation.pipeline.db_stats  # uses DATA_DIR env var
"""

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

# -- Lexical helpers -------------------------------------------------------

_LEXICAL_TASKS = ["Nouns", "Adjectives"]
_STYLES = ["realistic", "cartoon"]
_GRAMMATICAL_CATEGORIES = [
    "gram_subject_verb",
    "gram_subject_adjective",
    "gram_negation",
    "gram_order_matters",
    "gram_prepositions",
    "gram_comparatives",
    "gram_counting",
    "gram_embedded_relative",
]


def _count_seq_dirs(directory: Path) -> int:
    """Count seq_* directories directly under *directory*."""
    if not directory.is_dir():
        return 0
    return sum(1 for d in directory.iterdir() if d.is_dir() and d.name.startswith("seq_"))


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if not path.is_file():
        return None
    with Path(path).open() as f:
        return json.load(f)


def _word_count_from_word_list(data: dict, task: str) -> int:
    """Return the number of words in a word_list.json payload."""
    if task == "Nouns":
        categories = data.get("categories", {})
        return sum(len(v) for v in categories.values())
    words = data.get("words", {})
    return len(words)


def _word_set_from_word_list(data: dict, task: str) -> set[str]:
    """Return the set of words from a word_list.json payload."""
    if task == "Nouns":
        categories = data.get("categories", {})
        words = set()
        for cat_words in categories.values():
            words.update(cat_words)
        return words
    return set(data.get("words", {}).keys())


# -- Collectors ------------------------------------------------------------


def collect_lexical_stats(data_dir: Path) -> dict:
    """Gather per-task, per-style statistics for lexical tasks.

    Only counts examples that survived post-filtering (i.e. manifest items).
    Word counts come from the filtered word lists when available.
    """
    lexical_dir = data_dir / "Lexical"
    stats = {}
    all_words: set[str] = set()
    total_images = 0
    total_trials = 0

    for task in _LEXICAL_TASKS:
        task_dir = lexical_dir / task
        if not task_dir.is_dir():
            continue

        task_stats: dict = {}
        task_stats["per_style"] = {}

        for style in _STYLES:
            style_stats: dict = {}

            # Filtered word list (prefer hard-filtered, fall back to filtered)
            for wl_name in (
                f"word_list_filtered_hard_{style}.json",
                f"word_list_filtered_{style}.json",
            ):
                wl_data = _load_json(task_dir / wl_name)
                if wl_data:
                    style_stats["words_after_filtering"] = _word_count_from_word_list(wl_data, task)
                    all_words.update(_word_set_from_word_list(wl_data, task))
                    break

            # Manifest (trials that survived filtering + image generation)
            manifest_name = f"manifest_{task.lower()}_{style}.json"
            manifest = _load_json(task_dir / manifest_name)
            if manifest:
                meta = manifest.get("metadata", {})
                items = manifest.get("items", [])
                style_stats["num_trials"] = len(items)
                style_stats["words_per_frequency_bin"] = meta.get("items_per_frequency_bin", {})
                if task == "Nouns":
                    style_stats["words_per_category"] = meta.get("items_per_category", {})
                total_trials += len(items)
                # Collect unique words from manifest items
                for item in items:
                    w = item.get("word", "")
                    if w:
                        all_words.add(w)

            # Count images referenced by manifest items (2 per trial)
            n_trials = style_stats.get("num_trials", 0)
            style_stats["images"] = n_trials * 2
            total_images += n_trials * 2

            task_stats["per_style"][style] = style_stats

        stats[task.lower()] = task_stats

    return {
        "tasks": stats,
        "total_unique_words_lexical": len(all_words),
        "total_images_lexical": total_images,
        "total_trials_lexical": total_trials,
    }


def collect_grammatical_stats(data_dir: Path) -> dict:
    """Gather per-category, per-style statistics for grammatical tasks.

    Uses the grammatical manifests (post-filtered) rather than the raw
    sentence_list.json so that only validated trials are counted.
    """
    gram_dir = data_dir / "Grammatical"
    stats = {}
    total_images = 0
    total_trials = 0
    all_words: set[str] = set()

    for cat in _GRAMMATICAL_CATEGORIES:
        cat_dir = gram_dir / cat
        if not cat_dir.is_dir():
            continue

        # Extract short category name (e.g. "gram_counting" -> "counting")
        short_cat = cat.removeprefix("gram_")

        cat_stats: dict = {"per_style": {}}
        for style in _STYLES:
            style_stats: dict = {}

            # Grammatical manifest (filtered trials)
            manifest_name = f"manifest_grammatical_{short_cat}_{style}.json"
            manifest = _load_json(cat_dir / manifest_name)
            if manifest:
                meta = manifest.get("metadata", {})
                items = manifest.get("items", [])
                style_stats["num_trials"] = len(items)
                style_stats["num_original"] = meta.get("num_original", len(items))
                style_stats["num_removed"] = meta.get("num_removed", 0)
                total_trials += len(items)
                # Each trial has 2 images
                style_stats["images"] = len(items) * 2
                total_images += len(items) * 2
                # Collect unique words from manifest items
                for item in items:
                    w = item.get("word", "")
                    if w:
                        all_words.add(w)
            else:
                # Fall back to sentence_list.json if no manifest exists yet
                sl_data = _load_json(cat_dir / "sentence_list.json")
                if sl_data:
                    items = sl_data.get("items", [])
                    style_stats["num_trials"] = len(items)
                    total_trials += len(items)
                    imgs_style_dir = cat_dir / "imgs" / style
                    seq_count = _count_seq_dirs(imgs_style_dir)
                    style_stats["images"] = seq_count * 2
                    total_images += seq_count * 2
                    for item in items:
                        w = item.get("word", "")
                        if w:
                            all_words.add(w)

            cat_stats["per_style"][style] = style_stats

        # Aggregate across styles for convenience
        [cat_stats["per_style"][s].get("num_trials", 0) for s in _STYLES]
        cat_stats["num_unique_words"] = len(
            {
                item.get("word", "")
                for s in _STYLES
                for manifest_name in [f"manifest_grammatical_{short_cat}_{s}.json"]
                for manifest in [_load_json(cat_dir / manifest_name)]
                if manifest
                for item in manifest.get("items", [])
            }
            - {""}
        )

        stats[cat] = cat_stats

    return {
        "categories": stats,
        "total_unique_words_grammatical": len(all_words),
        "total_images_grammatical": total_images,
        "total_trials_grammatical": total_trials,
    }


def collect_vocabulary_stats(data_dir: Path) -> dict:
    """Compute statistics from the longtail_wordlist.csv."""
    csv_path = data_dir / "longtail_wordlist.csv"
    if not csv_path.is_file():
        return {}

    counts: list[int] = []
    pos_counts: dict[str, int] = collections.Counter()
    freq_bin_counts: dict[str, int] = collections.Counter()
    total = 0

    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            count_val = int(row.get("count", 0))
            counts.append(count_val)
            pos = row.get("pos", "UNKNOWN")
            pos_counts[pos] += 1
            freq_bin = row.get("freq_bin", "?")
            bin_label = row.get("bin_label", "")
            key = f"bin_{freq_bin}" + (f"_{bin_label}" if bin_label else "")
            freq_bin_counts[key] += 1

    # Percentiles
    percentiles_pct = [10, 25, 50, 75, 90, 95, 99]
    percentiles: dict[str, int] = {}
    if counts:
        counts.sort()
        n = len(counts)
        for p in percentiles_pct:
            idx = min(int(p / 100 * n), n - 1)
            percentiles[f"p{p}"] = counts[idx]

    return {
        "total_words": total,
        "words_per_pos": dict(sorted(pos_counts.items())),
        "count_percentiles": percentiles,
        "words_per_frequency_bin": dict(sorted(freq_bin_counts.items())),
    }


# -- Summary builder -------------------------------------------------------


def build_summary(lexical: dict, grammatical: dict, totals: dict) -> dict:
    """Build a compact summary with words/images/trials per task and subtask."""
    summary: dict = {}

    # Lexical tasks
    lex_tasks = lexical.get("tasks", {})
    for task_name, task_data in lex_tasks.items():
        images = 0
        trials = 0
        for style in _STYLES:
            sd = task_data.get("per_style", {}).get(style, {})
            images += sd.get("images", 0)
            trials += sd.get("num_trials", 0)
        # Use filtered word count (average across styles as proxy)
        word_counts = [task_data.get("per_style", {}).get(s, {}).get("words_after_filtering", 0) for s in _STYLES]
        max_words = max(word_counts) if word_counts else 0
        summary[task_name] = {"words": max_words, "images": images, "trials": trials}

    # Grammatical categories
    gram_cats = grammatical.get("categories", {})
    for cat_name, cat_data in gram_cats.items():
        images = 0
        trials = 0
        for style in _STYLES:
            sd = cat_data.get("per_style", {}).get(style, {})
            images += sd.get("images", 0)
            trials += sd.get("num_trials", 0)
        summary[cat_name] = {
            "words": cat_data.get("num_unique_words", 0),
            "images": images,
            "trials": trials,
        }

    summary["total"] = totals
    return summary


# -- Pretty printing -------------------------------------------------------


def _print_section(title: str) -> None:
    pass


def _print_kv(key: str, value: object, indent: int = 2) -> None:  # noqa: ARG001 -- placeholder helper used by pretty_print().
    " " * indent


def _print_dict(d: dict, indent: int = 4) -> None:
    " " * indent
    for _k, _v in d.items():
        pass


def pretty_print(all_stats: dict) -> None:  # noqa: C901, PLR0912 -- pipeline orchestration: complexity matches the spec it implements
    """Print a human-readable summary to stdout."""
    summary = all_stats.get("summary", {})
    lex = all_stats.get("lexical", {})
    gram = all_stats.get("grammatical", {})
    vocab = all_stats.get("vocabulary", {})
    totals = all_stats.get("totals", {})

    # -- Summary table --
    _print_section("Summary (words / images / trials)")
    for task in summary:
        if task == "total":
            continue
    summary.get("total", {})

    # -- Lexical --
    _print_section("Lexical Tasks")
    for task_data in lex.get("tasks", {}).values():
        for style in _STYLES:
            sd = task_data.get("per_style", {}).get(style, {})
            if not sd:
                continue
            _print_kv("  Words after filtering", sd.get("words_after_filtering", "N/A"), 4)
            _print_kv("  Trials (manifest items)", sd.get("num_trials", "N/A"), 4)
            _print_kv("  Images (trial x 2)", sd.get("images", "N/A"), 4)
            freq = sd.get("words_per_frequency_bin", {})
            if freq:
                _print_dict(freq, indent=6)
            cat = sd.get("words_per_category", {})
            if cat:
                _print_dict(cat, indent=6)

    # -- Grammatical --
    _print_section("Grammatical Tasks (post-filtered)")
    for cat_data in gram.get("categories", {}).values():
        _print_kv("Unique words (across styles)", cat_data.get("num_unique_words", "N/A"))
        for style in _STYLES:
            sd = cat_data.get("per_style", {}).get(style, {})
            if not sd:
                continue
            _print_kv("  Trials (filtered)", sd.get("num_trials", "N/A"), 4)
            if "num_original" in sd:
                _print_kv("  Original (pre-filter)", sd["num_original"], 4)
                _print_kv("  Removed by filter", sd.get("num_removed", 0), 4)
            _print_kv("  Images (trial x 2)", sd.get("images", "N/A"), 4)

    # -- Vocabulary --
    if vocab:
        _print_section("Source Vocabulary (longtail_wordlist.csv)")
        _print_kv("Total words", vocab.get("total_words", "N/A"))
        _print_dict(vocab.get("words_per_pos", {}))
        _print_dict(vocab.get("count_percentiles", {}))
        _print_dict(vocab.get("words_per_frequency_bin", {}))

    # -- Totals --
    _print_section("Totals")
    for k, v in totals.items():
        _print_kv(k, v)


# -- Main ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute comprehensive statistics for a benchmark_creation dataset.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", ""),
        help="Path to dataset directory (default: $DATA_DIR)",
    )
    args = parser.parse_args()

    if not args.data_dir:
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(1)

    lexical = collect_lexical_stats(data_dir)
    grammatical = collect_grammatical_stats(data_dir)
    vocabulary = collect_vocabulary_stats(data_dir)

    totals = {
        "total_unique_words": (lexical["total_unique_words_lexical"] + grammatical["total_unique_words_grammatical"]),
        "total_images": (lexical["total_images_lexical"] + grammatical["total_images_grammatical"]),
        "total_trials": (lexical["total_trials_lexical"] + grammatical["total_trials_grammatical"]),
    }

    summary = build_summary(lexical, grammatical, totals)

    all_stats = {
        "data_dir": str(data_dir),
        "summary": summary,
        "lexical": lexical,
        "grammatical": grammatical,
        "vocabulary": vocabulary,
        "totals": totals,
    }

    pretty_print(all_stats)

    output_path = data_dir / "db_stats.json"
    with Path(output_path).open("w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
