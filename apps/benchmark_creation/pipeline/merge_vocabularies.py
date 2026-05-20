# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Merge multiple vocab_sorted.csv files into a single vocabulary.

Computes the **intersection** of lemmas across all input datasets: a lemma
is kept only if at least one of its morphological forms appears in *every*
dataset.  For each retained word form, the count is the average across the
datasets (0 for datasets where that form is absent, then divided by the
total number of datasets).

Usage::

    python -m benchmark_creation.pipeline.merge_vocabularies \
        --vocab-csvs CSV1 CSV2 CSV3 CSV4 \
        --output /path/to/merged_vocab_sorted.csv
"""

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

from nltk.stem import WordNetLemmatizer

from apps.benchmark_creation.utils.vocabulary import ensure_nltk_resources, load_vocab_csv
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Regex to strip trailing possessive markers before lemmatisation.
_POSSESSIVE_RE = re.compile(r"'s?$")


def _normalise(word: str, lemmatizer: WordNetLemmatizer) -> str:
    """Return the lemma for *word* after stripping possessives."""
    cleaned = _POSSESSIVE_RE.sub("", word).lower()
    if not cleaned:
        return word.lower()
    return lemmatizer.lemmatize(cleaned)


def merge_vocabularies(
    csv_paths: list[str],
    output_path: str,
) -> Path:
    """Merge vocab CSVs and write the result to *output_path*.

    Returns the resolved output path.
    """
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    n_datasets = len(csv_paths)

    # Per-dataset: lemma -> set of word forms, word -> count
    # Global:      lemma -> set of dataset indices
    lemma_datasets: dict[str, set[int]] = defaultdict(set)
    # lemma -> dataset_idx -> {word: count}
    lemma_forms: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    for ds_idx, csv_path in enumerate(csv_paths):
        logger.info("Loading dataset %d/%d: %s", ds_idx + 1, n_datasets, csv_path)
        entries = load_vocab_csv(csv_path)
        for entry in entries:
            lemma = _normalise(entry.word, lemmatizer)
            lemma_datasets[lemma].add(ds_idx)
            lemma_forms[lemma][ds_idx][entry.word] = entry.count

    # Keep only lemmas present in ALL datasets
    intersected = {lemma for lemma, ds_set in lemma_datasets.items() if len(ds_set) == n_datasets}
    logger.info(
        "Lemma intersection: %d of %d lemmas present in all %d datasets",
        len(intersected),
        len(lemma_datasets),
        n_datasets,
    )

    # Collect all unique word forms for intersected lemmas and average counts
    word_avg_count: dict[str, float] = {}
    for lemma in intersected:
        all_forms: set[str] = set()
        for ds_forms in lemma_forms[lemma].values():
            all_forms.update(ds_forms.keys())
        for word in all_forms:
            total = sum(lemma_forms[lemma].get(ds_idx, {}).get(word, 0) for ds_idx in range(n_datasets))
            word_avg_count[word] = total / n_datasets

    # Sort by descending average count and assign ranks
    sorted_words = sorted(word_avg_count.items(), key=lambda x: -x[1])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count"])
        for rank, (word, avg_count) in enumerate(sorted_words, start=1):
            writer.writerow([rank, word, round(avg_count)])

    logger.info("Wrote %d merged entries to %s", len(sorted_words), out)
    return out


# -- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge vocab_sorted.csv files via lemma-based intersection.",
    )
    parser.add_argument(
        "--vocab-csvs",
        nargs="+",
        required=True,
        help="Paths to the vocab_sorted.csv files to merge.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the merged vocab_sorted.csv.",
    )
    args = parser.parse_args()

    setup_logging()

    merge_vocabularies(args.vocab_csvs, args.output)


if __name__ == "__main__":
    main()
