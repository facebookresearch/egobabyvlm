# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build a filtered, POS-tagged vocabulary from a raw vocab CSV.

Standalone first step of the benchmark pipeline.  Creates a new timestamped
data directory (e.g. ``MachineDevBench/COCO_20260413_120000/``) that all
downstream scripts (lexical, TROG, Winoground) consume via ``--vocab-dir``.

Usage::

    python scripts/01_Create_Vocabulary/create_vocabulary.py \
        --vocab-csv /path/to/vocab_sorted.csv \
        --output-dir MachineDevBench \
        --name COCO

Outputs::

    MachineDevBench/COCO_TIMESTAMP/
        longtail_wordlist.csv
        frequency_report.txt
"""

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from apps.benchmark_creation.utils.vocabulary import (
    DEFAULT_BIN_EDGES,
    assign_frequency_bins,
    deduplicate_entries,
    ensure_nltk_resources,
    filter_words,
    load_vocab_csv,
    pos_tag_words,
    write_frequency_report,
    write_longtail_csv,
)
from core.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("create_vocabulary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build filtered, POS-tagged vocabulary from a raw vocab CSV.",
    )
    parser.add_argument(
        "--vocab-csv",
        type=str,
        required=True,
        help="Path to vocab_sorted.csv (columns: rank, word, count).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory (e.g., MachineDevBench). "
        "A subdirectory named {name}_{timestamp} is created inside it.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Dataset",
        help="Dataset name for descriptions (e.g., 'COCO').",
    )
    parser.add_argument(
        "--bin-edges",
        default=None,
        help="Comma-separated frequency bin edges.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum word frequency to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    Path(args.vocab_csv)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir = output_dir / f"{args.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = [int(x) for x in args.bin_edges.split(",")] if args.bin_edges else DEFAULT_BIN_EDGES

    t0 = time.time()

    ensure_nltk_resources()

    logger.info("=== Stage 1: Loading vocabulary from %s ===", args.vocab_csv)
    entries = load_vocab_csv(args.vocab_csv)

    logger.info("Assigning frequency bins")
    entries = assign_frequency_bins(entries, bin_edges)

    logger.info("POS tagging %d words", len(entries))
    entries = pos_tag_words(entries)

    logger.info("Filtering words (min_freq=%d)", args.min_freq)
    entries = filter_words(entries, min_freq=args.min_freq)

    logger.info("Deduplicating entries (lowercase + lemmatise to base form)")
    entries = deduplicate_entries(entries)

    logger.info("Re-assigning frequency bins after deduplication")
    entries = assign_frequency_bins(entries, bin_edges)

    logger.info("Writing outputs to %s", output_dir)
    write_longtail_csv(entries, output_dir / "longtail_wordlist.csv")
    write_frequency_report(entries, output_dir / "frequency_report.txt", bin_edges)

    elapsed = time.time() - t0
    logger.info("Done. Total time: %.1f seconds.", elapsed)
    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
