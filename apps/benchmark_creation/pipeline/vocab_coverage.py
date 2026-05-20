# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute dataset vocabulary and measure coverage of benchmark_creation task words.

For a given dataset (HowTo100M, Ego4D, COCO Captions, BabyView), this script:
  1. Loads all utterances and computes word frequencies.
  2. Collects the full word inventory from all benchmark_creation task files.
  3. Reports which task words are covered and which are missing.
  4. Writes results to a timestamped subdirectory under the output root.

Usage:
    python -m apps.benchmark_creation.pipeline.vocab_coverage --dataset coco
    python -m apps.benchmark_creation.pipeline.vocab_coverage --dataset howto100m
"""

import argparse
import csv
import gzip
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from apps.benchmark_creation.paths import (
    get_babyview_manifest,
    get_coco_captions,
    get_ego4d_manifest,
    get_howto100m_manifest,
    get_outputs_root,
)
from apps.benchmark_creation.pipeline.merge_vocabularies import merge_vocabularies
from core.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("compute_vocab_coverage")

DATASET_CHOICES = ("howto100m", "ego4d", "coco", "babyview", "all")


# ---------------------------------------------------------------------------
#  Text dataset — loads utterances from JSON or plain-text manifests
# ---------------------------------------------------------------------------


class TextDataset:
    """Load utterances from a manifest file (JSON or plain text).

    JSON formats supported:
      - Karpathy COCO: ``{"images": [{"sentences": [{"tokens": [...]}]}]}``
      - Standard: ``[{"text": "..."}, ...]``
    Plain text: one utterance per line.
    """

    def __init__(self, manifest_path: str, min_word_count: int = 2) -> None:
        self.manifest_path = manifest_path
        logger.info("Loading dataset from %s...", manifest_path)
        self.metadata: list[dict] = self._load(manifest_path)

        if min_word_count > 0:
            before = len(self.metadata)
            self.metadata = [m for m in self.metadata if len(m["utterance"].split()) >= min_word_count]
            logger.info("  min_word_count=%d: %d -> %d utterances", min_word_count, before, len(self.metadata))

        self.word_counts: dict[str, int] = {}
        logger.info("Dataset ready: %d utterances", len(self.metadata))

    @staticmethod
    def _open(manifest_path: str) -> TextIO:
        """Open ``manifest_path`` transparently, handling gzip-compressed files."""
        if manifest_path.endswith(".gz"):
            return gzip.open(manifest_path, "rt", encoding="utf-8")
        return Path(manifest_path).open(encoding="utf-8")

    @staticmethod
    def _load(manifest_path: str) -> list[dict]:
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        # Strip ``.gz`` for the suffix check so ``foo.json.gz`` still routes through the JSON branch.
        logical_suffix = manifest_path.removesuffix(".gz")
        if logical_suffix.endswith(".json"):
            with TextDataset._open(manifest_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data:
                return [{"utterance": " ".join(s["tokens"])} for img in data["images"] for s in img["sentences"]]
            return [{"utterance": entry.get("text", "")} for entry in data]
        metadata = []
        with TextDataset._open(manifest_path) as f:
            for line in f:
                text = line.strip()
                if text:
                    metadata.append({"utterance": text})
        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def compute_vocab(self, min_freq: int = 1) -> dict[str, int]:
        """Compute word frequencies using NLTK tokenisation."""
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        logger.info("Computing vocabulary...")
        word_counts: Counter = Counter()
        for sample in self.metadata:
            word_counts.update(nltk.word_tokenize(sample["utterance"]))
        logger.info("  %d unique words, %d total tokens", len(word_counts), sum(word_counts.values()))
        if min_freq > 1:
            word_counts = Counter({w: c for w, c in word_counts.items() if c >= min_freq})
            logger.info("  After min_freq=%d: %d words", min_freq, len(word_counts))
        self.word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
        return self.word_counts


# ---------------------------------------------------------------------------
#  Task-word extraction from /data
# ---------------------------------------------------------------------------


def _extract_words_from_dir(subdir: Path) -> set[str]:  # noqa: C901, PLR0912 -- pipeline orchestration: complexity matches the spec it implements
    """Extract all task words from JSON files in a single task directory."""
    words: set[str] = set()

    # --- lexical tasks: word_list.json with "categories" or "words" ----
    word_list = subdir / "word_list.json"
    if word_list.exists():
        data = json.loads(word_list.read_text())
        # Noun format: {"categories": {cat_name: [words]}}
        for items in data.get("categories", {}).values():
            for item in items:
                # Multi-word entries (e.g. "ice cream") -> split into
                # individual tokens so we can check each one.
                for w in item.lower().split():
                    words.add(w)
        # Verb/Adjective format: {"words": {word: {"pos": ..., "neg": ...}}}
        for word in data.get("words", {}):
            for w in word.lower().split():
                words.add(w)

    # --- grammar tasks: sentence_list.json with "categories" -----------
    sent_list = subdir / "sentence_list.json"
    if sent_list.exists():
        data = json.loads(sent_list.read_text())
        for items in data.get("categories", {}).values():
            for item in items:
                for w in item.lower().split():
                    words.add(w)

    # --- sentence_pairs.json with "pair_types" ----------------------
    pairs_file = subdir / "sentence_pairs.json"
    if pairs_file.exists():
        data = json.loads(pairs_file.read_text())
        for pairs in data.get("pair_types", {}).values():
            for pair in pairs:
                for key in ("sentence_a", "sentence_b"):
                    for w in pair.get(key, "").lower().split():
                        words.add(w)

    return words


def _collect_task_words(task_dir: Path) -> dict[str, set[str]]:
    """
    Walk benchmark_creation task directories and collect every target word /
    content word from each task's JSON definition.

    Handles both flat layouts (``Grammatical/gram_*/sentence_list.json``) and
    nested layouts (``Lexical/Nouns/word_list.json``).
    contains no recognised JSON files, its children are checked recursively.

    Returns:
        ``{task_name: {word, ...}}`` -- one entry per leaf task directory.
    """
    task_words: dict[str, set[str]] = {}

    for subdir in sorted(task_dir.iterdir()):
        if not subdir.is_dir():
            continue

        words = _extract_words_from_dir(subdir)

        if words:
            task_words[subdir.name] = words
        else:
            # No JSON files at this level -- recurse into children
            # (handles Lexical/Nouns/, Lexical/Adjectives/, etc.)
            for child in sorted(subdir.iterdir()):
                if not child.is_dir():
                    continue
                child_words = _extract_words_from_dir(child)
                if child_words:
                    task_words[child.name] = child_words

    return task_words


# ---------------------------------------------------------------------------
#  Dataset loading helpers
# ---------------------------------------------------------------------------

_DATASET_MANIFEST_GETTERS = {
    "howto100m": ("HowTo100M", get_howto100m_manifest),
    "ego4d": ("Ego4D", get_ego4d_manifest),
    "coco": ("COCO", get_coco_captions),
    "babyview": ("BabyView", get_babyview_manifest),
}


def _load_dataset(args: argparse.Namespace) -> tuple["TextDataset", str]:
    """Instantiate a TextDataset and return (dataset, db_name)."""
    if args.dataset not in _DATASET_MANIFEST_GETTERS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    db_name, get_manifest = _DATASET_MANIFEST_GETTERS[args.dataset]
    manifest = args.manifest or str(get_manifest())
    ds = TextDataset(
        manifest_path=manifest,
        min_word_count=args.min_word_count,
    )
    return ds, db_name


# ---------------------------------------------------------------------------
#  Output writers
# ---------------------------------------------------------------------------


def _write_vocab_csv(path: Path, word_counts: dict[str, int]) -> None:
    """Write vocab_sorted.csv with rank, word, count columns."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "word", "count"])
        for rank, (word, count) in enumerate(word_counts.items(), start=1):
            writer.writerow([rank, word, count])
    logger.info("Wrote %s (%d words)", path, len(word_counts))


def _write_word_list_csv(path: Path, words_with_info: list[tuple], header: list[str]) -> None:
    """Write a CSV with an arbitrary header and rows."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in words_with_info:
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", path, len(words_with_info))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915 -- pipeline orchestration: complexity matches the spec it implements
    parser = argparse.ArgumentParser(
        description="Compute dataset vocabulary and task-word coverage.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset to analyse.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest file (overrides paths.yaml default).",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum word frequency for vocab inclusion.  Default: 1.",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=2,
        help="Minimum words per utterance (CSV datasets).  Default: 2.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=None,
        help="Path to benchmark task data directory (optional — if omitted, "
        "only vocab extraction is performed, no coverage analysis).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for results (default: outputs_root from paths.yaml).",
    )
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = get_outputs_root() / "vocab_coverage"

    # ---- "all" mode: extract each corpus then merge -----------------------
    if args.dataset == "all":
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"ALL_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ALL mode — extracting vocabularies for all datasets")

        per_dataset_csvs: list[str] = []
        for name, getter in _DATASET_MANIFEST_GETTERS.values():
            logger.info("Processing %s ...", name)
            ds = TextDataset(
                manifest_path=str(getter()),
                min_word_count=args.min_word_count,
            )
            wc = ds.compute_vocab(min_freq=args.min_freq)
            ds_dir = out_dir / name
            ds_dir.mkdir(parents=True, exist_ok=True)
            csv_path = ds_dir / "vocab_sorted.csv"
            _write_vocab_csv(csv_path, wc)
            per_dataset_csvs.append(str(csv_path))

        merged_path = out_dir / "vocab_sorted.csv"
        merge_vocabularies(per_dataset_csvs, str(merged_path))
        logger.info("Merged intersection vocabulary written to %s", merged_path)
        logger.info("Results saved to: %s", out_dir)
        return

    # ---- load dataset & compute vocab ------------------------------------
    dataset, db_name = _load_dataset(args)
    logger.info("Dataset: %s  |  %d utterances", db_name, len(dataset))

    word_counts = dataset.compute_vocab(min_freq=args.min_freq)
    total_tokens = sum(word_counts.values())
    logger.info("Vocab size: %d  |  Total tokens: %d", len(word_counts), total_tokens)

    # ---- write vocab CSV (always) ----------------------------------------
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"{db_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results saved to: %s", out_dir)

    _write_vocab_csv(out_dir / "vocab_sorted.csv", word_counts)

    # ---- task-word coverage (only if --task-dir provided) ----------------
    if args.task_dir is not None:
        # Build a lowercased lookup (dataset vocab may have mixed case)
        vocab_lower: Counter = Counter()
        for word, count in word_counts.items():
            vocab_lower[word.lower()] += count

        logger.info("Collecting task words from %s ...", args.task_dir)
        task_words = _collect_task_words(args.task_dir)
        all_task_words: set[str] = set()
        for words in task_words.values():
            all_task_words |= words
        logger.info(
            "Found %d unique task words across %d tasks",
            len(all_task_words),
            len(task_words),
        )

        covered = {w for w in all_task_words if w in vocab_lower}
        missing = all_task_words - covered
        coverage_pct = 100.0 * len(covered) / len(all_task_words) if all_task_words else 0.0
        logger.info(
            "Coverage: %d / %d task words (%.1f%%)",
            len(covered),
            len(all_task_words),
            coverage_pct,
        )

        per_task_rows = []
        for task_name in sorted(task_words):
            tw = task_words[task_name]
            tc = {w for w in tw if w in vocab_lower}
            tm = tw - tc
            pct = 100.0 * len(tc) / len(tw) if tw else 0.0
            per_task_rows.append((task_name, len(tw), len(tc), len(tm), f"{pct:.1f}"))

        covered_rows = sorted(
            [(w, vocab_lower[w]) for w in covered],
            key=lambda x: x[1],
            reverse=True,
        )
        _write_word_list_csv(
            out_dir / "covered_words.csv",
            covered_rows,
            ["word", "dataset_count"],
        )
        _write_word_list_csv(
            out_dir / "missing_words.csv",
            [(w,) for w in sorted(missing)],
            ["word"],
        )
        _write_word_list_csv(
            out_dir / "task_coverage_by_task.csv",
            per_task_rows,
            ["task", "total_words", "covered", "missing", "coverage_pct"],
        )


if __name__ == "__main__":
    main()
