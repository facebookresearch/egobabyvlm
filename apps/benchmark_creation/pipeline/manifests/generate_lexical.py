# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate per-task JSON manifests for downstream evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from apps.benchmark_creation.utils.vision_scoring import build_caption
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)

TASK_DIR_NAMES = {"nouns": "Nouns", "adjectives": "Adjectives"}

#: Minimum words required in a category for negative sampling.
_MIN_WORDS_PER_CATEGORY = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_word_list(task_dir: str | Path, style: str, filter_variant: str = "") -> dict:
    """Load the word list for a task, preferring the filtered variant.

    Parameters
    ----------
    filter_variant : str
        Which filtered word list to use.  ``""`` (default) tries
        ``word_list_filtered_{style}.json`` then
        ``word_list_filtered_hard_{style}.json``.  ``"hard"`` tries
        ``word_list_filtered_hard_{style}.json`` then
        ``word_list_filtered_{style}.json``.  Falls back to
        ``word_list.json`` only when no filtered file exists.
    """
    task_dir_path = Path(task_dir)
    if filter_variant == "hard":
        candidates = [
            task_dir_path / f"word_list_filtered_hard_{style}.json",
            task_dir_path / f"word_list_filtered_{style}.json",
        ]
    else:
        candidates = [
            task_dir_path / f"word_list_filtered_{style}.json",
            task_dir_path / f"word_list_filtered_hard_{style}.json",
        ]

    for filtered_path in candidates:
        if filtered_path.is_file():
            logger.info("Loading filtered word list: %s", filtered_path)
            with filtered_path.open() as f:
                return json.load(f)

    raw_path = task_dir_path / "word_list.json"
    logger.warning("No filtered word list found, falling back to raw: %s", raw_path)
    with raw_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Quantile computation
# ---------------------------------------------------------------------------


def compute_quantiles(word_bins: dict[str, dict]) -> dict[str, float]:
    """Compute percentile rank (0.0-1.0) for each word based on count.

    Words with the same count receive the same quantile (mean rank).
    """
    if not word_bins:
        return {}

    counts = {w: info["count"] for w, info in word_bins.items()}
    sorted_counts = sorted(counts.values())
    n = len(sorted_counts)

    # Map each unique count to its mean percentile rank.
    count_to_quantile: dict[int, float] = {}
    for c in sorted_counts:
        if c not in count_to_quantile:
            # Find all positions with this count and average them.
            positions = [j for j in range(n) if sorted_counts[j] == c]
            count_to_quantile[c] = sum(p / max(n - 1, 1) for p in positions) / len(positions)

    return {w: round(count_to_quantile[c], 4) for w, c in counts.items()}


# ---------------------------------------------------------------------------
# Manifest metadata
# ---------------------------------------------------------------------------


def build_metadata(
    items: list[dict],
    num_skipped: int,
    missing_words: list[str],
    total_words: int,
    *,
    categories: dict[str, list[str]] | None = None,
) -> dict:
    """Build summary metadata for a manifest."""
    bin_counts = Counter(it["frequency_bin"] for it in items)

    # Sort bins by lower bound (ascending) and rename to bin_01_..., bin_02_...
    def _bin_lower(label: str) -> float:
        """Extract the numeric lower bound from a bin label like '[4,8)'."""
        inner = label.strip("[]()").split(",")[0]
        try:
            return float(inner)
        except (ValueError, IndexError):
            return float("inf")

    sorted_labels = sorted(bin_counts.keys(), key=_bin_lower)
    sorted_bins = {f"bin_{i:02d}_{label}": bin_counts[label] for i, label in enumerate(sorted_labels, start=1)}

    meta: dict = {
        "num_items": len(items),
        "num_words_in_source": total_words,
        "num_skipped": num_skipped,
        "items_per_frequency_bin": sorted_bins,
    }

    if categories is not None:
        meta["num_categories"] = len(categories)
        cat_counts = Counter(it["category"] for it in items)
        meta["items_per_category"] = dict(sorted(cat_counts.items()))

    if missing_words:
        meta["missing_words"] = sorted(missing_words)

    return meta


# ---------------------------------------------------------------------------
# Noun manifest
# ---------------------------------------------------------------------------


def generate_noun_manifest(
    data_dir: str | Path,
    style: str,
    seed: int,
    filter_variant: str = "",
) -> dict:
    """Generate the noun manifest with random same-category negatives."""
    task_dir = Path(data_dir) / "Lexical" / "Nouns"
    data = load_word_list(task_dir, style, filter_variant)

    categories = data["categories"]
    word_bins = data.get("frequency_metadata", {}).get("word_bins", {})
    quantiles = compute_quantiles(word_bins)

    rng = random.Random(seed)
    items = []
    skipped = 0
    missing_words: list[str] = []
    total_words = sum(len(ws) for ws in categories.values())

    for cat_name, words in sorted(categories.items()):
        if len(words) < _MIN_WORDS_PER_CATEGORY:
            logger.warning("Category '%s' has < 2 words, skipping", cat_name)
            missing_words.extend(words)
            skipped += len(words)
            continue

        for word in sorted(words):
            # Check positive image exists. The image generator sanitizes
            # whitespace and slashes in word names, so the manifest must too
            # in order to point at files that actually exist on disk.
            safe_word = word.replace(" ", "_").replace("/", "_")
            img_pos_rel = Path(style) / cat_name / f"{safe_word}.png"
            img_pos_abs = task_dir / img_pos_rel
            if not img_pos_abs.is_file():
                logger.warning("Missing image for '%s': %s", word, img_pos_abs)
                skipped += 1
                missing_words.append(word)
                continue

            # Pick a random negative from the same category.
            candidates = [w for w in words if w != word]
            neg_word = rng.choice(candidates)

            # Check negative image exists.
            safe_neg_word = neg_word.replace(" ", "_").replace("/", "_")
            img_neg_rel = Path(style) / cat_name / f"{safe_neg_word}.png"
            img_neg_abs = task_dir / img_neg_rel
            if not img_neg_abs.is_file():
                logger.warning("Missing negative image for '%s': %s", neg_word, img_neg_abs)
                skipped += 1
                missing_words.append(word)
                continue

            # Frequency metadata.
            freq_info = word_bins.get(word, {})
            freq_bin = freq_info.get("bin_label", "")
            freq_count = freq_info.get("count", 0)
            freq_quantile = quantiles.get(word, 0.0)

            items.append(
                {
                    "word": word,
                    "category": cat_name,
                    "caption_positive": build_caption(word, category=cat_name, task="nouns"),
                    "caption_negative": build_caption(neg_word, category=cat_name, task="nouns"),
                    "negative_word": neg_word,
                    "image_positive": str(img_pos_rel),
                    "image_negative": str(img_neg_rel),
                    "frequency_bin": freq_bin,
                    "frequency_count": freq_count,
                    "frequency_quantile": freq_quantile,
                }
            )

    logger.info(
        "Nouns/%s: %d items generated, %d skipped (missing images)",
        style,
        len(items),
        skipped,
    )

    return {
        "description": f"Noun manifest for style '{style}'",
        "style": style,
        "metadata": build_metadata(
            items,
            skipped,
            missing_words,
            total_words,
            categories=categories,
        ),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Adjective manifest
# ---------------------------------------------------------------------------


def generate_adj_manifest(
    data_dir: str | Path,
    task: str,
    style: str,
    filter_variant: str = "",
) -> dict:
    """Generate an adjective manifest."""
    dir_name = TASK_DIR_NAMES[task]
    task_dir = Path(data_dir) / "Lexical" / dir_name
    data = load_word_list(task_dir, style, filter_variant)

    words = data["words"]
    word_bins = data.get("frequency_metadata", {}).get("word_bins", {})
    quantiles = compute_quantiles(word_bins)

    items = []
    skipped = 0
    missing_words: list[str] = []
    total_words = len(words)

    for word, sents in sorted(words.items()):
        # The adjective image generator sanitizes whitespace and slashes
        # in word names, so the manifest must too in order to point at
        # files that actually exist on disk.
        safe_word = word.replace(" ", "_").replace("/", "_")
        img_pos_rel = Path(style) / safe_word / "pos.png"
        img_neg_rel = Path(style) / safe_word / "neg.png"
        img_pos_abs = task_dir / img_pos_rel
        img_neg_abs = task_dir / img_neg_rel

        if not img_pos_abs.is_file() or not img_neg_abs.is_file():
            logger.warning("Missing image(s) for '%s' in %s/%s", word, task, style)
            skipped += 1
            missing_words.append(word)
            continue

        freq_info = word_bins.get(word, {})
        freq_bin = freq_info.get("bin_label", "")
        freq_count = freq_info.get("count", 0)
        freq_quantile = quantiles.get(word, 0.0)

        items.append(
            {
                "word": word,
                "caption_positive": sents["pos"],
                "caption_negative": sents["neg"],
                "image_positive": str(img_pos_rel),
                "image_negative": str(img_neg_rel),
                "frequency_bin": freq_bin,
                "frequency_count": freq_count,
                "frequency_quantile": freq_quantile,
            }
        )

    logger.info(
        "%s/%s: %d items generated, %d skipped (missing images)",
        dir_name,
        style,
        len(items),
        skipped,
    )

    return {
        "description": f"{dir_name} manifest for style '{style}'",
        "style": style,
        "metadata": build_metadata(items, skipped, missing_words, total_words),
        "items": items,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate per-task JSON manifests for evaluation.",
    )
    parser.add_argument("--data-dir", required=True, help="Root data directory")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["nouns", "adjectives"],
        choices=["nouns", "adjectives"],
        help="Tasks to generate manifests for",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic"],
        help="Image styles to process",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for noun negatives")
    parser.add_argument(
        "--filter-variant",
        type=str,
        default="",
        help="Filter variant to use: '' (default) or 'hard' for SigLIP2 hard filter.",
    )
    args = parser.parse_args()

    setup_logging()

    for task in args.tasks:
        dir_name = TASK_DIR_NAMES[task]
        task_dir = Path(args.data_dir) / "Lexical" / dir_name

        if not task_dir.is_dir():
            logger.warning(
                "Skipping task '%s': directory not found (%s)",
                task,
                task_dir,
            )
            continue

        for style in args.styles:
            logger.info("Generating manifest: %s / %s", task, style)

            if task == "nouns":
                manifest = generate_noun_manifest(
                    args.data_dir,
                    style,
                    args.seed,
                    args.filter_variant,
                )
            else:
                manifest = generate_adj_manifest(
                    args.data_dir,
                    task,
                    style,
                    args.filter_variant,
                )

            out_path = task_dir / f"manifest_{task}_{style}.json"
            with out_path.open("w") as f:
                json.dump(manifest, f, indent=2)

            logger.info("Wrote %d items to %s", len(manifest["items"]), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
