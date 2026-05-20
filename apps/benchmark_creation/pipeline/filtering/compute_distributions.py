# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute SigLIP2 score distributions for positive and negative image-text pairs."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from apps.benchmark_creation.utils.vision_scoring import (
    ScoringEngine,
    build_caption,
    load_model,
    score_image_text_pairs,
)
from core.utils.logging import setup_logging

#: Minimum noun count needed in a category to score it (need at least one negative).
_MIN_CAT_SIZE = 2

setup_logging()
logger = logging.getLogger("filter_distributions")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_stats(scores: list[float]) -> dict:
    """Compute summary statistics for a list of scores."""
    a = np.array(scores)
    return {
        "count": len(a),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "q05": round(float(np.percentile(a, 5)), 4),
        "q10": round(float(np.percentile(a, 10)), 4),
        "q25": round(float(np.percentile(a, 25)), 4),
        "q75": round(float(np.percentile(a, 75)), 4),
        "q90": round(float(np.percentile(a, 90)), 4),
        "q95": round(float(np.percentile(a, 95)), 4),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distribution(
    pos_scores: list[float],
    neg_scores: list[float],
    output_dir: Path,
    task: str,
    style: str,
) -> None:
    """Generate three histogram plots: pos-only, neg-only, and both overlaid."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(-1, 1, 51).tolist()  # covers SigLIP2 [0,1] and CLIP [-1,1]

    # --- Positive only ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_scores, bins=bins, color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{task.capitalize()} [{style}] — Positive Scores (n={len(pos_scores)})", fontsize=13)
    ax.set_xlim(-1, 1)
    pos_mean = float(np.mean(pos_scores))
    ax.axvline(pos_mean, color="#0D47A1", linestyle="--", linewidth=1.5, label=f"mean={pos_mean:.3f}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{style}_pos.png", dpi=150)
    plt.close(fig)

    # --- Negative only ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg_scores, bins=bins, color="#F44336", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{task.capitalize()} [{style}] — Negative Scores (n={len(neg_scores)})", fontsize=13)
    ax.set_xlim(-1, 1)
    neg_mean = float(np.mean(neg_scores))
    ax.axvline(neg_mean, color="#B71C1C", linestyle="--", linewidth=1.5, label=f"mean={neg_mean:.3f}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{style}_neg.png", dpi=150)
    plt.close(fig)

    # --- Both overlaid ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        pos_scores,
        bins=bins,
        color="#2196F3",
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        label=f"Positive (n={len(pos_scores)}, mean={pos_mean:.3f})",
    )
    ax.hist(
        neg_scores,
        bins=bins,
        color="#F44336",
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        label=f"Negative (n={len(neg_scores)}, mean={neg_mean:.3f})",
    )
    ax.axvline(pos_mean, color="#0D47A1", linestyle="--", linewidth=1.5)
    ax.axvline(neg_mean, color="#B71C1C", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{task.capitalize()} [{style}] — Score Distributions", fontsize=13)
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{style}_both.png", dpi=150)
    plt.close(fig)

    logger.info("Saved plots to %s", output_dir)


# ---------------------------------------------------------------------------
# Noun scoring: positive = own caption, negative = random same-category noun
# ---------------------------------------------------------------------------


def score_nouns(  # noqa: C901, PLR0913 -- pipeline-level orchestration: many parallel context fields
    engine: ScoringEngine,
    data_dir: Path,
    style: str,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[list[float], list[float], dict[str, tuple[list[float], list[float]]], list[dict]]:
    """Score noun images.

    Returns (pos_scores, neg_scores, per_category, pos_pairs) where
    *per_category* maps each category name to its own (pos_scores, neg_scores)
    lists, and *pos_pairs* is a list of ``{"word", "category", "caption", "score"}``
    dicts for every positive pair.
    """
    word_list_path = data_dir / "Lexical" / "Nouns" / "word_list.json"
    if not word_list_path.exists():
        logger.warning("Noun word list not found: %s", word_list_path)
        return [], [], {}, []

    with Path(word_list_path).open() as f:
        word_data = json.load(f)

    images_dir = data_dir / "Lexical" / "Nouns" / style
    rng = random.Random(seed)

    # Build items: (word, category, image_path, pos_caption, neg_caption)
    items: list[tuple[str, str, Path, str, str]] = []

    for category, words in word_data["categories"].items():
        # Gather all words in this category that have images
        available = []
        for w in words:
            safe = w.replace(" ", "_").replace("/", "_")
            img_path = images_dir / category / f"{safe}.png"
            if img_path.exists():
                available.append((w, img_path))

        if len(available) < _MIN_CAT_SIZE:
            logger.warning(
                "Nouns [%s/%s]: only %d images available, skipping category",
                style,
                category,
                len(available),
            )
            continue

        for w, img_path in available:
            pos_caption = build_caption(w, category, task="nouns")

            # Pick a random different noun from the same category as negative
            candidates = [other for other, _ in available if other != w]
            neg_word = rng.choice(candidates)
            neg_caption = build_caption(neg_word, category, task="nouns")

            items.append((w, category, img_path, pos_caption, neg_caption))

    logger.info("Nouns [%s]: scoring %d images", style, len(items))

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pos_pairs: list[dict] = []
    per_category: dict[str, tuple[list[float], list[float]]] = {}

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        images = [Image.open(p).convert("RGB") for _, _, p, _, _ in batch]

        # Positive: image vs own caption
        pos_texts = [pos_cap for _, _, _, pos_cap, _ in batch]
        pos_batch = score_image_text_pairs(engine, images, pos_texts, device)
        pos_scores.extend(pos_batch)

        # Negative: same image vs different noun's caption
        neg_texts = [neg_cap for _, _, _, _, neg_cap in batch]
        neg_batch = score_image_text_pairs(engine, images, neg_texts, device)
        neg_scores.extend(neg_batch)

        # Accumulate per-category scores and positive pair details
        for i, (word, cat, _, pos_cap, _) in enumerate(batch):
            if cat not in per_category:
                per_category[cat] = ([], [])
            per_category[cat][0].append(pos_batch[i])
            per_category[cat][1].append(neg_batch[i])
            pos_pairs.append(
                {
                    "word": word,
                    "category": cat,
                    "caption": pos_cap,
                    "score": round(pos_batch[i], 4),
                }
            )

        done = min(batch_start + batch_size, len(items))
        if done % 200 < batch_size or done == len(items):
            logger.info("Nouns [%s]: %d / %d scored", style, done, len(items))

    return pos_scores, neg_scores, per_category, pos_pairs


# ---------------------------------------------------------------------------
# Adjective scoring: pos_img->pos_cap vs pos_img->neg_cap (and vice versa)
# ---------------------------------------------------------------------------


def score_adj(
    engine: ScoringEngine,
    data_dir: Path,
    style: str,
    batch_size: int,
    device: str,
) -> tuple[list[float], list[float], list[dict]]:
    """Score adjective images. Returns (pos_scores, neg_scores, pos_pairs).

    Positive score: image scored against its *own* caption.
    Negative score: image scored against the *opposite polarity* caption.

    Both pos and neg images contribute one positive and one negative score each.
    *pos_pairs* is a list of ``{"word", "caption", "polarity", "score"}`` dicts
    for every positive pair.
    """
    task_dir_name = "Adjectives"
    word_list_path = data_dir / "Lexical" / task_dir_name / "word_list.json"
    if not word_list_path.exists():
        logger.warning("%s word list not found: %s", task_dir_name, word_list_path)
        return [], [], []

    with Path(word_list_path).open() as f:
        word_data = json.load(f)

    images_dir = data_dir / "Lexical" / task_dir_name / style

    # Collect items: (word, pos_img_path, neg_img_path, pos_caption, neg_caption)
    items: list[tuple[str, Path, Path, str, str]] = []
    missing = 0

    for word, sentences in word_data["words"].items():
        safe_name = word.replace(" ", "_").replace("/", "_")
        word_dir = images_dir / safe_name
        pos_path = word_dir / "pos.png"
        neg_path = word_dir / "neg.png"

        if not pos_path.exists() or not neg_path.exists():
            missing += 1
            continue

        items.append((word, pos_path, neg_path, sentences["pos"], sentences["neg"]))

    if missing:
        logger.warning("%s [%s]: %d words with missing images", task_dir_name, style, missing)

    logger.info("%s [%s]: scoring %d word pairs", task_dir_name, style, len(items))

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pos_pairs: list[dict] = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]

        # Collect all images and their matched/mismatched captions
        batch_imgs_pos: list[Image.Image] = []
        batch_imgs_neg: list[Image.Image] = []
        batch_pos_caps: list[str] = []
        batch_neg_caps: list[str] = []

        for _word, pos_path, neg_path, pos_cap, neg_cap in batch:
            batch_imgs_pos.append(Image.open(pos_path).convert("RGB"))
            batch_imgs_neg.append(Image.open(neg_path).convert("RGB"))
            batch_pos_caps.append(pos_cap)
            batch_neg_caps.append(neg_cap)

        # Positive scores: pos_img<->pos_cap, neg_img<->neg_cap
        pos_pos = score_image_text_pairs(engine, batch_imgs_pos, batch_pos_caps, device)
        neg_neg = score_image_text_pairs(engine, batch_imgs_neg, batch_neg_caps, device)
        pos_scores.extend(pos_pos)
        pos_scores.extend(neg_neg)

        # Negative scores: pos_img<->neg_cap, neg_img<->pos_cap
        pos_neg = score_image_text_pairs(engine, batch_imgs_pos, batch_neg_caps, device)
        neg_pos = score_image_text_pairs(engine, batch_imgs_neg, batch_pos_caps, device)
        neg_scores.extend(pos_neg)
        neg_scores.extend(neg_pos)

        # Accumulate positive pair details
        for i, (word, _, _, pos_cap, neg_cap) in enumerate(batch):
            pos_pairs.append({"word": word, "caption": pos_cap, "polarity": "pos", "score": round(pos_pos[i], 4)})
            pos_pairs.append({"word": word, "caption": neg_cap, "polarity": "neg", "score": round(neg_neg[i], 4)})

        done = min(batch_start + batch_size, len(items))
        if done % 100 < batch_size or done == len(items):
            logger.info("%s [%s]: %d / %d scored", task_dir_name, style, done, len(items))

    return pos_scores, neg_scores, pos_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SigLIP2 score distributions for positive and negative pairs.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory (e.g., data/coco_20260416_121733).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["nouns", "adjectives"],
        choices=["nouns", "adjectives"],
        help="Which sub-tasks to process (default: all).",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic", "cartoon"],
        help="Image styles to score (default: realistic cartoon).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/PE-Core-L14-336",
        help="Vision-language model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per forward pass (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative noun sampling (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915 -- pipeline orchestration: complexity matches the spec it implements
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    engine = load_model(args.model, args.device)

    t0 = time.time()

    for task in args.tasks:
        for style in args.styles:
            logger.info("=" * 60)
            logger.info("Computing distributions: %s [style=%s]", task, style)
            logger.info("=" * 60)

            per_category: dict[str, tuple[list[float], list[float]]] = {}
            pos_pairs: list[dict] = []
            if task == "nouns":
                pos_scores, neg_scores, per_category, pos_pairs = score_nouns(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                    args.seed,
                )
            else:
                pos_scores, neg_scores, pos_pairs = score_adj(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                )

            if not pos_scores:
                logger.warning("No scores for %s [%s], skipping", task, style)
                continue

            # Compute statistics
            pos_stats = compute_stats(pos_scores)
            neg_stats = compute_stats(neg_scores)

            logger.info(
                "%s [%s] — Positive: mean=%.3f, median=%.3f, std=%.3f, q05=%.3f, q95=%.3f",
                task,
                style,
                pos_stats["mean"],
                pos_stats["median"],
                pos_stats["std"],
                pos_stats["q05"],
                pos_stats["q95"],
            )
            logger.info(
                "%s [%s] — Negative: mean=%.3f, median=%.3f, std=%.3f, q05=%.3f, q95=%.3f",
                task,
                style,
                neg_stats["mean"],
                neg_stats["median"],
                neg_stats["std"],
                neg_stats["q05"],
                neg_stats["q95"],
            )

            # Save statistics
            task_dir_name = {"nouns": "Nouns", "adjectives": "Adjectives"}[task]
            dist_dir = data_dir / "Lexical" / task_dir_name / "distributions"
            dist_dir.mkdir(parents=True, exist_ok=True)

            stats_path = dist_dir / f"{task}_{style}_stats.json"
            stats_data = {
                "model": args.model,
                "task": task,
                "style": style,
                "seed": args.seed,
                "positive": pos_stats,
                "negative": neg_stats,
                "separation": {
                    "mean_diff": round(pos_stats["mean"] - neg_stats["mean"], 4),
                    "median_diff": round(pos_stats["median"] - neg_stats["median"], 4),
                },
            }
            with Path(stats_path).open("w") as f:
                json.dump(stats_data, f, indent=2)
            logger.info("Saved stats to %s", stats_path)

            # Save positive pair details (word, caption, score)
            if pos_pairs:
                pairs_path = dist_dir / f"{task}_{style}_pos_pairs.json"
                with Path(pairs_path).open("w") as f:
                    json.dump(pos_pairs, f, indent=2)
                logger.info("Saved %d positive pairs to %s", len(pos_pairs), pairs_path)

            # Generate plots
            plot_distribution(pos_scores, neg_scores, dist_dir, task, style)

            # Per-category distributions (nouns only)
            if per_category:
                cat_dir = dist_dir / "per_category"
                cat_dir.mkdir(parents=True, exist_ok=True)
                all_cat_stats: dict[str, dict] = {}

                for cat_name, (cat_pos, cat_neg) in sorted(per_category.items()):
                    if not cat_pos:
                        continue
                    cat_pos_stats = compute_stats(cat_pos)
                    cat_neg_stats = compute_stats(cat_neg)
                    all_cat_stats[cat_name] = {
                        "positive": cat_pos_stats,
                        "negative": cat_neg_stats,
                        "separation": {
                            "mean_diff": round(cat_pos_stats["mean"] - cat_neg_stats["mean"], 4),
                            "median_diff": round(cat_pos_stats["median"] - cat_neg_stats["median"], 4),
                        },
                    }
                    logger.info(
                        "  %s [%s/%s] — pos mean=%.3f, neg mean=%.3f, diff=%.3f (n=%d)",
                        task,
                        style,
                        cat_name,
                        cat_pos_stats["mean"],
                        cat_neg_stats["mean"],
                        cat_pos_stats["mean"] - cat_neg_stats["mean"],
                        cat_pos_stats["count"],
                    )
                    plot_distribution(cat_pos, cat_neg, cat_dir, f"nouns_{cat_name}", style)

                # Save aggregated per-category stats
                cat_stats_path = cat_dir / f"nouns_{style}_per_category_stats.json"
                with Path(cat_stats_path).open("w") as f:
                    json.dump(
                        {"model": args.model, "style": style, "seed": args.seed, "categories": all_cat_stats},
                        f,
                        indent=2,
                    )
                logger.info("Saved per-category stats to %s", cat_stats_path)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
