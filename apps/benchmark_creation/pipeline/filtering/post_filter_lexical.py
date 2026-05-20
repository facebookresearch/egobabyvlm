# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Post-filter generated images using SigLIP2 vision-language alignment.

Scores every generated image against its caption using a pretrained SigLIP2
model.  Images with low alignment scores are flagged and optionally removed
from the word list.

Supports two lexical sub-tasks (nouns, adjectives) and two image
styles (realistic, cartoon).

For all lexical tasks, an optional **contrastive check** verifies that SigLIP2
can correctly solve the task -- i.e., distinguish the positive image from its
negative counterpart.  For nouns, a random same-category negative is chosen;
for adjectives, the paired pos/neg images are compared.

Usage::

    # Score all tasks and styles:
    python -m benchmark_creation.pipeline.filtering.post_filter_lexical \\
        --data-dir data/coco_20260416_121733

    # Score only nouns, write filtered word list:
    python -m benchmark_creation.pipeline.filtering.post_filter_lexical \\
        --data-dir data/coco_20260416_121733 \\
        --tasks nouns \\
        --write-filtered

    # Strict filtering with contrastive check:
    python -m benchmark_creation.pipeline.filtering.post_filter_lexical \\
        --data-dir data/coco_20260416_121733 \\
        --min-score 0.2 \\
        --require-contrastive \\
        --write-filtered

Outputs::

    {data_dir}/Lexical/{Nouns|Adjectives}/
        siglip2_scores_{style}.json        # Per-word scores and pass/fail
        word_list_filtered_{style}.json    # (if --write-filtered)
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from apps.benchmark_creation.utils.vision_scoring import (
    ScoringEngine,
    build_caption,
    load_model,
    score_image_text_matrix,
)
from core.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("post_filter_lexical")


# ---------------------------------------------------------------------------
# Noun scoring
# ---------------------------------------------------------------------------


def score_nouns(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- pipeline-level orchestration: many parallel context fields
    engine: ScoringEngine,
    data_dir: Path,
    style: str,
    batch_size: int,
    device: str,
    existing_scores: dict | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """Score all noun images with contrastive check against a same-category negative.

    For each noun, picks a random same-category negative and computes a 2x2
    score matrix ``[pos_img, neg_img] x [pos_cap, neg_cap]``.  This lets us
    check both alignment (diagonal scores) and whether SigLIP2 can solve the
    task (contrastive: correct image scores higher than the distractor).

    Returns ``{word: {score, neg_score, contrastive_pos, contrastive_neg,
    negative_word, caption, neg_caption, category}}``.
    """
    import random as _random

    word_list_path = data_dir / "Lexical" / "Nouns" / "word_list.json"
    if not word_list_path.exists():
        logger.warning("Noun word list not found: %s", word_list_path)
        return {}

    with Path(word_list_path).open() as f:
        word_data = json.load(f)

    images_dir = data_dir / "Lexical" / "Nouns" / style

    # Build a lookup of available words per category (those with images).
    available_by_cat: dict[str, list[str]] = {}
    for category, words in word_data["categories"].items():
        avail = []
        for w in words:
            safe = w.replace(" ", "_").replace("/", "_")
            if (images_dir / category / f"{safe}.png").exists():
                avail.append(w)
        if avail:
            available_by_cat[category] = avail

    rng = _random.Random(seed)

    # Collect items: (word, category, pos_path, neg_path, pos_cap, neg_cap, neg_word)
    items: list[tuple[str, str, Path, Path, str, str, str]] = []
    missing = 0
    skipped_existing = 0

    for category, words in sorted(word_data["categories"].items()):
        avail = available_by_cat.get(category, [])
        for word in sorted(words):
            if existing_scores and word in existing_scores:
                skipped_existing += 1
                continue

            safe_name = word.replace(" ", "_").replace("/", "_")
            pos_path = images_dir / category / f"{safe_name}.png"
            if not pos_path.exists():
                missing += 1
                continue

            # Pick a random same-category negative.
            candidates = [w for w in avail if w != word]
            if not candidates:
                logger.warning("No negative candidate for '%s' in category '%s'", word, category)
                missing += 1
                continue

            neg_word = rng.choice(candidates)
            neg_safe = neg_word.replace(" ", "_").replace("/", "_")
            neg_path = images_dir / category / f"{neg_safe}.png"

            pos_cap = build_caption(word, category, task="nouns")
            neg_cap = build_caption(neg_word, category, task="nouns")
            items.append((word, category, pos_path, neg_path, pos_cap, neg_cap, neg_word))

    if skipped_existing:
        logger.info("Nouns [%s]: skipping %d already-scored words", style, skipped_existing)
    if missing:
        logger.warning("Nouns [%s]: %d images missing or no negative available", style, missing)

    logger.info("Nouns [%s]: scoring %d word pairs (batch_size=%d)", style, len(items), batch_size)

    scores = dict(existing_scores) if existing_scores else {}

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]

        for word, category, pos_path, neg_path, pos_cap, neg_cap, neg_word in batch:
            pos_img = Image.open(pos_path).convert("RGB")
            neg_img = Image.open(neg_path).convert("RGB")

            # 2x2 scoring: [pos_img, neg_img] x [pos_cap, neg_cap]
            score_matrix = score_image_text_matrix(
                engine,
                [pos_img, neg_img],
                [pos_cap, neg_cap],
                device,
            )

            pos_score = score_matrix[0, 0].item()
            neg_score = score_matrix[1, 1].item()
            pos_cap_neg_img = score_matrix[1, 0].item()
            contrastive_pos = score_matrix[0, 0].item() > score_matrix[1, 0].item()
            contrastive_neg = score_matrix[1, 1].item() > score_matrix[0, 1].item()

            scores[word] = {
                "score": round(pos_score, 4),
                "neg_score": round(neg_score, 4),
                "pos_cap_neg_img_score": round(pos_cap_neg_img, 4),
                "caption": pos_cap,
                "neg_caption": neg_cap,
                "category": category,
                "negative_word": neg_word,
                "contrastive_pos": contrastive_pos,
                "contrastive_neg": contrastive_neg,
            }

        done = min(batch_start + batch_size, len(items))
        if done % 200 < batch_size or done == len(items):
            logger.info("Nouns [%s]: %d / %d scored", style, done, len(items))

    return scores


# ---------------------------------------------------------------------------
# Adjective scoring (pos + neg pairs with contrastive check)
# ---------------------------------------------------------------------------


def score_adj(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    engine: ScoringEngine,
    data_dir: Path,
    style: str,
    batch_size: int,
    device: str,
    existing_scores: dict | None = None,
) -> dict[str, dict]:
    """Score adjective pos/neg image pairs.

    Returns ``{word: {pos_score, neg_score, contrastive_pos, contrastive_neg, status}}``.
    """
    task_dir_name = "Adjectives"
    word_list_path = data_dir / "Lexical" / task_dir_name / "word_list.json"
    if not word_list_path.exists():
        logger.warning("%s word list not found: %s", task_dir_name, word_list_path)
        return {}

    with Path(word_list_path).open() as f:
        word_data = json.load(f)

    images_dir = data_dir / "Lexical" / task_dir_name / style

    # Collect items: each word has pos + neg images
    items: list[tuple[str, Path, Path, str, str]] = []
    missing = 0
    skipped_existing = 0

    for word, sentences in word_data["words"].items():
        if existing_scores and word in existing_scores:
            skipped_existing += 1
            continue
        safe_name = word.replace(" ", "_").replace("/", "_")
        word_dir = images_dir / safe_name
        pos_path = word_dir / "pos.png"
        neg_path = word_dir / "neg.png"

        if not pos_path.exists() or not neg_path.exists():
            missing += 1
            continue

        pos_caption = sentences["pos"]
        neg_caption = sentences["neg"]
        items.append((word, pos_path, neg_path, pos_caption, neg_caption))

    if skipped_existing:
        logger.info("%s [%s]: skipping %d already-scored words", task_dir_name, style, skipped_existing)
    if missing:
        logger.warning("%s [%s]: %d words with missing images", task_dir_name, style, missing)

    logger.info(
        "%s [%s]: scoring %d word pairs (batch_size=%d)",
        task_dir_name,
        style,
        len(items),
        batch_size,
    )

    scores = dict(existing_scores) if existing_scores else {}

    # Process in batches -- each item needs 2 images and 2 texts,
    # so we do pairwise scoring per word (2x2 matrix per word).
    # Batch multiple words together for efficiency.
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]

        for word, pos_path, neg_path, pos_cap, neg_cap in batch:
            pos_img = Image.open(pos_path).convert("RGB")
            neg_img = Image.open(neg_path).convert("RGB")

            # 2x2 scoring: [pos_img, neg_img] x [pos_cap, neg_cap]
            score_matrix = score_image_text_matrix(
                engine,
                [pos_img, neg_img],
                [pos_cap, neg_cap],
                device,
            )

            pos_score = score_matrix[0, 0].item()
            neg_score = score_matrix[1, 1].item()
            pos_cap_neg_img = score_matrix[1, 0].item()
            contrastive_pos = score_matrix[0, 0].item() > score_matrix[1, 0].item()
            contrastive_neg = score_matrix[1, 1].item() > score_matrix[0, 1].item()

            scores[word] = {
                "pos_score": round(pos_score, 4),
                "neg_score": round(neg_score, 4),
                "pos_cap_neg_img_score": round(pos_cap_neg_img, 4),
                "pos_caption": pos_cap,
                "neg_caption": neg_cap,
                "contrastive_pos": contrastive_pos,
                "contrastive_neg": contrastive_neg,
            }

        done = min(batch_start + batch_size, len(items))
        if done % 100 < batch_size or done == len(items):
            logger.info("%s [%s]: %d / %d scored", task_dir_name, style, done, len(items))

    return scores


# ---------------------------------------------------------------------------
# Status assignment and summary
# ---------------------------------------------------------------------------


def assign_status_nouns(
    scores: dict[str, dict],
    min_score: float,
    *,
    require_contrastive: bool,
) -> tuple[dict, dict]:
    """Assign pass/fail status to noun scores.  Returns (updated_scores, summary)."""
    total = len(scores)
    failed_alignment = 0
    failed_contrastive = 0
    missing_images = 0

    for info in scores.values():
        if "score" not in info:
            info["status"] = "missing"
            missing_images += 1
            continue

        # Check alignment
        if info["score"] < min_score:
            info["status"] = "fail_alignment"
            failed_alignment += 1
            continue

        # Check contrastive (SigLIP2 can solve the task).
        # For nouns we only check contrastive_pos: given the target caption,
        # does the target image score higher than the random negative?
        # contrastive_neg is irrelevant since the negative is arbitrary.
        if require_contrastive and not info.get("contrastive_pos"):
            info["status"] = "fail_contrastive"
            failed_contrastive += 1
            continue

        info["status"] = "pass"

    passed = total - failed_alignment - failed_contrastive - missing_images
    summary = {
        "total": total,
        "passed": passed,
        "failed_alignment": failed_alignment,
        "failed_contrastive": failed_contrastive,
        "missing_images": missing_images,
    }
    return scores, summary


def assign_status_adj(
    scores: dict[str, dict],
    min_score: float,
    *,
    require_contrastive: bool,
) -> tuple[dict, dict]:
    """Assign pass/fail status to adjective scores.  Returns (updated_scores, summary)."""
    total = len(scores)
    failed_alignment = 0
    failed_contrastive = 0
    missing_images = 0

    for info in scores.values():
        if "pos_score" not in info:
            info["status"] = "missing"
            missing_images += 1
            continue

        # Check alignment
        if info["pos_score"] < min_score or info["neg_score"] < min_score:
            info["status"] = "fail_alignment"
            failed_alignment += 1
            continue

        # Check contrastive
        if require_contrastive and (not info["contrastive_pos"] or not info["contrastive_neg"]):
            info["status"] = "fail_contrastive"
            failed_contrastive += 1
            continue

        info["status"] = "pass"

    passed = total - failed_alignment - failed_contrastive - missing_images
    summary = {
        "total": total,
        "passed": passed,
        "failed_alignment": failed_alignment,
        "failed_contrastive": failed_contrastive,
        "missing_images": missing_images,
    }
    return scores, summary


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_scores(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    scores: dict[str, dict],
    summary: dict,
    output_path: Path,
    model_name: str,
    min_score: float,
    *,
    require_contrastive: bool,
) -> None:
    """Save scores JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed = sorted(w for w, info in scores.items() if info.get("status") != "pass")
    data = {
        "model": model_name,
        "min_score": min_score,
        "require_contrastive": require_contrastive,
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "summary": summary,
        "removed_words": removed,
        "scores": scores,
    }
    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved scores to %s", output_path)


def load_existing_scores(path: Path) -> dict[str, dict] | None:
    """Load existing scores for resume support."""
    if not path.exists():
        return None
    with Path(path).open() as f:
        data = json.load(f)
    logger.info("Loaded %d existing scores from %s", len(data.get("scores", {})), path)
    return data.get("scores", {})


# ---------------------------------------------------------------------------
# Filtered word list writing
# ---------------------------------------------------------------------------


def write_filtered_nouns(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    original_path: Path,
    output_path: Path,
    scores: dict[str, dict],
    model_name: str,
    min_score: float,
    *,
    require_contrastive: bool,
) -> None:
    """Write filtered noun word list, removing failed words."""
    with Path(original_path).open() as f:
        data = json.load(f)

    failed_words = {w for w, info in scores.items() if info.get("status") != "pass"}
    removed_by_category: dict[str, list[str]] = {}

    for category in data["categories"]:
        before = data["categories"][category]
        after = [w for w in before if w not in failed_words]
        removed = [w for w in before if w in failed_words]
        data["categories"][category] = after
        if removed:
            removed_by_category[category] = removed

    # Remove frequency metadata for failed words
    if "frequency_metadata" in data and "word_bins" in data["frequency_metadata"]:
        data["frequency_metadata"]["word_bins"] = {
            w: info for w, info in data["frequency_metadata"]["word_bins"].items() if w not in failed_words
        }

    data["filtering"] = {
        "model": model_name,
        "min_score": min_score,
        "require_contrastive": require_contrastive,
        "total_removed": len(failed_words),
        "removed_by_category": removed_by_category,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote filtered noun word list: %s (%d removed)", output_path, len(failed_words))


def write_filtered_adj(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    original_path: Path,
    output_path: Path,
    scores: dict[str, dict],
    model_name: str,
    min_score: float,
    *,
    require_contrastive: bool,
) -> None:
    """Write filtered adjective word list, removing failed words."""
    with Path(original_path).open() as f:
        data = json.load(f)

    failed_words = {w for w, info in scores.items() if info.get("status") != "pass"}

    data["words"] = {w: sentences for w, sentences in data["words"].items() if w not in failed_words}

    if "frequency_metadata" in data and "word_bins" in data["frequency_metadata"]:
        data["frequency_metadata"]["word_bins"] = {
            w: info for w, info in data["frequency_metadata"]["word_bins"].items() if w not in failed_words
        }

    data["filtering"] = {
        "model": model_name,
        "min_score": min_score,
        "require_contrastive": require_contrastive,
        "total_removed": len(failed_words),
        "removed_words": sorted(failed_words),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote filtered word list: %s (%d removed)", output_path, len(failed_words))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-filter generated images using SigLIP2 alignment scores.",
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
        help="Vision-language model checkpoint (default: facebook/PE-Core-L14-336).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Images per forward pass (default: 64).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Minimum alignment score to pass (default: 0.15). "
        "Calibrated for SigLIP2 sigmoid scores [0,1]; adjust for other backends.",
    )
    parser.add_argument(
        "--require-contrastive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require contrastive check (SigLIP2 can solve the task) for all lexical tasks (default: on).",
    )
    parser.add_argument(
        "--write-filtered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write word_list_filtered.json with failing words removed (default: on).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Load model once
    engine = load_model(args.model, args.device)

    t0 = time.time()

    for task in args.tasks:
        for style in args.styles:
            logger.info("=" * 60)
            logger.info("Processing: %s [style=%s]", task, style)
            logger.info("=" * 60)

            if task == "nouns":
                task_dir = data_dir / "Lexical" / "Nouns"
                scores_path = task_dir / f"siglip2_scores_{style}.json"

                existing = load_existing_scores(scores_path)
                scores = score_nouns(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                    existing,
                )

                if not scores:
                    logger.warning("No noun scores produced for style=%s", style)
                    continue

                scores, summary = assign_status_nouns(
                    scores,
                    args.min_score,
                    require_contrastive=args.require_contrastive,
                )
                save_scores(
                    scores,
                    summary,
                    scores_path,
                    args.model,
                    args.min_score,
                    require_contrastive=args.require_contrastive,
                )

                if args.write_filtered:
                    write_filtered_nouns(
                        task_dir / "word_list.json",
                        task_dir / f"word_list_filtered_{style}.json",
                        scores,
                        args.model,
                        args.min_score,
                        require_contrastive=args.require_contrastive,
                    )

            else:
                task_dir = data_dir / "Lexical" / "Adjectives"
                scores_path = task_dir / f"siglip2_scores_{style}.json"

                existing = load_existing_scores(scores_path)
                scores = score_adj(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                    existing,
                )

                if not scores:
                    logger.warning("No %s scores produced for style=%s", task, style)
                    continue

                scores, summary = assign_status_adj(
                    scores,
                    args.min_score,
                    require_contrastive=args.require_contrastive,
                )
                save_scores(
                    scores,
                    summary,
                    scores_path,
                    args.model,
                    args.min_score,
                    require_contrastive=args.require_contrastive,
                )

                if args.write_filtered:
                    write_filtered_adj(
                        task_dir / "word_list.json",
                        task_dir / f"word_list_filtered_{style}.json",
                        scores,
                        args.model,
                        args.min_score,
                        require_contrastive=args.require_contrastive,
                    )

            # Log summary
            logger.info(
                "Summary [%s, %s]: %d total, %d passed, %d failed_alignment, %d failed_contrastive",
                task,
                style,
                summary["total"],
                summary["passed"],
                summary["failed_alignment"],
                summary.get("failed_contrastive", 0),
            )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
