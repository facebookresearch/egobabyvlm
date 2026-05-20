# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hard post-filter: exclude all examples that SigLIP2 cannot solve.

Unlike the standard ``post_filter_lexical`` (which uses alignment thresholds
and contrastive checks), this filter **only** keeps examples that SigLIP2
correctly solves -- i.e., where the model assigns a higher score to the
positive image than the negative for the positive caption.  No alignment
threshold is applied.

Usage::

    # Score all tasks and styles:
    python -m benchmark_creation.pipeline.filtering.post_filter_lexical_hard \\
        --data-dir data/coco_20260416_121733

    # Score only nouns, realistic style:
    python -m benchmark_creation.pipeline.filtering.post_filter_lexical_hard \\
        --data-dir data/coco_20260416_121733 \\
        --tasks nouns \\
        --styles realistic

Outputs::

    {data_dir}/Lexical/{Nouns|Adjectives}/
        siglip2_scores_hard_{style}.json      # Per-word scores and pass/fail
        word_list_filtered_hard_{style}.json   # Filtered word list
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from apps.benchmark_creation.utils.vision_scoring import load_model
from core.utils.logging import setup_logging

from .post_filter_lexical import (
    save_scores,
    score_adj,
    score_nouns,
    write_filtered_adj,
    write_filtered_nouns,
)

setup_logging()
logger = logging.getLogger("post_filter_lexical_hard")


# ---------------------------------------------------------------------------
# Hard status assignment -- purely contrastive, no alignment threshold
# ---------------------------------------------------------------------------


def assign_status_nouns_hard(
    scores: dict[str, dict],
    margin: float = 0.0,
) -> tuple[dict, dict]:
    """Assign pass/fail based on whether SigLIP2 solves each noun example.

    An example passes only if the positive image scores higher than the
    negative image (for the positive caption) by at least *margin*:
    ``score - pos_cap_neg_img_score >= margin``.
    """
    total = len(scores)
    failed = 0
    missing = 0

    for info in scores.values():
        if "score" not in info:
            info["status"] = "missing"
            missing += 1
            continue

        pos_score = info["score"]
        neg_img_score = info.get("pos_cap_neg_img_score", 0.0)
        if pos_score - neg_img_score < margin:
            info["status"] = "fail_contrastive"
            failed += 1
            continue

        info["status"] = "pass"

    passed = total - failed - missing
    summary = {
        "total": total,
        "passed": passed,
        "failed_contrastive": failed,
        "missing_images": missing,
    }
    return scores, summary


def assign_status_adj_hard(
    scores: dict[str, dict],
    margin: float = 0.0,
) -> tuple[dict, dict]:
    """Assign pass/fail based on whether SigLIP2 solves each adjective example.

    An example passes only if the positive image scores higher than the
    negative image (for the positive caption) by at least *margin*:
    ``pos_score - pos_cap_neg_img_score >= margin``.
    """
    total = len(scores)
    failed = 0
    missing = 0

    for info in scores.values():
        if "pos_score" not in info:
            info["status"] = "missing"
            missing += 1
            continue

        pos_score = info["pos_score"]
        neg_img_score = info.get("pos_cap_neg_img_score", 0.0)
        if pos_score - neg_img_score < margin:
            info["status"] = "fail_contrastive"
            failed += 1
            continue

        info["status"] = "pass"

    passed = total - failed - missing
    summary = {
        "total": total,
        "passed": passed,
        "failed_contrastive": failed,
        "missing_images": missing,
    }
    return scores, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard post-filter: keep only examples SigLIP2 can solve.",
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
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Minimum score margin: sim(caption, pos_img) - sim(caption, neg_img) "
        "must be >= margin to pass (default: 0.0).",
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

    # Load model once.
    engine = load_model(args.model, args.device)

    t0 = time.time()

    for task in args.tasks:
        for style in args.styles:
            logger.info("=" * 60)
            logger.info("Hard filter: %s [style=%s]", task, style)
            logger.info("=" * 60)

            if task == "nouns":
                task_dir = data_dir / "Lexical" / "Nouns"
                hard_scores_path = task_dir / f"siglip2_scores_hard_{style}.json"

                scores = score_nouns(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                )

                if not scores:
                    logger.warning("No noun scores for style=%s", style)
                    continue

                scores, summary = assign_status_nouns_hard(scores, args.margin)
                save_scores(
                    scores,
                    summary,
                    hard_scores_path,
                    args.model,
                    0.0,
                    require_contrastive=False,
                )
                write_filtered_nouns(
                    task_dir / "word_list.json",
                    task_dir / f"word_list_filtered_hard_{style}.json",
                    scores,
                    args.model,
                    0.0,
                    require_contrastive=False,
                )

            else:
                task_dir = data_dir / "Lexical" / "Adjectives"
                hard_scores_path = task_dir / f"siglip2_scores_hard_{style}.json"

                scores = score_adj(
                    engine,
                    data_dir,
                    style,
                    args.batch_size,
                    args.device,
                )

                if not scores:
                    logger.warning("No %s scores for style=%s", task, style)
                    continue

                scores, summary = assign_status_adj_hard(scores, args.margin)
                save_scores(
                    scores,
                    summary,
                    hard_scores_path,
                    args.model,
                    0.0,
                    require_contrastive=False,
                )
                write_filtered_adj(
                    task_dir / "word_list.json",
                    task_dir / f"word_list_filtered_hard_{style}.json",
                    scores,
                    args.model,
                    0.0,
                    require_contrastive=False,
                )

            logger.info(
                "Summary [%s, %s]: %d total, %d passed, %d failed_contrastive",
                task,
                style,
                summary["total"],
                summary["passed"],
                summary["failed_contrastive"],
            )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
