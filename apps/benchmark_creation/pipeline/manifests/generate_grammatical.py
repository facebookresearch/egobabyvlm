# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate per-category JSON manifests for grammatical evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_categories(
    grammatical_base: str | Path,
    filter_list: list[str] | None = None,
) -> list[str]:
    """Scan for ``gram_*`` directories under the Grammatical base path."""
    base = Path(grammatical_base)
    if not base.is_dir():
        return []

    categories = sorted(
        entry.name.removeprefix("gram_")
        for entry in base.iterdir()
        if entry.is_dir() and entry.name.startswith("gram_")
    )

    if filter_list:
        categories = [c for c in categories if c in filter_list]

    return categories


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sentence_list(cat_dir: str | Path) -> dict:
    """Load sentence_list.json for a grammatical category."""
    path = Path(cat_dir) / "sentence_list.json"
    if not path.is_file():
        raise FileNotFoundError(f"sentence_list.json not found: {path}")
    with path.open() as f:
        return json.load(f)


def load_vlm_scores(cat_dir: str | Path, style: str) -> dict[str, dict] | None:
    """Load VLM scores for a category + style, if available."""
    path = Path(cat_dir) / f"vlm_scores_{style}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    scores = data.get("scores", {})
    logger.info("Loaded %d VLM scores from %s", len(scores), path)
    return scores


# ---------------------------------------------------------------------------
# Manifest metadata
# ---------------------------------------------------------------------------


def build_metadata(
    items: list[dict],
    num_skipped: int,
    total_items: int,
    *,
    vlm_filtered: bool,
    vlm_model: str | None = None,
) -> dict:
    """Build summary metadata for a grammatical manifest."""
    bin_counts = Counter(it.get("freq_bin") for it in items)

    # Sort bins numerically
    sorted_bins = {f"bin_{k}": v for k, v in sorted(bin_counts.items(), key=lambda x: (x[0] is None, x[0]))}

    meta: dict = {
        "num_items": len(items),
        "num_items_in_source": total_items,
        "num_skipped": num_skipped,
        "items_per_freq_bin": sorted_bins,
        "vlm_filtered": vlm_filtered,
    }

    if vlm_model:
        meta["vlm_filter_model"] = vlm_model

    return meta


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def generate_category_manifest(
    data_dir: str | Path,
    category: str,
    style: str,
    *,
    require_vlm_pass: bool = True,
) -> dict:
    """Generate the manifest for a single grammatical category + style.

    Parameters
    ----------
    require_vlm_pass : bool
        When True (default), only include trials that passed VLM filtering.
        When False, include all trials with existing images (ignoring VLM
        scores even if available).
    """
    cat_dir = Path(data_dir) / "Grammatical" / f"gram_{category}"
    imgs_dir = cat_dir / "imgs" / style

    sentence_data = load_sentence_list(cat_dir)
    original_items = sentence_data.get("items", [])
    total_items = len(original_items)

    # Load VLM scores for filtering (gracefully skip when absent)
    vlm_scores = load_vlm_scores(cat_dir, style) if require_vlm_pass else None
    vlm_model = None
    if vlm_scores is not None:
        # Try to get model name from the scores file
        scores_path = cat_dir / f"vlm_scores_{style}.json"
        with scores_path.open() as f:
            vlm_model = json.load(f).get("model")
    elif require_vlm_pass:
        logger.info(
            "%s/%s: no VLM scores found, including all trials with images",
            category,
            style,
        )

    items: list[dict] = []
    skipped = 0

    for idx, item in enumerate(original_items):
        seq_name = f"seq_{idx:02d}"
        seq_dir = imgs_dir / seq_name
        img_0_path = seq_dir / "img_0.png"
        img_1_path = seq_dir / "img_1.png"

        # Check images exist
        if not img_0_path.is_file() or not img_1_path.is_file():
            logger.debug("Missing images for %s/%s", category, seq_name)
            skipped += 1
            continue

        # Apply VLM filter if scores are available
        if vlm_scores is not None:
            score_info = vlm_scores.get(seq_name)
            if score_info is None or score_info.get("status") != "pass":
                skipped += 1
                continue

        # Relative image paths (from cat_dir)
        img_0_rel = str(Path("imgs") / style / seq_name / "img_0.png")
        img_1_rel = str(Path("imgs") / style / seq_name / "img_1.png")

        manifest_item = {
            "seq": seq_name,
            "caption_a": item["caption_a"],
            "caption_b": item["caption_b"],
            "word": item.get("word", ""),
            "image_0": img_0_rel,
            "image_1": img_1_rel,
        }

        if "freq_bin" in item:
            manifest_item["freq_bin"] = item["freq_bin"]

        items.append(manifest_item)

    vlm_filtered = vlm_scores is not None

    logger.info(
        "%s/%s: %d items generated, %d skipped (vlm_filtered=%s)",
        category,
        style,
        len(items),
        skipped,
        vlm_filtered,
    )

    return {
        "description": (f"Grammatical manifest for category '{category}', style '{style}'"),
        "category": category,
        "style": style,
        "metadata": build_metadata(
            items,
            skipped,
            total_items,
            vlm_filtered=vlm_filtered,
            vlm_model=vlm_model,
        ),
        "items": items,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate per-category JSON manifests for grammatical evaluation.",
    )
    parser.add_argument("--data-dir", required=True, help="Root data directory")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Grammatical categories to process (default: all discovered).",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic"],
        help="Image styles to process (default: realistic).",
    )
    parser.add_argument(
        "--no-vlm-filter",
        action="store_true",
        help="Include all trials with images, ignoring VLM scores.",
    )
    args = parser.parse_args()

    setup_logging()

    grammatical_base = Path(args.data_dir) / "Grammatical"
    categories = discover_categories(grammatical_base, args.categories)

    if not categories:
        logger.error("No grammatical categories found in %s", grammatical_base)
        sys.exit(1)

    logger.info("Categories: %s", categories)
    logger.info("Styles:     %s", args.styles)

    for category in categories:
        cat_dir = grammatical_base / f"gram_{category}"

        if not cat_dir.is_dir():
            logger.warning(
                "Skipping category '%s': directory not found (%s)",
                category,
                cat_dir,
            )
            continue

        for style in args.styles:
            logger.info("Generating manifest: %s / %s", category, style)

            manifest = generate_category_manifest(
                args.data_dir,
                category,
                style,
                require_vlm_pass=not args.no_vlm_filter,
            )

            out_path = cat_dir / f"manifest_grammatical_{category}_{style}.json"
            with out_path.open("w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(
                "Wrote %d items to %s",
                len(manifest["items"]),
                out_path,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
