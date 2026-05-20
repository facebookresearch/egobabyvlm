# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Post-filter grammatical images using a VLM (Gemma4) for depiction and distinction checks.

For each grammatical trial (seq_NN), the filter makes three VLM calls:
  1. Does img_0 depict caption_a?
  2. Does img_1 depict caption_b?
  3. (If both pass) Can you tell which image corresponds to which caption?

Trials that fail depiction or distinction are flagged and optionally removed
from the sentence list.

Usage::

    # Score all categories and styles:
    python -m benchmark_creation.pipeline.filtering.post_filter_grammatical \\
        --data-dir data/coco_20260416_121733

    # Score only negation, write filtered sentence list:
    python -m benchmark_creation.pipeline.filtering.post_filter_grammatical \\
        --data-dir data/coco_20260416_121733 \\
        --categories negation \\
        --styles realistic \\
        --write-filtered

Outputs::

    {data_dir}/Grammatical/gram_{category}/
        vlm_scores_{style}.json                    # Per-trial scores
        sentence_list_filtered_{style}.json        # (if --write-filtered)
"""

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.utils.logging import setup_logging

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

setup_logging()
logger = logging.getLogger("post_filter_grammatical")


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with Path(path).open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# VLM call (OpenAI multimodal format)
# ---------------------------------------------------------------------------


async def vlm_call(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    client: "AsyncOpenAI",
    model: str,
    text_prompt: str,
    image_b64_list: list[str],
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> str | None:
    """Multimodal VLM call with retry + exponential backoff.

    Sends one or more base64-encoded images plus a text prompt using the
    OpenAI vision message format.
    """
    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        for b64 in image_b64_list
    ]
    content.append({"type": "text", "text": text_prompt})

    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retries):
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]  # OpenAI's typed dicts don't model multimodal user content well; runtime accepts our shape.
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=180.0,
            )
            response_text = completion.choices[0].message.content
            return response_text.strip() if response_text is not None else None
        except Exception as e:  # noqa: BLE001 -- worker/retry boundary: must catch all errors to keep pipeline alive
            wait_time = 2**attempt
            logger.warning(
                "vlm_call attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                e,
                wait_time,
            )
            await asyncio.sleep(wait_time)
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_yes_no(response: str | None) -> tuple[bool, str]:
    """Extract yes/no and reasoning from a VLM response.

    Expects the first non-empty line to contain "yes" or "no" (case-insensitive,
    matched as a whole word). Returns (answer_bool, reasoning_text). Defaults
    to False if ambiguous.
    """
    if not response:
        return False, ""

    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    if not lines:
        return False, response

    first_line = lines[0].lower()
    reasoning = " ".join(lines[1:]) if len(lines) > 1 else lines[0]

    has_yes = re.search(r"\byes\b", first_line) is not None
    has_no = re.search(r"\bno\b", first_line) is not None
    if has_yes and not has_no:
        return True, reasoning
    if has_no and not has_yes:
        return False, reasoning
    if has_yes and has_no:
        # Both present — disambiguate by which appears first.
        yes_pos = first_line.find("yes")
        no_pos = first_line.find("no")
        return (yes_pos < no_pos), reasoning
    # Ambiguous -- default to False
    return False, reasoning


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DEPICTION_PROMPT = (
    "Does this image depict the general idea of the caption: '{caption}'?\n"
    "Your job is NOT to judge image quality. These are AI-generated images — "
    "visual artifacts, odd anatomy, wrong counts, merged objects, strange "
    "perspectives, and other quirks are completely expected and acceptable.\n"
    "Answer Yes as long as the image is *vaguely about* the right subject "
    "matter. Only answer No if the image is entirely unrelated (e.g., a "
    "landscape when the caption is about animals) or depicts the exact "
    "opposite meaning. Anything in between — even if messy, weird, or only "
    "loosely connected — should be Yes.\n"
    "Answer Yes/No on the first line, then 1-2 sentences why."
)

# Category-specific depiction prompts override the default when present.
DEPICTION_PROMPT_BY_CATEGORY: dict[str, str] = {
    "counting": (
        "Count the objects in this image and compare to the caption: '{caption}'.\n"
        "Ignore everything about the image EXCEPT the number of objects: ignore "
        "object identity, species, action, style, quality, anatomy, background.\n"
        "The count must be EXACT. If the caption says 'three', there must be "
        "exactly 3 distinct objects — not 2, not 4. Partially visible, overlapping, "
        "or merged objects that are clearly meant to be separate count individually.\n"
        "Answer Yes/No on the first line, then state how many objects you see."
    ),
}

DISTINCTION_PROMPT = (
    "Caption A: {caption_a}\n"
    "Caption B: {caption_b}\n\n"
    "The first image is meant to illustrate Caption A and the second Caption B. "
    "Ignore visual quality — focus only on whether the key semantic difference "
    "between the two captions is clearly visible in the images.\n"
    "Be strict: answer Yes ONLY if the distinguishing feature is obvious and "
    "unambiguous. A viewer should be able to match each image to its caption "
    "within seconds, with no doubt. If there is any ambiguity, if the images "
    "look too similar, if the wrong image could plausibly match the wrong "
    "caption, or if the key difference is subtle or hard to spot, answer No.\n"
    "Answer Yes/No on the first line, then 1-2 sentences explaining which "
    "visual cue makes them distinguishable (or why it fails)."
)

# Category-specific distinction prompts override the default when present.
DISTINCTION_PROMPT_BY_CATEGORY: dict[str, str] = {
    "counting": (
        "Caption A: {caption_a}\n"
        "Caption B: {caption_b}\n\n"
        "The first image is meant to illustrate Caption A and the second Caption B. "
        "The captions differ ONLY in the number of objects.\n"
        "Count the objects in each image. Answer Yes ONLY if the first image "
        "has the exact count specified in Caption A AND the second image has the "
        "exact count specified in Caption B.\n"
        "Answer Yes/No on the first line, then state how many objects you count "
        "in each image."
    ),
}


# ---------------------------------------------------------------------------
# Single trial scoring
# ---------------------------------------------------------------------------


async def score_trial(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    client: "AsyncOpenAI",
    model: str,
    img_0_path: Path,
    img_1_path: Path,
    caption_a: str,
    caption_b: str,
    category: str,
    temperature: float,
    max_retries: int,
    sem: asyncio.Semaphore,
    *,
    require_depiction: bool = True,
) -> dict:
    """Score a single grammatical trial with VLM calls.

    When *require_depiction* is True (default), three calls are made:
      1. Does img_0 depict caption_a?
      2. Does img_1 depict caption_b?
      3. (If both pass) Can you tell which image corresponds to which caption?

    When *require_depiction* is False, the depiction checks are skipped and
    only the distinction call (3) is made.

    Returns a dict with keys: caption_a, caption_b, img_0_depicts,
    img_1_depicts, distinguishable, status, reasoning.
    """
    depiction_tpl = DEPICTION_PROMPT_BY_CATEGORY.get(category, DEPICTION_PROMPT)

    async with sem:
        try:
            img_0_b64 = encode_image_base64(img_0_path)
            img_1_b64 = encode_image_base64(img_1_path)
        except Exception as e:  # noqa: BLE001 -- worker/retry boundary: must catch all errors to keep pipeline alive
            logger.warning("Failed to read images: %s", e)
            return {
                "caption_a": caption_a,
                "caption_b": caption_b,
                "img_0_depicts": None,
                "img_1_depicts": None,
                "distinguishable": None,
                "status": "error",
                "reasoning": {"img_0": str(e), "img_1": "", "distinction": ""},
            }

        img_0_depicts: bool | None = None
        img_1_depicts: bool | None = None
        reason_0 = ""
        reason_1 = ""

        # Calls 1+2: depiction checks (concurrent), skipped if not required
        if require_depiction:
            prompt_0 = depiction_tpl.format(caption=caption_a)
            prompt_1 = depiction_tpl.format(caption=caption_b)

            resp_0, resp_1 = await asyncio.gather(
                vlm_call(client, model, prompt_0, [img_0_b64], temperature, max_retries=max_retries),
                vlm_call(client, model, prompt_1, [img_1_b64], temperature, max_retries=max_retries),
            )

            img_0_depicts, reason_0 = parse_yes_no(resp_0)
            img_1_depicts, reason_1 = parse_yes_no(resp_1)

        reasoning = {"img_0": reason_0, "img_1": reason_1, "distinction": ""}

        # If depiction was checked and either failed, no need for distinction call
        if require_depiction and (not img_0_depicts or not img_1_depicts):
            return {
                "caption_a": caption_a,
                "caption_b": caption_b,
                "img_0_depicts": img_0_depicts,
                "img_1_depicts": img_1_depicts,
                "distinguishable": None,
                "status": "fail_depiction",
                "reasoning": reasoning,
            }

        # Call 3: distinction check (both images + both captions)
        distinction_tpl = DISTINCTION_PROMPT_BY_CATEGORY.get(category, DISTINCTION_PROMPT)
        prompt_dist = distinction_tpl.format(caption_a=caption_a, caption_b=caption_b)
        resp_dist = await vlm_call(
            client,
            model,
            prompt_dist,
            [img_0_b64, img_1_b64],
            temperature,
            max_retries=max_retries,
        )

        distinguishable, reason_dist = parse_yes_no(resp_dist)
        reasoning["distinction"] = reason_dist

        status = "pass" if distinguishable else "fail_distinction"
        return {
            "caption_a": caption_a,
            "caption_b": caption_b,
            "img_0_depicts": img_0_depicts,
            "img_1_depicts": img_1_depicts,
            "distinguishable": distinguishable,
            "status": status,
            "reasoning": reasoning,
        }


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------


async def score_category(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    client: "AsyncOpenAI",
    model: str,
    data_dir: Path,
    category: str,
    style: str,
    temperature: float,
    num_workers: int,
    max_retries: int,
    existing_scores: dict | None = None,
    *,
    require_depiction: bool = True,
) -> dict[str, dict]:
    """Score all trials for a grammatical category + style.

    Discovers ``seq_NN`` directories, skips already-scored trials (resume),
    and runs scoring with bounded concurrency.
    """
    imgs_dir = data_dir / "Grammatical" / f"gram_{category}" / "imgs" / style

    if not imgs_dir.exists():
        logger.warning("Images directory not found: %s", imgs_dir)
        return dict(existing_scores) if existing_scores else {}

    # Discover seq_NN directories
    seq_dirs = sorted(d for d in imgs_dir.iterdir() if d.is_dir() and d.name.startswith("seq_"))

    if not seq_dirs:
        logger.warning("No seq_* directories found in %s", imgs_dir)
        return dict(existing_scores) if existing_scores else {}

    scores = dict(existing_scores) if existing_scores else {}
    sem = asyncio.Semaphore(num_workers)

    tasks = []
    skipped = 0
    missing = 0

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name

        # Skip already-scored
        if seq_name in scores:
            skipped += 1
            continue

        metadata_path = seq_dir / "metadata.json"
        img_0_path = seq_dir / "img_0.png"
        img_1_path = seq_dir / "img_1.png"

        if not metadata_path.exists() or not img_0_path.exists() or not img_1_path.exists():
            missing += 1
            continue

        with Path(metadata_path).open() as f:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
            metadata = json.load(f)

        caption_a = metadata["caption_a"]
        caption_b = metadata["caption_b"]

        tasks.append(
            (
                seq_name,
                score_trial(
                    client,
                    model,
                    img_0_path,
                    img_1_path,
                    caption_a,
                    caption_b,
                    category,
                    temperature,
                    max_retries,
                    sem,
                    require_depiction=require_depiction,
                ),
            )
        )

    if skipped:
        logger.info("%s [%s]: skipping %d already-scored trials", category, style, skipped)
    if missing:
        logger.warning("%s [%s]: %d trials with missing files", category, style, missing)

    logger.info(
        "%s [%s]: scoring %d trials (workers=%d)",
        category,
        style,
        len(tasks),
        num_workers,
    )

    if not tasks:
        return scores

    # Run all trial tasks concurrently (bounded by semaphore)
    results = await asyncio.gather(*(t[1] for t in tasks))

    for (seq_name, _), result in zip(tasks, results, strict=False):
        scores[seq_name] = result

    return scores


def discover_categories(
    grammatical_base: Path,
    filter_list: list[str] | None = None,
) -> list[str]:
    """Scan for ``gram_*`` directories under the Grammatical base path.

    If *filter_list* is given, only return categories in that list.
    """
    if not grammatical_base.exists():
        return []

    categories = sorted(
        d.name.removeprefix("gram_") for d in grammatical_base.iterdir() if d.is_dir() and d.name.startswith("gram_")
    )

    if filter_list:
        categories = [c for c in categories if c in filter_list]

    return categories


# ---------------------------------------------------------------------------
# Summary and saving
# ---------------------------------------------------------------------------


def compute_summary(scores: dict[str, dict]) -> dict:
    """Count total / passed / failed_depiction / failed_distinction / error."""
    total = len(scores)
    passed = sum(1 for s in scores.values() if s.get("status") == "pass")
    failed_depiction = sum(1 for s in scores.values() if s.get("status") == "fail_depiction")
    failed_distinction = sum(1 for s in scores.values() if s.get("status") == "fail_distinction")
    errors = sum(1 for s in scores.values() if s.get("status") == "error")
    return {
        "total": total,
        "passed": passed,
        "failed_depiction": failed_depiction,
        "failed_distinction": failed_distinction,
        "errors": errors,
    }


def save_scores(
    scores: dict[str, dict],
    summary: dict,
    output_path: Path,
    model_name: str,
) -> None:
    """Write VLM scores JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_name,
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "summary": summary,
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
# Filtered sentence list writing
# ---------------------------------------------------------------------------


def write_filtered_sentence_list(
    original_path: Path,
    output_path: Path,
    scores: dict[str, dict],
    model_name: str,
) -> None:
    """Remove failing trials from sentence_list.json.

    Writes ``sentence_list_filtered_{style}.json`` with only passing items.
    """
    with Path(original_path).open() as f:
        data = json.load(f)

    # Build set of failing trial indices from seq_NN keys
    failed_indices: set[int] = set()
    for seq_name, info in scores.items():
        if info.get("status") != "pass":
            # seq_name is e.g. "seq_00" -> index 0
            try:
                idx = int(seq_name.split("_", 1)[1])
                failed_indices.add(idx)
            except (ValueError, IndexError):
                continue

    original_items = data.get("items", [])
    filtered_items = [item for i, item in enumerate(original_items) if i not in failed_indices]

    data["items"] = filtered_items
    data["filtering"] = {
        "model": model_name,
        "original_count": len(original_items),
        "filtered_count": len(filtered_items),
        "removed_count": len(original_items) - len(filtered_items),
        "removed_indices": sorted(failed_indices),
    }

    # Update metadata count if present
    if "metadata" in data:
        data["metadata"]["num_items"] = len(filtered_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote filtered sentence list: %s (%d -> %d items)",
        output_path,
        len(original_items),
        len(filtered_items),
    )


def save_top_level_summary(
    grammatical_base: Path,
    all_summaries: dict[str, dict[str, dict]],
    model_name: str,
) -> None:
    """Write a top-level summary JSON at ``Grammatical/vlm_filter_summary.json``.

    *all_summaries* maps ``category -> style -> summary_dict``.
    The file contains per-category/style breakdowns and an aggregated total.
    """
    # Aggregate across all categories and styles
    agg = {"total": 0, "passed": 0, "failed_depiction": 0, "failed_distinction": 0, "errors": 0}
    for cat_styles in all_summaries.values():
        for summary in cat_styles.values():
            for key in agg:
                agg[key] += summary.get(key, 0)

    output = {
        "model": model_name,
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "aggregate": agg,
        "categories": all_summaries,
    }

    out_path = grammatical_base / "vlm_filter_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved top-level summary to %s", out_path)


# ---------------------------------------------------------------------------
# Manifest creation (filtered)
# ---------------------------------------------------------------------------


def write_manifest(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    sentence_list_path: Path,
    cat_dir: Path,
    category: str,
    style: str,
    scores: dict[str, dict],
    model_name: str,
) -> None:
    """Create ``manifest_grammatical_{category}_{style}.json`` with only passing trials.

    Each manifest item records the original ``seq_NN`` index so that image
    paths stay correct even after failed trials are removed.
    """
    with Path(sentence_list_path).open() as f:
        data = json.load(f)

    original_items = data.get("items", [])
    imgs_rel = Path("imgs") / style  # relative to cat_dir

    passing_items: list[dict] = []
    for seq_name, info in sorted(scores.items()):
        if info.get("status") != "pass":
            continue
        try:
            idx = int(seq_name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if idx >= len(original_items):
            continue

        item = dict(original_items[idx])
        item["seq"] = seq_name
        item["image_0"] = str(imgs_rel / seq_name / "img_0.png")
        item["image_1"] = str(imgs_rel / seq_name / "img_1.png")
        passing_items.append(item)

    manifest = {
        "description": (
            f"Grammatical manifest for category '{category}', style '{style}'. "
            f"Contains only trials that passed VLM post-filtering."
        ),
        "category": category,
        "style": style,
        "metadata": {
            "num_items": len(passing_items),
            "num_original": len(original_items),
            "num_scored": len(scores),
            "num_removed": len(scores) - len(passing_items),
            "filter_model": model_name,
        },
        "items": passing_items,
    }

    out_path = cat_dir / f"manifest_grammatical_{category}_{style}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote manifest: %s (%d items from %d)",
        out_path,
        len(passing_items),
        len(original_items),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-filter grammatical images using a VLM (Gemma4).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory (e.g., data/coco_20260416_121733).",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic", "cartoon"],
        help="Image styles to score (default: realistic cartoon).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Grammatical categories to process (default: all discovered).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
        help="VLM model for vLLM (default: google/gemma-4-26B-A4B-it).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="VLM sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Max concurrent VLM calls (default: 20).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per VLM call (default: 3).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max model length (default: 4096).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization (default: 0.90).",
    )
    parser.add_argument(
        "--write-filtered",
        action="store_true",
        help="Write sentence_list_filtered_{style}.json with failing trials removed.",
    )
    parser.add_argument(
        "--require-depiction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run depiction checks before the distinction check (default: on). "
        "Use --no-require-depiction to skip depiction and only run distinction.",
    )
    parser.add_argument(
        "--skip-server-launch",
        action="store_true",
        help="Assume vLLM is already running (skip launch/shutdown).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score all trials from scratch, ignoring any existing scores.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901, PLR0912, PLR0915 -- pipeline orchestration: complexity matches the spec it implements
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    # Resolve model name for the served endpoint
    model_served_name = Path(args.model).name

    # Optionally launch vLLM
    vllm_proc = None
    if not args.skip_server_launch:
        from apps.benchmark_creation.utils.vllm_server import launch_local, wait_for_server

        vllm_proc = launch_local(
            model=args.model,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_args=["--limit-mm-per-prompt", '{"image": 2}'],
            log_file=str(data_dir / "vllm_grammatical_filter.log"),
        )

    try:
        if vllm_proc is not None and not wait_for_server(
            "localhost",
            args.port,
            timeout=600,
            proc=vllm_proc,
        ):
            logger.error("vLLM server failed to start")
            sys.exit(1)

        from apps.benchmark_creation.utils.vllm_server import get_client

        client = get_client("localhost", args.port)

        grammatical_base = data_dir / "Grammatical"
        categories = discover_categories(grammatical_base, args.categories)

        if not categories:
            logger.error("No grammatical categories found in %s", grammatical_base)
            sys.exit(1)

        logger.info("Categories to process: %s", categories)

        t0 = time.time()
        all_summaries: dict[str, dict[str, dict]] = {}

        for category in categories:
            cat_dir = grammatical_base / f"gram_{category}"

            for style in args.styles:
                logger.info("=" * 60)
                logger.info("Processing: %s [style=%s]", category, style)
                logger.info("=" * 60)

                scores_path = cat_dir / f"vlm_scores_{style}.json"
                existing = None if args.force else load_existing_scores(scores_path)

                scores = asyncio.run(
                    score_category(
                        client,
                        model_served_name,
                        data_dir,
                        category,
                        style,
                        args.temperature,
                        args.num_workers,
                        args.max_retries,
                        existing,
                        require_depiction=args.require_depiction,
                    )
                )

                if not scores:
                    logger.warning("No scores produced for %s [%s]", category, style)
                    continue

                summary = compute_summary(scores)
                save_scores(scores, summary, scores_path, model_served_name)

                # Collect for top-level summary
                all_summaries.setdefault(category, {})[style] = summary

                if args.write_filtered:
                    sentence_list_path = cat_dir / "sentence_list.json"
                    if sentence_list_path.exists():
                        write_filtered_sentence_list(
                            sentence_list_path,
                            cat_dir / f"sentence_list_filtered_{style}.json",
                            scores,
                            model_served_name,
                        )
                        write_manifest(
                            sentence_list_path,
                            cat_dir,
                            category,
                            style,
                            scores,
                            model_served_name,
                        )
                    else:
                        logger.warning("sentence_list.json not found: %s", sentence_list_path)

                logger.info(
                    "Summary [%s, %s]: %d total, %d passed, %d fail_depiction, %d fail_distinction, %d errors",
                    category,
                    style,
                    summary["total"],
                    summary["passed"],
                    summary["failed_depiction"],
                    summary["failed_distinction"],
                    summary["errors"],
                )

        # Write top-level summary across all categories and styles
        if all_summaries:
            save_top_level_summary(grammatical_base, all_summaries, model_served_name)

        elapsed = time.time() - t0
        logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    finally:
        if vllm_proc is not None:
            logger.info("Shutting down vLLM server...")
            try:
                vllm_proc.terminate()
                vllm_proc.wait(timeout=30)
            except Exception:  # noqa: BLE001 -- worker/retry boundary: must catch all errors to keep pipeline alive
                vllm_proc.kill()
            # Close the log file handle attached by launch_local() to avoid
            # leaking a file descriptor on every invocation.
            log_fh = getattr(vllm_proc, "_log_fh", None)
            if log_fh is not None:
                with contextlib.suppress(Exception):
                    log_fh.close()


if __name__ == "__main__":
    main()
