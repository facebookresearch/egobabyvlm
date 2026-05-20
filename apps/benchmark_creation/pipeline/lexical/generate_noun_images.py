# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate images for the lexical noun task.

Loads word lists from a task directory, generates one image per word using
a text-to-image model, and saves them into ``Lexical/Nouns/{style}/{category}/``.

Supports **batched generation** (multiple prompts per forward pass) and
**multi-GPU parallelism** (one pipeline per GPU) for significant speedups.

Supports resuming: existing images are skipped automatically.

Usage::

    # Single GPU, default settings
    python scripts/02_Create_Lexical/generate_lexical_noun_images.py \
        --data-dir data/coco_20260410_101010 \
        --styles realistic cartoon

    # Multi-GPU with torch.compile
    python scripts/02_Create_Lexical/generate_lexical_noun_images.py \
        --data-dir data/coco_20260410_101010 \
        --num-gpus 4 --batch-size 16 --compile
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import queue as _queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from nltk.corpus import wordnet as wn

from apps.benchmark_creation.pipeline.lexical.constants import (
    ABSTRACT_HYPERNYM_NAMES,
    PERSON_HYPERNYM_NAMES,
)
from apps.benchmark_creation.utils.flux_pipeline import FluxPipeline
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ImageWorkItem:
    """A single image to generate."""

    word: str
    category: str
    description: str
    output_path: Path


# ---------------------------------------------------------------------------
# Noun filtering -- remove person names, abstract concepts, non-visualizable
# ---------------------------------------------------------------------------

# Resolve synset name strings to actual wn.Synset objects
_ABSTRACT_HYPERNYMS = {wn.synset(n) for n in ABSTRACT_HYPERNYM_NAMES}
_PERSON_HYPERNYMS = {wn.synset(n) for n in PERSON_HYPERNYM_NAMES}


def _is_visualizable_noun(word: str) -> bool:
    """Return True if the noun is a concrete, visualizable object (not a
    person's name, abstract concept, or non-depictable thing).

    Uses WordNet hypernym lookup: if ALL synsets for the word lead to an
    abstract or person hypernym, the word is rejected.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False

    for ss in synsets[:3]:
        hypernym_set = set()
        for path in ss.hypernym_paths():
            hypernym_set.update(path)

        # If this synset does NOT have any abstract/person ancestors, keep it
        if not hypernym_set & _ABSTRACT_HYPERNYMS and not hypernym_set & _PERSON_HYPERNYMS:
            return True

    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for lexical noun benchmark.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory containing Lexical/Nouns/word_list.json.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=FluxPipeline.DEFAULT_MODEL,
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic", "cartoon"],
        help="Image styles from configs/styles.yaml (default: realistic cartoon).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to generate per forward pass (default: 16). "
        "Larger batches are faster but use more GPU memory.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in parallel (default: 1). Each GPU runs its own pipeline instance.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile on the transformer for faster inference (default: enabled).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=FluxPipeline.DEFAULT_STEPS,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=FluxPipeline.DEFAULT_GUIDANCE,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=FluxPipeline.DEFAULT_HEIGHT,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=FluxPipeline.DEFAULT_WIDTH,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: generate only the first 10 images per category.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Word list loading
# ---------------------------------------------------------------------------


def load_word_list(data_dir: Path, max_per_category: int | None = None) -> dict[str, list[str]]:
    """Load word_list.json, filter non-visualizable nouns, return {category: [words]}.

    Removes person names and abstract concepts via WordNet hypernym check.
    If *max_per_category* is set, only the first N words per category are kept.
    """
    path = data_dir / "Lexical" / "Nouns" / "word_list.json"
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {path}")
    with Path(path).open() as f:
        data = json.load(f)

    raw_categories = data["categories"]
    filtered: dict[str, list[str]] = {}
    total_removed = 0

    for cat, words in raw_categories.items():
        kept = []
        for w in words:
            if _is_visualizable_noun(w):
                kept.append(w)
            else:
                logger.warning("Filtered non-visualizable noun: '%s' (category: %s)", w, cat)
                total_removed += 1
        if max_per_category is not None:
            kept = kept[:max_per_category]
        if kept:
            filtered[cat] = kept

    total_kept = sum(len(v) for v in filtered.values())
    logger.info(
        "Noun filter: %d kept, %d removed (person names / abstract / non-visualizable)",
        total_kept,
        total_removed,
    )
    return filtered


def build_lexical_prompt(word: str, category: str) -> str:
    """Build a prompt for a single lexical item.

    The prompt asks for a clear, isolated depiction of the word so the
    resulting image is suitable for a forced-choice recognition task.
    The category provides context for ambiguous words (e.g. "bat" in
    animals vs. tools).
    """
    article = "an" if word[0].lower() in "aeiou" else "a"
    cat_label = category.replace("_", " ")
    return f"{article} {word} (from category '{cat_label}')"


# ---------------------------------------------------------------------------
# Work item collection
# ---------------------------------------------------------------------------


def collect_work_items(
    data_dir: Path,
    style: str,
    max_per_category: int | None = None,
) -> list[ImageWorkItem]:
    """Collect all images that need to be generated (skipping existing)."""
    categories = load_word_list(data_dir, max_per_category=max_per_category)
    images_dir = data_dir / "Lexical" / "Nouns" / style

    items = []
    for cat_name, words in sorted(categories.items()):
        cat_dir = images_dir / cat_name
        for word in words:
            safe_name = word.replace(" ", "_").replace("/", "_")
            out_path = cat_dir / f"{safe_name}.png"
            if out_path.exists():
                continue
            items.append(
                ImageWorkItem(
                    word=word,
                    category=cat_name,
                    description=build_lexical_prompt(word, cat_name),
                    output_path=out_path,
                )
            )

    return items


# ---------------------------------------------------------------------------
# Batched generation (single GPU)
# ---------------------------------------------------------------------------


def generate_batched(
    pipe: FluxPipeline,
    items: list[ImageWorkItem],
    style: str,
    batch_size: int,
    gen_kwargs: dict,
) -> dict:
    """Generate images in batches on a single GPU.

    Returns a summary dict with counts.
    """
    total = len(items)
    generated = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        descriptions = [item.description for item in batch]

        try:
            images = pipe.generate_batch(descriptions, style=style, **gen_kwargs)

            for item, image in zip(batch, images, strict=False):
                item.output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(item.output_path)
                generated += 1

        except Exception:
            logger.exception(
                "Batch generation failed (items %d-%d). Falling back to single.",
                batch_start,
                batch_start + len(batch) - 1,
            )
            for item in batch:
                try:
                    item.output_path.parent.mkdir(parents=True, exist_ok=True)
                    pipe.generate_and_save(
                        item.description,
                        item.output_path,
                        style=style,
                        **gen_kwargs,
                    )
                    generated += 1
                except Exception:
                    logger.exception("Failed to generate image for '%s'", item.word)
                    failed += 1

        if generated % 25 < batch_size or batch_start + batch_size >= total:
            logger.info(
                "Progress: %d/%d generated (%d failed)",
                generated,
                total,
                failed,
            )

    return {"total": total, "generated": generated, "failed": failed}


# ---------------------------------------------------------------------------
# Multi-GPU worker (runs in a subprocess -- no GIL contention)
# ---------------------------------------------------------------------------


def _gpu_worker(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    gpu_id: int,
    work_items: list[ImageWorkItem],
    model_id: str,
    *,
    compile_model: bool,
    style: str,
    batch_size: int,
    gen_kwargs: dict,
    results_queue: mp.Queue,
) -> None:
    """Worker process: loads model on its GPU and generates images."""
    setup_logging()
    wlog = logging.getLogger(str(gpu_id))

    pipe = None
    try:
        device = f"cuda:{gpu_id}"
        wlog.info("Loading model on %s (%d items)...", device, len(work_items))
        pipe = FluxPipeline(model_id=model_id, device=device, compile_model=compile_model)
        wlog.info("Model ready on %s", device)

        stats = generate_batched(pipe, work_items, style, batch_size, gen_kwargs)
        stats["gpu_id"] = gpu_id

        wlog.info("Done -- %d generated, %d failed", stats["generated"], stats["failed"])
        results_queue.put(stats)
    except Exception as e:  # noqa: BLE001 -- worker/retry boundary: must catch all errors to keep pipeline alive
        wlog.exception("Worker failed")
        results_queue.put(
            {
                "gpu_id": gpu_id,
                "total": len(work_items),
                "generated": 0,
                "failed": len(work_items),
                "error": str(e),
            }
        )
    finally:
        # Free FLUX VRAM. Just `del pipe` + `empty_cache()` is unreliable
        # because FLUX submodules and torch.compile graphs hold strong
        # refs in the CUDA caching allocator. Move to CPU first, then
        # force-exit so the OS reclaims the CUDA context.
        try:
            if pipe is not None:
                inner = getattr(pipe, "pipe", None)
                if inner is not None and hasattr(inner, "to"):
                    with contextlib.suppress(Exception):
                        inner.to("cpu")
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(gpu_id)
                torch.cuda.empty_cache()
            wlog.info("Released GPU resources on cuda:%d", gpu_id)
        finally:
            # Ensure the result reaches the parent before we exit.
            with contextlib.suppress(Exception):
                results_queue.close()
                results_queue.join_thread()
            # Force-exit so the OS reclaims the CUDA context immediately
            # without waiting for normal Python shutdown, which can hang
            # on cuFFT / NCCL teardown and leave VRAM held.
            os._exit(0)


def _collect_worker_results(
    processes: list,
    results_queue: "mp.Queue",
    expected: int,
    poll_interval: float = 5.0,
) -> list[dict]:
    """Robustly collect one result per worker process, even on worker death.

    A naive ``for _ in processes: results_queue.get()`` blocks forever if a
    worker is OOM-killed, segfaults, or otherwise exits without putting a
    result on the queue. This helper polls ``p.is_alive()`` so that a dead
    worker without a result is detected and a synthesized error stats dict
    is appended in its place.
    """
    received: list[dict] = []
    while len(received) < expected:
        try:
            stats = results_queue.get(timeout=poll_interval)
            received.append(stats)
            continue
        except _queue.Empty:
            pass
        alive = [p for p in processes if p.is_alive()]
        if not alive:
            while True:
                try:
                    received.append(results_queue.get_nowait())
                except _queue.Empty:
                    break
            missing = expected - len(received)
            if missing > 0:
                logger.error(
                    "Multi-GPU: %d worker(s) died without producing results",
                    missing,
                )
                received.extend(
                    {
                        "gpu_id": -1,
                        "total": 0,
                        "generated": 0,
                        "failed": 0,
                        "error": "worker died without producing results",
                    }
                    for _ in range(missing)
                )
            break
    return received


def generate_multi_gpu(  # noqa: C901, PLR0913 -- pipeline-level orchestration: many parallel context fields
    items: list[ImageWorkItem],
    model_id: str,
    *,
    compile_model: bool,
    style: str,
    batch_size: int,
    num_gpus: int,
    gen_kwargs: dict,
) -> dict:
    """Launch one process per GPU. Models load sequentially from page cache.

    Uses multiprocessing (not threads) to avoid Python GIL contention.
    Each process loads the model independently; after the first process reads
    weights from disk, the OS page cache makes subsequent loads near-instant.
    Processes are started one at a time with a short delay to stagger disk I/O.
    """
    available = torch.cuda.device_count()
    num_gpus = min(num_gpus, available, len(items))
    if num_gpus <= 0:
        logger.error("No CUDA GPUs available for multi-GPU generation.")
        return {"total": len(items), "generated": 0, "failed": len(items)}

    logger.info(
        "Multi-GPU: distributing %d items across %d GPUs (batch_size=%d)",
        len(items),
        num_gpus,
        batch_size,
    )

    # Split items across GPUs evenly
    chunks: list[list[ImageWorkItem]] = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        chunks[i % num_gpus].append(item)

    results_queue: mp.Queue = mp.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        logger.info("Starting worker for GPU %d (%d items)...", gpu_id, len(chunks[gpu_id]))
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, chunks[gpu_id], model_id),
            kwargs={
                "compile_model": compile_model,
                "style": style,
                "batch_size": batch_size,
                "gen_kwargs": gen_kwargs,
                "results_queue": results_queue,
            },
        )
        p.start()
        processes.append(p)
        # Stagger launches by 30s so each process loads from warm page cache
        if gpu_id < num_gpus - 1:
            logger.info("Waiting 30s for page cache to warm before next GPU...")
            time.sleep(30)

    all_stats = _collect_worker_results(processes, results_queue, len(processes))

    # Wait for workers to fully exit so the OS reclaims their CUDA
    # contexts before the next pipeline stage tries to allocate GPU
    # memory. Parent-side empty_cache() is a no-op here because the
    # parent never created CUDA contexts on those devices.
    for p in processes:
        p.join(timeout=60.0)
    for p in processes:
        if p.is_alive():
            logger.warning("Worker pid=%s still alive after 60s; terminating", p.pid)
            p.terminate()
            p.join(timeout=10.0)
        if p.is_alive():
            logger.warning("Worker pid=%s still alive after SIGTERM; killing", p.pid)
            p.kill()
            p.join(timeout=5.0)
    try:
        results_queue.close()
        results_queue.join_thread()
    except Exception:  # noqa: S110, BLE001 -- best-effort multiprocessing teardown
        # Best-effort cleanup; nothing actionable on failure.
        pass

    total_generated = sum(s["generated"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)

    return {
        "total": len(items),
        "generated": total_generated,
        "failed": total_failed,
        "per_gpu": all_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 -- pipeline orchestration: complexity matches the spec it implements
    args = parse_args()

    setup_logging()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    gen_kwargs = {}
    if args.num_inference_steps != FluxPipeline.DEFAULT_STEPS:
        gen_kwargs["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale != FluxPipeline.DEFAULT_GUIDANCE:
        gen_kwargs["guidance_scale"] = args.guidance_scale
    if args.height != FluxPipeline.DEFAULT_HEIGHT:
        gen_kwargs["height"] = args.height
    if args.width != FluxPipeline.DEFAULT_WIDTH:
        gen_kwargs["width"] = args.width
    if args.seed is not None:
        gen_kwargs["seed"] = args.seed

    use_multi_gpu = args.num_gpus > 1 and torch.cuda.device_count() > 1
    max_per_category = 10 if args.debug else None

    if args.debug:
        logger.info("DEBUG MODE: generating first 10 images per category")

    # For single-GPU mode, load the pipeline once upfront
    pipe: FluxPipeline | None = None
    if not use_multi_gpu:
        pipe = FluxPipeline(
            model_id=args.model_id,
            compile_model=args.compile,
        )

    t0 = time.time()

    for style in args.styles:
        logger.info("=" * 50)
        logger.info("Generating noun images  [style=%s]", style)
        logger.info("=" * 50)

        items = collect_work_items(data_dir, style, max_per_category=max_per_category)
        total_in_task = sum(len(w) for w in load_word_list(data_dir, max_per_category=max_per_category).values())
        skipped = total_in_task - len(items)

        if not items:
            logger.info("All %d images already exist, skipping.", total_in_task)
            continue

        logger.info(
            "%d images to generate (%d already exist)",
            len(items),
            skipped,
        )

        if use_multi_gpu:
            stats = generate_multi_gpu(
                items,
                args.model_id,
                compile_model=args.compile,
                style=style,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus,
                gen_kwargs=gen_kwargs,
            )
        else:
            assert pipe is not None
            stats = generate_batched(
                pipe,
                items,
                style,
                args.batch_size,
                gen_kwargs,
            )

        logger.info(
            "Done: %d generated, %d failed, %d skipped (existed).",
            stats["generated"],
            stats.get("failed", 0),
            skipped,
        )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes).", elapsed, elapsed / 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
