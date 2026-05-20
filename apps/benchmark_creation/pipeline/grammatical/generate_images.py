# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate images for the Grammatical 2-AFC benchmark.

Loads ``gram_{category}/sentence_list.json`` for each grammatical category and
generates 2 images per trial: one for ``caption_a`` and one for ``caption_b``.

Prompts are *contrastive*: each image prompt includes the other caption as
"avoid" guidance so the generated image clearly depicts only its target.

Directory layout::

    {data_dir}/Grammatical/gram_{category}/imgs/{style}/seq_00/
        metadata.json   — caption_a, caption_b, word, freq_bin, prompts
        img_0.png       — image for caption_a
        img_1.png       — image for caption_b

Supports **batched generation**, **multi-GPU parallelism**, and **resume**.

Usage::

    python scripts/03_Create_Grammatical/generate_grammatical_images.py \\
        --data-dir data/coco_20260413_120000 \\
        --styles realistic cartoon
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

from apps.benchmark_creation.pipeline.grammatical.rewriters import build_contrastive_prompt
from apps.benchmark_creation.utils.flux_pipeline import FluxPipeline
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ImageWorkItem:
    """A single image to generate."""

    prompt: str
    category: str
    caption_index: int
    image_index: int
    output_path: Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for the Grammatical 2-AFC benchmark.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory containing Grammatical/gram_{category}/sentence_list.json files.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=FluxPipeline.DEFAULT_MODEL,
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=["realistic", "cartoon"],
        help="Image style(s) from configs/styles.yaml (default: realistic cartoon).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images to generate per forward pass (default: 16).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use in parallel (default: 1).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile on the transformer (default: enabled).",
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
        "--max-images",
        type=int,
        default=None,
        help="Limit total images to generate (for quick testing). Default: all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: only generate the first 5 trials per category.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force regeneration: overwrite existing metadata.json and images.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sentence list loading
# ---------------------------------------------------------------------------


def load_2afc_data(data_dir: Path) -> dict[str, list[dict]]:
    """Load per-category sentence lists from gram_{category}/sentence_list.json.

    Returns {category: [{"caption_a": str, "caption_b": str,
                          "word": str|None, "freq_bin": int|None}, ...]}.
    """
    grammatical_base = data_dir / "Grammatical"
    categories: dict[str, list[dict]] = {}

    if not grammatical_base.exists():
        raise FileNotFoundError(f"Grammatical directory not found: {grammatical_base}")

    for cat_dir in sorted(grammatical_base.iterdir()):
        if not cat_dir.is_dir() or not cat_dir.name.startswith("gram_"):
            continue
        sl_path = cat_dir / "sentence_list.json"
        if not sl_path.exists():
            continue
        cat_name = cat_dir.name[len("gram_") :]  # strip "gram_" prefix
        with Path(sl_path).open() as f:
            data = json.load(f)
        categories[cat_name] = data.get("items", [])

    if not categories:
        raise FileNotFoundError(f"No gram_*/sentence_list.json files found in {grammatical_base}")

    return categories


# ---------------------------------------------------------------------------
# Directory setup & work item collection
# ---------------------------------------------------------------------------


def setup_sequence_dirs(data_dir: Path, style: str, *, force: bool = False) -> None:
    """Create sequence directories and write metadata.json files.

    metadata.json is always rewritten so it stays in sync with the current
    sentence_list.json contents. If the sentence list changes between runs,
    a stale metadata.json would otherwise misrepresent which prompt produced
    the on-disk images.
    """
    # `force` is accepted for API compatibility but no longer affects whether
    # metadata.json is rewritten — it is always rewritten.
    del force
    categories = load_2afc_data(data_dir)
    grammatical_base = data_dir / "Grammatical"

    for cat_name, items in sorted(categories.items()):
        for idx, item in enumerate(items):
            seq_dir = grammatical_base / f"gram_{cat_name}" / "imgs" / style / f"seq_{idx:02d}"
            seq_dir.mkdir(parents=True, exist_ok=True)
            meta_file = seq_dir / "metadata.json"
            caption_a = item["caption_a"]
            caption_b = item["caption_b"]
            item_antonym = item.get("antonym")
            meta = {
                "caption_a": caption_a,
                "caption_b": caption_b,
                "word": item.get("word"),
                "freq_bin": item.get("freq_bin"),
                "prompts": {
                    "img_0": build_contrastive_prompt(
                        caption_a,
                        caption_b,
                        0,
                        category=cat_name,
                        antonym=item_antonym,
                    ),
                    "img_1": build_contrastive_prompt(
                        caption_a,
                        caption_b,
                        1,
                        category=cat_name,
                        antonym=item_antonym,
                    ),
                },
            }
            with Path(meta_file).open("w") as f:
                json.dump(meta, f, indent=2)


def collect_work_items(
    data_dir: Path,
    style: str,
    max_per_category: int | None = None,
    *,
    force: bool = False,
) -> list[ImageWorkItem]:
    """Collect all images (caption_a + caption_b) that need to be generated."""
    categories = load_2afc_data(data_dir)
    grammatical_base = data_dir / "Grammatical"

    items = []
    for cat_name, cat_items in sorted(categories.items()):
        cap = max_per_category if max_per_category is not None else len(cat_items)
        for idx, entry in enumerate(cat_items[:cap]):
            seq_dir = grammatical_base / f"gram_{cat_name}" / "imgs" / style / f"seq_{idx:02d}"
            caption_a = entry["caption_a"]
            caption_b = entry["caption_b"]
            entry_antonym = entry.get("antonym")

            # img_0 = caption_a
            out_path = seq_dir / "img_0.png"
            if force or not out_path.exists():
                items.append(
                    ImageWorkItem(
                        prompt=build_contrastive_prompt(
                            caption_a,
                            caption_b,
                            0,
                            category=cat_name,
                            antonym=entry_antonym,
                        ),
                        category=cat_name,
                        caption_index=idx,
                        image_index=0,
                        output_path=out_path,
                    )
                )

            # img_1 = caption_b
            out_path = seq_dir / "img_1.png"
            if force or not out_path.exists():
                items.append(
                    ImageWorkItem(
                        prompt=build_contrastive_prompt(
                            caption_a,
                            caption_b,
                            1,
                            category=cat_name,
                            antonym=entry_antonym,
                        ),
                        category=cat_name,
                        caption_index=idx,
                        image_index=1,
                        output_path=out_path,
                    )
                )

    return items


def count_total_images(
    data_dir: Path,
    max_per_category: int | None = None,
) -> int:
    """Count total images to generate (2 per trial: one for each caption)."""
    categories = load_2afc_data(data_dir)
    total = 0
    for cat_items in categories.values():
        cap = max_per_category if max_per_category is not None else len(cat_items)
        total += min(cap, len(cat_items)) * 2
    return total


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
    """Generate images in batches on a single GPU."""
    total = len(items)
    generated = 0
    failed = 0

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        descriptions = [item.prompt for item in batch]

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
                        item.prompt,
                        item.output_path,
                        style=style,
                        **gen_kwargs,
                    )
                    generated += 1
                except Exception:
                    logger.exception(
                        "Failed to generate image for '%s'",
                        item.prompt[:60],
                    )
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
# Multi-GPU worker
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

        wlog.info("Done - %d generated, %d failed", stats["generated"], stats["failed"])
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
        # No result arrived in this interval — check whether any workers died.
        alive = [p for p in processes if p.is_alive()]
        if not alive:
            # Drain anything still queued (a worker may have produced a result
            # right before exiting).
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
    """Launch one process per GPU."""
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
    except Exception:  # noqa: S110, BLE001 -- best-effort multiprocessing teardown; nothing actionable on failure.
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915 -- pipeline orchestration: complexity matches the spec it implements
    args = parse_args()

    setup_logging()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    grammatical_base = data_dir / "Grammatical"
    gram_dirs = (
        [
            d
            for d in grammatical_base.iterdir()
            if d.is_dir() and d.name.startswith("gram_") and (d / "sentence_list.json").exists()
        ]
        if grammatical_base.exists()
        else []
    )
    if not gram_dirs:
        logger.error(
            "No gram_*/sentence_list.json files found in %s",
            grammatical_base,
        )
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

    max_per_cat = 5 if args.debug else None
    if args.debug:
        logger.info("DEBUG MODE: limiting to 5 trials per category.")
    if args.force:
        logger.info("FORCE MODE: overwriting existing metadata and images.")

    # --- Load model once (reused across all styles) ---
    pipe: FluxPipeline | None = None
    if not use_multi_gpu:
        pipe = FluxPipeline(
            model_id=args.model_id,
            compile_model=args.compile,
        )

    for style in args.styles:
        # --- Setup sequence directories and write metadata.json files ---
        logger.info("Setting up sequence directories for style '%s'...", style)
        setup_sequence_dirs(data_dir, style, force=args.force)

        # --- Collect all images to generate ---
        items = collect_work_items(
            data_dir,
            style,
            max_per_category=max_per_cat,
            force=args.force,
        )
        total_images = count_total_images(data_dir, max_per_category=max_per_cat)
        skipped = total_images - len(items)

        if args.max_images is not None and len(items) > args.max_images:
            logger.info(
                "Limiting to %d images (out of %d pending).",
                args.max_images,
                len(items),
            )
            items = items[: args.max_images]
            # Recompute skipped to reflect what is actually being processed.
            skipped = total_images - len(items)

        logger.info("=" * 50)
        logger.info("Generating Grammatical 2-AFC images  [style=%s]", style)
        logger.info("=" * 50)

        if not items:
            logger.info("All %d images already exist, nothing to do.", total_images)
        else:
            n_a = sum(1 for it in items if it.image_index == 0)
            n_b = sum(1 for it in items if it.image_index == 1)
            logger.info(
                "%d images to generate (%d caption_a, %d caption_b, %d not processed this run)",
                len(items),
                n_a,
                n_b,
                skipped,
            )

            t0 = time.time()

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

            elapsed = time.time() - t0
            logger.info(
                "Done: %d generated, %d failed, %d skipped (existed). Time: %.1fs",
                stats["generated"],
                stats.get("failed", 0),
                skipped,
                elapsed,
            )

        logger.info(
            "Output structure: %s/Grammatical/gram_{category}/imgs/%s/seq_NN/",
            data_dir,
            style,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
