# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate images for the lexical adjective task.

Loads ``Lexical/Adjectives/word_list.json`` (which contains word->phrase mappings
produced by ``build_adjectives.py``) and generates images for
each adjective using a text-to-image model.

Usage::

    python scripts/02_Create_Lexical/generate_lexical_adj_imgs.py \
        --data-dir data/coco_20260416_121733 \
        --styles realistic cartoon

    # Debug mode (first 50 images only):
    python scripts/02_Create_Lexical/generate_lexical_adj_imgs.py \
        --data-dir data/coco_20260416_121733 \
        --debug
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import queue as _queue
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

from apps.benchmark_creation.utils.flux_pipeline import FluxPipeline
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdjImageItem:
    """A single adjective image to generate."""

    adjective: str
    phrase: str
    output_path: Path
    polarity: str  # "pos" or "neg"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images for the lexical adjective task.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset directory containing Lexical/Adjectives/word_list.json.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=FluxPipeline.DEFAULT_MODEL,
        help="Image generation model ID or local path.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["realistic", "cartoon"],
        help="Image styles (default: realistic cartoon).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per forward pass (default: 16).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for image generation (default: 1).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile (default: enabled).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=FluxPipeline.DEFAULT_STEPS,
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=FluxPipeline.DEFAULT_GUIDANCE,
    )
    parser.add_argument("--height", type=int, default=FluxPipeline.DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=FluxPipeline.DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only the first 50 adjectives.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Word list loading
# ---------------------------------------------------------------------------


def load_adjective_list(data_dir: Path) -> tuple[dict[str, dict[str, str]], dict]:
    """Load word_list.json and return ({adjective: {"pos": ..., "neg": ...}}, full_data)."""
    path = data_dir / "Lexical" / "Adjectives" / "word_list.json"
    if not path.exists():
        raise FileNotFoundError(f"Adjective word list not found: {path}")
    with Path(path).open() as f:
        data = json.load(f)
    return data["words"], data


# ---------------------------------------------------------------------------
# Work item collection
# ---------------------------------------------------------------------------


def collect_work_items(
    adjs_with_phrases: dict[str, dict[str, str]],
    data_dir: Path,
    style: str,
) -> tuple[list[AdjImageItem], int]:
    """Collect images to generate. Returns (items, skipped).

    Each adjective produces two items (pos and neg) in a per-word subdirectory:
      style/word/pos.png
      style/word/neg.png
    """
    images_dir = data_dir / "Lexical" / "Adjectives" / style
    items: list[AdjImageItem] = []
    skipped = 0

    for adj, phrases in adjs_with_phrases.items():
        safe_name = adj.replace(" ", "_").replace("/", "_")
        word_dir = images_dir / safe_name

        for polarity in ("pos", "neg"):
            out_path = word_dir / f"{polarity}.png"
            if out_path.exists():
                skipped += 1
                continue
            items.append(
                AdjImageItem(
                    adjective=adj,
                    phrase=phrases[polarity],
                    output_path=out_path,
                    polarity=polarity,
                )
            )

    return items, skipped


# ---------------------------------------------------------------------------
# Meta JSON writing
# ---------------------------------------------------------------------------


def write_meta_json(
    adjs_with_phrases: dict[str, dict[str, str]],
    full_data: dict,
    data_dir: Path,
    style: str,
) -> None:
    """Write a meta.json per word subdirectory with captions and bin info."""
    images_dir = data_dir / "Lexical" / "Adjectives" / style
    word_bins = full_data.get("frequency_metadata", {}).get("word_bins", {})

    for adj, phrases in adjs_with_phrases.items():
        safe_name = adj.replace(" ", "_").replace("/", "_")
        word_dir = images_dir / safe_name
        word_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "word": adj,
            "pos_caption": phrases["pos"],
            "neg_caption": phrases["neg"],
            "bin_info": word_bins.get(adj, {}),
        }
        meta_path = word_dir / "meta.json"
        with Path(meta_path).open("w") as f:
            json.dump(meta, f, indent=2)

    logger.info("Wrote meta.json for %d adjectives [style=%s]", len(adjs_with_phrases), style)


# ---------------------------------------------------------------------------
# Batched generation (single GPU)
# ---------------------------------------------------------------------------


def generate_batched(
    pipe: FluxPipeline,
    items: list[AdjImageItem],
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
        prompts = [item.phrase for item in batch]

        try:
            images = pipe.generate_batch(prompts, style=style, **gen_kwargs)
            for item, image in zip(batch, images, strict=False):
                item.output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(item.output_path)
                generated += 1
        except Exception:
            logger.exception(
                "Batch failed (items %d-%d). Falling back to single.",
                batch_start,
                batch_start + len(batch) - 1,
            )
            for item in batch:
                try:
                    item.output_path.parent.mkdir(parents=True, exist_ok=True)
                    pipe.generate_and_save(
                        item.phrase,
                        item.output_path,
                        style=style,
                        **gen_kwargs,
                    )
                    generated += 1
                except Exception:
                    logger.exception("Failed: %s (%s)", item.adjective, item.phrase[:40])
                    failed += 1

        if generated % 25 < batch_size or batch_start + batch_size >= total:
            logger.info("Progress: %d/%d generated (%d failed)", generated, total, failed)

    return {"total": total, "generated": generated, "failed": failed}


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------


def _gpu_worker(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    gpu_id: int,
    work_items: list[AdjImageItem],
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
    items: list[AdjImageItem],
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
        logger.error("No CUDA GPUs available.")
        return {"total": len(items), "generated": 0, "failed": len(items)}

    logger.info(
        "Multi-GPU: %d items across %d GPUs (batch_size=%d)",
        len(items),
        num_gpus,
        batch_size,
    )

    chunks: list[list[AdjImageItem]] = [[] for _ in range(num_gpus)]
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
            logger.info("Waiting 30s for page cache to warm...")
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

    return {
        "total": len(items),
        "generated": sum(s["generated"] for s in all_stats),
        "failed": sum(s["failed"] for s in all_stats),
        "per_gpu": all_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 -- pipeline orchestration: complexity matches the spec it implements
    args = parse_args()

    setup_logging()

    data_dir = Path(args.data_dir)

    # Stage 1: Load adjectives with phrases
    logger.info("Stage 1: Loading adjective list with phrases")
    adj_phrases, full_data = load_adjective_list(data_dir)
    if args.debug:
        adj_phrases = dict(list(adj_phrases.items())[:50])
        logger.info("DEBUG MODE: using first 50 adjectives")
    logger.info("Loaded %d adjectives", len(adj_phrases))

    # Stage 2: Generate images
    logger.info("Stage 2: Generating images")

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

    pipe: FluxPipeline | None = None
    if not use_multi_gpu:
        pipe = FluxPipeline(model_id=args.model_id, compile_model=args.compile)

    t0 = time.time()

    for style in args.styles:
        logger.info("=" * 50)
        logger.info("Generating adjective images  [style=%s]", style)
        logger.info("=" * 50)

        # Write meta.json for each word subdirectory
        write_meta_json(adj_phrases, full_data, data_dir, style)

        items, skipped = collect_work_items(adj_phrases, data_dir, style)

        if not items:
            logger.info("All %d images already exist, skipping.", len(adj_phrases))
            continue

        logger.info("%d images to generate (%d already exist)", len(items), skipped)

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
            stats = generate_batched(pipe, items, style, args.batch_size, gen_kwargs)

        logger.info(
            "Done: %d generated, %d failed, %d skipped.",
            stats["generated"],
            stats.get("failed", 0),
            skipped,
        )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes).", elapsed, elapsed / 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
