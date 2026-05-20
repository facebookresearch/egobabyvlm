# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Async LLM worker pool that drives the LT-Swap pipeline against a vLLM server.

This is a slimmed reimplementation of the upstream
``apps/swapbench/third_party/lt_swap/generate_task/mp_main.py`` worker pool
that uses the existing ``apps/benchmark_creation/utils/vllm_server.llm_call``
client (OpenAI-compatible, with retry + backoff) instead of Meta-internal
``matrix`` / ``metagen`` clients. The on-disk file format produced is
identical to upstream ``mp_main``, so the downstream upstream scripts
(``wordswap_pairs_and_filtering_prompts``, ``retrieve_correct_pairs``, …)
consume the output unchanged.

Each input line is either ``<metadata>|<prompt>`` or
``<metadata>|<prompt1>/<prompt2>/<answer1>/<answer2>`` (a "/" delimits
multiple prompts and their expected answers; matches upstream's
``mp_main.main`` parsing). Output is one ``<idx>|<metadata>|<response>``
line per call (or ``<idx>|<metadata>|<answer>|<response>`` when answers
are present), append-only so reruns are idempotent.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING

from apps.benchmark_creation.utils.vllm_server import llm_call

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

#: Sentinel placed on the queue to signal a worker that no more work is coming.
_DONE = None


def _read_done_indices(output_path: Path) -> tuple[set[str], int]:
    """Walk an existing output file to find which input lines are already done.

    Returns ``(done_indices, error_indices_count)``. Error rows (responses
    starting with ``<ERROR``) are NOT considered done so the next run retries
    them; this matches upstream ``mp_main`` behavior.
    """
    if not output_path.exists():
        return set(), 0
    done: set[str] = set()
    errors = 0
    with output_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                idx = line.split("|", 1)[0]
            except ValueError:
                continue
            try:
                response = line.rsplit("|", 1)[1]
            except IndexError:
                continue
            if response.startswith("<ERROR"):
                errors += 1
            else:
                done.add(idx)
    return done, errors


def _iter_jobs(
    input_path: Path,
    done: set[str],
) -> list[tuple[str, str, str, str | None]]:
    """Parse an upstream-format input file into ``(idx, metadata, prompt, answer)`` jobs.

    A line ``<metadata>|<prompts_field>`` is split on ``|`` once from the
    right. ``<prompts_field>`` is then either a single prompt, or a slash-
    separated list with the form ``p1/p2/.../pN/a1/a2/.../aN`` matching
    upstream ``mp_main.main``. Indices ``done`` are skipped.
    """
    jobs: list[tuple[str, str, str, str | None]] = []
    with input_path.open(encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            line, _, prompts_field = raw.rstrip("\n").rpartition("|")
            if not prompts_field:
                continue
            slash_count = prompts_field.count("/")
            if slash_count == 0:
                idx = str(line_idx)
                if idx in done or not prompts_field.strip():
                    continue
                jobs.append((idx, line, prompts_field.strip(), None))
                continue
            if slash_count % 2 != 1:
                logger.warning(
                    "Line %d has even slash count %d; skipping (upstream mp_main does the same).",
                    line_idx,
                    slash_count,
                )
                continue
            parts = prompts_field.split("/")
            half = slash_count // 2 + 1
            prompts, answers = parts[:half], parts[half:]
            for j, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
                idx = f"{line_idx}-{j}"
                if idx in done or not prompt.strip():
                    continue
                jobs.append((idx, line, prompt.strip(), answer.strip()))
    return jobs


async def _worker(  # noqa: PLR0913 -- async pool plumbing; collapsing to a dataclass would just move the args.
    name: int,
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    lock: asyncio.Lock,
    output_handle: IO[str],
) -> None:
    """Drain ``queue`` until ``_DONE``, calling ``llm_call`` for each job."""
    processed = 0
    while True:
        item = await queue.get()
        if item is _DONE:
            queue.task_done()
            logger.info("Worker %d finished after %d items", name, processed)
            return
        idx, metadata, prompt, answer = item
        try:
            response = await llm_call(
                client=client,
                model=model,
                prompt=prompt.replace("\\n", "\n"),
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        except BaseException as e:  # noqa: BLE001
            # Any unhandled error becomes an <ERROR> row so the pool can
            # progress; otherwise an unhandled exception in a worker leaves
            # ``queue.task_done()`` uncalled and ``queue.join()`` deadlocks.
            logger.warning("Worker %d uncaught exception on idx %s: %r", name, idx, e)
            response = f"<ERROR uncaught {type(e).__name__}: {str(e)[:200]}>"
        if response is None:
            response = f"<ERROR after {max_retries} retries>"
        else:
            response = response.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        async with lock:
            if answer is None:
                output_handle.write(f"{idx}|{metadata}|{response}\n")
            else:
                output_handle.write(f"{idx}|{metadata}|{answer}|{response}\n")
            output_handle.flush()
        processed += 1
        queue.task_done()


async def run_llm_pool(  # noqa: PLR0913 -- knobs are all user-facing tuning; no obvious grouping.
    *,
    client: AsyncOpenAI,
    model: str,
    input_path: str | Path,
    output_path: str | Path,
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
    num_workers: int = 16,
    queue_size: int = 64,
) -> int:
    """Drive an upstream LT-Swap prompt file through the vLLM endpoint.

    Resumes append-only against an existing ``output_path``. Returns the
    number of jobs actually processed (excluding skipped already-done
    indices). Output rows match ``mp_main`` byte-for-byte so the upstream
    downstream-filter scripts can consume them unchanged.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done, error_count = _read_done_indices(output_path)
    if done or error_count:
        logger.info(
            "Resuming from %s (%d completed, %d errored will be retried)",
            output_path,
            len(done),
            error_count,
        )
    jobs = _iter_jobs(input_path, done)
    if not jobs:
        logger.info("Nothing to do — all %d input lines already processed.", len(done))
        return 0

    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    lock = asyncio.Lock()
    mode = "a" if output_path.exists() else "w"
    # The open() and the per-write flush() are sync, but the writes themselves
    # are guarded by ``lock`` so contention is bounded; using aiofiles here
    # would be the correct async-purist alternative but adds a dep for what
    # is effectively log-line-rate I/O.
    with output_path.open(mode, encoding="utf-8") as handle:  # noqa: ASYNC230 -- see comment above.
        workers = [
            asyncio.create_task(
                _worker(
                    name=i,
                    queue=queue,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    lock=lock,
                    output_handle=handle,
                ),
            )
            for i in range(num_workers)
        ]
        for job in jobs:
            await queue.put(job)
        for _ in workers:
            await queue.put(_DONE)
        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    logger.info("Processed %d jobs into %s", len(jobs), output_path)
    return len(jobs)
