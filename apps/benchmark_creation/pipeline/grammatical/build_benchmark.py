# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Create the Grammatical 2-AFC benchmark via a unified 3-LLM-call pipeline.

Expects a pre-built vocabulary directory (from create_vocabulary.py) containing
``longtail_wordlist.csv``.  For each item the pipeline makes three LLM calls:
  1. LLM selects a word from the vocabulary pool
  2. LLM generates a caption pair (or single sentence for deterministic tasks)
  3. LLM validates both captions for grammar and visual representability

Usage::

    python scripts/03_Create_Grammatical/create_grammatical_benchmark.py \
        --vocab-dir data/coco_20260410_120000 \
        --name COCO \
        --api-base http://localhost:8000/v1 \
        --model google/gemma-4-26B-A4B-it

Outputs::

    {output_dir}/
        Grammatical/prompts/gram_grammatical_responses.jsonl
        Grammatical/gram_{category}/sentence_list.json   (one per category)
"""

import argparse
import asyncio
import contextlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, TextIO

from apps.benchmark_creation.pipeline.grammatical.constants import (
    COUNTING_SAFE_VERBS,
    EMBEDDED_RELATIVE_SAFE_VERBS,
    ORDER_MATTERS_SAFE_VERBS,
    SUBJECT_VERB_SAFE_VERBS,
    check_embedded_relative_sanity,
    check_order_matters_sanity,
    check_subject_adjective_sanity,
    get_comparative_forms,
    get_past_participle,
    get_third_person_singular,
    pluralize,
    singularize,
)
from apps.benchmark_creation.pipeline.grammatical.constants import (
    get_gerund as _get_gerund,
)
from apps.benchmark_creation.pipeline.grammatical.diversity import (
    _MAX_NOUN_PAIR_USES_PER_CATEGORY,
    _MAX_NOUN_USES_PER_CATEGORY,
    _MAX_VERB_USES_PER_CATEGORY,
    _SKIP_WORDS,
    check_noun_diversity,
    check_noun_pair_diversity,
    check_verb_diversity,
    check_vocab_coverage,
    extract_nouns_from_caption,
    update_noun_counts,
    update_noun_pair_counts,
    update_verb_counts,
)
from apps.benchmark_creation.pipeline.grammatical.parsers import (
    clean_text,
    parse_embedded_relative_response,
    parse_pair_response,
    parse_validation_response,
    validate_response_text,
)
from apps.benchmark_creation.pipeline.grammatical.prompts import (
    _ANIMATE_NOUNS,
    _COMPARATIVE_NOUNS,
    GRAMMATICAL_TEMPLATES,
    build_pair_generation_prompt,
    build_post_verification_prompt,
    build_validation_prompt,
    build_word_suitability_prompt,
)
from apps.benchmark_creation.pipeline.grammatical.word_filters import (
    _BAD_NEGATION_ADJECTIVES,
    _GLOBAL_BLOCKED_WORDS,
    _SUBJECTIVE_ADJECTIVES,
    _SUPPLEMENTAL_ADJECTIVES,
    derive_deterministic_negative,
)
from apps.benchmark_creation.utils.vocabulary import (
    VocabEntry,
    load_longtail_csv,
)
from core.utils.logging import setup_logging

try:
    from openai import AsyncOpenAI  # type: ignore[attr-defined]
except ImportError as exc:
    msg = (
        "openai package is required (used as client for any OpenAI-compatible "
        "API, including vLLM). Install it via the dev pixi env: "
        "`pixi install -e dev` from the repo root."
    )
    raise ImportError(msg) from exc

logger = logging.getLogger("create_grammatical")


# ===========================================================================
#  Data structures
# ===========================================================================


@dataclass
class CategoryState:
    """Per-category mutable state shared across async workers."""

    category: str
    pos: str
    template: str
    pair_mode: str
    word_pool: list[VocabEntry]
    selected_words: set[str] = field(default_factory=set)
    bucket_counts: dict[int, int] = field(default_factory=dict)
    bucket_targets: dict[int, int] = field(default_factory=dict)
    noun_counts: dict[str, int] = field(default_factory=dict)
    noun_pair_counts: dict[str, int] = field(default_factory=dict)
    verb_counts: dict[str, int] = field(default_factory=dict)
    suggested_noun_pool: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WorkItem(NamedTuple):
    """A single work item for the grammatical pipeline."""

    category: str
    item_index: int
    global_idx: int


# ===========================================================================
#  Bucket / pool helpers
# ===========================================================================


def compute_bucket_targets(pool: list[VocabEntry], n: int) -> dict[int, int]:
    """Compute proportional target counts per frequency bin."""
    by_bin: dict[int, int] = {}
    for e in pool:
        by_bin[e.freq_bin] = by_bin.get(e.freq_bin, 0) + 1

    total = len(pool)
    if total == 0:
        return {}

    targets: dict[int, int] = {}
    remainder_entries: list[tuple[float, int]] = []

    for bin_idx, count in sorted(by_bin.items()):
        exact = (count / total) * n
        floor_n = int(exact)
        targets[bin_idx] = min(floor_n, count)
        remainder_entries.append((exact - floor_n, bin_idx))

    remaining = n - sum(targets.values())
    remainder_entries.sort(key=lambda x: x[0], reverse=True)
    for _, bin_idx in remainder_entries:
        if remaining <= 0:
            break
        if targets[bin_idx] < by_bin[bin_idx]:
            targets[bin_idx] += 1
            remaining -= 1

    return targets


# ===========================================================================
#  I/O helpers
# ===========================================================================


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with Path(path).open() as f:
        for raw in f:
            line = raw.strip()
            if line:
                items.append(json.loads(line))
    return items


# ===========================================================================
#  LLM call helper
# ===========================================================================


async def llm_call(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> str | None:
    """Single LLM call with retry + exponential backoff."""
    # enable_thinking is Qwen3-specific; skip for other models (e.g. Gemma4)
    extra_body = {}
    if "qwen" in model.lower():
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(max_retries):
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                ),
                timeout=180.0,
            )
            content = completion.choices[0].message.content
            return content.strip() if content is not None else None
        except Exception as e:  # noqa: BLE001 -- worker/retry boundary: must catch all errors to keep pipeline alive
            wait_time = 2**attempt
            logger.warning(
                "llm_call attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                e,
                wait_time,
            )
            await asyncio.sleep(wait_time)
    return None


# ===========================================================================
#  process_item helpers — response recording & word undo
# ===========================================================================


async def _write_record(
    record: dict,
    write_lock: asyncio.Lock,
    response_file: TextIO,
    **fields: object,
) -> None:
    """Update *record* with *fields*, then write it to *response_file*."""
    record.update(fields)
    async with write_lock:
        response_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        response_file.flush()


async def _undo_word_selection(
    state: CategoryState,
    word: str,
    word_bin: int,
) -> None:
    """Remove *word* from selected set and decrement its bucket count."""
    async with state.lock:
        state.selected_words.discard(word)
        state.bucket_counts[word_bin] = max(
            0,
            state.bucket_counts.get(word_bin, 0) - 1,
        )


# ===========================================================================
#  Core 3-call processing
# ===========================================================================


async def process_item(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- pipeline-level orchestration: many parallel context fields
    work_item: WorkItem,
    state: CategoryState,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_retries: int,
    max_validation_retries: int,
    response_file: TextIO,
    write_lock: asyncio.Lock,
    vocab_set: frozenset[str] | None = None,
) -> bool:
    """Process a single grammatical item with the 3-call flow.

    Returns True if an accepted caption pair was produced.
    """
    category = work_item.category
    item_index = work_item.item_index
    global_idx = work_item.global_idx

    for attempt in range(max_validation_retries):
        record: dict = {
            "idx": global_idx,
            "category": category,
            "item_index": item_index,
            "attempt": attempt + 1,
        }

        # --- Call 1: Word selection ---
        chosen_entry: VocabEntry | None = None
        call_1_prompt = ""
        call_1_response = ""

        # Sample word directly from the pool (no LLM call).
        # LLM word selection was a bottleneck: it fixates on a few "favorite"
        # words and ignores the rest, causing pool exhaustion.  Direct sampling
        # gives uniform coverage and never wastes retries on word selection.
        async with state.lock:
            # Exclude words already selected AND verbs/nouns that hit diversity cap.
            exhausted_verbs = {
                _get_gerund(e.word)
                for e in state.word_pool
                if e.pos == "VERB" and state.verb_counts.get(_get_gerund(e.word), 0) >= _MAX_VERB_USES_PER_CATEGORY
            }
            exhausted_base = {
                e.word for e in state.word_pool if e.pos == "VERB" and _get_gerund(e.word) in exhausted_verbs
            }
            available = [
                e for e in state.word_pool if e.word not in state.selected_words and e.word not in exhausted_base
            ]
            if not available:
                # All words either selected or exhausted — allow reuse of
                # selected words but still respect diversity caps.
                available = [e for e in state.word_pool if e.word not in exhausted_base]
            if not available:
                logger.info(
                    "[%s] All %d pool words exhausted (diversity caps) — allowing full reuse.",
                    category,
                    len(state.word_pool),
                )
                state.selected_words.clear()
                available = list(state.word_pool)
            chosen_entry = random.choice(available)
            state.selected_words.add(chosen_entry.word)
            state.bucket_counts[chosen_entry.freq_bin] = state.bucket_counts.get(chosen_entry.freq_bin, 0) + 1
        call_1_prompt = "(sampled directly from word pool)"
        call_1_response = chosen_entry.word

        if chosen_entry is None:
            await _write_record(
                record,
                write_lock,
                response_file,
                status="error",
                call_1={
                    "prompt": call_1_prompt,
                    "response": call_1_response or "",
                    "word": None,
                    "word_bin": None,
                },
                caption_a=None,
                caption_b=None,
                word=None,
                word_bin=None,
            )
            if attempt < max_validation_retries - 1:
                continue
            return False

        word = chosen_entry.word
        word_bin = chosen_entry.freq_bin

        call_1_dict = {
            "prompt": call_1_prompt,
            "response": call_1_response,
            "word": word,
            "word_bin": word_bin,
        }

        # --- Reject subjective adjectives for comparatives ---
        if category == "comparatives" and word in _SUBJECTIVE_ADJECTIVES:
            # Keep the word in selected_words so it's never picked again
            async with state.lock:
                state.bucket_counts[word_bin] = max(
                    0,
                    state.bucket_counts.get(word_bin, 0) - 1,
                )
            await _write_record(
                record,
                write_lock,
                response_file,
                status="rejected",
                call_1=call_1_dict,
                call_3={
                    "prompt": "",
                    "response": f"skipped (subjective adjective '{word}' not suitable for comparatives)",
                    "accepted": False,
                    "reason": f"subjective adjective: {word}",
                },
                caption_a=None,
                caption_b=None,
                word=word,
                word_bin=word_bin,
            )
            continue

        # --- Reject bad adjectives for negation ---
        if category == "negation" and word in _BAD_NEGATION_ADJECTIVES:
            async with state.lock:
                state.bucket_counts[word_bin] = max(
                    0,
                    state.bucket_counts.get(word_bin, 0) - 1,
                )
            await _write_record(
                record,
                write_lock,
                response_file,
                status="rejected",
                call_1=call_1_dict,
                call_3={
                    "prompt": "",
                    "response": f"skipped (bad negation adjective '{word}')",
                    "accepted": False,
                    "reason": f"bad negation adjective: {word}",
                },
                caption_a=None,
                caption_b=None,
                word=word,
                word_bin=word_bin,
            )
            continue

        # --- LLM suitability check: reject abstract / inappropriate words ---
        suitability_prompt = build_word_suitability_prompt(
            word,
            state.pos,
            category,
        )
        suitability_response = await llm_call(
            client,
            model,
            suitability_prompt,
            temperature=0.0,
            max_tokens=64,
            max_retries=max_retries,
        )
        suitability_ok = suitability_response is not None and suitability_response.strip().upper().startswith("YES")
        if not suitability_ok:
            reason = (suitability_response or "no response").strip()
            logger.debug(
                "Word '%s' rejected by suitability check for %s: %s",
                word,
                category,
                reason,
            )
            async with state.lock:
                state.bucket_counts[word_bin] = max(
                    0,
                    state.bucket_counts.get(word_bin, 0) - 1,
                )
            await _write_record(
                record,
                write_lock,
                response_file,
                status="rejected",
                call_1=call_1_dict,
                call_3={
                    "prompt": suitability_prompt,
                    "response": reason,
                    "accepted": False,
                    "reason": f"suitability check failed: {reason}",
                },
                caption_a=None,
                caption_b=None,
                word=word,
                word_bin=word_bin,
            )
            continue

        # --- Call 2: Pair generation ---
        # Collect overused nouns to steer LLM toward diversity
        async with state.lock:
            overused = [n for n, c in state.noun_counts.items() if c >= _MAX_NOUN_USES_PER_CATEGORY - 1]
            overused_verbs = [v for v, c in state.verb_counts.items() if c >= _MAX_VERB_USES_PER_CATEGORY - 1]
            # Collect overused noun pairs to steer LLM away from repeated combos
            overused_pairs = [
                key for key, c in state.noun_pair_counts.items() if c >= _MAX_NOUN_PAIR_USES_PER_CATEGORY - 1
            ]
        call_2_prompt = build_pair_generation_prompt(
            category,
            state.template,
            word,
            state.pos,
            item_index,
            overused_nouns=overused if overused else None,
            overused_verbs=overused_verbs if overused_verbs else None,
            overused_noun_pairs=overused_pairs if overused_pairs else None,
            suggested_noun_pool=state.suggested_noun_pool or None,
        )
        call_2_response = await llm_call(
            client,
            model,
            call_2_prompt,
            temperature,
            max_tokens=256,
            max_retries=max_retries,
        )
        if call_2_response is None:
            await _undo_word_selection(state, word, word_bin)
            await _write_record(
                record,
                write_lock,
                response_file,
                status="error",
                call_1=call_1_dict,
                call_2={"prompt": call_2_prompt, "response": ""},
                caption_a=None,
                caption_b=None,
                word=word,
                word_bin=word_bin,
            )
            if attempt < max_validation_retries - 1:
                continue
            return False

        # --- Parse Call 2 response ---
        caption_a: str | None = None
        caption_b: str | None = None
        antonym: str | None = None
        sanity_fail_reason: str | None = None

        if state.pair_mode == "llm":
            if category == "embedded_relative":
                caption_a, caption_b, antonym = parse_embedded_relative_response(call_2_response)
            else:
                caption_a, caption_b = parse_pair_response(call_2_response)
            if caption_a is None or caption_b is None:
                sanity_fail_reason = "failed to parse caption_a/caption_b from LLM response"
            elif validate_response_text(caption_a) is None:
                sanity_fail_reason = f"caption_a failed sanity: {caption_a[:60]}"
            elif validate_response_text(caption_b) is None:
                sanity_fail_reason = f"caption_b failed sanity: {caption_b[:60]}"
            elif caption_a == caption_b:
                sanity_fail_reason = "caption_a and caption_b are identical"
        else:
            # Deterministic mode: LLM generates only the positive sentence
            sentence_clean = clean_text(call_2_response)
            if validate_response_text(sentence_clean) is None:
                sanity_fail_reason = "sentence failed basic sanity check"
            else:
                caption_a = sentence_clean
                caption_b = derive_deterministic_negative(
                    sentence_clean,
                    category,
                    item_index,
                )
                if caption_b is None:
                    sanity_fail_reason = f"failed to derive deterministic negative for {category}"

        call_2_dict = {"prompt": call_2_prompt, "response": call_2_response}

        if sanity_fail_reason is not None:
            await _undo_word_selection(state, word, word_bin)
            await _write_record(
                record,
                write_lock,
                response_file,
                status="rejected",
                call_1=call_1_dict,
                call_2=call_2_dict,
                call_3={
                    "prompt": "",
                    "response": f"skipped ({sanity_fail_reason})",
                    "accepted": False,
                    "reason": sanity_fail_reason,
                },
                caption_a=None,
                caption_b=None,
                word=word,
                word_bin=word_bin,
            )
            continue

        assert caption_a is not None
        assert caption_b is not None

        # --- Noun diversity check ---
        # Categories with constrained noun pools (e.g., order_matters requires
        # animate nouns only) get a higher threshold to avoid excessive rejections.
        relaxed_noun_categories = {
            "embedded_relative",
            "subject_verb",
            "order_matters",
            "counting",
            "subject_adjective",
            "comparatives",
        }
        noun_limit = (
            _MAX_NOUN_USES_PER_CATEGORY * 2 if category in relaxed_noun_categories else _MAX_NOUN_USES_PER_CATEGORY
        )
        async with state.lock:
            diversity_reason = check_noun_diversity(
                caption_a,
                caption_b,
                state.noun_counts,
                max_uses=noun_limit,
            )
            # Check noun-pair diversity (prevent same pair of nouns repeating)
            if diversity_reason is None:
                diversity_reason = check_noun_pair_diversity(
                    caption_a,
                    caption_b,
                    state.noun_pair_counts,
                )
            # Also check verb diversity for categories with two verbs
            if diversity_reason is None and category in ("subject_verb", "counting"):
                diversity_reason = check_verb_diversity(
                    caption_a,
                    caption_b,
                    state.verb_counts,
                )
        if diversity_reason:
            await _undo_word_selection(state, word, word_bin)
            await _write_record(
                record,
                write_lock,
                response_file,
                status="rejected",
                call_1=call_1_dict,
                call_2=call_2_dict,
                call_3={
                    "prompt": "",
                    "response": f"skipped (diversity: {diversity_reason})",
                    "accepted": False,
                    "reason": diversity_reason,
                },
                caption_a=caption_a,
                caption_b=caption_b,
                word=word,
                word_bin=word_bin,
            )
            continue

        # --- Vocabulary coverage check ---
        if vocab_set is not None:
            oov_reason = check_vocab_coverage(
                caption_a,
                caption_b,
                vocab_set,
                _SKIP_WORDS,
            )
            if oov_reason:
                await _undo_word_selection(state, word, word_bin)
                await _write_record(
                    record,
                    write_lock,
                    response_file,
                    status="rejected",
                    call_1=call_1_dict,
                    call_2=call_2_dict,
                    call_3={
                        "prompt": "",
                        "response": f"skipped (vocab coverage: {oov_reason})",
                        "accepted": False,
                        "reason": oov_reason,
                    },
                    caption_a=caption_a,
                    caption_b=caption_b,
                    word=word,
                    word_bin=word_bin,
                )
                continue

        # --- order_matters: hard-coded invertibility check ---
        if category == "order_matters":
            om_reason = check_order_matters_sanity(caption_a)
            if om_reason:
                await _undo_word_selection(state, word, word_bin)
                await _write_record(
                    record,
                    write_lock,
                    response_file,
                    status="rejected",
                    call_1=call_1_dict,
                    call_2=call_2_dict,
                    call_3={
                        "prompt": "",
                        "response": f"skipped (order_matters sanity: {om_reason})",
                        "accepted": False,
                        "reason": om_reason,
                    },
                    caption_a=caption_a,
                    caption_b=caption_b,
                    word=word,
                    word_bin=word_bin,
                )
                continue

        # --- subject_adjective: verify adjective swap ---
        if category == "subject_adjective":
            sa_reason = check_subject_adjective_sanity(caption_a, caption_b, word)
            if sa_reason:
                await _undo_word_selection(state, word, word_bin)
                await _write_record(
                    record,
                    write_lock,
                    response_file,
                    status="rejected",
                    call_1=call_1_dict,
                    call_2=call_2_dict,
                    call_3={
                        "prompt": "",
                        "response": f"skipped (subject_adjective sanity: {sa_reason})",
                        "accepted": False,
                        "reason": sa_reason,
                    },
                    caption_a=caption_a,
                    caption_b=caption_b,
                    word=word,
                    word_bin=word_bin,
                )
                continue

        # --- embedded_relative: verify same words in both captions ---
        if category == "embedded_relative":
            er_reason = check_embedded_relative_sanity(caption_a, caption_b)
            if er_reason:
                await _undo_word_selection(state, word, word_bin)
                await _write_record(
                    record,
                    write_lock,
                    response_file,
                    status="rejected",
                    call_1=call_1_dict,
                    call_2=call_2_dict,
                    call_3={
                        "prompt": "",
                        "response": f"skipped (embedded_relative sanity: {er_reason})",
                        "accepted": False,
                        "reason": er_reason,
                    },
                    caption_a=caption_a,
                    caption_b=caption_b,
                    word=word,
                    word_bin=word_bin,
                )
                continue

        # --- Call 3: Validation ---
        call_3_prompt = build_validation_prompt(caption_a, caption_b, category, word)
        call_3_response = await llm_call(
            client,
            model,
            call_3_prompt,
            temperature,
            max_tokens=64,
            max_retries=max_retries,
        )

        if call_3_response is None:
            accepted, reason = False, "LLM call failed"
        else:
            accepted, reason = parse_validation_response(call_3_response)

        if accepted:
            # Update noun diversity tracker
            async with state.lock:
                update_noun_counts(caption_a, caption_b, state.noun_counts)
                update_noun_pair_counts(caption_a, caption_b, state.noun_pair_counts)
                update_verb_counts(caption_a, caption_b, state.verb_counts)
            await _write_record(
                record,
                write_lock,
                response_file,
                status="accepted",
                call_1=call_1_dict,
                call_2=call_2_dict,
                call_3={
                    "prompt": call_3_prompt,
                    "response": call_3_response or "",
                    "accepted": True,
                    "reason": "",
                },
                caption_a=caption_a,
                caption_b=caption_b,
                word=word,
                word_bin=word_bin,
                antonym=antonym,
            )
            return True
        await _undo_word_selection(state, word, word_bin)
        await _write_record(
            record,
            write_lock,
            response_file,
            status="rejected",
            call_1=call_1_dict,
            call_2=call_2_dict,
            call_3={
                "prompt": call_3_prompt,
                "response": call_3_response or "",
                "accepted": False,
                "reason": reason,
            },
            caption_a=None,
            caption_b=None,
            word=word,
            word_bin=word_bin,
        )
        continue

    # All validation retries exhausted
    return False


# ===========================================================================
#  Worker and orchestrator
# ===========================================================================


async def worker(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    worker_id: int,  # noqa: ARG001 -- `worker_id` kept for parallel-worker signature uniformity
    queue: asyncio.Queue,
    category_states: dict[str, CategoryState],
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_retries: int,
    max_validation_retries: int,
    response_file: TextIO,
    write_lock: asyncio.Lock,
    progress: dict,
    vocab_set: frozenset[str] | None = None,
) -> None:
    """Worker coroutine: pulls WorkItems from the queue."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        work_item: WorkItem = item
        state = category_states[work_item.category]

        await process_item(
            work_item,
            state,
            client,
            model,
            temperature,
            max_retries,
            max_validation_retries,
            response_file,
            write_lock,
            vocab_set=vocab_set,
        )

        progress["done"] += 1
        if progress["done"] % 20 == 0 or progress["done"] == progress["total"]:
            logger.info(
                "Progress: %d / %d (%.1f%%)",
                progress["done"],
                progress["total"],
                100.0 * progress["done"] / progress["total"],
            )

        queue.task_done()


async def run_grammatical_generation(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- pipeline-level orchestration: many parallel context fields
    entries: list[VocabEntry],
    output_dir: Path,
    dataset_name: str,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    items_per_category: int,
    num_workers: int,
    max_retries: int,
    max_validation_retries: int,
    min_freq_bin: int = 4,
    category_items: dict[str, int] | None = None,
) -> None:
    """Unified grammatical generation pipeline: word selection + pair gen + validation."""
    response_path = output_dir / "Grammatical" / "prompts" / "gram_grammatical_responses.jsonl"
    grammatical_base = output_dir / "Grammatical"

    response_path.parent.mkdir(parents=True, exist_ok=True)

    # Create per-category directories
    for cat_name in GRAMMATICAL_TEMPLATES:
        cat_dir = grammatical_base / f"gram_{cat_name}"
        cat_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Grammatical unified 3-call pipeline ===")

    # --- Build vocabulary lookup set from ALL valid entries ---
    # Used to validate that LLM-generated words belong to the dataset vocabulary.
    all_vocab_words: set[str] = set()
    for e in entries:
        if e.is_valid:
            all_vocab_words.add(e.word)
    # Add common morphological variants using existing helpers
    morphological_expansions: set[str] = set()
    for w in list(all_vocab_words):
        for variant_fn in [pluralize, singularize, _get_gerund, get_past_participle, get_third_person_singular]:
            with contextlib.suppress(Exception):
                morphological_expansions.add(variant_fn(w))
        # Add ALL plausible comparative forms (both -er and "more X")
        with contextlib.suppress(Exception):
            morphological_expansions.update(get_comparative_forms(w))
    # Also expand curated safe-verb lists so their gerund forms (wiggling,
    # tiptoeing, etc.) pass the OOV check.  We add ALL safe-verb morphological
    # variants unconditionally — these verbs are hand-curated for the benchmark,
    # so their gerund/past/3rd-person forms should always be accepted.
    for w in ORDER_MATTERS_SAFE_VERBS + SUBJECT_VERB_SAFE_VERBS + COUNTING_SAFE_VERBS + EMBEDDED_RELATIVE_SAFE_VERBS:
        for variant_fn in [_get_gerund, get_past_participle, get_third_person_singular]:
            with contextlib.suppress(Exception):
                morphological_expansions.add(variant_fn(w))
    all_vocab_words |= morphological_expansions
    vocab_set = frozenset(all_vocab_words)

    # --- Build POS pools (filtered by minimum frequency bin) ---
    valid = [e for e in entries if e.is_valid and e.freq_bin >= min_freq_bin]
    verbs = [e for e in valid if e.pos == "VERB"]
    adjs = [e for e in valid if e.pos == "ADJ"]
    nouns = [e for e in valid if e.pos == "NOUN"]
    pos_pools = {"VERB": verbs, "ADJ": adjs, "NOUN": nouns}

    # order_matters uses a hardcoded safe-verb list — filter against vocabulary.
    order_matters_pool = [
        VocabEntry(word=w, count=0, rank=0, freq_bin=-1, pos="VERB")
        for w in ORDER_MATTERS_SAFE_VERBS
        if w in vocab_set
    ]

    # subject_verb also uses a curated verb list — filter against vocabulary.
    subject_verb_pool = [
        VocabEntry(word=w, count=0, rank=0, freq_bin=-1, pos="VERB") for w in SUBJECT_VERB_SAFE_VERBS if w in vocab_set
    ]

    # counting uses a curated intransitive verb list — filter against vocabulary.
    counting_pool = [
        VocabEntry(word=w, count=0, rank=0, freq_bin=-1, pos="VERB") for w in COUNTING_SAFE_VERBS if w in vocab_set
    ]

    # embedded_relative uses a curated transitive verb list.
    embedded_relative_pool = [
        VocabEntry(word=w, count=0, rank=0, freq_bin=-1, pos="VERB")
        for w in EMBEDDED_RELATIVE_SAFE_VERBS
        if w in vocab_set
    ]

    # Supplement adjective pools — COCO vocab has few adjectives, so we add
    # curated visual adjectives that work well for comparatives and negation.
    coco_adj_words = {e.word for e in adjs}
    supplemental_entries = [
        VocabEntry(word=w, count=0, rank=0, freq_bin=-1, pos="ADJ")
        for w in _SUPPLEMENTAL_ADJECTIVES
        if w not in coco_adj_words and w in vocab_set
    ]
    adjs_augmented = adjs + supplemental_entries
    pos_pools["ADJ"] = adjs_augmented

    logger.info(
        "Vocabulary lookup: %d words (including morphological variants)",
        len(vocab_set),
    )
    logger.info(
        "Vocabulary filter: order_matters verbs %d→%d, subject_verb verbs %d→%d, "
        "counting verbs %d→%d, embedded_relative verbs %d→%d, "
        "supplemental adjectives %d→%d",
        len(ORDER_MATTERS_SAFE_VERBS),
        len(order_matters_pool),
        len(SUBJECT_VERB_SAFE_VERBS),
        len(subject_verb_pool),
        len(COUNTING_SAFE_VERBS),
        len(counting_pool),
        len(EMBEDDED_RELATIVE_SAFE_VERBS),
        len(embedded_relative_pool),
        len(_SUPPLEMENTAL_ADJECTIVES),
        len(supplemental_entries),
    )
    logger.info(
        "Vocabulary (freq_bin >= %d): %d valid (%d verbs, %d adjectives [+%d supplemental], %d nouns)",
        min_freq_bin,
        len(valid),
        len(verbs),
        len(adjs),
        len(supplemental_entries),
        len(nouns),
    )

    # --- Filter noun suggestion pools against the dataset vocabulary ---
    animate_noun_pool = [n for n in _ANIMATE_NOUNS if n in vocab_set]
    comparative_noun_pool = [n for n in _COMPARATIVE_NOUNS if n in vocab_set]
    logger.info(
        "Noun suggestion pools (vocab-filtered): animate %d/%d, comparative %d/%d",
        len(animate_noun_pool),
        len(_ANIMATE_NOUNS),
        len(comparative_noun_pool),
        len(_COMPARATIVE_NOUNS),
    )

    # --- Create CategoryState for each category ---
    category_states: dict[str, CategoryState] = {}
    for cat_name, gram_cat in GRAMMATICAL_TEMPLATES.items():
        pool = pos_pools.get(gram_cat.pos, [])
        if cat_name == "order_matters":
            pool = order_matters_pool
        elif cat_name == "subject_verb":
            pool = subject_verb_pool
        elif cat_name == "counting":
            pool = counting_pool
        elif cat_name == "embedded_relative":
            pool = embedded_relative_pool
        elif cat_name == "comparatives":
            # Pre-filter subjective adjectives so they never enter the pool
            pool = [e for e in pool if e.word not in _SUBJECTIVE_ADJECTIVES]
        elif cat_name == "negation":
            pool = [e for e in pool if e.word not in _BAD_NEGATION_ADJECTIVES]
        # Global pre-filter: remove abstract and inappropriate words
        pool = [e for e in pool if e.word not in _GLOBAL_BLOCKED_WORDS]
        targets = compute_bucket_targets(pool, items_per_category)
        # Pick the vocab-filtered noun suggestion pool for this category
        if cat_name in ("subject_verb", "order_matters", "counting", "embedded_relative"):
            noun_suggestion_pool = animate_noun_pool
        elif cat_name in ("comparatives", "subject_adjective"):
            noun_suggestion_pool = comparative_noun_pool
        else:
            noun_suggestion_pool = []
        category_states[cat_name] = CategoryState(
            category=cat_name,
            pos=gram_cat.pos,
            template=gram_cat.template,
            pair_mode=gram_cat.pair_mode,
            word_pool=list(pool),
            bucket_targets=targets,
            suggested_noun_pool=noun_suggestion_pool,
        )

    # --- Clear previous run (always generate from scratch) ---
    if response_path.exists():
        logger.info("Removing previous response log: %s", response_path)
        response_path.unlink()
    for cat_name in GRAMMATICAL_TEMPLATES:
        old_output = grammatical_base / f"gram_{cat_name}" / "sentence_list.json"
        if old_output.exists():
            logger.info("Removing previous sentence list: %s", old_output)
            old_output.unlink()

    # --- Build work queue ---
    work_items: list[WorkItem] = []
    global_idx = 0
    _cat_items = category_items or {}
    for cat_name in GRAMMATICAL_TEMPLATES:
        n = _cat_items.get(cat_name, items_per_category)
        for item_idx in range(n):
            work_items.append(WorkItem(cat_name, item_idx, global_idx))
            global_idx += 1

    logger.info("Generating %d items.", len(work_items))

    random.shuffle(work_items)

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(work_items)}

    for wi in work_items:
        await queue.put(wi)
    for _ in range(num_workers):
        await queue.put(None)

    with Path(response_path).open("w") as response_file:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        workers = [
            asyncio.create_task(
                worker(
                    i,
                    queue,
                    category_states,
                    client,
                    model,
                    temperature,
                    max_retries,
                    max_validation_retries,
                    response_file,
                    write_lock,
                    progress,
                    vocab_set=vocab_set,
                )
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

    logger.info("Generation complete: %d items processed.", progress["done"])

    # --- Post-generation verification ---
    responses = load_jsonl(response_path)
    verified = await post_verify_responses(
        responses,
        client,
        model,
        temperature=0.0,  # deterministic for reproducibility
        num_workers=num_workers,
        max_retries=max_retries,
    )

    # Overwrite response log with verified results
    with Path(response_path).open("w") as f:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        for record in verified:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- Assemble: write one sentence_list.json per category ---
    per_category = assemble_grammatical(verified, dataset_name)
    for cat_name, cat_data in per_category.items():
        cat_output = grammatical_base / f"gram_{cat_name}" / "sentence_list.json"
        with Path(cat_output).open("w") as f:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
            json.dump(cat_data, f, indent=2)
        logger.info("Wrote %s (%d pairs)", cat_output, len(cat_data.get("items", [])))

    # --- Write summary.json with per-category stats ---
    summary: dict[str, object] = {}
    total_items = 0
    total_attempts = 0
    total_rejected_validation = 0
    total_rejected_post = 0
    categories_summary = {}
    for cat_name, cat_data in sorted(per_category.items()):
        meta = cat_data.get("metadata", {})
        stats = meta.get("generation_stats", {})
        n_items = stats.get("accepted", 0)
        n_attempts = stats.get("total_attempts", 0)
        n_rej_val = stats.get("rejected_validation", 0)
        n_rej_post = stats.get("rejected_post_verification", 0)
        total_items += n_items
        total_attempts += n_attempts
        total_rejected_validation += n_rej_val
        total_rejected_post += n_rej_post
        categories_summary[cat_name] = {
            "items": n_items,
            "attempts": n_attempts,
            "rejected_validation": n_rej_val,
            "rejected_post_verification": n_rej_post,
            "acceptance_rate": stats.get("acceptance_rate", 0),
        }
    summary["total"] = {
        "items": total_items,
        "categories": len(per_category),
        "attempts": total_attempts,
        "rejected_validation": total_rejected_validation,
        "rejected_post_verification": total_rejected_post,
        "acceptance_rate": round(
            total_items / max(1, total_attempts),
            3,
        ),
    }
    summary["per_category"] = categories_summary
    summary_path = grammatical_base / "summary.json"
    with Path(summary_path).open("w") as f:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        json.dump(summary, f, indent=2)
    logger.info("Wrote generation summary to %s", summary_path)


# ===========================================================================
#  Post-generation verification
# ===========================================================================


async def _verify_one(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    record: dict,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_retries: int,
    sem: asyncio.Semaphore,
) -> dict:
    """Verify a single accepted record. Adds verification fields in-place."""
    async with sem:
        caption_a = record.get("caption_a", "")
        caption_b = record.get("caption_b", "")
        category = record.get("category", "")
        word = record.get("word", "")

        prompt = build_post_verification_prompt(
            caption_a,
            caption_b,
            category,
            word,
        )
        response = await llm_call(
            client,
            model,
            prompt,
            temperature,
            max_tokens=256,
            max_retries=max_retries,
        )

        if response is None:
            # LLM call failed — keep the item (don't penalize for infra issues)
            record["post_verification"] = {
                "response": "",
                "accepted": True,
                "reason": "LLM call failed — kept",
            }
            return record

        accepted, reason = parse_validation_response(response)
        record["post_verification"] = {
            "response": response.strip(),
            "accepted": accepted,
            "reason": reason,
        }

        if not accepted:
            record["status"] = "rejected_post_verify"

        return record


async def post_verify_responses(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    responses: list[dict],
    client: AsyncOpenAI,
    model: str,
    temperature: float = 0.0,
    num_workers: int = 20,
    max_retries: int = 3,
) -> list[dict]:
    """Run a strict common-sense verification pass on all accepted items.

    Items that fail are marked as ``status="rejected_post_verify"`` so they
    are excluded during assembly.  The full response list (with verification
    annotations) is returned.
    """
    accepted = [r for r in responses if r.get("status") == "accepted"]
    logger.info(
        "=== Post-generation verification: %d accepted items ===",
        len(accepted),
    )
    if not accepted:
        return responses

    sem = asyncio.Semaphore(num_workers)
    tasks = [_verify_one(record, client, model, temperature, max_retries, sem) for record in accepted]
    await asyncio.gather(*tasks)

    kept = sum(1 for r in accepted if r.get("status") == "accepted")
    rejected = sum(1 for r in accepted if r.get("status") == "rejected_post_verify")
    logger.info(
        "Post-verification: %d kept, %d rejected (%.1f%% rejection rate)",
        kept,
        rejected,
        100.0 * rejected / max(1, len(accepted)),
    )

    # Log per-category stats
    cat_stats: dict[str, dict[str, int]] = {}
    for r in accepted:
        cat = r.get("category", "")
        cs = cat_stats.setdefault(cat, {"kept": 0, "rejected": 0})
        if r.get("status") == "accepted":
            cs["kept"] += 1
        else:
            cs["rejected"] += 1
    for cat, cs in sorted(cat_stats.items()):
        logger.info(
            "  %s: %d kept, %d rejected",
            cat,
            cs["kept"],
            cs["rejected"],
        )

    return responses


# ===========================================================================
#  Assembly
# ===========================================================================


def assemble_grammatical(  # noqa: C901 -- pipeline orchestration: complexity matches the spec it implements
    responses: list[dict],
    dataset_name: str,
) -> dict[str, dict]:
    """Assemble per-category sentence_list.json dicts from accepted records.

    Returns ``{category_name: {"metadata": ..., "items": [...]}}`` — one
    entry per category, each written to its own ``gram_{category}/sentence_list.json``.
    """
    categories: dict[str, list[dict]] = {}
    seen_per_cat: dict[str, set[str]] = {}
    # Track rejection stats per category
    rejection_counts: dict[str, int] = {}
    post_verify_rejections: dict[str, int] = {}
    attempt_counts: dict[str, int] = {}

    for record in responses:
        cat = record.get("category", "")
        attempt_counts[cat] = attempt_counts.get(cat, 0) + 1

        status = record.get("status", "")
        if status == "rejected_post_verify":
            post_verify_rejections[cat] = post_verify_rejections.get(cat, 0) + 1
            continue
        if status != "accepted":
            rejection_counts[cat] = rejection_counts.get(cat, 0) + 1
            continue
        caption_a = record.get("caption_a", "")
        caption_b = record.get("caption_b", "")
        if not caption_a or not caption_b:
            continue
        seen = seen_per_cat.setdefault(cat, set())
        pair_key = f"{caption_a}|{caption_b}"
        if pair_key in seen:
            continue
        seen.add(pair_key)
        categories.setdefault(cat, [])

        entry: dict = {
            "caption_a": caption_a,
            "caption_b": caption_b,
            "word": record.get("word"),
            "freq_bin": record.get("word_bin"),
        }
        # Store antonym if present (used by embedded_relative for image generation)
        if record.get("antonym"):
            entry["antonym"] = record["antonym"]

        categories[cat].append(entry)

    total = sum(len(v) for v in categories.values())
    logger.info(
        "Grammatical assembly: %d pairs across %d categories.",
        total,
        len(categories),
    )
    for cat, items in sorted(categories.items()):
        logger.info("  %s: %d pairs", cat, len(items))

    result: dict[str, dict] = {}
    for cat, items in categories.items():
        # Frequency bin distribution
        bin_dist: dict[int, int] = {}
        for item in items:
            b = item.get("freq_bin")
            if b is not None:
                bin_dist[b] = bin_dist.get(b, 0) + 1
        bin_dist_sorted = {k: bin_dist[k] for k in sorted(bin_dist.keys())}

        # Unique words used
        words_used = sorted({item["word"] for item in items if item.get("word")})

        # Unique nouns (heuristic: extract content words from captions)
        all_nouns: set[str] = set()
        for item in items:
            all_nouns.update(extract_nouns_from_caption(item["caption_a"]))
            all_nouns.update(extract_nouns_from_caption(item["caption_b"]))

        # Category template info
        gram_cat = GRAMMATICAL_TEMPLATES.get(cat)
        cat_pos = gram_cat.pos if gram_cat else "unknown"
        cat_pair_mode = gram_cat.pair_mode if gram_cat else "unknown"

        result[cat] = {
            "description": (
                f"Caption pairs for Grammatical 2-AFC category '{cat}', "
                f"generated from {dataset_name} vocabulary via LLM."
            ),
            "metadata": {
                "category": cat,
                "dataset": dataset_name,
                "part_of_speech": cat_pos,
                "pair_mode": cat_pair_mode,
                "num_items": len(items),
                "num_unique_words": len(words_used),
                "words_used": words_used,
                "num_unique_nouns": len(all_nouns),
                "freq_bin_distribution": bin_dist_sorted,
                "generation_stats": {
                    "total_attempts": attempt_counts.get(cat, 0),
                    "accepted": len(items),
                    "rejected_validation": rejection_counts.get(cat, 0),
                    "rejected_post_verification": post_verify_rejections.get(cat, 0),
                    "acceptance_rate": round(
                        len(items) / max(1, attempt_counts.get(cat, 0)),
                        3,
                    ),
                },
            },
            "items": items,
        }
    return result


# ===========================================================================
#  CLI & Main
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Grammatical benchmark via unified 3-LLM-call pipeline.",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        required=True,
        help="Path to vocabulary directory (containing longtail_wordlist.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to --vocab-dir (in-place). When given, a timestamp suffix is appended.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Dataset",
        help="Dataset name for descriptions (e.g., 'COCO').",
    )
    parser.add_argument(
        "--items-per-category",
        type=int,
        default=250,
        help="Number of items per grammatical category.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL (default: local vLLM server).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="dummy",
        help="API key (default: 'dummy' for local vLLM).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
        help="Model name as served by the API (default: google/gemma-4-26B-A4B-it).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of concurrent async workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM call.",
    )
    parser.add_argument(
        "--max-validation-retries",
        type=int,
        default=20,
        help="Max validation retry loops per item.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--category-items",
        type=str,
        default=None,
        help="Per-category item overrides as comma-separated key=value pairs. "
        "E.g. 'comparatives=300,subject_verb=300'. Categories not listed "
        "fall back to --items-per-category.",
    )
    parser.add_argument(
        "--min-freq-bin",
        type=int,
        default=4,
        help="Minimum frequency bin (1-10). Only words with freq_bin >= this "
        "value are included in the pool. Default 4 (top 7 bins).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    if args.output_dir is None:
        output_dir = Path(args.vocab_dir)
    else:
        output_dir = Path(args.output_dir)
        response_log = output_dir / "Grammatical" / "prompts" / "gram_grammatical_responses.jsonl"
        if not response_log.exists():
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir / f"{args.name}_{timestamp}"

    random.seed(args.seed)
    t0 = time.time()

    logger.info("=== Loading vocabulary ===")
    vocab_dir = Path(args.vocab_dir)
    wordlist_path = vocab_dir / "longtail_wordlist.csv"
    entries = load_longtail_csv(wordlist_path)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    # Parse per-category item overrides
    category_items = None
    if args.category_items:
        category_items = {}
        for pair in args.category_items.split(","):
            cat, count = pair.strip().split("=")
            cat = cat.strip()
            if cat not in GRAMMATICAL_TEMPLATES:
                logger.warning(
                    "Unknown category '%s' in --category-items (valid: %s). Ignoring.",
                    cat,
                    ", ".join(GRAMMATICAL_TEMPLATES),
                )
                continue
            category_items[cat] = int(count.strip())

    asyncio.run(
        run_grammatical_generation(
            entries,
            output_dir,
            args.name,
            client,
            args.model,
            args.temperature,
            args.items_per_category,
            args.num_workers,
            args.max_retries,
            args.max_validation_retries,
            args.min_freq_bin,
            category_items=category_items,
        )
    )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
