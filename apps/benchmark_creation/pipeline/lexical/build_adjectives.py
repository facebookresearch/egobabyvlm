# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build frequency-stratified word lists for the lexical adjective task."""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from core.utils.logging import setup_logging

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from apps.benchmark_creation.pipeline.lexical.constants import (
    ADJ_ANTONYMS as _ADJ_ANTONYMS,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    LLM_VISUALIZABLE_PROMPT,
    NEG_PHRASE_PROMPT,
    PHRASE_PROMPT,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    POS_ARTICLES as _POS_ARTICLES,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    POS_CHECK_PROMPT as _POS_CHECK_PROMPT,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    POS_GUIDANCE as _POS_GUIDANCE,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    POS_NAMES as _POS_NAMES,
)
from apps.benchmark_creation.pipeline.lexical.constants import (
    WN_POS as _WN_POS,
)
from apps.benchmark_creation.utils.vllm_server import llm_call
from apps.benchmark_creation.utils.vocabulary import (
    DEFAULT_BIN_EDGES,
    VocabEntry,
    assign_frequency_bins,
    load_longtail_csv,
    stratified_sample,
    write_json,
)

setup_logging()
logger = logging.getLogger("build_lexical_adj")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


#: Minimum lemma length to keep after lemmatisation.
_MIN_LEMMA_LEN = 2
#: Maximum number of words allowed in an LLM-generated adjective phrase.
_MAX_PHRASE_WORDS = 6
#: Minimum word count for a phrase to safely extract a head noun.
_MIN_PHRASE_PARTS = 2


# ---------------------------------------------------------------------------
#  Lemmatization: merge inflected forms into canonical form
# ---------------------------------------------------------------------------


def lemmatize_and_merge(  # noqa: C901 -- pipeline orchestration: complexity matches the spec it implements
    entries: list[VocabEntry],
    target_pos: str,
    bin_edges: list[int] | None = None,
) -> list[VocabEntry]:
    """Lemmatize all entries to canonical form for *target_pos* and merge counts.

    For adjectives this converts inflected forms to their base form and sums
    their counts.  We try lemmatizing *every* entry (regardless of its POS tag)
    and keep those whose lemma has a valid WordNet synset for *target_pos*.

    Returns a new list of VocabEntry objects -- one per unique lemma -- with
    merged counts and freshly-assigned frequency bins.
    """
    wnl = WordNetLemmatizer()
    wn_pos = _WN_POS[target_pos]

    # All POS tags for cross-POS dominance check
    all_pos = [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]

    # Cache synset lookups
    _synset_cache: dict[str, bool] = {}

    def _has_synsets(word: str) -> bool:
        if word not in _synset_cache:
            _synset_cache[word] = bool(wn.synsets(word, pos=wn_pos))
        return _synset_cache[word]

    # Cache primary-POS check
    _primary_pos_cache: dict[str, bool] = {}

    def _is_primary_pos(word: str) -> bool:
        """Return True if *target_pos* is the dominant POS for *word* in WordNet.

        A word is rejected when another POS has strictly more synsets than the
        target POS.  E.g. "cow" has 6 noun synsets vs 1 verb -> rejected as verb.
        """
        if word not in _primary_pos_cache:
            target_count = len(wn.synsets(word, pos=wn_pos))
            dominant = True
            for other_pos in all_pos:
                if other_pos == wn_pos:
                    continue
                if len(wn.synsets(word, pos=other_pos)) > target_count:
                    dominant = False
                    break
            _primary_pos_cache[word] = dominant
        return _primary_pos_cache[word]

    # Lemmatize every entry and accumulate counts per lemma
    lemma_counts: dict[str, int] = {}
    skipped_primary_pos = 0
    for e in entries:
        if not e.is_valid:
            continue
        lemma = wnl.lemmatize(e.word, pos=wn_pos)
        if not lemma or len(lemma) < _MIN_LEMMA_LEN:
            continue
        if not lemma.isalpha():
            continue
        if not _has_synsets(lemma):
            continue
        if not _is_primary_pos(lemma):
            skipped_primary_pos += 1
            continue
        lemma_counts[lemma] = lemma_counts.get(lemma, 0) + e.count

    # Build new VocabEntry list (one per lemma)
    merged = [
        VocabEntry(word=lemma, count=count, rank=0, pos=target_pos, is_valid=True)
        for lemma, count in lemma_counts.items()
    ]

    # Assign frequency bins based on merged counts
    merged = assign_frequency_bins(merged, bin_edges)

    logger.info(
        "Lemmatization (%s): %d raw entries -> %d unique lemmas (%d skipped: primary POS mismatch)",
        target_pos,
        len(entries),
        len(merged),
        skipped_primary_pos,
    )

    return merged


# ---------------------------------------------------------------------------
#  LLM response parsing
# ---------------------------------------------------------------------------


def _parse_visualizable_response(response: str) -> tuple[bool, str]:
    """Parse LLM response into (is_visualizable, reason)."""
    is_visualizable = True
    reason = ""

    for line in response.splitlines():
        line_stripped = line.strip()
        line_upper = line_stripped.upper()
        if line_upper.startswith("VISUALIZABLE:"):
            value = line_upper.split(":", 1)[1].strip()
            if value == "NO":
                is_visualizable = False
        elif line_upper.startswith("REASON:"):
            reason = line_stripped.split(":", 1)[1].strip()

    return is_visualizable, reason


# ---------------------------------------------------------------------------
#  Async LLM filtering worker
# ---------------------------------------------------------------------------


async def _llm_filter_worker(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    worker_id: int,  # noqa: ARG001 -- `worker_id` kept for parallel-worker signature uniformity
    queue: asyncio.Queue,
    client: "AsyncOpenAI",
    model: str,
    temperature: float,
    max_retries: int,
    log_file: TextIO,
    write_lock: asyncio.Lock,
    results: dict,
    progress: dict,
) -> None:
    """Worker coroutine: pulls (word, pos) pairs from the queue."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        word, pos = item
        pos_guidance = _POS_GUIDANCE.get(pos, "")
        prompt = LLM_VISUALIZABLE_PROMPT.format(
            word=word,
            pos=pos,
            pos_guidance=pos_guidance,
        )

        response = await llm_call(
            client,
            model,
            prompt,
            temperature,
            max_tokens=128,
            max_retries=max_retries,
        )

        if response is None:
            is_visualizable, reason = True, "LLM_FAILURE_KEPT"
            action = "LLM_FAILURE_KEPT"
        else:
            is_visualizable, reason = _parse_visualizable_response(response)
            action = "KEPT" if is_visualizable else "REMOVED"

        results[word] = is_visualizable

        if is_visualizable:
            logger.info("'%s' (%s) -- visualizable", word, pos)
        else:
            logger.info("'%s' (%s) -- NOT visualizable: %s", word, pos, reason)

        log_entry = {
            "word": word,
            "pos": pos,
            "llm_response": response,
            "action": action,
            "reason": reason,
        }
        async with write_lock:
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        progress["done"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            logger.info(
                "LLM filter progress: %d / %d (%.1f%%)",
                progress["done"],
                progress["total"],
                100.0 * progress["done"] / progress["total"],
            )
        queue.task_done()


async def _run_llm_filtering(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    entries: list[VocabEntry],
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    num_workers: int,
    max_retries: int,
) -> list[VocabEntry]:
    """Run async LLM filtering on adjective entries for visual representability."""
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    log_dir = output_dir / "Lexical" / "prompts"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_visualizable_log.jsonl"

    # Always start fresh -- delete previous log if it exists
    if log_path.exists():
        logger.info("Removing previous LLM filter log: %s", log_path)
        log_path.unlink()

    to_process = [e for e in entries if e.is_valid and e.pos == "ADJ"]

    if not to_process:
        logger.info("No words to process")
        return entries

    logger.info("LLM filtering %d words with %d workers", len(to_process), num_workers)

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    results: dict[str, bool] = {}
    progress = {"done": 0, "total": len(to_process)}

    for e in to_process:
        await queue.put((e.word, e.pos))
    for _ in range(num_workers):
        await queue.put(None)

    with Path(log_path).open("w") as log_file:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        workers = [
            asyncio.create_task(
                _llm_filter_worker(
                    i,
                    queue,
                    client,
                    model,
                    temperature,
                    max_retries,
                    log_file,
                    write_lock,
                    results,
                    progress,
                )
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

    removed = 0
    for entry in entries:
        if entry.word in results and not results[entry.word]:
            entry.is_valid = False
            removed += 1

    logger.info(
        "LLM filter complete: %d removed, %d kept",
        removed,
        len(to_process) - removed,
    )

    return entries


def clean_llm_text(text: str) -> str:
    """Clean LLM output to a single short sentence or phrase."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text.strip()
    if len(text) >= _MIN_LEMMA_LEN and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text.lower()


def _wordnet_antonym(word: str) -> str | None:
    """Look up an antonym via WordNet lemma antonyms."""
    wn_pos = _WN_POS["ADJ"]
    for synset in wn.synsets(word, pos=wn_pos):
        for lemma in synset.lemmas():
            if lemma.name().lower() == word:
                for ant in lemma.antonyms():
                    return ant.name().lower().replace("_", " ")
    return None


def _get_antonym(word: str) -> str:
    """Return an antonym for *word*: WordNet -> hardcoded -> 'not {word}'."""
    ant = _wordnet_antonym(word)
    if ant:
        return ant
    ant = _ADJ_ANTONYMS.get(word)
    if ant:
        return ant
    return f"not {word}"


# ---------------------------------------------------------------------------
#  Async phrase generation workers
# ---------------------------------------------------------------------------


async def _phrase_worker(  # noqa: PLR0913, PLR0915 -- pipeline-level orchestration: many parallel context fields
    worker_id: int,  # noqa: ARG001 -- `worker_id` kept for parallel-worker signature uniformity
    queue: asyncio.Queue,
    client: "AsyncOpenAI",
    model: str,
    temperature: float,
    max_retries: int,
    results: dict[str, dict[str, str]],
    write_lock: asyncio.Lock,
    log_file: TextIO,
    progress: dict,
) -> None:
    """Worker that generates pos/neg phrase pairs for adjectives."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        word = item

        # --- Positive phrase ---
        prompt = PHRASE_PROMPT.format(adjective=word)

        response = await llm_call(
            client,
            model,
            prompt,
            temperature,
            max_tokens=64,
            max_retries=max_retries,
        )

        if response is None:
            article = "an" if word[0].lower() in "aeiou" else "a"
            pos_phrase = f"{article} {word} object"
            pos_action = "LLM_FAILURE_FALLBACK"
        else:
            pos_phrase = clean_llm_text(response)
            if len(pos_phrase.split()) > _MAX_PHRASE_WORDS:
                article = "an" if word[0].lower() in "aeiou" else "a"
                pos_phrase = f"{article} {word} object"
                pos_action = "TOO_LONG_FALLBACK"
            elif word not in pos_phrase:
                article = "an" if word[0].lower() in "aeiou" else "a"
                pos_phrase = f"{article} {word} object"
                pos_action = "WRONG_ADJ_FALLBACK"
            else:
                pos_action = "OK"

        # --- Negative phrase ---
        neg_prompt = NEG_PHRASE_PROMPT.format(adjective=word, pos_phrase=pos_phrase)

        neg_response = await llm_call(
            client,
            model,
            neg_prompt,
            temperature,
            max_tokens=64,
            max_retries=max_retries,
        )

        if neg_response is None:
            antonym = _get_antonym(word)
            article = "an" if antonym[0].lower() in "aeiou" else "a"
            parts = pos_phrase.split()
            noun = parts[-1] if len(parts) >= _MIN_PHRASE_PARTS else "object"
            neg_phrase = f"{article} {antonym} {noun}"
            neg_action = "LLM_FAILURE_ANTONYM_FALLBACK"
        else:
            neg_phrase = clean_llm_text(neg_response)
            if len(neg_phrase.split()) > _MAX_PHRASE_WORDS:
                antonym = _get_antonym(word)
                article = "an" if antonym[0].lower() in "aeiou" else "a"
                parts = pos_phrase.split()
                noun = parts[-1] if len(parts) >= _MIN_PHRASE_PARTS else "object"
                neg_phrase = f"{article} {antonym} {noun}"
                neg_action = "TOO_LONG_ANTONYM_FALLBACK"
            else:
                neg_action = "OK"

        results[word] = {"pos": pos_phrase, "neg": neg_phrase}

        log_entry = {
            "adjective": word,
            "pos_phrase": pos_phrase,
            "neg_phrase": neg_phrase,
            "llm_response_pos": response,
            "llm_response_neg": neg_response,
            "pos_action": pos_action,
            "neg_action": neg_action,
        }
        async with write_lock:
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        progress["done"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            logger.info(
                "Phrase generation: %d / %d (%.1f%%)",
                progress["done"],
                progress["total"],
                100.0 * progress["done"] / progress["total"],
            )
        queue.task_done()


async def generate_phrases_for_words(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    words: list[str],
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    num_workers: int,
    max_retries: int,
) -> dict[str, dict[str, str]]:
    """Generate short pos/neg phrase pairs for all adjectives via LLM.

    Returns ``{word: {"pos": ..., "neg": ...}}`` dict.
    """
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    log_dir = output_dir / "Lexical" / "prompts"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "adj_phrases_log.jsonl"

    if log_path.exists():
        logger.info("Removing previous phrase log: %s", log_path)
        log_path.unlink()

    logger.info(
        "Generating phrases for %d adjective words with %d workers",
        len(words),
        num_workers,
    )

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    results: dict[str, dict[str, str]] = {}
    progress = {"done": 0, "total": len(words)}

    for w in words:
        await queue.put(w)
    for _ in range(num_workers):
        await queue.put(None)

    with Path(log_path).open("w") as log_file:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        workers = [
            asyncio.create_task(
                _phrase_worker(
                    i,
                    queue,
                    client,
                    model,
                    temperature,
                    max_retries,
                    results,
                    write_lock,
                    log_file,
                    progress,
                )
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

    logger.info("Phrase generation complete: %d phrases", len(results))
    return results


def _fallback_phrases(words: list[str]) -> dict[str, dict[str, str]]:
    """Generate deterministic fallback pos/neg phrase pairs without LLM."""
    result: dict[str, dict[str, str]] = {}
    for w in words:
        antonym = _get_antonym(w)
        article_pos = "an" if w[0].lower() in "aeiou" else "a"
        pos_sent = f"{article_pos} {w} object"
        article_neg = "an" if antonym[0].lower() in "aeiou" else "a"
        neg_sent = f"{article_neg} {antonym} object"
        result[w] = {"pos": pos_sent, "neg": neg_sent}
    return result


# ---------------------------------------------------------------------------
#  POS sanity check (LLM-based)
# ---------------------------------------------------------------------------


def _parse_pos_check_response(response: str, target_pos: str) -> tuple[bool, str]:
    """Parse LLM POS-check response into (is_primary, reason)."""
    is_primary = True
    reason = ""
    tag = f"PRIMARY_{target_pos}:"

    for line in response.splitlines():
        line_stripped = line.strip()
        line_upper = line_stripped.upper()
        if line_upper.startswith(tag):
            value = line_upper.split(":", 1)[1].strip()
            if value == "NO":
                is_primary = False
        elif line_upper.startswith("REASON:"):
            reason = line_stripped.split(":", 1)[1].strip()

    return is_primary, reason


async def _pos_check_worker(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    worker_id: int,  # noqa: ARG001 -- `worker_id` kept for parallel-worker signature uniformity
    queue: asyncio.Queue,
    client: "AsyncOpenAI",
    model: str,
    temperature: float,
    max_retries: int,
    results: dict[str, tuple[bool, str]],
    write_lock: asyncio.Lock,
    log_file: TextIO,
    progress: dict,
    target_pos: str,
) -> None:
    """Worker that verifies a word is primarily used as *target_pos*."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        word = item
        prompt = _POS_CHECK_PROMPT.format(
            word=word,
            pos_article=_POS_ARTICLES[target_pos],
            pos_name=_POS_NAMES[target_pos],
            pos_tag=target_pos,
        )

        response = await llm_call(
            client,
            model,
            prompt,
            temperature,
            max_tokens=128,
            max_retries=max_retries,
        )

        if response is None:
            is_primary, reason = True, "LLM_FAILURE_KEPT"
            action = "LLM_FAILURE_KEPT"
        else:
            is_primary, reason = _parse_pos_check_response(response, target_pos)
            action = "PASS" if is_primary else "FLAGGED"

        results[word] = (is_primary, reason)

        log_entry = {
            "word": word,
            "pos": target_pos,
            "is_primary": is_primary,
            "llm_response": response,
            "action": action,
            "reason": reason,
        }
        async with write_lock:
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        progress["done"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            logger.info(
                "POS sanity check (%s): %d / %d (%.1f%%)",
                target_pos,
                progress["done"],
                progress["total"],
                100.0 * progress["done"] / progress["total"],
            )
        queue.task_done()


async def _run_pos_sanity_check(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    words: list[str],
    target_pos: str,
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    num_workers: int,
    max_retries: int,
) -> list[str]:
    """Verify via LLM that each word is primarily used as *target_pos*.

    Returns the list of words that pass the check.  Flagged words are logged
    as warnings and removed.
    """
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    log_dir = output_dir / "Lexical" / "prompts"
    log_dir.mkdir(parents=True, exist_ok=True)
    pos_label = _POS_NAMES[target_pos]
    log_path = log_dir / f"{pos_label}_pos_check_log.jsonl"

    if log_path.exists():
        logger.info("Removing previous POS check log: %s", log_path)
        log_path.unlink()

    logger.info(
        "POS sanity check: verifying %d words as %s with %d workers",
        len(words),
        target_pos,
        num_workers,
    )

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    results: dict[str, tuple[bool, str]] = {}
    progress = {"done": 0, "total": len(words)}

    for w in words:
        await queue.put(w)
    for _ in range(num_workers):
        await queue.put(None)

    with Path(log_path).open("w") as log_file:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        workers = [
            asyncio.create_task(
                _pos_check_worker(
                    i,
                    queue,
                    client,
                    model,
                    temperature,
                    max_retries,
                    results,
                    write_lock,
                    log_file,
                    progress,
                    target_pos,
                )
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

    # Partition into passed / flagged
    passed = []
    flagged = []
    for w in words:
        is_primary, reason = results.get(w, (True, ""))
        if is_primary:
            passed.append(w)
        else:
            flagged.append((w, reason))

    if flagged:
        logger.warning(
            "POS sanity check (%s): %d / %d words REMOVED:",
            target_pos,
            len(flagged),
            len(words),
        )
        for word, reason in flagged:
            logger.warning("  '%s' -- %s", word, reason)
    else:
        logger.info(
            "POS sanity check (%s): all %d words pass",
            target_pos,
            len(words),
        )

    return passed


# ---------------------------------------------------------------------------
#  Word list generation
# ---------------------------------------------------------------------------


def generate_word_list(  # noqa: PLR0913 -- pipeline-level orchestration: many parallel context fields
    entries: list[VocabEntry],
    sentences: dict[str, dict[str, str]],
    max_words: int = 80,
    name: str = "Dataset",
    seed: int = 42,
    bin_edges: list[int] | None = None,
) -> dict:
    """Generate word_list.json from already-lemmatized adjective entries.

    *sentences* maps each word to ``{"pos": ..., "neg": ...}``.
    When *max_words* <= 0, all valid entries are kept (no sampling).
    """
    valid = [e for e in entries if e.is_valid]

    sampled = stratified_sample(valid, max_words, seed=seed) if max_words > 0 else valid

    words = {e.word: sentences.get(e.word, {"pos": "", "neg": ""}) for e in sampled}

    word_bins: dict[str, dict] = {}
    for e in sampled:
        word_bins[e.word] = {
            "bin_index": e.freq_bin,
            "bin_label": e.bin_label,
            "count": e.count,
        }

    logger.info("lex_adjectives: %d words selected", len(words))

    return {
        "description": (
            f"Adjectives for lexical 4-AFC trials, auto-discovered from {name} vocabulary. Frequency-stratified."
        ),
        "words": words,
        "frequency_metadata": {
            "bin_edges": bin_edges or DEFAULT_BIN_EDGES,
            "word_bins": word_bins,
        },
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915 -- pipeline orchestration: complexity matches the spec it implements
    parser = argparse.ArgumentParser(
        description="Build frequency-stratified word lists for the lexical adjective task.",
    )
    parser.add_argument(
        "--vocab-dir",
        required=True,
        help="Path to vocabulary directory (containing longtail_wordlist.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory root. Defaults to --vocab-dir (in-place). "
        "When given, a subdirectory named {name}_{timestamp} is created inside it.",
    )
    parser.add_argument(
        "--name",
        default="Dataset",
        help="Dataset name for descriptions (e.g., 'COCO').",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=80,
        help="Maximum words to select per POS (default: 80). Set to 0 to keep all valid words (no sampling).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    # LLM filtering arguments
    llm_group = parser.add_argument_group(
        "LLM filtering",
        "Optional LLM-based filtering for visual representability. "
        "Requires a running vLLM server. Skipped if --api-base is not provided.",
    )
    llm_group.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="vLLM server base URL (e.g., http://localhost:8000/v1). If not provided, LLM filtering is skipped.",
    )
    llm_group.add_argument(
        "--api-key",
        type=str,
        default="dummy",
        help="API key for the vLLM server (default: 'dummy').",
    )
    llm_group.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-26B-A4B-it",
        help="Model name served by the vLLM server (default: google/gemma-4-26B-A4B-it).",
    )
    llm_group.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of async LLM workers (default: 20).",
    )
    llm_group.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM call (default: 3).",
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0).",
    )

    args = parser.parse_args()

    if args.api_base and not args.model:
        parser.error("--model is required when --api-base is provided")

    if args.output_dir is None:
        output_dir = Path(args.vocab_dir)
    else:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir = output_dir / f"{args.name}_{timestamp}"
    assert output_dir.parent.exists(), f"Parent directory does not exist: {output_dir.parent}"
    bin_edges = DEFAULT_BIN_EDGES

    vocab_dir = Path(args.vocab_dir)
    wordlist_path = vocab_dir / "longtail_wordlist.csv"

    # Stage 1: Load vocabulary
    logger.info("Stage 1: Loading vocabulary from %s", wordlist_path)
    entries = load_longtail_csv(wordlist_path)
    logger.info("Loaded %d raw entries", len(entries))

    # Stage 2: Lemmatize and merge inflected forms
    logger.info("Stage 2: Lemmatizing to canonical forms and merging counts")
    lemmatized = lemmatize_and_merge(entries, "ADJ", bin_edges)

    # Stage 3: LLM filtering (optional)
    if args.api_base:
        logger.info(
            "Stage 3: LLM filtering for visual representability (api_base=%s, model=%s)",
            args.api_base,
            args.model,
        )
        lemmatized = asyncio.run(
            _run_llm_filtering(
                lemmatized,
                output_dir=output_dir,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                num_workers=args.num_workers,
                max_retries=args.max_retries,
            )
        )
    else:
        logger.info("Stage 3: LLM filtering skipped (no --api-base provided)")

    # Stage 3.5: Phrase generation
    if args.api_base:
        logger.info(
            "Stage 3.5: Generating phrases via LLM (api_base=%s, model=%s)",
            args.api_base,
            args.model,
        )
        valid_words = [e.word for e in lemmatized if e.is_valid]
        phrases = asyncio.run(
            generate_phrases_for_words(
                valid_words,
                output_dir=output_dir,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                num_workers=args.num_workers,
                max_retries=args.max_retries,
            )
        )
    else:
        logger.info("Stage 3.5: Generating fallback phrases (no LLM)")
        valid_words = [e.word for e in lemmatized if e.is_valid]
        phrases = _fallback_phrases(valid_words)

    # Stage 3.7: POS sanity check (LLM-based, optional)
    if args.api_base:
        logger.info("Stage 3.7: LLM POS sanity check")
        valid_words = [e.word for e in lemmatized if e.is_valid]
        passed = asyncio.run(
            _run_pos_sanity_check(
                valid_words,
                target_pos="ADJ",
                output_dir=output_dir,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                num_workers=args.num_workers,
                max_retries=args.max_retries,
            )
        )
        passed_set = set(passed)
        removed = 0
        for entry in lemmatized:
            if entry.is_valid and entry.word not in passed_set:
                entry.is_valid = False
                removed += 1
        # Also remove phrases for dropped words
        phrases = {w: s for w, s in phrases.items() if w in passed_set}
        if removed:
            logger.info("POS sanity check removed %d ADJ words", removed)
    else:
        logger.info("Stage 3.7: POS sanity check skipped (no --api-base provided)")

    # Stage 4: Generate word list (delete existing output first)
    logger.info("Stage 4: Generating word list")

    out_path = output_dir / "Lexical" / "Adjectives" / "word_list.json"
    if out_path.exists():
        logger.info("Removing previous output: %s", out_path)
        out_path.unlink()
    data = generate_word_list(
        lemmatized,
        sentences=phrases,
        max_words=args.max_words,
        name=args.name,
        seed=args.seed,
        bin_edges=bin_edges,
    )
    write_json(out_path, data)

    logger.info("Done. Output written to %s", output_dir)


if __name__ == "__main__":
    main()
