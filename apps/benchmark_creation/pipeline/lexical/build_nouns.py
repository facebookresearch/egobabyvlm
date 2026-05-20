# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build frequency-stratified word lists for the lexical noun task."""

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

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from apps.benchmark_creation.pipeline.lexical.constants import (
    LLM_FILTER_PROMPT,
    VALID_CATEGORIES,
    WORDNET_SUPER_CATEGORIES,
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
logger = logging.getLogger("build_lexical")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


#: Minimum lemma length to keep after lemmatisation.
_MIN_LEMMA_LEN = 2
#: Categories with fewer than this many entries get folded into "miscellaneous".
_MIN_CATEGORY_SIZE = 5

_CATEGORY_SYNSETS: dict[str, str] | None = None


# ---------------------------------------------------------------------------
#  Lemmatization: merge noun variants (e.g. "dog"/"dogs") into one entry
# ---------------------------------------------------------------------------


def _is_named_entity(word: str) -> bool:
    """Return True if the word's primary WordNet noun sense is a named entity.

    Catches proper nouns like city names (Venice), country names (France),
    people (Shakespeare), etc. that have WordNet entries as *instances*
    rather than generic concepts.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    first = synsets[0]
    # Instance hypernyms indicate a specific named entity
    return bool(first.instance_hypernyms())


def lemmatize_nouns(
    entries: list[VocabEntry],
    bin_edges: list[int] | None = None,
) -> list[VocabEntry]:
    """Collapse morphological noun variants into canonical lemmas.

    E.g. "dog" and "dogs" are merged into a single "dog" entry whose count
    is the sum of both.  Only entries tagged as NOUN with a valid WordNet
    noun synset are kept.

    Also filters out:
    - Named entities whose primary WordNet sense is an instance (e.g. Venice,
      Shakespeare) rather than a generic concept.

    Returns a new list of VocabEntry objects -- one per unique lemma -- with
    merged counts and freshly-assigned frequency bins.
    """
    wnl = WordNetLemmatizer()

    named_entities: list[str] = []
    lemma_counts: dict[str, int] = {}
    for e in entries:
        if not e.is_valid or e.pos != "NOUN":
            continue
        # Lowercase before lemmatizing so "Animals" -> "animal", not "Animals"
        lemma = wnl.lemmatize(e.word.lower(), pos=wn.NOUN)
        if not lemma or len(lemma) < _MIN_LEMMA_LEN or not lemma.isalpha():
            continue
        if not wn.synsets(lemma, pos=wn.NOUN):
            continue
        # Filter named entities (cities, countries, people, etc.)
        if _is_named_entity(lemma):
            named_entities.append(lemma)
            continue
        lemma_counts[lemma] = lemma_counts.get(lemma, 0) + e.count

    merged = [
        VocabEntry(word=lemma, count=count, rank=0, pos="NOUN", is_valid=True) for lemma, count in lemma_counts.items()
    ]

    merged = assign_frequency_bins(merged, bin_edges)

    raw_count = sum(1 for e in entries if e.is_valid and e.pos == "NOUN")
    logger.info(
        "Noun lemmatization: %d raw NOUN entries -> %d unique lemmas (%d named entities removed: %s)",
        raw_count,
        len(merged),
        len(named_entities),
        ", ".join(sorted(set(named_entities))[:20]) or "(none)",
    )
    return merged


# ---------------------------------------------------------------------------
#  WordNet noun categorization
# ---------------------------------------------------------------------------


def _get_category_synsets() -> dict:
    """Lazily build {wn.synset: category_name} mapping."""
    global _CATEGORY_SYNSETS  # noqa: PLW0603 -- module-level lazy cache populated once on first call.
    if _CATEGORY_SYNSETS is None:
        _CATEGORY_SYNSETS = {}
        for synset_name, cat in WORDNET_SUPER_CATEGORIES.items():
            try:
                ss = wn.synset(synset_name)
                _CATEGORY_SYNSETS[ss] = cat
            except nltk.corpus.reader.wordnet.WordNetError:
                logger.warning("WordNet synset not found: %s", synset_name)
    return _CATEGORY_SYNSETS


def categorize_nouns(entries: list[VocabEntry]) -> list[VocabEntry]:  # noqa: C901, PLR0912 -- pipeline orchestration: complexity matches the spec it implements
    """Categorize NOUNs via WordNet hypernym lookup."""
    cat_synsets = _get_category_synsets()

    for entry in entries:
        if not entry.is_valid or entry.pos != "NOUN":
            continue
        synsets = wn.synsets(entry.word, pos=wn.NOUN)
        if not synsets:
            entry.category = "miscellaneous"
            continue

        matched = False
        for ss in synsets[:2]:
            hypernyms = ss.hypernym_paths()
            for path in hypernyms:
                for ancestor in reversed(path):
                    if ancestor in cat_synsets:
                        entry.category = cat_synsets[ancestor]
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if not matched:
            entry.category = "miscellaneous"

    cat_counts: dict[str, int] = {}
    for e in entries:
        if e.is_valid and e.pos == "NOUN" and e.category:
            cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
    logger.info("Noun categories: %s", cat_counts)

    tiny_cats = {c for c, n in cat_counts.items() if n < _MIN_CATEGORY_SIZE and c != "miscellaneous"}
    if tiny_cats:
        logger.info("Merging tiny categories into miscellaneous: %s", tiny_cats)
        for entry in entries:
            if entry.is_valid and entry.category in tiny_cats:
                entry.category = "miscellaneous"

    return entries


# ---------------------------------------------------------------------------
#  LLM-based word filtering
# ---------------------------------------------------------------------------


def _parse_llm_filter_response(response: str) -> tuple[bool, str | None]:
    """Parse LLM filter response into (is_safe, new_category_or_None).

    Returns:
        (is_safe, action) where action is one of:
        - None       -> keep current category (CORRECT or parse failure)
        - "WRONG"    -> move to miscellaneous
        - "<cat>"    -> recategorize to the named category
    """
    is_safe = True
    action = None

    for raw_line in response.splitlines():
        line = raw_line.strip().upper()
        if line.startswith("SAFE:"):
            value = line.split(":", 1)[1].strip()
            if value == "NO":
                is_safe = False
        elif line.startswith("CATEGORY:"):
            value = line.split(":", 1)[1].strip()
            if value == "CORRECT":
                action = None
            elif value == "WRONG":
                action = "WRONG"
            elif value.startswith("RECATEGORIZE:"):
                new_cat = value.split(":", 1)[1].strip().lower()
                action = new_cat

    return is_safe, action


async def _llm_filter_worker(  # noqa: C901, PLR0912, PLR0913, PLR0915 -- pipeline-level orchestration: many parallel context fields
    worker_id: int,  # noqa: ARG001 -- `worker_id` kept for parallel-worker signature uniformity
    queue: asyncio.Queue,
    client: "AsyncOpenAI",
    model: str,
    temperature: float,
    max_retries: int,
    log_file: TextIO,
    filtered_file: TextIO,
    recategorized_file: TextIO,
    write_lock: asyncio.Lock,
    results: dict,
    progress: dict,
) -> None:
    """Worker coroutine: pulls (word, category) pairs from the queue."""
    categories_str = ", ".join(VALID_CATEGORIES)

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        word, category = item
        prompt = LLM_FILTER_PROMPT.format(
            word=word,
            category=category,
            categories=categories_str,
        )

        response = await llm_call(
            client,
            model,
            prompt,
            temperature,
            max_tokens=64,
            max_retries=max_retries,
        )

        if response is None:
            is_safe, action_taken = True, None
            action_label = "LLM_FAILURE_KEPT"
        else:
            is_safe, action = _parse_llm_filter_response(response)
            if not is_safe:
                action_taken = "REMOVED_UNSAFE"
                action_label = "REMOVED_UNSAFE"
            elif action == "WRONG":
                action_taken = "RECATEGORIZE_MISC"
                action_label = "RECATEGORIZE_MISC"
            elif action is not None and action in VALID_CATEGORIES:
                action_taken = f"RECATEGORIZE_{action}"
                action_label = f"RECATEGORIZE_{action}"
            elif action is not None:
                action_taken = None
                action_label = "INVALID_CATEGORY_KEPT"
            else:
                action_taken = None
                action_label = "KEPT"

        results[word] = (is_safe, action_taken)

        if not is_safe:
            logger.info("Removing '%s' as not safe (was: %s)", word, category)
        elif action_taken is not None and action_taken == "RECATEGORIZE_MISC":
            logger.info("'%s' recategorized to miscellaneous (was: %s)", word, category)
        elif action_taken is not None and action_taken.startswith("RECATEGORIZE_"):
            new_cat = action_taken[len("RECATEGORIZE_") :]
            logger.info("'%s' recategorized to %s (was: %s)", word, new_cat, category)
        else:
            logger.info("'%s' correctly categorized as %s", word, category)

        log_entry = {
            "word": word,
            "original_category": category,
            "llm_response": response,
            "action": action_label,
        }
        async with write_lock:
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            if not is_safe:
                filtered_file.write(json.dumps(log_entry) + "\n")
                filtered_file.flush()
            elif action_taken is not None:
                recat_entry = {
                    "word": word,
                    "original_category": category,
                    "new_category": action_taken.replace("RECATEGORIZE_", ""),
                    "llm_response": response,
                }
                recategorized_file.write(json.dumps(recat_entry) + "\n")
                recategorized_file.flush()

        progress["done"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            logger.info(
                "LLM filter progress: %d / %d (%.1f%%)",
                progress["done"],
                progress["total"],
                100.0 * progress["done"] / progress["total"],
            )
        queue.task_done()


async def _run_llm_filtering(  # noqa: C901, PLR0913 -- pipeline-level orchestration: many parallel context fields
    entries: list[VocabEntry],
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    num_workers: int,
    max_retries: int,
) -> list[VocabEntry]:
    """Run async LLM filtering on all valid noun entries."""
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    log_dir = output_dir / "Lexical" / "prompts"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_filter_log.jsonl"
    filtered_path = log_dir / "filtered_words.jsonl"
    recategorized_path = log_dir / "recategorized_words.jsonl"

    prior_results: dict[str, tuple[bool, str | None]] = {}
    if log_path.exists():
        with Path(log_path).open() as f:  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
            for raw in f:
                line = raw.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        word = entry["word"]
                        action_label = entry.get("action")
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if action_label == "REMOVED_UNSAFE":
                        prior_results[word] = (False, "REMOVED_UNSAFE")
                    elif action_label == "RECATEGORIZE_MISC":
                        prior_results[word] = (True, "RECATEGORIZE_MISC")
                    elif action_label and action_label.startswith("RECATEGORIZE_"):
                        prior_results[word] = (True, action_label)
                    else:
                        # KEPT, LLM_FAILURE_KEPT, INVALID_CATEGORY_KEPT, etc.
                        prior_results[word] = (True, None)
        logger.info("Resuming: %d words already processed", len(prior_results))

    noun_entries = [
        e for e in entries if e.is_valid and e.pos == "NOUN" and e.category and e.word not in prior_results
    ]

    results: dict[str, tuple[bool, str | None]] = dict(prior_results)

    if not noun_entries:
        logger.info("All words already processed, applying logged decisions only")
        _apply_llm_results(entries, results, total_processed=len(prior_results))
        return entries

    logger.info("LLM filtering %d words with %d workers", len(noun_entries), num_workers)

    queue: asyncio.Queue = asyncio.Queue()
    write_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(noun_entries)}

    for e in noun_entries:
        await queue.put((e.word, e.category))
    for _ in range(num_workers):
        await queue.put(None)

    with (
        Path(log_path).open("a") as log_file,  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        Path(filtered_path).open("a") as filtered_file,  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
        Path(recategorized_path).open("a") as recategorized_file,  # noqa: ASYNC230 -- short startup/teardown I/O dwarfed by LLM call latency
    ):
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
                    filtered_file,
                    recategorized_file,
                    write_lock,
                    results,
                    progress,
                )
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

    _apply_llm_results(entries, results, total_processed=len(noun_entries))

    return entries


def _apply_llm_results(
    entries: list[VocabEntry],
    results: dict[str, tuple[bool, str | None]],
    total_processed: int,
) -> None:
    """Apply (is_safe, action) decisions from the LLM filter to the entry list."""
    removed = 0
    recategorized = 0
    for entry in entries:
        if entry.word not in results:
            continue
        is_safe, action = results[entry.word]
        if not is_safe:
            entry.is_valid = False
            removed += 1
        elif action is not None:
            if action == "RECATEGORIZE_MISC":
                entry.category = "miscellaneous"
            elif action.startswith("RECATEGORIZE_"):
                new_cat = action[len("RECATEGORIZE_") :]
                entry.category = new_cat
            recategorized += 1

    logger.info(
        "LLM filter complete: %d removed, %d recategorized, %d kept",
        removed,
        recategorized,
        max(0, total_processed - removed - recategorized),
    )


# ---------------------------------------------------------------------------
#  Word list generation
# ---------------------------------------------------------------------------


def generate_noun_word_list(
    entries: list[VocabEntry],
    max_per_category: int = 50,
    name: str = "Dataset",
    seed: int = 42,
    bin_edges: list[int] | None = None,
) -> dict:
    """Generate word_list.json for the noun task."""
    nouns = [e for e in entries if e.is_valid and e.pos == "NOUN" and e.category]

    by_cat: dict[str, list[VocabEntry]] = {}
    for e in nouns:
        by_cat.setdefault(e.category, []).append(e)

    categories: dict[str, list[str]] = {}
    word_bins: dict[str, dict] = {}
    for cat, cat_entries in sorted(by_cat.items()):
        sampled = stratified_sample(cat_entries, max_per_category, seed=seed)
        words = [e.word for e in sampled]
        categories[cat] = words
        for e in sampled:
            word_bins[e.word] = {
                "bin_index": e.freq_bin,
                "bin_label": e.bin_label,
                "count": e.count,
            }

    total = sum(len(v) for v in categories.values())
    logger.info("Nouns: %d words across %d categories", total, len(categories))

    return {
        "description": (
            f"Nouns for lexical 4-AFC trials, auto-discovered from {name} vocabulary. Frequency-stratified."
        ),
        "categories": categories,
        "frequency_metadata": {
            "bin_edges": bin_edges or DEFAULT_BIN_EDGES,
            "word_bins": word_bins,
        },
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frequency-stratified word lists for the lexical noun task.",
    )
    parser.add_argument(
        "--vocab-dir",
        required=True,
        help="Path to vocabulary directory (output of create_vocabulary.py, containing longtail_wordlist.csv).",
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
        "--max-per-category",
        type=int,
        default=50,
        help="Maximum words per semantic category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    # LLM filtering arguments (optional -- skipped if --api-base not provided)
    llm_group = parser.add_argument_group(
        "LLM filtering",
        "Optional LLM-based word filtering for safety and category accuracy. "
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

    # Stage 1.5: Lemmatize nouns (merge variants like "dog"/"dogs")
    logger.info("Stage 1.5: Lemmatizing nouns to merge morphological variants")
    entries = lemmatize_nouns(entries, bin_edges)

    # Stage 2: Categorize nouns via WordNet
    logger.info("Stage 2: Categorizing nouns via WordNet")
    entries = categorize_nouns(entries)

    # Stage 3: LLM filtering (optional)
    if args.api_base:
        logger.info("Stage 3: LLM filtering (api_base=%s, model=%s)", args.api_base, args.model)
        entries = asyncio.run(
            _run_llm_filtering(
                entries,
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

    # Stage 4: Generate noun word list
    logger.info("Stage 4: Generating noun word list")
    noun_data = generate_noun_word_list(
        entries,
        max_per_category=args.max_per_category,
        name=args.name,
        seed=args.seed,
        bin_edges=bin_edges,
    )
    write_json(output_dir / "Lexical" / "Nouns" / "word_list.json", noun_data)

    logger.info("Done. Output written to %s", output_dir)


if __name__ == "__main__":
    main()
