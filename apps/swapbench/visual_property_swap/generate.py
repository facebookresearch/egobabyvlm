# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Hydra runner for VP-Swap (Visual Property Swap).

Runs the four-property variant introduced in the EgoBabyVLM paper
(App. methods:vp-swap): for each of ``color``, ``material``,
``relative_size``, ``shape``, generate sentence pairs from the corpus's
long-tail visual-word list, swap the property descriptions between the
two words, and ask an LLM whether the swap broke physical accuracy.

Pipeline (per property):

1. Read ``longtail_visualnouns`` (produced by
   ``apps.swapbench.longtail_swap.build_word_lists``); one ``word,freq``
   row per candidate visual noun.
2. Filter "is X representing something physical?" via the LLM gate
   (this stage is shared across properties; we cache the gate output).
3. Within each frequency bin, build per-property sentence-generation
   prompts for word pairs.
4. LLM-generate the (s1, s2) sentence pair per word pair.
5. Swap the per-property descriptions between the two sentences and
   build A/B filter prompts: an unswapped sentence should still be
   judged the more physically accurate one.
6. LLM-filter.
7. Collect the surviving rows into one ``vp_swap_<property>_pairs.txt``.

The output row format matches LT-Swap's ``visualswap`` convention so
that downstream eval code can consume it the same way.

VP-Swap is first-party code: the upstream LT-Swap repo contains an
earlier "size/weight/shape" variant; here we instead cover the four
properties reported in the paper.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore

from apps.benchmark_creation.utils.vllm_server import get_client
from apps.swapbench.utils.llm_runner import run_llm_pool
from apps.swapbench.visual_property_swap.prompts import (
    SUPPORTED_PROPERTIES,
    filter_prompt,
    generation_prompt,
    physical_object_prompt,
)
from core.utils.logging import setup_logging

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

#: Frequency bin edges used by the LT-Swap pipeline; we mirror them so the
#: VP-Swap output bins line up with the WordSwap / InflectionSwap bins.
_FREQ_BIN_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, np.inf])

#: Maximum number of (w1, w2) pairs we generate per frequency bin per property.
_MAX_PAIRS_PER_BIN = 2000

#: Frequency bin distance allowed between the two words of a pair. Legacy
#: defaults to 0; raise this for very small corpora that would otherwise
#: produce too few same-bin pairs.
_MAX_BIN_DISTANCE = 0

#: Number of comma-separated columns we expect in the longtail_visualnouns file.
_VISUALNOUNS_COLS = 2

#: Field count of a worker-pool response row (idx | metadata | response, 4 = 3 + 1).
_PHYSICAL_OBJECT_RESPONSE_COLS = 4

#: Minimum length (in characters) for each half of a split sentence pair.
_MIN_SPLIT_SENTENCE_LEN = 3

#: Minimum field count of a stage-3 generation row before the trailing response.
_GENERATION_MIN_COLS = 7

#: Minimum field count of a stage-6 metadata block (after stripping idx + ground-truth + response).
#: Layout from ``_write_filter_prompts``: ``bin|VISUAL|w1|s1|i1|w2|s2|i2`` (8 fields).
_METADATA_MIN_COLS = 8


@dataclass
class VPSwapConfig:
    """Configuration for one VP-Swap pipeline run."""

    #: Path to the ``longtail_visualnouns`` file emitted by build_word_lists
    #: (one ``word,freq`` row per noun).
    visualnouns_path: str = "???"

    #: Output directory; one file per property is written.
    output_dir: str = "???"

    #: Visual property to generate. Use ``all`` to run all four sequentially.
    visual_property: str = "all"

    #: Random seed for the (w1, w2) pair sampling.
    seed: int = 42

    #: vLLM (or other OpenAI-compatible) server endpoint and model.
    api_host: str = "localhost"
    api_port: int = 8000
    api_key: str = "dummy"
    model: str = "meta-llama/Llama-3.1-405B-Instruct"

    #: Generation parameters.
    temperature: float = 0.7
    max_tokens: int = 256
    max_retries: int = 3
    num_workers: int = 16
    queue_size: int = 64


@dataclass
class VPSwapPipelineConfig:
    """Top-level Hydra config; processor only."""

    processor: VPSwapConfig = field(default_factory=VPSwapConfig)


cs = ConfigStore.instance()
cs.store(name="vp_swap_pipeline", node=VPSwapPipelineConfig)


def _bin_for_freq(freq: int) -> int:
    """Index of the frequency bin ``freq`` falls into."""
    return int(np.where(freq >= _FREQ_BIN_EDGES)[0][-1])


def _read_visual_words(path: Path) -> dict[str, int]:
    """Parse ``longtail_visualnouns`` into ``{word: freq_bin}``.

    Each line is ``word,freq``. Duplicate words are dropped (first
    occurrence wins); rows whose freq doesn't fall in any bin are skipped.
    """
    words: dict[str, int] = {}
    with path.open() as f:
        for raw in f:
            parts = raw.rstrip().split(",")
            if len(parts) != _VISUALNOUNS_COLS:
                continue
            word, freq_str = parts
            try:
                freq_int = int(freq_str)
            except ValueError:
                continue
            try:
                bin_idx = _bin_for_freq(freq_int)
            except (IndexError, ValueError):
                continue
            words.setdefault(word, bin_idx)
    return words


def _write_physical_object_prompts(words: dict[str, int], output: Path) -> None:
    """Stage-1 prompt file: ``word|freq_bin|prompt`` per row."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for word, bin_idx in words.items():
            f.write(f"{word}|{bin_idx}|{physical_object_prompt(word)}\n")


def _parse_yes_no_response(response: str) -> bool:
    """Best-effort parse of an ``[yes]/[no]`` LLM gate response."""
    start = response.rfind("[")
    end = response.rfind("]")
    payload = response[start + 1 : end] if start != -1 and end != -1 else response
    return payload.strip().lower().startswith("y")


def _read_physical_object_responses(path: Path) -> dict[str, int]:
    """Filter ``words`` down to those the LLM judged physical."""
    kept: dict[str, int] = {}
    with path.open() as f:
        for raw in f:
            parts = raw.rstrip().split("|")
            if len(parts) != _PHYSICAL_OBJECT_RESPONSE_COLS:
                continue
            _idx, word, bin_str, response = parts
            if not response or response.startswith("<ERROR"):
                continue
            if _parse_yes_no_response(response):
                try:
                    kept[word] = int(bin_str)
                except ValueError:
                    continue
    return kept


def _write_generation_prompts(
    physical_words: dict[str, int],
    visual_property: str,
    output: Path,
    seed: int,
) -> None:
    """Stage-3 prompt file for ``visual_property``: pair words within bins."""
    pairs_per_bin: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    sorted_words = sorted(physical_words)
    for i, w1 in enumerate(sorted_words):
        b1 = physical_words[w1]
        for w2 in sorted_words[i + 1 :]:
            b2 = physical_words[w2]
            if abs(b2 - b1) > _MAX_BIN_DISTANCE:
                continue
            key = "|".join(sorted((w1, w2)))
            if key in seen:
                continue
            seen.add(key)
            pairs_per_bin[min(b1, b2)].append((w1, w2))

    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w") as f:
        for bin_idx in sorted(pairs_per_bin):
            pairs = pairs_per_bin[bin_idx]
            rng.shuffle(pairs)
            for w1, w2 in pairs[:_MAX_PAIRS_PER_BIN]:
                prompt = generation_prompt(visual_property, w1, w2)
                f.write(f"{bin_idx}|{w1}|{w2}|NOUN|VISUAL|{prompt}\n")
                written += 1
    logger.info("Wrote %d generation prompts to %s", written, output)


def _split_two_sentences(generation: str) -> tuple[str, str] | None:
    """Best-effort split of an LLM ``[s1 . s2]`` response into ``(s1, s2)``.

    Pull the bracketed content, then split on the first hit from a small
    list of sentence-boundary patterns. Heuristic — handles the common LLM
    output shapes for the per-property generation prompts.
    """
    start = generation.rfind("[")
    end = generation.rfind("]")
    if start == -1 or end == -1:
        return None
    body = generation[start + 1 : end]
    body = body.replace("\\", "").replace('"', "").replace("'", "")
    body = " ".join(filter(None, body.split(" ")))
    for pattern in (".", "!", "?", "/", ", but", ", while", ", whereas", ", and ", ",", ";"):
        if pattern in body[:-1]:
            idx = body.find(pattern)
            s1, s2 = body[:idx].strip(), body[idx + len(pattern) + 1 :].strip()
            if len(s1) >= _MIN_SPLIT_SENTENCE_LEN and len(s2) >= _MIN_SPLIT_SENTENCE_LEN:
                return s1, s2
    return None


def _word_indices(s1: str, s2: str, w1: str, w2: str) -> tuple[int, int] | None:
    """Locate ``(w1 in s1)`` and ``(w2 in s2)``, accepting word-pair swap."""
    i1 = s1.find(w1)
    i2 = s2.find(w2)
    if i1 == -1 or i2 == -1:
        i1 = s1.find(w2)
        i2 = s2.find(w1)
        if i1 == -1 or i2 == -1:
            return None
        return i1, i2
    return i1, i2


def _write_filter_prompts(
    generations_path: Path,
    visual_property: str,
    output: Path,
) -> None:
    """Build A/B filter prompts after swapping the two words between sentences."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with generations_path.open() as src, output.open("w") as dst:
        for raw in src:
            parts = raw.rstrip().split("|")
            if len(parts) < _GENERATION_MIN_COLS:
                continue
            _idx, bin_str, w1, w2, _pos, _rule, *generation_parts = parts
            generation = "|".join(generation_parts)
            sentences = _split_two_sentences(generation)
            if sentences is None:
                continue
            s1, s2 = sentences
            indices = _word_indices(s1, s2, w1, w2)
            if indices is None:
                continue
            i1, i2 = indices
            # Swap the words between the two sentences. After the swap, the
            # original (s1, s2) should still read as more physically accurate.
            ss1 = s1[:i1] + w2 + s1[i1 + len(w1) :]
            ss2 = s2[:i2] + w1 + s2[i2 + len(w2) :]
            if w1 in ss1 or w2 in ss2:
                continue
            p1 = filter_prompt(visual_property, s1, ss1)
            p11 = filter_prompt(visual_property, ss1, s1)
            p2 = filter_prompt(visual_property, s2, ss2)
            p22 = filter_prompt(visual_property, ss2, s2)
            prompts = f"{p1}/{p11}/{p2}/{p22}/A/B/A/B"
            dst.write(f"{bin_str}|VISUAL|{w1}|{s1}|{i1}|{w2}|{s2}|{i2}|{prompts}\n")
            written += 1
    logger.info("Wrote %d filter prompts to %s", written, output)


def _format_answer(response: str) -> str | None:
    """Pull an A/B verdict out of an LLM response (looks for a single-letter token in brackets)."""
    start = response.rfind("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return response if response in ("A", "B") else None
    payload = response[start + 1 : end].replace(" ", "").upper()
    return payload if payload in ("A", "B") else None


def _retrieve_correct_visualswap_pairs(  # noqa: C901, PLR0912  -- linear stage glue, splitting hurts readability
    filter_responses_path: Path,
    final_pairs_path: Path,
) -> int:
    """Group worker-pool responses by input-line idx; emit metadata for accepted pairs.

    Each input row to ``run_llm_pool`` had 4 ``A``/``B`` filter prompts;
    the worker pool writes one ``{i}-{j}|{metadata}|{ground_truth}|{response}``
    row per prompt. We regroup by ``i``, sort by ``j``, and accept the
    pair only if every response's bracketed verdict equals its expected
    ground truth — same logic as ``mp_utils.process_filtering_file``.
    """
    if not filter_responses_path.exists():
        msg = f"missing filter responses: {filter_responses_path}"
        raise FileNotFoundError(msg)
    grouped: dict[str, list[tuple[int, str, str, str]]] = {}
    with filter_responses_path.open() as src:
        for raw in src:
            line = raw.rstrip("\n")
            if not line:
                continue
            head, sep, _ = line.partition("|")
            if not sep or "-" not in head:
                continue
            try:
                i_str, j_str = head.split("-", 1)
                j = int(j_str)
            except ValueError:
                continue
            after_idx = line[len(head) + 1 :]
            try:
                metadata, ground_truth, response = after_idx.rsplit("|", 2)
            except ValueError:
                continue
            grouped.setdefault(i_str, []).append((j, metadata, ground_truth, response))

    accepted = 0
    seen_pairs: set[str] = set()
    final_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with final_pairs_path.open("w") as dst:
        for items in grouped.values():
            items.sort(key=lambda x: x[0])
            store_metadata: str | None = None
            all_match = True
            for _j, metadata, ground_truth, response in items:
                if _format_answer(response) != ground_truth:
                    all_match = False
                    break
                if store_metadata is None:
                    store_metadata = metadata
            if not all_match or store_metadata is None:
                continue
            # metadata layout from _write_filter_prompts: bin|VISUAL|w1|s1|i1|w2|s2|i2
            parts = store_metadata.split("|")
            if len(parts) < _METADATA_MIN_COLS:
                continue
            bin_str, _rule, w1, g1, ig1, w2, g2, ig2 = parts[:_METADATA_MIN_COLS]
            if w1 == w2:
                continue
            key = "-".join(sorted((w1, w2)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            dst.write(f"{bin_str}|VISUAL|{w1}|{g1}|{ig1}|{w2}|{g2}|{ig2}\n")
            accepted += 1
    logger.info("Accepted %d pairs into %s", accepted, final_pairs_path)
    return accepted


async def _filter_physical_objects(
    cfg: VPSwapConfig,
    visualnouns_path: Path,
    output_dir: Path,
    client: AsyncOpenAI,
) -> dict[str, int]:
    """Stage 1 (shared across properties): ask the LLM gate which words are physical."""
    physical_prompts = output_dir / "physical_object_prompts.txt"
    physical_responses = output_dir / "physical_object_responses.txt"

    words = _read_visual_words(visualnouns_path)
    logger.info("Loaded %d candidate visual words", len(words))
    _write_physical_object_prompts(words, physical_prompts)

    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=physical_prompts,
        output_path=physical_responses,
        temperature=0.0,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )
    physical_words = _read_physical_object_responses(physical_responses)
    logger.info(
        "LLM gate kept %d / %d candidate words as physical",
        len(physical_words),
        len(words),
    )
    return physical_words


async def _vp_swap_one_property(
    cfg: VPSwapConfig,
    physical_words: dict[str, int],
    visual_property: str,
    output_dir: Path,
    client: AsyncOpenAI,
) -> None:
    """Stages 3-7 for one visual property."""
    sentence_prompts = output_dir / f"vp_swap_{visual_property}_sentence_prompts.txt"
    sentence_generations = output_dir / f"vp_swap_{visual_property}_sentence_generations.txt"
    pairs_filtering_prompts = output_dir / f"vp_swap_{visual_property}_pairs_filtering_prompts.txt"
    pairs_to_be_filtered = output_dir / f"vp_swap_{visual_property}_pairs_to_be_filtered.txt"
    final_pairs = output_dir / f"vp_swap_{visual_property}_pairs.txt"

    logger.info("[%s] Stage 3/7: build sentence-generation prompts", visual_property)
    _write_generation_prompts(physical_words, visual_property, sentence_prompts, cfg.seed)

    logger.info("[%s] Stage 4/7: LLM-generate sentence pairs", visual_property)
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=sentence_prompts,
        output_path=sentence_generations,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )

    logger.info("[%s] Stage 5/7: build A/B filter prompts after swap", visual_property)
    _write_filter_prompts(sentence_generations, visual_property, pairs_filtering_prompts)

    logger.info("[%s] Stage 6/7: LLM-filter pairs", visual_property)
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=pairs_filtering_prompts,
        output_path=pairs_to_be_filtered,
        temperature=0.0,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )

    logger.info("[%s] Stage 7/7: collect correct pairs", visual_property)
    _retrieve_correct_visualswap_pairs(pairs_to_be_filtered, final_pairs)


@hydra.main(version_base=None, config_name="vp_swap_pipeline")
def main(config: VPSwapPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    cfg = config.processor

    if cfg.visual_property != "all" and cfg.visual_property not in SUPPORTED_PROPERTIES:
        msg = f"visual_property must be 'all' or one of {SUPPORTED_PROPERTIES}, got {cfg.visual_property!r}"
        raise ValueError(msg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    properties = SUPPORTED_PROPERTIES if cfg.visual_property == "all" else (cfg.visual_property,)

    async def _run() -> None:
        client = get_client(host=cfg.api_host, port=cfg.api_port, api_key=cfg.api_key)
        physical_words = await _filter_physical_objects(cfg, Path(cfg.visualnouns_path), output_dir, client)
        for prop in properties:
            await _vp_swap_one_property(cfg, physical_words, prop, output_dir, client)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
