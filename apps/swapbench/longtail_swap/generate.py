# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Hydra runner for the LT-Swap generators.

Orchestrates the upstream LT-Swap generation pipeline against an
OpenAI-compatible vLLM server, replacing the upstream notebook
(``generate_task/generation_notebook.ipynb``) with our async
``apps.swapbench.utils.llm_runner.run_llm_pool`` worker pool. The
on-disk intermediate-file layout matches upstream ``mp_main``
byte-for-byte, so the upstream prep + filter scripts run unchanged.

Per-task pipeline stages:

* ``wordswap``:
    1. ``wordswap_sentence_prompts`` -> ``wordswap_sentence_prompts.txt``
    2. LLM generation -> ``wordswap_sentence_generations.txt``
    3. ``wordswap_pairs_and_filtering_prompts`` -> ``wordswap_sentence_pairs_filtering_prompts.txt``
    4. LLM filtering -> ``wordswap_sentence_pairs_to_be_filtered.with_idx.txt``
    5. ``_collect_pairs_from_filter_responses`` -> ``wordswap_pairs.txt``

* ``syntax`` (covers InflectionSwap + AgreementSwap, which share a pipeline):
    1. ``inflpairs_filtering_prompts`` -> ``syntax_words_filtering_prompts.txt``
    2. LLM generation -> ``syntax_words_to_be_filtered.txt``
    3. ``syntax_sentence_prompts`` -> ``syntax_sentence_pairs_prompts.txt``
    4. LLM generation -> ``syntax_sentence_generations.txt``
    5. ``syntax_get_generation_and_filtering_prompts`` -> ``syntax_sentence_pairs_filtering_prompts.txt``
    6. LLM filtering -> ``syntax_sentence_pairs_to_be_filtered.with_idx.txt``
    7. ``_collect_pairs_from_filter_responses`` -> ``syntax_sentence_pairs_filtered.txt``
    8. ``split_agrswap_infswap`` -> ``inflswap_pairs.txt`` + ``agrswap_pairs.txt``

Each upstream script is invoked as a subprocess so we never have to
modify the upstream code; this keeps the diff against upstream small and
makes refresh trivial.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from apps.benchmark_creation.utils.vllm_server import get_client
from apps.swapbench.utils.llm_runner import run_llm_pool
from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)

#: Filesystem path of the upstream generate_task directory. We prepend
#: it to ``PYTHONPATH`` when invoking upstream scripts so their sibling-
#: relative imports (``from preprocessing_utils import …``) resolve.
_UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "lt_swap" / "generate_task"

#: Allowed values for ``processor.task``.
_ALLOWED_TASKS = ("wordswap", "syntax")


@dataclass
class LTSwapConfig:
    """Configuration for one LT-Swap pipeline run."""

    #: Either ``wordswap`` (runs WordSwap end-to-end) or ``syntax``
    #: (runs InflectionSwap + AgreementSwap end-to-end, which share a pipeline).
    task: str = "wordswap"

    #: Directory of the wordlist artifacts produced by ``build_word_lists``.
    #: Must contain ``longtail_wordlist``, ``longtail_inflpairs``, and
    #: ``vocabulary``. Generate it once per corpus, then reuse for each task.
    wordlists_dir: str = "???"

    #: Directory the run writes its prompt/output files to. Restartable;
    #: completed prompts are skipped on rerun.
    output_dir: str = "???"

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
class LTSwapPipelineConfig:
    """Top-level Hydra config; processor only."""

    processor: LTSwapConfig = field(default_factory=LTSwapConfig)


cs = ConfigStore.instance()
cs.store(name="lt_swap_pipeline", node=LTSwapPipelineConfig)


def _run_upstream(script: str, args: list[str]) -> None:
    """Invoke an upstream ``generate_task/*.py`` script as a subprocess.

    Subprocess (rather than direct import) so the upstream code stays
    unchanged — refresh is just a re-copy. PYTHONPATH is extended
    so the upstream sibling-relative imports
    (``from preprocessing_utils import …``) resolve.
    """
    cmd = [sys.executable, "-m", script, *args]
    logger.info("Running upstream script: %s", " ".join(cmd))
    env = os.environ.copy()
    pythonpath = str(_UPSTREAM_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0:
        msg = (
            f"Upstream script {script} failed (exit {completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        raise RuntimeError(msg)
    if completed.stdout.strip():
        logger.debug("[%s stdout]\n%s", script, completed.stdout.strip())


def _strip_idx_prefix(src_with_idx: Path, dst_clean: Path) -> None:
    """Read ``src_with_idx`` and write ``dst_clean`` with the leading ``<digits>|`` stripped.

    Upstream parsers (``wordswap_pairs_and_filtering_prompts``,
    ``syntax_sentence_prompts``, etc.) expect rows in the same field count as
    the original input file; our async LLM driver prepends an ``<idx>|`` for
    resumability. We keep the with-idx file on disk (so subsequent
    ``run_llm_pool`` calls can detect already-done rows) and write a stripped
    sibling for the upstream stages to read.
    """
    if not src_with_idx.exists():
        msg = f"missing input for strip: {src_with_idx}"
        raise FileNotFoundError(msg)
    n_total = 0
    with src_with_idx.open() as src, dst_clean.open("w") as dst:
        for raw in src:
            n_total += 1
            head, sep, rest = raw.partition("|")
            if sep and head.isdigit():
                dst.write(rest)
            else:
                dst.write(raw)
    logger.info("Wrote stripped %s (%d rows) for upstream consumption", dst_clean.name, n_total)


def _format_answer(response: str) -> str | None:
    """Pull an A/B verdict out of an LLM response.

    Looks for a single-letter token in brackets; falls back to bare ``A``/``B``.
    Mirrors upstream ``mp_utils.format_answer`` and the same logic used by
    VP-Swap's filter step, kept in one place so all three pipelines treat
    LLM verdicts identically.
    """
    start = response.rfind("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return response if response in ("A", "B") else None
    payload = response[start + 1 : end].replace(" ", "").upper()
    return payload or None


def _collect_pairs_from_filter_responses(filter_with_idx: Path, final_pairs: Path) -> int:  # noqa: C901  -- linear stage glue, splitting hurts readability
    """Group worker-pool responses by input-line idx; emit metadata for rows where all verdicts match.

    Reimplements upstream ``mp_utils.process_filtering_file`` against our
    with-idx output format. The upstream ``retrieve_correct_pairs.py`` script
    in ``third_party`` is broken — it expects 24 unsplit fields per row, but
    the worker pool splits each input row into N per-prompt calls. The
    notebook glues them back together with ``process_filtering_file``; we do
    the same thing here so the wordswap / syntax pipelines emit the
    8-field metadata rows downstream eval consumes (and that
    ``split_agrswap_infswap`` reads).

    Each worker output row is ``{i}-{j}|{metadata}|{ground_truth}|{response}``.
    We group by ``i``, sort by ``j``, and accept the row only if every
    response's bracketed verdict equals its expected ground truth.
    """
    if not filter_with_idx.exists():
        msg = f"missing filter responses: {filter_with_idx}"
        raise FileNotFoundError(msg)
    grouped: dict[str, list[tuple[int, str, str, str]]] = {}
    with filter_with_idx.open() as src:
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
    final_pairs.parent.mkdir(parents=True, exist_ok=True)
    with final_pairs.open("w") as dst:
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
            if all_match and store_metadata is not None:
                dst.write(f"{store_metadata}\n")
                accepted += 1
    logger.info("Accepted %d / %d filter groups into %s", accepted, len(grouped), final_pairs)
    return accepted


async def _wordswap(cfg: LTSwapConfig, output_dir: Path) -> None:
    """Run the full WordSwap pipeline."""
    wordlists_dir = Path(cfg.wordlists_dir)
    sentence_prompts = output_dir / "wordswap_sentence_prompts.txt"
    sentence_generations_with_idx = output_dir / "wordswap_sentence_generations.with_idx.txt"
    sentence_generations = output_dir / "wordswap_sentence_generations.txt"
    pairs_filtering_prompts = output_dir / "wordswap_sentence_pairs_filtering_prompts.txt"
    pairs_to_be_filtered_with_idx = output_dir / "wordswap_sentence_pairs_to_be_filtered.with_idx.txt"
    final_pairs = output_dir / "wordswap_pairs.txt"

    logger.info("Stage 1/5: build sentence-generation prompts")
    _run_upstream(
        "wordswap_sentence_prompts",
        [
            f"--wordlist={wordlists_dir / 'longtail_wordlist'}",
            f"--output_file={sentence_prompts}",
        ],
    )

    logger.info("Stage 2/5: LLM-generate sentences")
    client = get_client(host=cfg.api_host, port=cfg.api_port, api_key=cfg.api_key)
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=sentence_prompts,
        output_path=sentence_generations_with_idx,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )
    _strip_idx_prefix(sentence_generations_with_idx, sentence_generations)

    logger.info("Stage 3/5: build pair-filtering prompts")
    _run_upstream(
        "wordswap_pairs_and_filtering_prompts",
        [
            f"--input_file={sentence_generations}",
            f"--output_file={pairs_filtering_prompts}",
            f"--voc_file={wordlists_dir / 'vocabulary'}",
        ],
    )

    logger.info("Stage 4/5: LLM-filter pairs")
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=pairs_filtering_prompts,
        output_path=pairs_to_be_filtered_with_idx,
        temperature=0.0,  # filtering is single-token A/B classification
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )

    logger.info("Stage 5/5: collect correct pairs")
    _collect_pairs_from_filter_responses(pairs_to_be_filtered_with_idx, final_pairs)
    logger.info("WordSwap pipeline complete: %s", final_pairs)


async def _syntax(cfg: LTSwapConfig, output_dir: Path) -> None:
    """Run the full InflectionSwap + AgreementSwap pipeline (shared stages)."""
    wordlists_dir = Path(cfg.wordlists_dir)
    inflpairs_prompts = output_dir / "syntax_words_filtering_prompts.txt"
    inflpairs_filtered_with_idx = output_dir / "syntax_words_to_be_filtered.with_idx.txt"
    inflpairs_filtered = output_dir / "syntax_words_to_be_filtered.txt"
    sentence_prompts = output_dir / "syntax_sentence_pairs_prompts.txt"
    sentence_generations_with_idx = output_dir / "syntax_sentence_generations.with_idx.txt"
    sentence_generations = output_dir / "syntax_sentence_generations.txt"
    pairs_filtering_prompts = output_dir / "syntax_sentence_pairs_filtering_prompts.txt"
    pairs_to_be_filtered_with_idx = output_dir / "syntax_sentence_pairs_to_be_filtered.with_idx.txt"
    syntax_filtered_pairs = output_dir / "syntax_sentence_pairs_filtered.txt"
    inflswap_final = output_dir / "inflswap_pairs.txt"
    agrswap_final = output_dir / "agrswap_pairs.txt"

    logger.info("Stage 1/8: build inflpairs filtering prompts")
    _run_upstream(
        "inflpairs_filtering_prompts",
        [
            f"--inflpairs={wordlists_dir / 'longtail_inflpairs'}",
            f"--output_file={inflpairs_prompts}",
        ],
    )

    logger.info("Stage 2/8: LLM-filter inflection pairs")
    client = get_client(host=cfg.api_host, port=cfg.api_port, api_key=cfg.api_key)
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=inflpairs_prompts,
        output_path=inflpairs_filtered_with_idx,
        temperature=0.0,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )
    _strip_idx_prefix(inflpairs_filtered_with_idx, inflpairs_filtered)

    logger.info("Stage 3/8: build sentence-generation prompts")
    _run_upstream(
        "syntax_sentence_prompts",
        [
            f"--inflpairs={inflpairs_filtered}",
            f"--output_file={sentence_prompts}",
        ],
    )

    logger.info("Stage 4/8: LLM-generate sentences")
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=sentence_prompts,
        output_path=sentence_generations_with_idx,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )
    _strip_idx_prefix(sentence_generations_with_idx, sentence_generations)

    logger.info("Stage 5/8: build pair-filtering prompts")
    _run_upstream(
        "syntax_get_generation_and_filtering_prompts",
        [
            f"--input_file={sentence_generations}",
            f"--output_file={pairs_filtering_prompts}",
            f"--voc_file={wordlists_dir / 'vocabulary'}",
        ],
    )

    logger.info("Stage 6/8: LLM-filter pairs")
    await run_llm_pool(
        client=client,
        model=cfg.model,
        input_path=pairs_filtering_prompts,
        output_path=pairs_to_be_filtered_with_idx,
        temperature=0.0,
        max_tokens=cfg.max_tokens,
        max_retries=cfg.max_retries,
        num_workers=cfg.num_workers,
        queue_size=cfg.queue_size,
    )

    logger.info("Stage 7/8: collect correct inflswap + agrswap pairs")
    _collect_pairs_from_filter_responses(pairs_to_be_filtered_with_idx, syntax_filtered_pairs)

    logger.info("Stage 8/8: split into inflswap + agrswap files")
    _run_upstream(
        "split_agrswap_infswap",
        [str(syntax_filtered_pairs), str(inflswap_final), str(agrswap_final)],
    )
    logger.info("Syntax pipeline complete: %s, %s", inflswap_final, agrswap_final)


@hydra.main(version_base=None, config_name="lt_swap_pipeline")
def main(config: LTSwapPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    cfg = config.processor
    if cfg.task not in _ALLOWED_TASKS:
        msg = f"task must be one of {_ALLOWED_TASKS}, got {cfg.task!r}"
        raise ValueError(msg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = _wordswap if cfg.task == "wordswap" else _syntax
    asyncio.run(runner(cfg, output_dir))


if __name__ == "__main__":
    main()
