#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download Zorro grammatical evaluation data and convert it to BLiMP-style JSONs.

Zorro ships ``.txt`` files of paired good/bad sentences in
``sentences/babyberta/<paradigm>.txt`` (one paradigm per file). The pipeline
expects per-task JSONL files (``<task>.json``) bucketing those paradigms by
phenomenon. This script clones the Zorro repo, runs the conversion, and writes
the bucketed JSONs into the cache.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

ZORRO_REPO_URL = "https://github.com/phueb/Zorro.git"

# Map Zorro paradigm filename prefix -> task bucket name expected by the pipeline.
# Paradigm name = filename without .txt; longest-prefix wins.
PARADIGM_TO_TASK: dict[str, str] = {
    "agreement_subject_verb-": "subject_verb_agreement",
    "agreement_determiner_noun-": "determiner_noun_agreement",
    "anaphor_agreement-": "anaphor_agreement",
    "argument_structure-": "argument_structure",
    "binding-": "binding",
    "case-": "case_subjective_pronoun",
    "ellipsis-": "ellipsis",
    "filler-gap-": "filler_gap",
    "irregular-": "irregular_forms",
    "island-effects-": "island_effects",
    "local_attractor-": "local_attractor",
    "npi_licensing-": "npi_licensing",
    "quantifiers-": "quantifiers",
}


def _format_sentence(line: str) -> str:
    return line.strip().replace(" .", ".").replace(" ?", "?")


def _bucket_for_paradigm(paradigm: str) -> str | None:
    """Return the pipeline task name for a paradigm, or ``None`` if no mapping fits."""
    matches = [task for prefix, task in PARADIGM_TO_TASK.items() if paradigm.startswith(prefix)]
    if not matches:
        return None
    # Longest prefix wins to avoid e.g. ``agreement_subject_verb-`` matching ``agreement_determiner_noun-``.
    return max(
        matches,
        key=lambda task: max(len(p) for p, t in PARADIGM_TO_TASK.items() if t == task),
    )


def _convert(sentences_dir: Path, out_dir: Path) -> None:
    """Convert per-paradigm .txt files into per-task BLiMP-style JSONLs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict[str, str]]] = {}

    for txt in sorted(sentences_dir.glob("*.txt")):
        paradigm = txt.stem
        task = _bucket_for_paradigm(paradigm)
        if task is None:
            logger.warning("Skipping unmapped paradigm: %s", paradigm)
            continue

        lines = [_format_sentence(line) for line in txt.read_text().splitlines()]
        # Pairs are alternating bad/good lines (matching the lm_training converter).
        pairs = []
        for i in range(0, len(lines) - 1, 2):
            sentence_bad, sentence_good = lines[i], lines[i + 1]
            pairs.append({"sentence_good": sentence_good, "sentence_bad": sentence_bad, "phenomena": task})

        by_task.setdefault(task, []).extend(pairs)
        logger.info("  %s -> %s (%d pairs)", paradigm, task, len(pairs))

    for task, pairs in by_task.items():
        out_path = out_dir / f"{task}.json"
        with out_path.open("w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Wrote %d pairs to %s", len(pairs), out_path)


def main() -> None:
    args = cache_argparser("zorro").parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("Zorro already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "Zorro", env_var="ZORRO_DATA_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = args.cache_dir / "_repo"

    if repo_dir.exists() and args.force:
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        logger.info("Cloning %s -> %s", ZORRO_REPO_URL, repo_dir)
        subprocess.run(["git", "clone", "--depth=1", ZORRO_REPO_URL, str(repo_dir)], check=True)

    sentences_dir = repo_dir / "sentences" / "babyberta"
    if not sentences_dir.is_dir():
        raise RuntimeError(f"Zorro repo missing sentences/babyberta/: {sentences_dir}")

    logger.info("Converting Zorro paradigms to per-task JSONLs")
    _convert(sentences_dir, args.cache_dir)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "Zorro", env_var="ZORRO_DATA_ROOT")


if __name__ == "__main__":
    main()
