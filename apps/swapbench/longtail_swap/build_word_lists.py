# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the LT-Swap word/inflpair lists from a text corpus.

Wraps the upstream ``get_word_lists.py`` + ``build_longtail.py`` scripts
(under ``apps/swapbench/third_party/lt_swap/generate_task/``) in a single
Hydra entry point. Run this once per training corpus; the resulting
``wordlists_dir/`` is then passed as input to
``apps.swapbench.longtail_swap.generate`` for each task (WordSwap,
InflectionSwap, AgreementSwap).

Outputs under ``output_dir``:

* ``wordlists/`` - per-input-file intermediate JSONs from get_word_lists
* ``longtail_wordlist`` - WordSwap candidate words
* ``longtail_inflpairs`` - InflectionSwap / AgreementSwap candidate pairs
* ``longtail_visualnouns`` - candidate words for visual / VP-Swap probes
  (``word,freq`` per row, nouns only; derived locally from the per-shard
  wordlists since upstream ``build_longtail.py`` does not emit a
  noun-only file in this format)
* ``vocabulary`` - corpus vocabulary with raw frequency counts
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)

#: Module path of the upstream generate_task package.
_UPSTREAM_PKG = "apps.swapbench.third_party.lt_swap.generate_task"

#: Filesystem path of the upstream generate_task directory. We prepend
#: it to ``PYTHONPATH`` when invoking upstream scripts so their sibling-
#: relative imports (``from preprocessing_utils import …``) resolve.
_UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "lt_swap" / "generate_task"


@dataclass
class BuildWordListsConfig:
    """Config for the wordlist build pipeline."""

    #: Directory of plain ``.txt`` files (one corpus shard per file).
    data_dir: str = "???"

    #: Output directory; populated with ``wordlists/`` plus the four
    #: top-level files listed in the module docstring.
    output_dir: str = "???"

    #: Number of CPU workers for ``get_word_lists`` (matches the upstream
    #: ``--ncpus`` flag; ideally one per shard).
    num_workers: int = 8


@dataclass
class BuildWordListsPipelineConfig:
    """Top-level Hydra config; processor only."""

    processor: BuildWordListsConfig = field(default_factory=BuildWordListsConfig)


cs = ConfigStore.instance()
cs.store(name="lt_swap_build_word_lists_pipeline", node=BuildWordListsPipelineConfig)


def _run_upstream(script: str, args: list[str]) -> None:
    """Invoke an upstream ``generate_task/*.py`` script as a subprocess.

    The upstream scripts use sibling-relative imports
    (``from preprocessing_utils import …``); we prepend the upstream
    directory to ``PYTHONPATH`` and run the script as
    ``python -m <script>`` so those imports resolve without modifying
    the upstream files.
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


def _derive_visual_nouns(wordlists_dir: Path, output: Path) -> int:
    """Emit ``word,freq`` rows for every NOUN seen across the per-shard wordlists.

    The upstream ``build_longtail.py`` does not produce a noun-only file in
    the format the VP-Swap generator consumes (``word,freq``), so we derive
    equivalent rows here from the per-shard JSONs by:

    1. summing each word's NOUN frequency across all shards (a word may appear
       with NOUN POS in some shards and not in others), and
    2. emitting one ``word,total_noun_freq`` row per unique noun.

    Returns the number of nouns written.
    """
    noun_freq: dict[str, int] = {}
    for path in sorted(wordlists_dir.glob("*.txt")):
        if path.name.endswith(".voc"):
            continue
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for word, info in data.items():
            pos_dict = info.get("POS", {}) if isinstance(info, dict) else {}
            for pos, pos_info in pos_dict.items():
                if not pos.startswith("NOUN"):
                    continue
                freq = pos_info.get("freq", 0) if isinstance(pos_info, dict) else 0
                noun_freq[word] = noun_freq.get(word, 0) + int(freq)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for word, freq in sorted(noun_freq.items()):
            f.write(f"{word},{freq}\n")
    return len(noun_freq)


@hydra.main(version_base=None, config_name="lt_swap_build_word_lists_pipeline")
def main(config: BuildWordListsPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    cfg = config.processor

    output_dir = Path(cfg.output_dir)
    wordlists_dir = output_dir / "wordlists"
    wordlists_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1/3: tokenize + extract per-file wordlists")
    _run_upstream(
        "get_word_lists",
        [
            f"--data={cfg.data_dir}",
            f"--output_wordlists_dir={wordlists_dir}",
            f"--ncpus={cfg.num_workers}",
        ],
    )

    logger.info("Stage 2/3: merge into longtail wordlist + inflpairs")
    _run_upstream(
        "build_longtail",
        [
            f"--wordlists_dir={wordlists_dir}",
            f"--output_wordlist={output_dir / 'longtail_wordlist'}",
            f"--output_inflpairs={output_dir / 'longtail_inflpairs'}",
            f"--output_voc={output_dir / 'vocabulary'}",
        ],
    )

    logger.info("Stage 3/3: derive longtail_visualnouns for VP-Swap")
    written = _derive_visual_nouns(wordlists_dir, output_dir / "longtail_visualnouns")
    logger.info("Wrote %d unique nouns to %s/longtail_visualnouns", written, output_dir)
    logger.info("Wordlist build complete: %s", output_dir)


if __name__ == "__main__":
    main()
