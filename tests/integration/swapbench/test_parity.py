# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Behavioral parity tests for the swapbench Hydra wrappers vs the upstream LT-Swap.

Marked ``integration`` (not run by default) because they shell out to
the upstream scripts under ``apps/swapbench/third_party/lt_swap/`` and
require nltk corpora (``punkt_tab``, ``averaged_perceptron_tagger_eng``)
to be available locally.

Two parity checks:

* ``test_build_word_lists_byte_parity_with_upstream`` runs both the
  Hydra wrapper and the raw upstream scripts on the same tiny corpus
  and asserts the produced ``longtail_wordlist`` / ``longtail_inflpairs``
  / ``vocabulary`` / per-shard JSON files are byte-identical. This
  catches the kind of bug where the wrapper passes args that upstream
  silently ignores or rejects.

* ``test_llm_runner_output_schema_matches_mp_main`` drives the same
  upstream-format input file through both ``run_llm_pool`` and the
  upstream ``mp_main.main`` (against deterministic stub LLMs) and
  asserts the ``idx`` set + per-row field count match. Schema parity
  matters because the upstream filter scripts (which consume these
  rows) parse positionally.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from apps.swapbench.utils.llm_runner import run_llm_pool

if TYPE_CHECKING:
    from collections.abc import Iterable

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_DIR = REPO_ROOT / "apps" / "swapbench" / "third_party" / "lt_swap" / "generate_task"


def _has_required_nltk_data() -> bool:
    try:
        import nltk

        nltk.word_tokenize("hello world")
        nltk.pos_tag(["hello"])
    except Exception:  # noqa: BLE001 -- intentional broad catch; we just want a yes/no for the skip.
        return False
    return True


_SAMPLE_TEXT = """\
The cat sat on the mat.
The dog barked at the cat.
A small dog ran fast.
The big dog jumped over the fence.
The cats climbed the tree.
The dogs followed the cats.
A red apple fell from the tree.
The green apple is ripe.
She ate three apples for lunch.
The bananas are yellow.
"""


def _make_corpus(corpus_dir: Path, num_shards: int = 3) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_shards + 1):
        (corpus_dir / f"shard_{i}.txt").write_text(_SAMPLE_TEXT)


def _run_upstream_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", script, *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0:
        msg = (
            f"Upstream script {script} failed (exit {completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        raise RuntimeError(msg)


@pytest.mark.skipif(not _has_required_nltk_data(), reason="nltk punkt_tab + perceptron_tagger required")
def test_build_word_lists_byte_parity_with_upstream(tmp_path: Path) -> None:
    """``egobabyvlm-swapbench-build-word-lists`` must produce upstream-byte-identical outputs."""
    corpus = tmp_path / "corpus"
    upstream_out = tmp_path / "upstream"
    my_out = tmp_path / "mine"
    _make_corpus(corpus)

    # 1) Drive the upstream scripts directly to produce the baseline.
    (upstream_out / "wordlists").mkdir(parents=True, exist_ok=True)
    _run_upstream_script(
        "get_word_lists",
        [
            f"--data={corpus}",
            f"--output_wordlists_dir={upstream_out / 'wordlists'}",
            "--ncpus=1",
        ],
    )
    _run_upstream_script(
        "build_longtail",
        [
            f"--wordlists_dir={upstream_out / 'wordlists'}",
            f"--output_wordlist={upstream_out / 'longtail_wordlist'}",
            f"--output_inflpairs={upstream_out / 'longtail_inflpairs'}",
            f"--output_voc={upstream_out / 'vocabulary'}",
        ],
    )

    # 2) Drive my Hydra wrapper with the same corpus.
    cmd = [
        sys.executable,
        "-m",
        "apps.swapbench.longtail_swap.build_word_lists",
        f"processor.data_dir={corpus}",
        f"processor.output_dir={my_out}",
        "processor.num_workers=1",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=REPO_ROOT)
    assert completed.returncode == 0, f"Wrapper failed:\n{completed.stdout}\n{completed.stderr}"

    # 3) Diff the upstream-produced files; they must be byte-identical.
    upstream_files: Iterable[str] = ("longtail_wordlist", "longtail_inflpairs", "vocabulary")
    for name in upstream_files:
        assert (my_out / name).read_bytes() == (upstream_out / name).read_bytes(), (
            f"{name} differs between wrapper and upstream"
        )
    for shard in ("shard_1.txt", "shard_2.txt", "shard_3.txt"):
        assert (my_out / "wordlists" / shard).read_bytes() == (upstream_out / "wordlists" / shard).read_bytes()
        assert (my_out / "wordlists" / f"{shard}.voc").read_bytes() == (
            upstream_out / "wordlists" / f"{shard}.voc"
        ).read_bytes()

    # 4) The wrapper additionally derives longtail_visualnouns; sanity-check it.
    visualnouns = (my_out / "longtail_visualnouns").read_text().splitlines()
    assert len(visualnouns) > 0, "expected at least one noun in longtail_visualnouns"
    for row in visualnouns:
        word, freq = row.split(",", 1)
        assert word
        assert int(freq) > 0


def test_llm_runner_output_schema_matches_mp_main(tmp_path: Path) -> None:
    """``run_llm_pool`` must produce the same idx-set + per-row field counts as upstream ``mp_main``."""
    input_path = tmp_path / "input.txt"
    upstream_out = tmp_path / "upstream_out.txt"
    my_out = tmp_path / "my_out.txt"
    rows = [
        "meta_a|prompt one",
        "meta_b|prompt two",
        "meta_quad|p1/p2/p3/p4/A/B/A/B",
        "meta_pair|q1/q2/A/B",
    ]
    input_path.write_text("\n".join(rows) + "\n")

    # Upstream mp_main does ``from __main__ import worker``; inject it.
    sys.path.insert(0, str(UPSTREAM_DIR))
    try:
        import mp_main  # type: ignore[import-untyped]
    finally:
        sys.path.pop(0)
    sys.modules["__main__"].worker = mp_main.worker

    args = MagicMock()
    args.client = "test"
    args.model = "ignored"
    args.api_key = ""
    args.app_name = ""
    args.input_file = str(input_path)
    args.output_file = str(upstream_out)
    args.num_workers = 4
    args.queue_size = 16
    args.max_retries = 1
    args.temperature = 0.0
    asyncio.run(mp_main.main(args))

    # My runner against a deterministic AsyncOpenAI stub.
    async def stub_create(*, model: str, messages: list[dict], **kwargs: object) -> object:  # noqa: ARG001
        prompt = messages[0]["content"]
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=f"stub: {prompt[:30]}"))]
        return completion

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=stub_create)
    asyncio.run(
        run_llm_pool(
            client=client,
            model="stub",
            input_path=input_path,
            output_path=my_out,
            temperature=0.0,
            num_workers=4,
            queue_size=16,
        ),
    )

    def _idx_and_field_count(path: Path) -> set[tuple[str, int]]:
        out: set[tuple[str, int]] = set()
        for raw in path.read_text().splitlines():
            if not raw:
                continue
            idx = raw.split("|", 1)[0]
            out.add((idx, raw.count("|") + 1))
        return out

    assert _idx_and_field_count(my_out) == _idx_and_field_count(upstream_out)
