# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the BERT config + tokenizer helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---- create_bert_config -------------------------------------------------


def _run_create_bert_config(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apps.baselines.lm_training.scripts.create_bert_config", *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_create_bert_config_default_matches_bert_base(tmp_path: Path) -> None:
    out = tmp_path / "bert_base"
    _run_create_bert_config([str(out)])

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "bert"
    assert cfg["vocab_size"] == 30522  # bert-base-cased
    assert cfg["hidden_size"] == 768
    assert cfg["num_hidden_layers"] == 12
    assert cfg["num_attention_heads"] == 12
    assert cfg["intermediate_size"] == 3072
    assert cfg["hidden_act"] == "gelu"
    assert cfg["max_position_embeddings"] == 512
    assert cfg["type_vocab_size"] == 2
    assert cfg["pad_token_id"] == 0


def test_create_bert_config_overrides_apply(tmp_path: Path) -> None:
    out = tmp_path / "bert_tiny"
    _run_create_bert_config(
        [
            str(out),
            "--vocab_size",
            "8000",
            "--hidden_size",
            "128",
            "--num_hidden_layers",
            "4",
            "--num_attention_heads",
            "4",
            "--intermediate_size",
            "512",
            "--max_position_embeddings",
            "256",
        ]
    )
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["vocab_size"] == 8000
    assert cfg["hidden_size"] == 128
    assert cfg["num_hidden_layers"] == 4
    assert cfg["num_attention_heads"] == 4
    assert cfg["intermediate_size"] == 512
    assert cfg["max_position_embeddings"] == 256


def test_create_bert_config_creates_parent_dirs(tmp_path: Path) -> None:
    """The output dir is created even when its parents don't exist yet."""
    out = tmp_path / "deeply" / "nested" / "config"
    _run_create_bert_config([str(out)])
    assert (out / "config.json").is_file()


# ---- train_bert_tokenizer ----------------------------------------------


_TINY_CORPUS = """\
the cat sits on the mat
a dog runs in the park
she reads a long book
we walked across the wide bridge
he caught a small red fish
they painted the wooden door
birds fly south for the winter
children laugh in the playground
the clock ticks slowly each minute
rain falls on the dry leaves
she opened the heavy old gate
we will meet at the cafe
"""


def _write_tiny_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Write a 12-line training corpus and a 4-line val corpus to ``tmp_path``."""
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text(_TINY_CORPUS)
    val.write_text("a quick brown fox\nthe lazy dog\n")
    return train, val


@pytest.mark.integration
def test_train_bert_tokenizer_round_trip(tmp_path: Path) -> None:
    """End-to-end: train a tiny WordPiece tokenizer, reload it, encode + decode a sentence.

    Marked ``integration`` because tokenizer training pulls down ``bert-base-cased``
    (~436 MB) on first run. Skipped by the default pytest invocation.
    """
    from transformers import AutoTokenizer

    train, val = _write_tiny_corpus(tmp_path)
    out = tmp_path / "tokenizer_tiny"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.baselines.lm_training.scripts.train_bert_tokenizer",
            str(out),
            "--train_file",
            str(train),
            "--val_file",
            str(val),
            "--vocab_size",
            "500",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    tok = AutoTokenizer.from_pretrained(out)
    assert tok.pad_token is not None
    assert tok.cls_token is not None
    assert tok.mask_token is not None
    assert tok.sep_token is not None
    assert tok.unk_token is not None

    encoded = tok("the cat", return_tensors=None)
    assert "input_ids" in encoded
    decoded = tok.decode(encoded["input_ids"], skip_special_tokens=True)
    assert "cat" in decoded
