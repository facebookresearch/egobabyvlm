# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retrain a WordPiece BERT tokenizer from ``bert-base-cased`` on a custom corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

#: Default WordPiece vocab size — matches bert-base-cased.
_DEFAULT_VOCAB_SIZE = 30522
#: Batch size for the tokenizer-training iterator.
_TOKENIZER_BATCH_SIZE = 1000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to save the trained tokenizer to.")
    parser.add_argument(
        "--train_file",
        type=Path,
        required=True,
        help="Plain-text training corpus (one line per utterance).",
    )
    parser.add_argument(
        "--val_file",
        type=Path,
        default=None,
        help="Optional validation corpus to mix into the tokenizer-training stream.",
    )
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="bert-base-cased",
        help="HuggingFace model id whose tokenizer's algorithm + special-token layout to inherit.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=_DEFAULT_VOCAB_SIZE,
        help="Target vocabulary size (default: 30522).",
    )
    return parser.parse_args()


def _iter_corpus_in_batches(paths: list[Path], batch_size: int = _TOKENIZER_BATCH_SIZE) -> Iterator[list[str]]:
    """Yield successive ``batch_size``-line chunks across the given files."""
    buffer: list[str] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                buffer.append(line)
                if len(buffer) >= batch_size:
                    yield buffer
                    buffer = []
    if buffer:
        yield buffer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    paths: list[Path] = [args.train_file]
    if args.val_file is not None:
        paths.append(args.val_file)
    logger.info("Training tokenizer on %s", [str(p) for p in paths])

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        _iter_corpus_in_batches(paths),
        vocab_size=args.vocab_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_tokenizer.save_pretrained(args.output_dir)
    logger.info("Tokenizer saved to %s", args.output_dir)

    logger.info("Vocab size:     %d", len(new_tokenizer))
    logger.info("Special tokens: %s", new_tokenizer.all_special_tokens)
    logger.info("PAD:  %s", new_tokenizer.pad_token)
    logger.info("UNK:  %s", new_tokenizer.unk_token)
    logger.info("SEP:  %s", new_tokenizer.sep_token)
    logger.info("CLS:  %s", new_tokenizer.cls_token)
    logger.info("MASK: %s", new_tokenizer.mask_token)


if __name__ == "__main__":
    main()
