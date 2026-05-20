# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build a fresh BERT config (HuggingFace ``BertConfig``) and save it to disk."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import BertConfig

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to write the config to.")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size (default: bert-base-cased).")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()

    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # `do_lower_case` is a tokenizer setting; mirrored here for symmetry with bert-base-cased.
        do_lower_case=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(args.output_dir)
    logger.info("BERT config saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
