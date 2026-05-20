# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export the BERT text encoder + MLM head from a self-contained .pt checkpoint to a HF directory.

The text-side eval pipeline (Zorro / LT-Swap, both via ``lm_eval``'s
``AutoMaskedLM``) loads a HuggingFace ``BertForMaskedLM`` from a directory
on disk. A self-contained contrastive checkpoint stores the same weights
but under different keys: ``model_state_dict["text_embed.model.*"]`` for
the BERT backbone and ``mlm_head_state_dict["head.predictions.*"]`` for
the MLM head. This script converts those tensors into a directory layout
``BertForMaskedLM.from_pretrained(...)`` understands.

Usage::

    egobabyvlm-export-text-encoder-to-hf \\
        --checkpoint /path/to/contrastive.pt \\
        --output-dir /path/to/hf_bert_dir

The output directory contains ``config.json``, ``tokenizer.json``,
``vocab.txt``, ``model.safetensors`` (BERT backbone + ``cls.*`` head), and
the rest of the tokenizer files copied unchanged from the ``hf_model_name``
the checkpoint was originally trained with.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, BertForMaskedLM

logger = logging.getLogger(__name__)


def export(checkpoint_path: Path, output_dir: Path) -> None:
    """Write a HF ``BertForMaskedLM`` directory from a self-contained .pt checkpoint."""
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    text_cfg = ckpt["config"]["model"]["text_encoder"]
    hf_model_name = text_cfg["hf_model_name"]
    logger.info("Original HF model name: %s", hf_model_name)

    config = AutoConfig.from_pretrained(hf_model_name)
    model = BertForMaskedLM(config)

    bert_state = {
        k.removeprefix("text_embed.model."): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("text_embed.model.")
    }
    if not bert_state:
        msg = "no text_embed.model.* keys in model_state_dict; checkpoint is not a triple/contrastive ckpt"
        raise RuntimeError(msg)
    # ``BertForMaskedLM`` expects backbone keys under ``bert.*``.
    bert_state = {f"bert.{k}": v for k, v in bert_state.items()}
    bert_missing, bert_unexpected = model.load_state_dict(bert_state, strict=False)
    bert_unexpected = [k for k in bert_unexpected if not k.startswith("cls.")]
    cls_missing = [k for k in bert_missing if k.startswith("cls.")]
    if bert_unexpected:
        logger.warning("Unexpected backbone keys (ignored): %s", bert_unexpected[:5])

    trained_mlm = ckpt.get("mlm_head_state_dict")
    if trained_mlm:
        cls_state = {f"cls.{k.removeprefix('head.')}": v for k, v in trained_mlm.items() if k.startswith("head.")}
        cls_missing_after, cls_unexpected = model.load_state_dict(cls_state, strict=False)
        non_bert_missing = [k for k in cls_missing_after if not k.startswith("bert.")]
        if non_bert_missing:
            logger.warning("MLM head keys missing after load: %s", non_bert_missing[:5])
        if cls_unexpected:
            logger.warning("Unexpected MLM head keys (ignored): %s", cls_unexpected[:5])
        logger.info("Loaded trained MLM head from checkpoint (%d tensors)", len(cls_state))
    else:
        logger.warning(
            "Checkpoint has no mlm_head_state_dict — MLM head will fall back to the pretrained %s weights below.",
            hf_model_name,
        )
        if cls_missing:
            # Fall back to the pretrained MLM head from the original HF dir on disk.
            try:
                pretrained_mlm = BertForMaskedLM.from_pretrained(hf_model_name)
                pretrained_cls = {k: v for k, v in pretrained_mlm.state_dict().items() if k.startswith("cls.")}
                model.load_state_dict(pretrained_cls, strict=False)
                logger.info("Initialized MLM head from pretrained %s (%d tensors)", hf_model_name, len(pretrained_cls))
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Could not initialize MLM head from %s: %s — head stays at random init",
                    hf_model_name,
                    e,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info("Saved BertForMaskedLM weights to %s", output_dir)

    # Tokenizer + extra files. Tokenizer comes from the original BERT dir
    # (the contrastive run did not retrain the vocabulary).
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(output_dir)
    src_dir = Path(hf_model_name) if Path(hf_model_name).is_dir() else None
    if src_dir is not None:
        for fname in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = src_dir / fname
            if src.is_file():
                shutil.copy(src, output_dir / fname)
    logger.info("Saved tokenizer to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
