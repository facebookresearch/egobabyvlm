#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 Phase 0 training on COCO captions.

Trains a GPT-2 Small from scratch on COCO captions; the resulting checkpoint is
the LM backbone for the multimodal LLaVA stack. The tokenizer is retrained with
HuggingFace's ``train_new_from_iterator`` from the standard GPT-2 base, which
preserves byte-level BPE structure and case sensitivity.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerBase

# W&B settings — override via the standard `WANDB_*` env vars (e.g.
# `WANDB_PROJECT`, `WANDB_DIR`, `WANDB_ENTITY`, `WANDB_BASE_URL`) before launch.
import os as _os

_os.environ.setdefault("WANDB_PROJECT", "egobabyvlm-phase0-gpt2")

logger = logging.getLogger(__name__)

# Defaults / magic numbers documented as named constants.
_DEFAULT_BLOCK_SIZE = 1024
_TOKENIZER_BATCH_SIZE = 1000
_NUM_JSON_ARGV = 2  # ``python train_gpt2.py path/to/args.json`` form.
_LABEL_IGNORE_INDEX = -100  # HF / cross-entropy ignore index for masked label tokens.


@dataclass
class ModelArguments:
    """Arguments for model/config/tokenizer."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Pretrained model checkpoint for weight init. Leave unset to train from scratch."},
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Path to a pre-trained tokenizer folder. If not set, a new tokenizer is trained."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from HF."},
    )
    model_type: str | None = field(
        default="gpt2",
        metadata={"help": "Model type when training from scratch (default: gpt2)."},
    )
    train_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to train a new tokenizer from the GPT-2 base tokenizer on the training data."},
    )
    vocab_size: int = field(
        default=52000,
        metadata={"help": "Vocabulary size for the custom tokenizer (default: 52000)."},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for resid_pdrop, embd_pdrop, and attn_pdrop (default: 0.1)."},
    )


@dataclass
class DataArguments:
    """Arguments for data."""

    train_file: str | None = field(
        default=None,
        metadata={"help": "Training data file (a text file, one sentence per line)."},
    )
    validation_file: str | None = field(
        default=None,
        metadata={"help": "Validation data file (a text file, one sentence per line)."},
    )
    block_size: int | None = field(
        default=None,
        metadata={"help": "Sequence length after tokenization. Defaults to model max length (1024)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached tokenized datasets."})
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing."},
    )
    keep_linebreaks: bool = field(default=True, metadata={"help": "Whether to keep line breaks when using TXT files."})


def train_tokenizer_from_gpt2(
    train_file: str | None,
    val_file: str | None,
    save_path: str,
    vocab_size: int = 52000,
) -> PreTrainedTokenizerBase:
    """Retrain a byte-level BPE tokenizer from the GPT-2 base on the given corpus.

    Inherits GPT-2's algorithm and case-sensitivity, retrains BPE merges on the
    domain corpus, and persists the result to ``save_path``.
    """
    logger.info("Training new tokenizer from GPT-2 base (vocab_size=%d)", vocab_size)

    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    lines: list[str] = []
    for fn in (train_file, val_file):
        if fn and Path(fn).exists():
            with Path(fn).open() as f:
                lines.extend(f.readlines())

    logger.info("  Corpus: %d lines", len(lines))

    def get_training_corpus(raw_lines: list[str]) -> Iterator[list[str]]:
        for i in range(0, len(raw_lines), _TOKENIZER_BATCH_SIZE):
            yield raw_lines[i : i + _TOKENIZER_BATCH_SIZE]

    corpus_iter = get_training_corpus(lines)

    tokenizer = old_tokenizer.train_new_from_iterator(corpus_iter, vocab_size)

    # GPT-2 has no pad token by default; reuse EOS as a safe pad.
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(save_path)

    logger.info("  Tokenizer saved to %s", save_path)
    logger.info("  Vocab size: %d", len(tokenizer))
    logger.info("  BOS token: %s (id=%s)", tokenizer.bos_token, tokenizer.bos_token_id)
    logger.info("  EOS token: %s (id=%s)", tokenizer.eos_token, tokenizer.eos_token_id)
    logger.info("  PAD token: %s (id=%s)", tokenizer.pad_token, tokenizer.pad_token_id)

    return tokenizer


def main() -> None:  # noqa: PLR0915 -- linear HF Trainer setup; splitting hurts readability
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore[arg-type]  # HF accepts a tuple of dataclass types at runtime

    if len(sys.argv) == _NUM_JSON_ARGV and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).resolve()))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    last_checkpoint: str | None = None
    if Path(training_args.output_dir).is_dir() and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this, change --output_dir.",
                last_checkpoint,
            )

    set_seed(training_args.seed)

    data_files: dict[str, str] = {}
    dataset_args: dict[str, Any] = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    extension = "text"
    dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        **dataset_args,
    )

    tokenizer_path = Path(training_args.output_dir) / "tokenizer"

    tokenizer: PreTrainedTokenizerBase
    if model_args.tokenizer_name:
        logger.info("Loading tokenizer from %s", model_args.tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif (tokenizer_path / "tokenizer_config.json").exists():
        logger.info("Loading existing tokenizer from %s", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer = train_tokenizer_from_gpt2(
            data_args.train_file,
            data_args.validation_file,
            str(tokenizer_path),
            vocab_size=model_args.vocab_size,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            activation_function="gelu_new",
            resid_pdrop=model_args.dropout,
            embd_pdrop=model_args.dropout,
            attn_pdrop=model_args.dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        logger.info("Created GPT-2 Small config from scratch.")

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info("Training new model from scratch - Total size=%.2fM params", n_params / 2**20)

    embedding_size = model.get_input_embeddings().weight.shape[0]  # type: ignore[union-attr,index]  # nn.Module __getattr__ returns Tensor|Module|Size; .weight is Tensor here
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning("Please ignore the warning above - long inputs will be chunked.")
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "Tokenizer model_max_length > %d. Using block_size=%d. Override with --block_size.",
                _DEFAULT_BLOCK_SIZE,
                _DEFAULT_BLOCK_SIZE,
            )
            block_size = _DEFAULT_BLOCK_SIZE
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                "block_size (%d) > tokenizer.model_max_length (%d). Using %d.",
                data_args.block_size,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples: dict[str, Any]) -> dict[str, Any]:
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[next(iter(examples))])
        # Drop the trailing remainder so all sequences hit `block_size` exactly
        # (default_data_collator requires uniform length).
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {k: [] for k in examples} | {"labels": []}
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"] if training_args.do_train else None
    eval_dataset = lm_datasets["validation"] if training_args.do_eval else None

    def preprocess_logits_for_metrics(logits: Any, _labels: Any) -> Any:  # noqa: ANN401
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds: tuple[Any, Any]) -> dict[str, float]:
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        mask = labels != _LABEL_IGNORE_INDEX
        accuracy = (preds[mask] == labels[mask]).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    if training_args.do_train:
        checkpoint: str | None = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        assert train_dataset is not None  # do_train was True, so train_dataset is set
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        assert eval_dataset is not None  # do_eval was True, so eval_dataset is set
        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Done!")


if __name__ == "__main__":
    main()
