# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""Masked-language-model pretraining for BERT, ported from HuggingFace's ``run_mlm.py``."""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

if TYPE_CHECKING:
    from collections.abc import Mapping

check_min_version("4.29.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

_NUM_JSON_ARGV = 2
_DEFAULT_BLOCK_SIZE = 1024
_LABEL_IGNORE_INDEX = -100
_MAX_MISSING_KEYS_OK = 2


@dataclass
class ModelArguments:
    """Arguments controlling which model / config / tokenizer to fine-tune or train from scratch."""

    clip_model_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to CLIP-finetuned text encoder weights (.pt file)"},
    )
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            ),
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            ),
        },
    )
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            ),
        },
    )
    torch_dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "Create the model as an empty shell, then materialize parameters when pretrained weights load. "
                "Reduces RAM and speeds up loading for large LMs."
            ),
        },
    )
    dropout: float | None = field(
        default=None,
        metadata={"help": "Override attention_probs_dropout_prob and hidden_dropout_prob in config"},
    )

    def __post_init__(self) -> None:
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path",
            )

        if self.clip_model_checkpoint is not None and not Path(self.clip_model_checkpoint).exists():
            raise ValueError(f"CLIP checkpoint file {self.clip_model_checkpoint} does not exist")


@dataclass
class DataTrainingArguments:
    """Arguments controlling the training/eval data fed to the model."""

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: str | None = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: str | None = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: int | None = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split",
        },
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            ),
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            ),
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            ),
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            ),
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    def __post_init__(self) -> None:
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            if extension not in {"csv", "json", "txt"}:
                raise ValueError("`train_file` should be a csv, a json or a txt file.")
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            if extension not in {"csv", "json", "txt"}:
                raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def load_clip_finetuned_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load CLIP-finetuned weights into a BERT-for-MLM model.

    Mirrors the key remapping from ``evaluate_clip_text_encoder``: the CVCL projection
    head is dropped, ``model.*`` keys are remapped to ``bert.*``, and ``cls.*`` MLM
    head keys are kept unchanged.

    Args:
        model: BERT-for-MLM module to load into.
        checkpoint_path: Path to a CVCL ``.pt`` state dict.

    Returns:
        The same model, with weights loaded in place.
    """
    device = model.device
    state_dict = torch.load(checkpoint_path, map_location=device)  # type: ignore[arg-type]

    new_state_dict: dict[str, torch.Tensor] = {}
    projection_keys = [k for k in state_dict if "projection" in k]
    if projection_keys:
        logger.info("Found projection keys that will be excluded: %s", projection_keys)

    for key, value in state_dict.items():
        if "projection" in key:
            continue
        if key.startswith("cls"):
            new_key = key
        elif key.startswith("model."):
            new_key = "bert." + key[len("model.") :]
        else:
            new_key = "bert." + key
        new_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    logger.info("Loaded CLIP-finetuned weights with:")
    logger.info("Missing keys: %s", missing_keys)
    logger.info("Unexpected keys: %s", unexpected_keys)

    if len(missing_keys) > _MAX_MISSING_KEYS_OK:
        logger.warning("Warning: Multiple keys are still missing. Model might not work correctly.")

    return model


def freeze_bert_weights(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze the BERT encoder so only the MLM head trains.

    Args:
        model: BERT-for-MLM module.

    Returns:
        The same model, with ``requires_grad`` flipped in place.
    """
    for param in model.bert.parameters():  # type: ignore[union-attr]
        param.requires_grad = False
    for param in model.cls.parameters():  # type: ignore[union-attr]
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Percentage trainable: %.2f%%", 100 * trainable_params / total_params)

    return model


class TokenCountingCollator:
    """Wraps a base collator and accumulates the count of non-pad tokens it emits."""

    def __init__(self, base_collator: Any, pad_token_id: int) -> None:  # noqa: ANN401
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id
        self.total_tokens = 0

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        self.total_tokens += (batch["input_ids"] != self.pad_token_id).sum().item()
        return batch


class TokenCountingCallback(transformers.TrainerCallback):
    """Logs ``tokens_consumed`` to W&B at every Trainer log step."""

    def __init__(self, collator: TokenCountingCollator, world_size: int = 1) -> None:
        self.collator = collator
        self.world_size = world_size

    def on_log(
        self,
        args: TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> None:
        if logs is not None and state.is_world_process_zero:
            tokens = self.collator.total_tokens * self.world_size
            report_to = args.report_to or []
            if "wandb" in report_to:
                # Lazy import — wandb is optional.
                import wandb

                if wandb.run is not None:
                    wandb.log({"tokens_consumed": tokens}, step=state.global_step)


def main() -> None:  # noqa: PLR0915 -- linear HF Trainer setup; splitting hurts readability
    # W&B project — override via the standard `WANDB_*` env vars before launch.
    os.environ.setdefault("WANDB_PROJECT", "egobabyvlm-bert")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))  # type: ignore[arg-type]
    if len(sys.argv) == _NUM_JSON_ARGV and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(  # type: ignore[misc]
            json_file=str(Path(sys.argv[1]).resolve()),
        )
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
    if Path(training_args.output_dir).is_dir() and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(Path(training_args.output_dir).iterdir()):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.",
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                last_checkpoint,
            )

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files: dict[str, str] = {}
        extension = "text"
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=False, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info("Overriding config: %s", model_args.config_overrides)
            config.update_from_string(model_args.config_overrides)
            logger.info("New config: %s", config)

    if model_args.dropout is not None:
        config.attention_probs_dropout_prob = model_args.dropout
        config.hidden_dropout_prob = model_args.dropout
        logger.info("Overriding dropout to %s", model_args.dropout)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name.",
        )

    if model_args.model_name_or_path:
        logger.info("Loading pretrained model from: %s", model_args.model_name_or_path)
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in {"auto", None}
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("No model_name_or_path specified, initializing model from scratch from config")
        model = AutoModelForMaskedLM.from_config(config)

    if model_args.clip_model_checkpoint is not None:
        logger.info("Loading CLIP-finetuned weights from: %s", model_args.clip_model_checkpoint)
        model = load_clip_finetuned_weights(model, model_args.clip_model_checkpoint)
        logger.info("Freezing BERT weights, keeping only MLM head trainable")
        model = freeze_bert_weights(model)

    embedding_size = model.get_input_embeddings().weight.shape[0]  # type: ignore[union-attr,index]
    if len(tokenizer) > embedding_size:
        logger.info("Resizing token embeddings to avoid index error")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` longer than the default `block_size` of %d. "
                "Override with `--block_size xxx` to use a longer chunk.",
                _DEFAULT_BLOCK_SIZE,
            )
            max_seq_length = _DEFAULT_BLOCK_SIZE
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                "The max_seq_length passed (%d) is larger than the model's max length (%d). Using %d.",
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # `return_special_tokens_mask=True` lets DataCollatorForLanguageModeling skip recomputation.
        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[next(iter(examples))])
            # Drop the trailing remainder so all chunks hit `max_seq_length` exactly.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            return {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    train_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    compute_metrics = None
    preprocess_logits_for_metrics = None
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits: Any, _labels: Any) -> Any:  # noqa: ANN401
            if isinstance(logits, tuple):
                # Some model heads return (logits, *extras); logits always come first.
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds: tuple[Any, Any]) -> dict[str, float]:
            preds, labels = eval_preds
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != _LABEL_IGNORE_INDEX
            labels = labels[mask]
            preds = preds[mask]
            return {"accuracy": (preds == labels).astype(float).mean().item()}

    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    token_counting_collator = TokenCountingCollator(data_collator, pad_token_id)
    token_counting_callback = TokenCountingCallback(token_counting_collator, training_args.world_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=token_counting_collator,
        callbacks=[token_counting_callback],
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

        assert train_dataset is not None
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        assert eval_dataset is not None
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs: dict[str, str] = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(_index: int) -> None:
    # Entry point for `xla_spawn` (TPUs); the process index is unused here.
    main()


if __name__ == "__main__":
    main()
