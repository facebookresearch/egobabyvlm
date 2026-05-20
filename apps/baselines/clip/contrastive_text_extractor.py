# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only feature extractor wrapping a contrastive-trainer ``.pt`` checkpoint.

Adapts a self-contained ``.pt`` from ``egobabyvlm-train-contrastive`` for
MLM-scoring evals (Zorro, LT-Swap) that expect a HuggingFace
``BertForMaskedLM`` directory.

First construction for a given ``(checkpoint_path, config)`` extracts the
BERT backbone (and any trained MLM head) into a fresh ``BertForMaskedLM``,
optionally finetunes the head on a text corpus, and caches the resulting
HF directory. Without the finetune step Zorro/LT-Swap scores fall well below
the contrastive run's real text capability — the per-cycle MLM updates
during contrastive training are too few to keep the head calibrated.
Subsequent constructions hit the cache and skip straight to load.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from lm_eval.models.huggingface import AutoMaskedLM
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import BertForMaskedLM, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    """Project-wide cache root (mirrors ``scripts/eval_data/_common.py``)."""
    root = Path(os.environ.get("EGOBABYVLM_CACHE", Path.home() / ".cache" / "egobabyvlm"))
    return root / "contrastive_text"


class ContrastiveTextFeatureExtractor(AutoMaskedLM):
    """``AutoMaskedLM`` backed by a contrastive-trainer ``.pt`` checkpoint.

    Args:
        checkpoint_path: Self-contained ``.pt`` from ``egobabyvlm-train-contrastive``.
        finetune: Freeze BERT and finetune the MLM head on ``train_file``
            before serving the model. Default ``True``.
        train_file: Sentence-per-line text file. Required when ``finetune=True``.
        validation_file: Optional held-out text file for MLM eval during finetune.
        mlm_head_checkpoint: Override the HF dir whose ``cls.*`` weights seed
            the MLM head, when the embedded ``hf_model_name`` isn't reachable.
        cache_dir: Where to write the prepared ``BertForMaskedLM`` dir.
            Defaults to ``$EGOBABYVLM_CACHE/contrastive_text/``
            (or ``~/.cache/egobabyvlm/contrastive_text/`` if unset).
        finetune_epochs / finetune_lr / finetune_batch_size / max_seq_length /
        mlm_probability / seed: Finetune knobs.
        device: Where to place the eval model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        finetune: bool = True,
        train_file: str | None = None,
        validation_file: str | None = None,
        mlm_head_checkpoint: str | None = None,
        cache_dir: str | Path | None = None,
        finetune_epochs: int = 30,
        finetune_lr: float = 1e-4,
        finetune_batch_size: int = 128,
        max_seq_length: int = 512,
        mlm_probability: float = 0.15,
        seed: int = 42,
        device: int | str | torch.device | None = None,
    ) -> None:
        if finetune and not train_file:
            msg = "finetune=True requires train_file"
            raise ValueError(msg)

        self.checkpoint_path = Path(checkpoint_path)
        cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
        cache_config = {
            "checkpoint_path": str(self.checkpoint_path),
            "finetune": finetune,
            "train_file": train_file,
            "validation_file": validation_file,
            "mlm_head_checkpoint": mlm_head_checkpoint,
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "finetune_batch_size": finetune_batch_size,
            "max_seq_length": max_seq_length,
            "mlm_probability": mlm_probability,
            "seed": seed,
        }
        config_hash = hashlib.sha256(json.dumps(cache_config, sort_keys=True, default=str).encode()).hexdigest()[:12]
        model_dir = cache_root / f"{self.checkpoint_path.stem}_{config_hash}"

        if (model_dir / "config.json").is_file():
            logger.info("Reusing cached text model at %s", model_dir)
        else:
            model, tokenizer = self._build_bert_mlm(mlm_head_checkpoint)
            model_dir.mkdir(parents=True, exist_ok=True)
            if finetune:
                assert train_file is not None  # checked above
                self._finetune_mlm_head(
                    model,
                    tokenizer,
                    model_dir=model_dir,
                    train_file=train_file,
                    validation_file=validation_file,
                    epochs=finetune_epochs,
                    learning_rate=finetune_lr,
                    batch_size=finetune_batch_size,
                    max_seq_length=max_seq_length,
                    mlm_probability=mlm_probability,
                    seed=seed,
                )
            else:
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                logger.info("Wrote text model (no finetune) to %s", model_dir)

        self._model_dir = model_dir
        super().__init__(
            pretrained=str(model_dir),
            device=str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
            max_length=max_seq_length,
        )

    def _build_bert_mlm(
        self,
        mlm_head_checkpoint: str | None,
    ) -> tuple[BertForMaskedLM, PreTrainedTokenizerBase]:
        """Build a ``BertForMaskedLM`` with backbone+head loaded from the .pt."""
        payload: Mapping = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        text_cfg = payload["config"]["model"]["text_encoder"]
        hf_dir = mlm_head_checkpoint or text_cfg["hf_model_name"]
        if not Path(hf_dir).is_dir():
            msg = (
                f"hf_model_name from checkpoint config does not exist on disk: {hf_dir!r}. "
                "Pass mlm_head_checkpoint=... to override."
            )
            raise FileNotFoundError(msg)

        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        model = AutoModelForMaskedLM.from_pretrained(hf_dir)

        bert_state = {
            k.removeprefix("text_embed.model."): v
            for k, v in payload["model_state_dict"].items()
            if k.startswith("text_embed.model.")
        }
        if not bert_state:
            msg = "no text_embed.model.* keys in checkpoint; not a contrastive ckpt?"
            raise RuntimeError(msg)
        # Pooler keys can be missing here: the contrastive trainer's BERT runs
        # without a pooler, but BertForMaskedLM construction adds one back.
        # Anything else missing or unexpected is a real divergence.
        missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
        if any("pooler" not in k for k in missing) or unexpected:
            logger.warning(
                "BERT load: %d missing (%s) %d unexpected (%s)",
                len(missing),
                missing[:3],
                len(unexpected),
                unexpected[:3],
            )

        # Trained MLM head from triple / interleaved_lm runs takes precedence
        # over the original HF head.
        mlm_state = payload.get("mlm_head_state_dict") or {}
        cls_state = {f"cls.{k.removeprefix('head.')}": v for k, v in mlm_state.items() if k.startswith("head.")}
        if cls_state:
            model.load_state_dict(cls_state, strict=False)
            logger.info("Loaded trained MLM head from checkpoint (%d tensors)", len(cls_state))

        return model, tokenizer

    def _finetune_mlm_head(
        self,
        model: BertForMaskedLM,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_dir: Path,
        train_file: str,
        validation_file: str | None,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        max_seq_length: int,
        mlm_probability: float,
        seed: int,
    ) -> None:
        """Freeze BERT, train MLM head on ``train_file``, save to ``model_dir``."""
        from datasets import load_dataset  # local import; HF datasets is heavy

        set_seed(seed)
        for p in model.bert.parameters():
            p.requires_grad = False
        for p in model.cls.parameters():
            p.requires_grad = True

        data_files = {"train": train_file}
        if validation_file:
            data_files["validation"] = validation_file
        raw = load_dataset("text", data_files=data_files)

        def _tokenize(batch: dict) -> dict:
            batch["text"] = [line for line in batch["text"] if line and not line.isspace()]
            return tokenizer(
                batch["text"],
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        tokenized = raw.map(
            _tokenize, batched=True, num_proc=os.cpu_count() or 1, remove_columns=["text"], desc="Tokenizing"
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(model_dir),
                learning_rate=learning_rate,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                seed=seed,
                eval_strategy="epoch" if validation_file else "no",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=bool(validation_file),
                logging_steps=10,
                report_to=[],
            ),
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation"),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability),
        )
        logger.info("Finetuning MLM head for %d epochs (lr=%.0e, bs=%d)", epochs, learning_rate, batch_size)
        trainer.train()
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        logger.info("Saved finetuned text model to %s", model_dir)
