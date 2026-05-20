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

"""Main LLaVA training script for EgoBabyLLaVA.

Supports:

* Phase 1 — projector pretraining (vision + LM frozen, projector trainable).
* Phase 2 — full finetuning (vision frozen, LM + projector trainable).

Example invocations live in ``scripts/`` and the package README.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, Trainer

from apps.baselines.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from apps.baselines.llava.model import LlavaGPT2ForCausalLM

if TYPE_CHECKING:
    from collections.abc import Sequence

# W&B settings — override via the standard `WANDB_*` env vars (e.g.
# `WANDB_PROJECT`, `WANDB_DIR`, `WANDB_ENTITY`, `WANDB_BASE_URL`) before launch.
os.environ.setdefault("WANDB_PROJECT", "egobabyvlm")

logger = logging.getLogger(__name__)

# Local rank for distributed training, set in `train()` before any logging.
local_rank: int | None = None


def rank0_print(*args: Any) -> None:  # noqa: ANN401
    """Print only on rank 0 (or when not in a distributed run)."""
    if local_rank == 0 or local_rank is None:
        print(*args)  # noqa: T201 -- rank-0 console echo for SLURM logs


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str | None = field(default="./checkpoints/phase0_gpt2_coco")
    version: str | None = field(default="v1")

    vision_tower: str | None = field(default="dinov2_vitb14")
    vision_tower_path: str | None = field(default=None)

    mm_projector_type: str | None = field(default="mlp2x_gelu")
    pretrain_mm_mlp_adapter: str | None = field(default=None)

    freeze_vision_tower: bool = field(default=True)
    freeze_llm_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)

    mm_vision_select_layer: int | None = field(default=-1)
    mm_vision_select_feature: str | None = field(default="patch")
    mm_patch_merge_type: str | None = field(default="flat")

    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)


@dataclass
class DataArguments:
    """Arguments for data loading."""

    data_path: str | None = field(default=None, metadata={"help": "Path to training data JSON"})
    image_folder: str | None = field(default=None)
    image_aspect_ratio: str = field(default="square")
    lazy_preprocess: bool = field(default=False)

    prompt_setup: str | None = field(
        default=None,
        metadata={
            "help": (
                "When set to 'yes_no', uses on-the-fly CLIP-style NxN contrastive "
                "batch generation instead of loading a pre-generated JSON. "
                "data_path should point to COCO annotations JSON "
                "(e.g. captions_train2017.json). Other values (None, 'random', "
                "'image_description') use LazySupervisedDataset as before."
            ),
        },
    )
    permutation_percent: int = field(
        default=0,
        metadata={
            "help": "Percentage of image-caption pairs to permute (0-100). Only used with --prompt_setup yes_no.",
        },
    )
    contrastive_batch_size: int = field(
        default=32,
        metadata={"help": "N for the NxN contrastive matrix. Only used with --prompt_setup yes_no."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments."""

    cache_dir: str | None = field(default=None)
    remove_unused_columns: bool = field(default=False)

    optim: str = field(default="adamw_torch")
    group_by_modality_length: bool = field(default=False)

    model_max_length: int = field(default=512, metadata={"help": "Max sequence length"})

    # LoRA fields kept for upstream-LLaVA compatibility; not used here.
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")

    bf16: bool = field(default=False)
    fp16: bool = field(default=True)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """Save model for HuggingFace Trainer, handling distributed training."""
    assert trainer.model is not None, "trainer.model must be initialized before saving"
    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def preprocess_multimodal(
    sources: list[list[dict[str, Any]]],
    data_args: DataArguments,  # noqa: ARG001 -- kept for upstream LLaVA call-site parity
) -> list[list[dict[str, Any]]]:
    """Preprocess multimodal data sources."""
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
    return sources


def preprocess(
    sources: list[list[dict[str, Any]]],
    tokenizer: PreTrainedTokenizerFast,
    *,
    has_image: bool = False,  # noqa: ARG001 -- kept for upstream LLaVA call-site parity
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of conversations into ``input_ids`` / ``labels`` tensors.

    Standardized format: ``"<image>\\nDescribe this image. {caption}"``.
    Prefix tokens are masked with ``IGNORE_INDEX``; only caption tokens
    contribute to the next-token-prediction loss.
    """
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for source in sources:
        human_value = source[0]["value"]
        gpt_value = source[1]["value"]

        if DEFAULT_IMAGE_TOKEN in human_value:
            prefix_text = human_value.replace(DEFAULT_IMAGE_TOKEN, "") + " "

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            caption_ids = tokenizer.encode(gpt_value, add_special_tokens=False)

            input_ids_seq = [IMAGE_TOKEN_INDEX, *prefix_ids, *caption_ids]
            labels_seq = [IGNORE_INDEX] * (1 + len(prefix_ids)) + caption_ids
        else:
            conversation = f"{human_value} {gpt_value}"
            input_ids_seq = tokenizer.encode(conversation, add_special_tokens=True)
            labels_seq = list(input_ids_seq)

        input_ids_list.append(torch.tensor(input_ids_seq, dtype=torch.long))
        labels_list.append(torch.tensor(labels_seq, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

    return {"input_ids": input_ids, "labels": labels}


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with lazy loading."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, data_args: DataArguments) -> None:
        super().__init__()

        rank0_print(f"Loading data from {data_path}...")
        with pathlib.Path(data_path).open() as f:
            self.list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.data_args = data_args

        # Local-FS only — remote-storage image loading is not bundled in OSS.
        if data_args.image_folder and "://" in data_args.image_folder:
            msg = (
                "Remote-storage image loading is not bundled in the open-source "
                "release. Provide a local path for image_folder."
            )
            raise NotImplementedError(msg)

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.list_data_dict[idx]

        conversations = sample.get("conversations", [])

        image_file = sample.get("image", None)
        has_image = image_file is not None

        sources = preprocess_multimodal([conversations], self.data_args) if has_image else [conversations]

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        result: dict[str, Any] = {"input_ids": data_dict["input_ids"][0], "labels": data_dict["labels"][0]}

        if has_image and self.data_args.image_folder:
            image_folder = self.data_args.image_folder
            try:
                image_path = pathlib.Path(image_folder) / image_file
                image = Image.open(image_path).convert("RGB")
                result["images"] = image
            except Exception as e:  # noqa: BLE001 -- one corrupt sample shouldn't kill training
                # Recover by serving the next sample instead of returning ``None`` —
                # ``prepare_inputs_labels_for_multimodal`` only strips IMAGE_TOKEN_INDEX
                # when ``images is not None``, so a None here would leave the -200
                # sentinel in input_ids and crash embed_tokens with a CUDA index OOB.
                rank0_print(f"Failed to load image {image_file}: {e}; skipping to next sample")
                return self.__getitem__((idx + 1) % len(self))

        return result


class EpochReshuffleCallback(transformers.TrainerCallback):
    """Trainer callback that rebuilds dataset entries each epoch (when ``set_epoch`` exists)."""

    def on_epoch_begin(
        self,
        args: Any,  # noqa: ANN401, ARG002 -- HF callback signature
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401, ARG002
        train_dataloader: Any = None,  # noqa: ANN401
        **_kwargs: Any,  # noqa: ANN401
    ) -> None:
        if train_dataloader is None:
            return
        dataset = train_dataloader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(int(state.epoch))


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizerFast
    image_processor: Any = None

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        # Pair each example with an image; substitute a zero tensor for None
        # so the IMAGE_TOKEN count matches the image count.
        images = [inst.get("images") for inst in instances]
        if any(img is not None for img in images) and self.image_processor is not None:
            ref_image = next(img for img in images if img is not None)
            filled_images = [img if img is not None else ref_image for img in images]
            processed = self.image_processor(images=filled_images, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            for i, img in enumerate(images):
                if img is None:
                    pixel_values[i] = 0.0
            batch["images"] = pixel_values

        return batch


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizerFast,
    data_args: DataArguments,
    image_processor: Any = None,  # noqa: ANN401 -- HF image processor; varies by model
    seed: int = 42,  # noqa: ARG001 -- kept for upstream LLaVA call-site parity
) -> dict[str, Any]:
    """Create data module for supervised training."""
    train_dataset: Dataset
    if data_args.prompt_setup == "yes_no":
        msg = (
            "prompt_setup='yes_no' relies on COCO data-prep utilities that are not "
            "bundled in the open-source release. Use the default supervised setup."
        )
        raise NotImplementedError(msg)
    assert data_args.data_path is not None, "data_args.data_path is required"
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, image_processor=image_processor)

    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": data_collator}


def train() -> None:  # noqa: PLR0915 -- linear HF Trainer setup; splitting hurts readability
    """Main training function."""
    global local_rank  # noqa: PLW0603 -- single-process global toggled by HF Trainer's local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore[arg-type]  # HF accepts a tuple of dataclass types at runtime
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("=" * 60)
    rank0_print("EgoBabyLLaVA Training")
    rank0_print("=" * 60)
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"Vision tower: {model_args.vision_tower}")
    rank0_print(f"Freeze vision: {model_args.freeze_vision_tower}")
    rank0_print(f"Freeze LLM: {model_args.freeze_llm_backbone}")
    rank0_print(f"Tune projector: {model_args.tune_mm_mlp_adapter}")
    rank0_print("=" * 60)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    rank0_print("Loading model...")
    model = LlavaGPT2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
    )
    model.config.use_cache = False

    rank0_print("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor if vision_tower else None

    if model_args.freeze_llm_backbone:
        rank0_print("Freezing LLM backbone...")
        for name, param in model.named_parameters():
            if "mm_projector" not in name and "vision_tower" not in name:
                param.requires_grad = False

    if model_args.tune_mm_mlp_adapter:
        rank0_print("Ensuring projector is trainable...")
        for name, param in model.named_parameters():
            if "mm_projector" in name:
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M")

    model.initialize_vision_tokenizer(model_args, tokenizer)

    rank0_print("Creating data module...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        image_processor=image_processor,
        seed=training_args.seed,
    )

    callbacks: list[transformers.TrainerCallback] = []
    if data_args.prompt_setup == "yes_no":
        callbacks.append(EpochReshuffleCallback())
        rank0_print("Registered EpochReshuffleCallback for on-the-fly contrastive regeneration")

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    rank0_print("Starting training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Save the projector separately so Phase 2 can re-load it via --pretrain_mm_mlp_adapter.
    if model_args.tune_mm_mlp_adapter:
        mm_projector_state = {k: v.cpu() for k, v in model.state_dict().items() if "mm_projector" in k}
        projector_path = pathlib.Path(training_args.output_dir) / "mm_projector.bin"
        torch.save(mm_projector_state, projector_path)
        rank0_print(f"Saved projector to {projector_path}")

    rank0_print("Training complete!")


if __name__ == "__main__":
    train()
