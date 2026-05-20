# Copyright (c) Meta Platforms, Inc. and affiliates.
# Type errors here belong to the upstream perception_models source we
# adapted from; we don't refactor types in this file because it would
# diverge from upstream and make future refresh updates painful.
# mypy: ignore-errors

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import torch
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from stopes.core import Requirements, StopesModule
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from apps.alignment_scoring.configs import PLMGenerationConfig
from apps.alignment_scoring.data import CaptionsPathDataset
from apps.alignment_scoring.third_party.perception_models.apps.plm.tokenizer import (
    PLMTokenizer,
    Tokenizer,
    build_tokenizer,
)
from apps.alignment_scoring.third_party.perception_models.apps.plm.transformer import (
    LMTransformer,
    LMTransformerArgs,
)
from apps.alignment_scoring.third_party.perception_models.core.args import dataclass_from_dict
from apps.alignment_scoring.third_party.perception_models.core.checkpoint import load_consolidated_checkpoint
from apps.alignment_scoring.third_party.perception_models.core.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)

logger = logging.getLogger(__name__)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample_top_k(probs: torch.Tensor, k: float) -> torch.Tensor:
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return torch.multinomial(probs, num_samples=1)


def sample_tokens(
    logits: torch.Tensor, temperature: float = 0.0, top_p: float | None = None, top_k: float | None = None
) -> torch.Tensor:
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.view(shape[:-1])


def pack_prompts(prompts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    res = []
    lengths = []
    for _, p in enumerate(prompts):
        prompt_tensor = torch.tensor(p, dtype=torch.long)
        length = prompt_tensor.size(0)
        res.append(prompt_tensor)
        lengths.append(length)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts: list[Any], max_elements: int, lengths: list[int] | None = None) -> list[list[Any]]:
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


class KVCache(nn.Module):
    def __init__(
        self, bsz: int, seqlen: int, n_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device | str
    ) -> None:
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor, tok_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        return self.k_cache, self.v_cache


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: float | None = None
    min_gen_len: int = 0
    max_gen_len: int = 256
    max_tokens: int = 9920  # 11264
    until: list[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    dtype: str | None = "bf16"
    device: str | None = "cuda"


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ) -> None:
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.min_gen_len = cfg.min_gen_len
        self.max_tokens = cfg.max_tokens
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not cfg.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            backend="inductor",
            fullgraph=True,
            mode="reduce-overhead",  # Other available mode is "max-autotune"
            disable=not cfg.reduce_generation_overhead,
        )

        self.show_progress = cfg.show_progress
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

    def clear_cache(self, offset: torch.Tensor) -> None:
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor) -> None:
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        self.prefill_mask = create_block_mask(doc_mask_mod, 1, None, lengths.sum(), max_tokens)

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        self.padded_doc_id, self.padded_tok_id = lengths_to_local_ids(padded_lengths)

    @torch.compiler.disable
    def setup_generation(self, lengths: torch.Tensor) -> None:
        # KV Cache offset is set to the start of the padded documents
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = self.padded_doc_start
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = lengths.clone()
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = torch.arange(lengths.size(0), device=lengths.device)

    @torch.compiler.disable
    def prepare_media_inputs(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        images: list[torch.Tensor] | None,
        image_patch_text_ids: list[list[int]] | None,
        num_image_chunks: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int]]:
        image_pos_index = None
        num_chunks = []
        if images is not None and len(images) > 0:
            assert image_patch_text_ids is not None
            assert len(image_patch_text_ids) == len(images)
            assert num_image_chunks is not None
            assert len(num_image_chunks) == len(images)
            image_pos_index = torch.full(tokens.shape, -1, dtype=torch.int).to(self.device)
            assert tokens.shape[0] == 1
            # offsets = torch.roll(lengths.cpu(), shifts=1, dims=-1).numpy()
            offsets = torch.roll(lengths.cpu(), shifts=1, dims=-1)
            offsets[0] = 0
            offsets = torch.cumsum(offsets, dim=0).numpy()
            num_chunks_seq = 0
            image_id_offset = 0
            for image_id, offset in enumerate(offsets):
                num_image_tokens = len(image_patch_text_ids[image_id])
                image_indices = torch.arange(num_image_tokens, dtype=torch.int).to(self.device) + image_id_offset
                text_indices = [i + offset for i in image_patch_text_ids[image_id]]
                image_pos_index[0, text_indices] = image_indices
                image_id_offset += num_image_tokens
                num_chunks_seq += num_image_chunks[image_id]
            num_chunks.append(num_chunks_seq)
            # Move images to the same device and dtype as model parameters
            model_param = next(self.model.parameters())
            images = torch.cat(images).to(model_param)
        else:
            images = None
        return images, image_pos_index, num_chunks

    # From here on some methods for generation
    def prefill(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        images: list[torch.Tensor] | None = None,
        image_patch_text_ids: list[list[int]] | None = None,
        num_image_chunks: list[int] | None = None,
    ) -> torch.Tensor:
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        images, image_pos_index, num_chunks = self.prepare_media_inputs(
            tokens, lengths, images, image_patch_text_ids, num_image_chunks
        )
        is_batched = lengths.size(0) > 1
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask if is_batched else "causal",
            images=images,
            image_pos_index=image_pos_index,
            num_chunks=num_chunks,
            attn_impl="flex_attention" if is_batched else "sdpa",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token: torch.Tensor) -> torch.Tensor:
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        out = self.model.forward(
            current_token,
            tok_idx=self.current_tok_id,  # n_seqs
            mask=mask,
            attn_impl="sdpa",
        )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(
        self, prompts: list[Any], *, max_gen_len: int | None = None
    ) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        max_gen_len = max_gen_len or self.max_gen_len

        images = []
        image_patch_text_ids = []
        num_image_chunks = []
        last_prompt_logits = []
        # Tokenize
        if isinstance(self.tokenizer, PLMTokenizer):
            encoded_prompts = []
            for p in prompts:
                assert isinstance(p, (tuple, list))
                assert len(p) == 2
                question, image = p

                images.append(image)
                text_ids, image_pos = self.tokenizer._tokenize_for_generation(question, image)
                num_chunks = image.size(0)

                encoded_prompts.append(text_ids)
                image_patch_text_ids.append(image_pos)
                num_image_chunks.append(num_chunks)
            prompts = encoded_prompts
        else:
            prompts = [self.tokenizer.encode(p, add_bos=False, add_eos=False) for p in prompts]

        # Account for the generation in lengths
        padded_lengths = [len(p) + max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it, desc="Generating")
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)
            current_images = images[:n_seqs]
            current_image_patch_text_ids = image_patch_text_ids[:n_seqs]
            current_num_image_chunks = num_image_chunks[:n_seqs]
            images = images[n_seqs:]
            image_patch_text_ids = image_patch_text_ids[n_seqs:]
            num_image_chunks = num_image_chunks[n_seqs:]

            # Prefilling cache
            prompt_logits = self.prefill(
                packed_batch.unsqueeze(0),
                lengths,
                images=current_images,
                image_patch_text_ids=current_image_patch_text_ids,
                num_image_chunks=current_num_image_chunks,
            )

            # Store last prompt logits used for VQA scoring
            last_prompt_logits.append(prompt_logits[0, lengths.cumsum(0) - 1, :])

            # Selecting last token in each prompt
            all_tokens = sample_tokens(prompt_logits, self.temperature, self.top_p, self.top_k)

            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, max_gen_len):
                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(next_logits.clone(), self.temperature, self.top_p, self.top_k)

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        # Only check for stopping conditions if we've reached minimum length
                        if len(generated_tokens[seq_id]) >= self.min_gen_len:
                            current_end_str = self.tokenizer.decode(generated_tokens[seq_id][-self.max_until_size :])
                            contains_end_string = any(e in current_end_str for e in self.until)
                            is_eos_token = tok in {self.tokenizer.eot_id, self.tokenizer.eos_id}
                            is_done[seq_id] = contains_end_string or is_eos_token
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(batch, prompt_logits.squeeze(0).split(lengths.tolist()), strict=False):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        generation = [response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "") for response in generation]

        last_prompt_logits = torch.cat(last_prompt_logits, dim=0)

        return generation, loglikelihood, greedy, last_prompt_logits

    @cached_property
    def yes_token_id(self) -> int:
        return self.tokenizer.encode("Yes", add_bos=False, add_eos=False)[0]

    @cached_property
    def no_token_id(self) -> int:
        return self.tokenizer.encode("No", add_bos=False, add_eos=False)[0]

    @torch.inference_mode()
    def compute_vqa_score(self, prompts: list[Any]) -> list[float]:
        generations, _, _, last_logits = self.generate(prompts, max_gen_len=1)

        logger.debug("Generated outputs: %s", generations)

        probs = torch.softmax(last_logits, dim=-1)
        yes_probs = probs[:, self.yes_token_id]
        no_probs = probs[:, self.no_token_id]
        max_probs, _ = probs.max(dim=-1)

        logger.debug("VQA 'Yes' probabilities: %s", yes_probs.cpu().tolist())
        logger.debug("VQA 'No' probabilities: %s", no_probs.cpu().tolist())
        logger.debug("VQA max probabilities: %s", max_probs.cpu().tolist())

        return yes_probs.cpu().tolist()


def load_consolidated_model_and_tokenizer(ckpt: str) -> tuple[Any, Any, Any] | None:
    """Resolve ``ckpt`` (HF id or local path) and load the consolidated PLM checkpoint."""
    if Path(ckpt).exists():
        ckpt_path: str | Path = ckpt
    else:
        try:
            logger.info("Downloading %s from Hugging Face Hub...", ckpt)
            ckpt_path = snapshot_download(ckpt)
            ckpt_path = Path(ckpt_path) / "original"
            logger.info("Downloaded to: %s", ckpt_path)
        except OSError:
            logger.exception("An error occurred while downloading %s", ckpt)
            return None

    config_path = Path(ckpt_path) / "params.json"
    with config_path.open() as f:
        config = OmegaConf.load(f)

    tokenizer_path = config.data.tokenizer_path
    if not Path(tokenizer_path).exists():
        tokenizer_path = str(Path(ckpt_path) / config.data.tokenizer_path)

    tokenizer = build_tokenizer(
        config.data.tokenizer_name,
        tokenizer_path,
        pooling_ratio=config.model.pooling_ratio,
        patch_size=config.model.vision_model.patch_size,
    )

    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    model = LMTransformer(model_args)
    load_consolidated_checkpoint(model, ckpt_path)
    param_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[config.distributed.model_dtype]
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    return model, tokenizer, config


class PromptBatch(NamedTuple):
    prompts: list[tuple[str, torch.Tensor]]
    media_paths: list[str]
    texts: list[str]
    media_ids: list[int] | list[str]


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: CaptionsPathDataset | torch.utils.data.Subset,
        transform: Callable[[Any], tuple[torch.Tensor, Any]],
        question: str,
        max_video_frames: int | None = None,
        *,
        vqa_scoring: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.question = question
        self.vqa_scoring = vqa_scoring
        self.max_video_frames = max_video_frames

        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset  # Unwrap subset to access attributes
        self.is_video_dataset = getattr(base_dataset, "is_video_dataset", False)

        self.failed_indices = []

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict | None:
        media_path = None
        try:
            media_path, text, media_id = self.base_dataset[idx]
            if self.is_video_dataset:
                video_info = (media_path, self.max_video_frames, None, None, None)
                media_tensor, _ = self.transform(video_info)
                question_str = self.question.format(text=text) if self.vqa_scoring else self.question
            else:
                with Path(media_path).open("rb") as f:
                    image = Image.open(f).convert("RGB")
                media_tensor, _ = self.transform(image)
                question_str = self.question.format(text=text) if self.vqa_scoring else self.question
        except Exception:
            logger.exception(
                "Error processing item %d (path: %s)",
                idx,
                media_path,
            )
            self.failed_indices.append(idx)
            return None
        else:
            return {
                "prompt": (question_str, media_tensor),
                "media_path": media_path,
                "text": text,
                "media_id": media_id,
            }


def collate_prompts(batch: list[dict | None]) -> PromptBatch:
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return PromptBatch([], [], [], [])

    return PromptBatch(
        prompts=[item["prompt"] for item in batch],
        media_paths=[item["media_path"] for item in batch],
        texts=[item["text"] for item in batch],
        media_ids=[item["media_id"] for item in batch],
    )


class PLMGenerationModule(StopesModule):
    """Stopes module for parallel image/video captioning"""

    def __init__(self, config: PLMGenerationConfig) -> None:
        super().__init__(config, PLMGenerationConfig)

        dataset: CaptionsPathDataset = hydra.utils.instantiate(self.config.dataset)
        self.num_items = len(dataset)

    def requirements(self) -> Requirements:
        """Requirements for each submitted job"""
        return Requirements(
            nodes=1,
            mem_gb=140,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=10,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        return f"plm_gen_{self.config.name}"

    @property
    def num_chunks(self) -> int:
        return math.ceil(self.num_items / self.config.num_items_per_chunk)

    def array(self) -> list[tuple[int, int]]:
        """Array job indices as (start_idx, end_idx) tuples"""
        return [
            (
                self.config.num_items_per_chunk * i,
                min(self.config.num_items_per_chunk * (i + 1), self.num_items),
            )
            for i in range(self.num_chunks)
        ]

    def get_dataloader(
        self,
        indices: tuple[int, int] | None,
        model_config: DictConfig,
        image_res: int,
    ) -> torch.utils.data.DataLoader:
        from apps.alignment_scoring.third_party.perception_models.core.transforms.image_transform import (
            get_image_transform,
        )
        from apps.alignment_scoring.third_party.perception_models.core.transforms.video_transform import (
            get_video_transform,
        )

        dataset: CaptionsPathDataset = hydra.utils.instantiate(self.config.dataset)
        is_video_dataset = getattr(dataset, "is_video_dataset", False)

        if indices is not None:
            start_idx, end_idx = indices
            end_idx = min(end_idx, len(dataset))
            logger.info("Dataset size: %d, using indices %d-%d", len(dataset), start_idx, end_idx)
            dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))

        if is_video_dataset:
            transform = get_video_transform(image_res=image_res)
        else:
            transform = get_image_transform(
                vision_input_type=model_config.data.vision_input_type,
                image_res=image_res,
                max_num_tiles=model_config.data.max_num_tiles,
            )

        prompt_dataset = PromptDataset(
            base_dataset=dataset,
            transform=transform,
            question=self.config.question,
            max_video_frames=model_config.data.max_video_frames if is_video_dataset else None,
            vqa_scoring=self.config.vqa_scoring,
        )

        return torch.utils.data.DataLoader(
            prompt_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_prompts,
            pin_memory=True,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )

    def run(self, iteration_value: tuple[int, int], iteration_index: int) -> list[dict]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device %s", device)

        model, tokenizer, config = load_consolidated_model_and_tokenizer(self.config.ckpt)
        logger.info("Model and tokenizer loaded successfully from %s", self.config.ckpt)

        dataloader = self.get_dataloader(
            indices=iteration_value, model_config=config, image_res=model.vision_model.image_size
        )

        gen_cfg = PackedCausalTransformerGeneratorArgs(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_gen_len=self.config.max_gen_len,
            dtype=self.config.dtype,
            device=device,
            show_progress=False,
        )
        generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

        chunk_results = []

        for batch in tqdm(dataloader, desc="Processing batches"):
            prompts, media_paths, texts, media_ids = batch

            if len(prompts) == 0:
                logger.warning("Empty batch encountered, skipping")
                continue

            if self.config.vqa_scoring:
                scores = generator.compute_vqa_score(prompts)

                for i, score in enumerate(scores):
                    result = {
                        "index": media_ids[i],
                        "image_path": media_paths[i],
                        "text": texts[i],
                        "vqa_score": score,
                    }
                    chunk_results.append(result)
            else:
                generations, _, _, _ = generator.generate(prompts)

                for i, caption in enumerate(generations):
                    result = {"index": media_ids[i], "generated_caption": caption}
                    chunk_results.append(result)

        logger.info("Processed %d items total in chunk %d", len(chunk_results), iteration_index)
        logger.info("Example results: %s", chunk_results[:3])

        return chunk_results


# Use ``pytest -m gpu tests/apps/alignment_scoring/test_pipelines_smoke.py``
# to exercise this file end-to-end.
