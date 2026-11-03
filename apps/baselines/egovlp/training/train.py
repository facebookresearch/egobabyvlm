# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Contrastive fine-tuning entrypoint for the EgoVLP baseline.

Fine-tunes the EgoVLP dual encoder, initialized from the released EgoClip
checkpoint (``--pretrained_checkpoint egovlp.pth``), with a symmetric
image-text InfoNCE loss and a learnable temperature on one of 6 dataset
settings:

    babyview | coco_p0 | coco_p25 | coco_p50 | coco_p75 | coco_p100

``coco_pX`` = COCO-MC with X% of training captions permuted (``coco_p0`` is the
aligned corpus). Images are fed as single-frame (T=1) clips. The defaults are
the paper's settings (lr 3e-5, batch 32, 10 epochs, AdamW wd 0.2; seeds 42/43/44,
one GPU per run).

Checkpoints: after every epoch the trainer writes ``latest.pth`` and, if the
loss of the epoch's **last training batch** is the lowest so far, ``best.pth``.
No validation pass is run. The paper's EgoVLP numbers were evaluated on
``best.pth``.

Example (one paper run)::

    python -m apps.baselines.egovlp.training.train \\
        --dataset coco_p50 --seed 42 \\
        --pretrained_checkpoint /path/to/egovlp.pth \\
        --extracted_root /path/to/scratch/coco_metaclip/extracted \\
        --output_dir /path/to/checkpoints/egovlp/coco_p50/seed42

Smoke test (3 steps on 32 BabyView samples)::

    python -m apps.baselines.egovlp.training.train \\
        --dataset babyview --pretrained_checkpoint /path/to/egovlp.pth \\
        --output_dir /tmp/egovlp_smoke --batch_size 4 --max_steps 3 \\
        --limit_train_samples 32 --num_workers 2
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from apps.baselines.egovlp.data import (
    BABYVIEW_IMAGE_ROOT,
    BABYVIEW_TRAIN_JSON,
    BABYVIEW_VAL_JSON,
    BabyViewCaptionsDataset,
    build_coco_metaclip_datasets,
    build_egovlp_transform,
    egovlp_collate,
)
from apps.baselines.egovlp.data.collate import KEY_CAPTIONS, KEY_CLIP
from apps.baselines.egovlp.model import (
    DEFAULT_TEXT_MODEL,
    EgoVLPConfig,
    build_egovlp_model,
    save_egovlp_checkpoint,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATASET_CHOICES = ("babyview", "coco_p0", "coco_p25", "coco_p50", "coco_p75", "coco_p100")


# --------------------------------------------------------------------- distributed
def _setup_distributed() -> tuple[int, int, int, torch.device]:
    """Init torch.distributed from env (torchrun); no-op for single process.

    Returns ``(rank, world_size, local_rank, device)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------- datasets
def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    """Build (train, val) datasets for the requested ``--dataset`` setting."""
    train_tf = build_egovlp_transform(image_size=args.image_size, augment=args.augment)
    eval_tf = build_egovlp_transform(image_size=args.image_size, augment=False)

    if args.dataset == "babyview":
        missing = [
            flag
            for flag, value in (
                ("--babyview_train_json", args.babyview_train_json),
                ("--babyview_val_json", args.babyview_val_json),
                ("--babyview_image_root", args.babyview_image_root),
            )
            if not value
        ]
        if missing:
            msg = f"Set $BABYVIEW_ROOT or pass {', '.join(missing)}."
            raise ValueError(msg)
        train_ds: Dataset = BabyViewCaptionsDataset(
            args.babyview_train_json,
            args.babyview_image_root,
            transform=train_tf,
        )
        val_ds: Dataset | None = BabyViewCaptionsDataset(
            args.babyview_val_json,
            args.babyview_image_root,
            transform=eval_tf,
            multiple_frames=False,
        )
        return train_ds, val_ds

    percent = int(args.dataset.split("_p")[1])
    train_ds, val_ds = build_coco_metaclip_datasets(
        train_tf,
        val_transform=eval_tf,
        percent=percent,
        seed=args.perm_seed,
        extracted_root=args.extracted_root,
        check_train_size=not args.smoke,
    )
    return train_ds, val_ds


def _maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):  # type: ignore[arg-type]
        return dataset
    return torch.utils.data.Subset(dataset, list(range(limit)))


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=egovlp_collate,
        drop_last=drop_last,
    )


# --------------------------------------------------------------------- loss
def clip_contrastive_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric image<->text InfoNCE over the batch (CLIP objective)."""
    image_emb = F.normalize(image_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    logits = logit_scale.exp() * image_emb @ text_emb.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class _ContrastiveWrapper(nn.Module):
    """Wraps the EgoVLP model + learnable temperature so DDP can sync grads.

    DDP hooks fire on the wrapper's ``forward``; computing the loss inside
    ``forward`` (rather than calling ``model.encode_*`` on ``.module``) is what
    makes multi-GPU gradient all-reduce correct.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, clip: torch.Tensor, captions: list[str]) -> torch.Tensor:
        image_emb = self.model.encode_image(clip)
        text_emb = self.model.encode_text(captions)
        return clip_contrastive_loss(image_emb, text_emb, self.logit_scale)

    def clamp_temperature(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100.0))  # CLIP-style temp cap


# --------------------------------------------------------------------- train
def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = _setup_distributed()
    torch.manual_seed(args.seed + rank)

    config = EgoVLPConfig(
        embed_dim=args.embed_dim,
        num_frames=1,
        image_size=args.image_size,
        text_model=args.text_model,
        max_text_len=args.max_text_len,
        pretrained_checkpoint=args.pretrained_checkpoint,
        pretrained_text=args.pretrained_text,
    )
    model = build_egovlp_model(config).to(device)

    wrapper = _ContrastiveWrapper(model).to(device)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds, val_ds = build_datasets(args)
    train_ds = _maybe_subset(train_ds, args.limit_train_samples)
    if _is_main(rank):
        n_val = len(val_ds) if val_ds is not None else 0  # type: ignore[arg-type]
        logger.info("Dataset=%s | train=%d val=%d samples", args.dataset, len(train_ds), n_val)  # type: ignore[arg-type]

    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
    )

    step_module: nn.Module = wrapper
    if world_size > 1:
        step_module = nn.parallel.DistributedDataParallel(
            wrapper,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )

    output_dir = Path(args.output_dir)
    if _is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    last_loss = float("inf")
    for epoch in range(args.epochs):
        step_module.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            clip = batch[KEY_CLIP].to(device)
            captions = batch[KEY_CAPTIONS]

            loss = step_module(clip, captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wrapper.clamp_temperature()
            last_loss = float(loss.item())

            if _is_main(rank) and global_step % args.log_every == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, global_step, last_loss)

            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if _is_main(rank):
            save_egovlp_checkpoint(
                model,
                config,
                output_dir / "latest.pth",
                epoch=epoch,
                step=global_step,
                args=vars(args),
            )
            # "best" = lowest last-batch training loss (no validation pass).
            # Kept as-is because the paper's EgoVLP numbers use this selection.
            if last_loss < best_loss:
                best_loss = last_loss
                save_egovlp_checkpoint(
                    model,
                    config,
                    output_dir / "best.pth",
                    epoch=epoch,
                    step=global_step,
                    args=vars(args),
                )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if _is_main(rank):
        logger.info("Done. Checkpoints in %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EgoVLP contrastive fine-tuning")
    # Core sweep knobs.
    p.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--perm_seed",
        type=int,
        default=42,
        help="Derangement seed for the COCO-MC permutation map (fixed across training "
        "seeds; only seed42 permutation files exist). Decoupled from --seed so the 3 "
        "training seeds share the same permuted dataset per alignment level.",
    )
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32, help="Per-process batch size.")
    p.add_argument("--output_dir", type=str, required=True)
    # Optimization.
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    # Model.
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--text_model", type=str, default=DEFAULT_TEXT_MODEL)
    p.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Released EgoClip egovlp.pth to init from (omit only for tests: random video encoder).",
    )
    p.add_argument("--no_pretrained_text", dest="pretrained_text", action="store_false")
    p.set_defaults(pretrained_text=True)
    p.add_argument("--augment", action="store_true")
    # Data.
    p.add_argument("--babyview_train_json", type=str, default=BABYVIEW_TRAIN_JSON)
    p.add_argument("--babyview_val_json", type=str, default=BABYVIEW_VAL_JSON)
    p.add_argument("--babyview_image_root", type=str, default=BABYVIEW_IMAGE_ROOT)
    p.add_argument(
        "--extracted_root",
        type=str,
        default=None,
        help="Pre-staged MetaCLIP jpegs ({scratch_root}/extracted from data.stage_metaclip).",
    )
    # Smoke-test knobs.
    p.add_argument("--max_steps", type=int, default=None, help="Cap total optimizer steps (smoke tests).")
    p.add_argument("--limit_train_samples", type=int, default=None, help="Truncate train dataset (smoke tests).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Relax the COCO-MC train-size assert for subsetted manifests.",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
