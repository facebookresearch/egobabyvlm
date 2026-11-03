# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""COCO + MetaCLIP (COCO-MC) image-text dataset for EgoVLP fine-tuning.

Uses the same pair ordering and train/val routing as the CLIP+ and LLaVA
COCO-MC runs, so the shared permutation-metadata JSONs
(``permutation_metadata_cocomc_train_p{25,50,75,100}_seed42.json``) apply
index-for-index without regeneration.

Pairing: every COCO caption is paired with its original COCO image plus each of
its 5 retrieved MetaCLIP images (6 pairs sharing the caption). Train/val is
routed by the BERT-MLM split files (zero caption leakage). The train split must
contain exactly ``EXPECTED_TRAIN_PAIRS`` pairs (asserted) so permutation maps
line up.

Permutations (alignment levels {0,25,50,75,100}%): serving train pair ``i`` when
``i`` is in the map, use the *donor* pair's caption (a guaranteed true mismatch)
while keeping pair ``i``'s image. ``p0`` uses no permutation file.

Images:
  * COCO     -> ``{COCO_IMAGE_DIR}/{coco_id:012d}.jpg``
  * MetaCLIP -> ``{extracted_root}/{uuid[:2]}/{uuid}.jpeg`` if pre-staged to
    scratch, else read directly from the source tar (slow; fine for smoke
    tests). Pre-stage full runs with ``apps.baselines.egovlp.data.stage_metaclip``.

Data locations are read from environment variables (see the EgoVLP README):

  * ``COCO_MC_MANIFEST``         ``coco_to_metaclip.jsonl`` (one row per COCO
    caption with its ``matches``)
  * ``COCO_MC_SPLIT_DIR``        dir with ``coco_metaclip_val.txt`` (held-out
    validation captions; every other caption is train)
  * ``COCO_MC_PERMUTATION_DIR``  dir with the permutation-metadata JSONs
  * ``COCO_IMAGE_DIR``           COCO images named ``{coco_id:012d}.jpg``
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch
    from torchvision import transforms

logger = logging.getLogger(__name__)


def _env_path(name: str) -> str:
    """Return the path in env var ``name`` or raise a helpful error."""
    value = os.environ.get(name)
    if not value:
        msg = f"Set ${name} (see apps/baselines/egovlp/README.md, 'Data')."
        raise RuntimeError(msg)
    return value


#: The shared contract: the train split MUST have this many pairs for the
#: permutation maps to align. Matches CLIP+/LLaVA.
EXPECTED_TRAIN_PAIRS = 495_380

#: Valid alignment (permutation) percentages.
PERMUTATION_PERCENTS = (0, 25, 50, 75, 100)


@dataclass
class _Pair:
    caption: str
    kind: str  # "coco" | "metaclip"
    locator: str  # coco_id (str) for coco, uuid for metaclip
    tar_path: str | None = None


def permutation_metadata_path(percent: int, seed: int = 42) -> str | None:
    """Path to the permutation JSON for ``percent`` (``None`` for p0)."""
    if percent == 0:
        return None
    if percent not in PERMUTATION_PERCENTS:
        msg = f"percent must be one of {PERMUTATION_PERCENTS}, got {percent}"
        raise ValueError(msg)
    return f"{_env_path('COCO_MC_PERMUTATION_DIR')}/permutation_metadata_cocomc_train_p{percent}_seed{seed}.json"


def build_pair_index(
    manifest_path: str | None = None,
    split_dir: str | None = None,
) -> tuple[list[_Pair], list[_Pair]]:
    """Stream the manifest into (train, val) pair lists (same order as CLIP+).

    Routing: caption listed in ``coco_metaclip_val.txt`` -> val; every other
    caption -> train.

    Args:
        manifest_path: ``coco_to_metaclip.jsonl`` (default ``$COCO_MC_MANIFEST``).
        split_dir: Dir with ``coco_metaclip_val.txt`` (default ``$COCO_MC_SPLIT_DIR``).
    """
    manifest_path = manifest_path or _env_path("COCO_MC_MANIFEST")
    split_dir = split_dir or _env_path("COCO_MC_SPLIT_DIR")
    with (Path(split_dir) / "coco_metaclip_val.txt").open() as f:
        val_caps = {line.strip() for line in f if line.strip()}

    train_pairs: list[_Pair] = []
    val_pairs: list[_Pair] = []
    with Path(manifest_path).open() as f:
        for line in f:
            row = json.loads(line)
            caption = row["caption"].strip()
            target = val_pairs if caption in val_caps else train_pairs
            target.append(_Pair(caption=caption, kind="coco", locator=str(row["coco_id"])))
            for m in row["matches"]:
                target.append(
                    _Pair(caption=caption, kind="metaclip", locator=m["uuid"], tar_path=m["tar_path"]),
                )
    logger.info("[coco-mc] train pairs=%d  val pairs=%d", len(train_pairs), len(val_pairs))
    return train_pairs, val_pairs


def load_permutation_map(path: str, num_train_pairs: int) -> dict[int, int]:
    """Load + validate a permutation map against the current train pair count."""
    with Path(path).open() as f:
        meta = json.load(f)
    if meta.get("split") != "train":
        msg = "permutation metadata must be for the train split"
        raise ValueError(msg)
    if meta["num_pairs"] != num_train_pairs:
        msg = (
            f"permutation metadata num_pairs={meta['num_pairs']} != current train split "
            f"{num_train_pairs}; manifest or BERT split files changed"
        )
        raise ValueError(msg)
    logger.info("[coco-mc] permutation: %d%% permuted (%s)", meta["percent_permuted"], path)
    return {int(k): int(v) for k, v in meta["permutation_map"].items()}


class _MetaclipImageReader:
    """Read MetaCLIP jpegs from scratch (if staged) else directly from tar.

    Keeps a thread-local cache of open tar handles. For full runs, pre-stage
    with ``python -m apps.baselines.egovlp.data.stage_metaclip`` and pass
    ``extracted_root``.
    """

    def __init__(self, extracted_root: str | None) -> None:
        self.extracted_root = extracted_root
        self._tls = threading.local()

    def _get_tar(self, tar_path: str) -> tarfile.TarFile:
        cache = getattr(self._tls, "cache", None)
        if cache is None:
            cache = {}
            self._tls.cache = cache
        tf = cache.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, mode="r")  # noqa: SIM115 - cached, kept open for reuse
            cache[tar_path] = tf
        return tf

    def read(self, uuid: str, tar_path: str | None) -> Image.Image:
        if self.extracted_root is not None:
            scratch = Path(self.extracted_root) / uuid[:2] / f"{uuid}.jpeg"
            if scratch.is_file():
                return Image.open(scratch).convert("RGB")
        if tar_path is None:
            msg = f"metaclip uuid {uuid} not in scratch and has no tar_path"
            raise FileNotFoundError(msg)
        tf = self._get_tar(tar_path)
        # Tar members may be stored as "./uuid.jpeg" or "uuid.jpeg" (as in stage_metaclip).
        for name in (f"./{uuid}.jpeg", f"{uuid}.jpeg"):
            try:
                member = tf.extractfile(name)
            except KeyError:
                continue
            if member is not None:
                return Image.open(io.BytesIO(member.read())).convert("RGB")
        msg = f"could not extract {uuid}.jpeg from {tar_path}"
        raise FileNotFoundError(msg)


class CocoMetaclipCaptionsDataset(Dataset):
    """One row per (image, caption) pair; emits ``(image, caption, id)``.

    Args:
        pairs: Pair list from :func:`build_pair_index` (train or val).
        transform: Image transform (applied to the PIL image).
        coco_image_dir: COCO image root (default ``$COCO_IMAGE_DIR``).
        extracted_root: Scratch root for pre-staged MetaCLIP jpegs (or ``None``
            to read from tars on the fly).
        permutation_map: Optional ``{pair_idx: donor_idx}`` (train only).
    """

    def __init__(
        self,
        pairs: list[_Pair],
        transform: transforms.Compose | None,
        *,
        coco_image_dir: str | None = None,
        extracted_root: str | None = None,
        permutation_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.transform = transform
        self.coco_image_dir = coco_image_dir or _env_path("COCO_IMAGE_DIR")
        self.permutation_map = permutation_map or {}
        self._mc_reader = _MetaclipImageReader(extracted_root)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, pair: _Pair) -> Image.Image:
        if pair.kind == "coco":
            path = Path(self.coco_image_dir) / f"{int(pair.locator):012d}.jpg"
            return Image.open(path).convert("RGB")
        return self._mc_reader.read(pair.locator, pair.tar_path)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, str, str]:
        pair = self.pairs[idx]
        image = self._load_image(pair)
        if self.transform is not None:
            image = self.transform(image)
        caption = self.pairs[self.permutation_map[idx]].caption if idx in self.permutation_map else pair.caption
        return image, caption, f"{pair.kind}:{pair.locator}"


def build_coco_metaclip_datasets(
    transform: transforms.Compose,
    *,
    val_transform: transforms.Compose | None = None,
    percent: int = 0,
    seed: int = 42,
    extracted_root: str | None = None,
    manifest_path: str | None = None,
    check_train_size: bool = True,
) -> tuple[CocoMetaclipCaptionsDataset, CocoMetaclipCaptionsDataset]:
    """Build (train, val) COCO-MC datasets for the given permutation ``percent``.

    Set ``check_train_size=False`` only for tiny smoke tests where the manifest
    is subsetted (the ``EXPECTED_TRAIN_PAIRS`` assert would otherwise fail).
    """
    train_pairs, val_pairs = build_pair_index(manifest_path=manifest_path)
    if check_train_size and len(train_pairs) != EXPECTED_TRAIN_PAIRS:
        msg = (
            f"train split has {len(train_pairs)} pairs, expected {EXPECTED_TRAIN_PAIRS}; "
            f"permutation maps will not align. Pass check_train_size=False only for smoke tests."
        )
        raise ValueError(msg)

    permutation_map: dict[int, int] | None = None
    perm_path = permutation_metadata_path(percent, seed=seed)
    if perm_path is not None:
        permutation_map = load_permutation_map(perm_path, len(train_pairs))

    train_ds = CocoMetaclipCaptionsDataset(
        train_pairs,
        transform,
        extracted_root=extracted_root,
        permutation_map=permutation_map,
    )
    val_ds = CocoMetaclipCaptionsDataset(
        val_pairs,
        val_transform or transform,
        extracted_root=extracted_root,
    )
    return train_ds, val_ds
