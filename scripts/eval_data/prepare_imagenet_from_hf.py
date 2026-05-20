#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert HuggingFace ``ILSVRC/imagenet-1k`` parquet to the wnid layout DINOv2 expects.

Writes the original JPEG bytes from each parquet row directly into
``<root>/{train,val}/<wnid>/<basename>.JPEG`` — byte-identical to what the
official image-net.org tarballs would produce. After this, run
``build_imagenet_extra`` to produce the ``.npy`` sidecar files DINOv2's
loader needs.

Requires:

1. A HuggingFace account that has accepted the ImageNet ToS at
   https://huggingface.co/datasets/ILSVRC/imagenet-1k.
2. ``hf auth login`` once with a read token.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

from scripts.eval_data._common import announce, setup_logging

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_SAMPLES = 1_281_167
EXPECTED_VAL_SAMPLES = 50_000
EXPECTED_CLASSES = 1000

HF_REPO = "ILSVRC/imagenet-1k"
SHARD_COUNTS = {"train": 294, "validation": 14, "test": 28}

#: HF stores filenames with a duplicated wnid suffix (e.g.
#: ``n01440764_10026_n01440764.JPEG``); strip the suffix to recover the
#: canonical basename (``n01440764_10026.JPEG``).
_PATH_RE = re.compile(r"^(?P<basename>.+?)_(?P<wnid>n\d{8})\.JPEG$")


def _load_imagenet_classes() -> dict[str, str]:
    """Fetch ``classes.py`` from the HF dataset repo and return its wnid → name map."""
    from huggingface_hub import hf_hub_download

    classes_py = hf_hub_download(repo_id=HF_REPO, filename="classes.py", repo_type="dataset")
    namespace: dict = {}
    exec(Path(classes_py).read_text(), namespace)  # noqa: S102 -- upstream module shipped by the HF dataset
    mapping = namespace["IMAGENET2012_CLASSES"]
    if len(mapping) != EXPECTED_CLASSES:
        msg = f"expected {EXPECTED_CLASSES} entries in IMAGENET2012_CLASSES, got {len(mapping)}"
        raise RuntimeError(msg)
    return mapping


def _write_labels(root: Path, classes: dict[str, str]) -> None:
    """Write ``<root>/labels.txt`` as ``<wnid>,<short_name>`` per row."""
    out = root / "labels.txt"
    rows = [f"{wnid},{name.split(',', 1)[0].strip().replace(chr(39), '')}" for wnid, name in classes.items()]
    out.write_text("\n".join(rows) + "\n")
    logger.info("Wrote %s (%d rows)", out, len(rows))


def _download_shards(split: str, cache_dir: Path | None) -> list[Path]:
    """Download the parquet shards for one HF split (train / validation / test)."""
    from huggingface_hub import hf_hub_download

    n = SHARD_COUNTS[split]
    paths = []
    for i in range(n):
        fname = f"data/{split}-{i:05d}-of-{n:05d}.parquet"
        local = hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        paths.append(Path(local))
        logger.info("[%s] shard %d/%d ready", split, i + 1, n)
    return paths


def _extract_shard(args: tuple[Path, Path]) -> int:
    """Worker: write each row's raw JPEG bytes into the wnid layout."""
    import pyarrow.parquet as pq

    shard_path, out_root = args
    table = pq.read_table(shard_path, columns=["image"])
    images = table.column("image").to_pylist()
    written = 0
    for img_struct in images:
        path = img_struct["path"]
        m = _PATH_RE.match(path)
        if m is None:
            msg = f"unexpected HF image path format: {path!r}"
            raise RuntimeError(msg)
        wnid = m.group("wnid")
        basename = f"{m.group('basename')}.JPEG"
        (out_root / wnid / basename).write_bytes(img_struct["bytes"])
        written += 1
    return written


def _extract_split(  # noqa: PLR0913
    *,
    hf_split: str,
    out_subdir: str,
    root: Path,
    wnids: list[str],
    cache_dir: Path | None,
    expected: int,
    workers: int,
) -> None:
    """Download + extract parquet shards into ``<root>/<out_subdir>/<wnid>/``."""
    shards = _download_shards(hf_split, cache_dir)

    out_root = root / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    for w in wnids:
        (out_root / w).mkdir(exist_ok=True)

    work = [(shard, out_root) for shard in shards]
    if workers <= 1:
        results = [_extract_shard(w) for w in work]
    else:
        with Pool(processes=workers) as pool:
            results = []
            for i, n in enumerate(pool.imap_unordered(_extract_shard, work), start=1):
                results.append(n)
                logger.info("[%s] %d/%d shards done (%d rows)", hf_split, i, len(shards), n)

    total = sum(results)
    if total != expected:
        logger.warning("[%s] wrote %d JPEGs; expected %d", hf_split, total, expected)
    logger.info("[%s] wrote %d JPEGs under %s", hf_split, total, out_root)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HF ImageNet-1k parquet to the wnid layout.")
    parser.add_argument("--root", type=Path, required=True, help="Destination $IMAGENET_ROOT.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace hub cache (defaults to $HF_HOME/hub or ~/.cache/huggingface/hub).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Which splits to materialize. HF 'validation' split is mapped to 'val'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel processes for shard extraction.",
    )
    parser.add_argument(
        "--purge-hf-cache",
        action="store_true",
        help="Delete the HF parquet shard cache after extraction.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    setup_logging()

    classes = _load_imagenet_classes()
    wnids = list(classes.keys())
    args.root.mkdir(parents=True, exist_ok=True)
    _write_labels(args.root, classes)

    if "train" in args.splits:
        _extract_split(
            hf_split="train",
            out_subdir="train",
            root=args.root,
            wnids=wnids,
            cache_dir=args.cache_dir,
            expected=EXPECTED_TRAIN_SAMPLES,
            workers=args.workers,
        )
    if "val" in args.splits:
        _extract_split(
            hf_split="validation",
            out_subdir="val",
            root=args.root,
            wnids=wnids,
            cache_dir=args.cache_dir,
            expected=EXPECTED_VAL_SAMPLES,
            workers=args.workers,
        )

    if args.purge_hf_cache and args.cache_dir is not None and args.cache_dir.exists():
        logger.info("Purging HF cache at %s", args.cache_dir)
        shutil.rmtree(args.cache_dir)

    announce(args.root, "ImageNet (wnid layout)", env_var="IMAGENET_ROOT")
    logger.info(
        "  Next: pixi run -e dev python -m scripts.eval_data.build_imagenet_extra --root %s --extra <IMAGENET_EXTRA>",
        args.root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
