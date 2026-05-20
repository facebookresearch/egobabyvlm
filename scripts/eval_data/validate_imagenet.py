#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validate that ``$IMAGENET_ROOT`` and ``$IMAGENET_EXTRA`` are set up correctly.

Catches the common mistakes that break DINOv2's ``ImageNet`` loader:

- Wrong number of ``<wnid>/`` class directories (≠ 1000)
- Train images not following ``<wnid>_<digits>.JPEG`` (DINOv2's
  filename → actual_index parser silently fails otherwise)
- Missing ``labels.txt`` or wrong row count
- ``extra/`` ``.npy`` files missing, wrong shape, or wrong sample count
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from scripts.eval_data._common import setup_logging

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_SAMPLES = 1_281_167
EXPECTED_VAL_SAMPLES = 50_000
EXPECTED_CLASSES = 1000

#: DINOv2's ImageNet split.parse_image_relpath assumes train basenames have
#: the form ``<wnid>_<actual_index>.JPEG``; val basenames have the form
#: ``ILSVRC2012_val_<8d>.JPEG``.
_TRAIN_BASENAME = re.compile(r"^n\d{8}_\d+\.JPEG$")
_VAL_BASENAME = re.compile(r"^ILSVRC2012_val_\d{8}\.JPEG$")


def _check(name: str, fn: Callable[[], str]) -> bool:
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001
        logger.error("[FAIL] %s: %s", name, e)
        return False
    logger.info("[OK]   %s: %s", name, result)
    return True


def _check_train_layout(root: Path) -> str:
    train = root / "train"
    if not train.is_dir():
        msg = f"missing {train}"
        raise FileNotFoundError(msg)
    wnids = sorted(p.name for p in train.iterdir() if p.is_dir())
    if len(wnids) != EXPECTED_CLASSES:
        msg = f"expected {EXPECTED_CLASSES} wnid dirs in {train}, got {len(wnids)}"
        raise RuntimeError(msg)
    sample_dir = train / wnids[0]
    images = sorted(sample_dir.glob("*.JPEG"))
    if not images:
        msg = f"no .JPEG files in {sample_dir}"
        raise FileNotFoundError(msg)
    if not _TRAIN_BASENAME.match(images[0].name):
        msg = f"basename {images[0].name!r} does not match DINOv2 pattern <wnid>_<digits>.JPEG"
        raise RuntimeError(msg)
    return f"{len(wnids)} wnids; first dir has {len(images)} JPEGs (e.g. {images[0].name})"


def _check_val_layout(root: Path) -> str:
    val = root / "val"
    if not val.is_dir():
        msg = f"missing {val}"
        raise FileNotFoundError(msg)
    wnids = sorted(p.name for p in val.iterdir() if p.is_dir())
    if len(wnids) != EXPECTED_CLASSES:
        msg = f"expected {EXPECTED_CLASSES} wnid dirs in {val}, got {len(wnids)}"
        raise RuntimeError(msg)
    sample_dir = val / wnids[0]
    images = sorted(sample_dir.glob("*.JPEG"))
    if not images:
        msg = f"no .JPEG files in {sample_dir}"
        raise FileNotFoundError(msg)
    if not _VAL_BASENAME.match(images[0].name):
        msg = f"basename {images[0].name!r} does not match ILSVRC2012_val_<8d>.JPEG"
        raise RuntimeError(msg)
    return f"{len(wnids)} wnids; first dir has {len(images)} JPEGs (e.g. {images[0].name})"


def _check_labels(root: Path) -> str:
    labels = root / "labels.txt"
    if not labels.exists():
        msg = f"missing {labels}; see docs/eval_data.md"
        raise FileNotFoundError(msg)
    rows = [line.strip() for line in labels.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_CLASSES:
        msg = f"expected {EXPECTED_CLASSES} rows in {labels}, got {len(rows)}"
        raise RuntimeError(msg)
    if "," not in rows[0]:
        msg = f"row 0 of {labels} is not CSV: {rows[0]!r}"
        raise RuntimeError(msg)
    return f"{len(rows)} rows (e.g. {rows[0]!r})"


def _check_extra(extra: Path) -> str:
    expected_arrays = {
        "entries-TRAIN.npy": (EXPECTED_TRAIN_SAMPLES, "structured"),
        "entries-VAL.npy": (EXPECTED_VAL_SAMPLES, "structured"),
        "class-ids-TRAIN.npy": (EXPECTED_CLASSES, "U"),
        "class-ids-VAL.npy": (EXPECTED_CLASSES, "U"),
        "class-names-TRAIN.npy": (EXPECTED_CLASSES, "U"),
        "class-names-VAL.npy": (EXPECTED_CLASSES, "U"),
    }
    summary = []
    for fname, (expected_n, expected_kind) in expected_arrays.items():
        path = extra / fname
        if not path.exists():
            msg = f"missing {path}; run scripts.eval_data.build_imagenet_extra"
            raise FileNotFoundError(msg)
        arr = np.load(path)
        if arr.shape != (expected_n,):
            msg = f"{path}: expected shape ({expected_n},), got {arr.shape}"
            raise RuntimeError(msg)
        if expected_kind == "structured" and arr.dtype.names is None:
            msg = f"{path}: expected structured dtype, got {arr.dtype}"
            raise RuntimeError(msg)
        if expected_kind == "U" and arr.dtype.kind != "U":
            msg = f"{path}: expected unicode dtype, got {arr.dtype}"
            raise RuntimeError(msg)
        summary.append(f"{fname}({arr.shape[0]})")
    return ", ".join(summary)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate $IMAGENET_ROOT + $IMAGENET_EXTRA.")
    parser.add_argument("--root", type=Path, required=True, help="$IMAGENET_ROOT")
    parser.add_argument("--extra", type=Path, required=True, help="$IMAGENET_EXTRA")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    setup_logging()

    if not args.root.exists():
        logger.error("ImageNet root does not exist: %s", args.root)
        return 1
    if not args.extra.exists():
        logger.error("ImageNet extra dir does not exist: %s", args.extra)
        return 1

    results = [
        _check("train layout", lambda: _check_train_layout(args.root)),
        _check("val layout", lambda: _check_val_layout(args.root)),
        _check("labels.txt", lambda: _check_labels(args.root)),
        _check("extra/ artifacts", lambda: _check_extra(args.extra)),
    ]
    if all(results):
        logger.info("ImageNet setup looks correct.")
        return 0
    logger.error("%d/%d checks failed.", sum(1 for r in results if not r), len(results))
    return 1


if __name__ == "__main__":
    sys.exit(main())
