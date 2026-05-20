#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate DINOv2's ``extra/`` ImageNet metadata in one command.

DINOv2's :class:`ImageNet` loader needs per-split ``.npy`` index files
(``class-ids-{TRAIN,VAL}.npy``, ``class-names-{TRAIN,VAL}.npy``,
``entries-{TRAIN,VAL,TEST}.npy``) under a sidecar directory. This script
wraps DINOv2's own ``ImageNet(...).dump_extra()`` so users don't have to
copy the snippet from the upstream README.

Prerequisite layout (already prepared per ``docs/eval_data.md``):

    <IMAGENET_ROOT>/
        labels.txt                            # CSV "<wnid>,<class_name>" per line
        train/<wnid>/<wnid>_<idx>.JPEG
        val/<wnid>/ILSVRC2012_val_<8d>.JPEG
        test/ILSVRC2012_test_<8d>.JPEG        # only required for the TEST split
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dinov2.data.datasets import ImageNet

from scripts.eval_data._common import setup_logging

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DINOv2 ImageNet extra/ metadata.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="ImageNet root with train/, val/, optional test/, and labels.txt.",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        required=True,
        help="Destination directory for the .npy metadata. Created if missing.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[s.value for s in ImageNet.Split],
        default=["train", "val"],
        help="Splits to process. Add 'test' if you have it.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    setup_logging()

    if not (args.root / "labels.txt").exists():
        logger.error("Missing %s — required CSV of '<wnid>,<class_name>'.", args.root / "labels.txt")
        return 1

    args.extra.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        split = ImageNet.Split(split_name)
        logger.info("Generating extra/ metadata for split=%s", split_name)
        dataset = ImageNet(split=split, root=str(args.root), extra=str(args.extra))
        dataset.dump_extra()

    logger.info("✓ Extra metadata written to %s", args.extra)
    logger.info("  Set IMAGENET_ROOT=%s and IMAGENET_EXTRA=%s.", args.root, args.extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
