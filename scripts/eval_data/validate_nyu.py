#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validate that ``$NYU_ROOT`` is set up correctly for the depth eval.

Catches the common mistakes:

- Missing or wrong-length ``nyu_train.txt`` / ``nyu_test.txt``
- Lines that don't have 3 space-separated fields (rgb depth focal)
- Sample lines whose ``rgb_*.jpg`` / ``sync_depth_*.png`` files don't
  resolve relative to ``$NYU_ROOT``
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path

from scripts.eval_data._common import setup_logging

logger = logging.getLogger(__name__)

EXPECTED_TRAIN_LINES = 24231
EXPECTED_TEST_LINES = 654
SPOT_CHECK_N = 5

#: Each split-file line is "<rgb_path> <depth_path> <focal_length>".
EXPECTED_FIELDS_PER_LINE = 3


def _check(name: str, fn: Callable[[], str]) -> bool:
    try:
        result = fn()
    except Exception as e:  # noqa: BLE001
        logger.error("[FAIL] %s: %s", name, e)
        return False
    logger.info("[OK]   %s: %s", name, result)
    return True


def _check_split(root: Path, split_file: str, expected_lines: int) -> str:
    path = root / split_file
    if not path.exists():
        msg = f"missing {path}; run scripts.eval_data.prepare_nyu --write-splits"
        raise FileNotFoundError(msg)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) != expected_lines:
        msg = f"{path}: expected {expected_lines} lines, got {len(lines)}"
        raise RuntimeError(msg)
    bad = [(i, ln) for i, ln in enumerate(lines[:50]) if len(ln.split()) != EXPECTED_FIELDS_PER_LINE]
    if bad:
        msg = f"{path}: malformed lines (expected 'rgb depth focal'): {bad[:3]}"
        raise RuntimeError(msg)

    # Spot-check a few random rows resolve to real files on disk.
    rng = random.Random(42)
    sample = rng.sample(lines, min(SPOT_CHECK_N, len(lines)))
    missing: list[str] = []
    for ln in sample:
        rgb_rel, depth_rel, _focal = ln.split()
        # Some legacy splits have a leading "/"; strip it before joining.
        rgb_path = root / rgb_rel.lstrip("/")
        depth_path = root / depth_rel.lstrip("/")
        if not rgb_path.exists():
            missing.append(str(rgb_path))
        if not depth_path.exists():
            missing.append(str(depth_path))
    if missing:
        msg = f"{path}: {len(missing)} of {2 * len(sample)} sampled paths missing, e.g. {missing[:2]}"
        raise FileNotFoundError(msg)

    return f"{len(lines)} entries; spot-checked {len(sample)} rows OK"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate $NYU_ROOT for the depth eval.")
    parser.add_argument("--nyu-root", type=Path, required=True, help="$NYU_ROOT")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    setup_logging()

    if not args.nyu_root.exists():
        logger.error("NYU root does not exist: %s", args.nyu_root)
        return 1

    results = [
        _check("nyu_train.txt", lambda: _check_split(args.nyu_root, "nyu_train.txt", EXPECTED_TRAIN_LINES)),
        _check("nyu_test.txt", lambda: _check_split(args.nyu_root, "nyu_test.txt", EXPECTED_TEST_LINES)),
    ]
    if all(results):
        logger.info("NYUv2 setup looks correct.")
        return 0
    logger.error("%d/%d checks failed.", sum(1 for r in results if not r), len(results))
    return 1


if __name__ == "__main__":
    sys.exit(main())
