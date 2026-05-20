# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for ``scripts/eval_data/`` download scripts."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

DEFAULT_CACHE_ROOT = Path(os.environ.get("EGOBABYVLM_CACHE", Path.home() / ".cache" / "egobabyvlm"))
DEFAULT_DATA_ROOT = DEFAULT_CACHE_ROOT / "eval_data"

logger = logging.getLogger(__name__)


def cache_argparser(dataset_name: str) -> argparse.ArgumentParser:
    """Build a standard argument parser for a download script.

    Args:
        dataset_name: Short name for ``--help`` text and the cache subdirectory.

    Returns:
        Parser with ``--cache-dir`` and ``--force`` already wired.
    """
    parser = argparse.ArgumentParser(
        description=f"Download the {dataset_name} eval dataset into the local cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / dataset_name,
        help=f"Destination directory (default: {DEFAULT_DATA_ROOT / dataset_name}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists in the cache.",
    )
    return parser


def setup_logging() -> None:
    """Configure stdout logging for download scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def already_downloaded_marker(cache_dir: Path) -> Path:
    """Path to the sentinel file that marks a dataset as fully downloaded."""
    return cache_dir / ".downloaded"


def is_already_downloaded(cache_dir: Path) -> bool:
    """Whether ``cache_dir`` contains a sentinel marker from a prior successful run."""
    return already_downloaded_marker(cache_dir).exists()


def mark_downloaded(cache_dir: Path) -> None:
    """Drop the sentinel marker so the next run can short-circuit."""
    already_downloaded_marker(cache_dir).touch()


def announce(cache_dir: Path, dataset_name: str, env_var: str) -> None:
    """Print the cache path and the env var that should point at it."""
    logger.info("✓ %s ready at %s", dataset_name, cache_dir)
    logger.info("  Set %s=%s to use it from eval YAMLs.", env_var, cache_dir)


def flatten_single_wrapper_dir(root: Path, expected_subdirs: Iterable[str]) -> bool:
    """Strip a single top-level wrapper directory if the tarball was built with one.

    Some release tarballs are built with a redundant top-level directory
    (``tar -cf foo.tar Foo/`` instead of ``cd Foo && tar -cf ../foo.tar .``).
    If ``root`` contains exactly one entry — a directory holding every name in
    ``expected_subdirs`` — move its contents up one level and remove the
    wrapper. No-op when the layout is already flat or doesn't match the
    expected pattern, so this is safe to call after every extract.

    Returns:
        ``True`` if a wrapper was flattened, ``False`` otherwise.
    """
    entries = [p for p in root.iterdir() if p.name != ".downloaded"]
    if len(entries) != 1 or not entries[0].is_dir():
        return False
    wrapper = entries[0]
    if not all((wrapper / sub).is_dir() for sub in expected_subdirs):
        return False
    logger.info("Flattening wrapper directory %s -> %s", wrapper, root)
    for child in wrapper.iterdir():
        target = root / child.name
        if target.exists():
            logger.warning("Cannot flatten: %s already exists at destination.", target)
            return False
        shutil.move(str(child), str(target))
    wrapper.rmdir()
    return True
