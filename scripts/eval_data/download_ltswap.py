#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download and extract the LT-Swap evaluation tarball.

LT-Swap pair files are *model/corpus-conditional* — the same script run
against different training corpora yields different pair files. The
release ships one bundle covering all four training corpora the paper
uses; submitters select the subdir matching their training data.

Output layout matches what
:class:`evaluation.text.ltswap.LTSwapEvalModule` expects (one
``{wordswap,agrswap,inflswap,vp_swap_combined}_pairs.txt`` per corpus
subdir)::

    <cache_dir>/
    ├── babyview/
    │   ├── wordswap_pairs.txt
    │   ├── agrswap_pairs.txt
    │   ├── inflswap_pairs.txt
    │   └── vp_swap_combined_pairs.txt
    ├── ego4d/   (same 4 files)
    ├── howto/   (same 4 files)
    └── coco_mc/ (same 4 files)

Point ``LTSWAP_DATA_ROOT`` at ``<cache_dir>/<your_corpus>`` (NOT at
``<cache_dir>`` itself — the eval reads pair files directly from
``data_dir``).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    flatten_single_wrapper_dir,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://github.com/facebookresearch/egobabyvlm/releases/download/Eval-Data/LTSwap.tar"

EXPECTED_CORPORA = ("babyview", "ego4d", "howto", "coco_mc")
EXPECTED_FILES = (
    "wordswap_pairs.txt",
    "agrswap_pairs.txt",
    "inflswap_pairs.txt",
    "vp_swap_combined_pairs.txt",
)


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (wget when available, urllib otherwise)."""
    logger.info("Downloading %s -> %s", url, dest)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310


def _extract(archive: Path, dest_dir: Path) -> None:
    """Extract a ``.tar`` / ``.tar.gz`` into ``dest_dir`` (creating it if needed)."""
    logger.info("Extracting %s -> %s", archive, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir, filter="data")


def _looks_like_extracted(root: Path) -> bool:
    """Quick sanity check: at least one expected corpus subdir with all 4 pair files."""
    for corpus in EXPECTED_CORPORA:
        sub = root / corpus
        if sub.is_dir() and all((sub / fn).is_file() for fn in EXPECTED_FILES):
            return True
    return False


def main() -> None:
    parser: argparse.ArgumentParser = cache_argparser("ltswap")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to a locally-staged LTSwap.tar (skips the network download).",
    )
    parser.add_argument(
        "--url",
        default=ARCHIVE_URL,
        help=f"Override the download URL (default: {ARCHIVE_URL}).",
    )
    args = parser.parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("LT-Swap already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "LT-Swap", env_var="LTSWAP_DATA_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.archive is not None:
        if not args.archive.is_file():
            logger.error("--archive points at %s but no such file exists.", args.archive)
            sys.exit(1)
        _extract(args.archive, args.cache_dir)
    else:
        tmp_archive = args.cache_dir / "_ltswap.tar"
        try:
            _download(args.url, tmp_archive)
            _extract(tmp_archive, args.cache_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("LT-Swap download failed: %s", e)
            sys.exit(1)
        finally:
            if tmp_archive.exists():
                tmp_archive.unlink()

    if not _looks_like_extracted(args.cache_dir):
        # Tarball may be wrapped in a single top-level dir
        # (`tar -cf foo.tar LTSwap/`); strip it if so.
        flatten_single_wrapper_dir(args.cache_dir, EXPECTED_CORPORA)

    if not _looks_like_extracted(args.cache_dir):
        logger.error(
            "Extraction finished but %s does not contain the expected per-corpus subdirs. "
            "The archive layout may have changed.",
            args.cache_dir,
        )
        sys.exit(1)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "LT-Swap", env_var="LTSWAP_DATA_ROOT")
    logger.info(
        "Note: LTSWAP_DATA_ROOT should point at one of the per-corpus subdirs "
        "(e.g. %s/babyview), not the cache root itself.",
        args.cache_dir,
    )
    logger.warning(
        "These shipped pair files were generated against the paper's exact training "
        "corpus snapshots and are intended primarily for reproducing / comparing to "
        "the paper. If your training data differs (different snapshot, subset, "
        "preprocessing, tokenization), regenerate the pair files from your own "
        "corpus via apps/swapbench/ so the long-tail vocabulary matches what your "
        "model actually saw."
    )


if __name__ == "__main__":
    main()
