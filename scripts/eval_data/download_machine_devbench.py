#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download and extract the Machine-DevBench evaluation tarball.

Output layout matches what
:class:`evaluation.data.machine_devbench.BenchmarkData` expects::

    <cache_dir>/
    ├── Lexical/
    │   ├── Nouns/manifest_nouns_<style>.json + images
    │   ├── Verbs/manifest_verbs_<style>.json + images
    │   └── Adjectives/manifest_adjectives_<style>.json + images
    └── Grammatical/
        └── gram_<category>/manifest_grammatical_<category>_<style>.json + images

Point ``MACHINE_DEVBENCH_DATA_ROOT`` at the resulting directory and the
multimodal eval YAMLs Just Work.
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

ARCHIVE_URL = "https://github.com/facebookresearch/egobabyvlm/releases/download/Eval-Data/MachineDevBench.tar"


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
        # Python 3.12+ ships the ``data`` filter which rejects path-traversal
        # entries; safe for trusted release assets and avoids the upcoming
        # default-filter DeprecationWarning.
        tf.extractall(dest_dir, filter="data")


def _looks_like_extracted(root: Path) -> bool:
    """Quick sanity check that ``root`` already has the expected top-level layout."""
    return (root / "Lexical").is_dir() and (root / "Grammatical").is_dir()


def main() -> None:
    parser: argparse.ArgumentParser = cache_argparser("machine_devbench")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=("Path to a locally-staged MachineDevBench.tar (skips the network download)."),
    )
    parser.add_argument(
        "--url",
        default=ARCHIVE_URL,
        help=f"Override the download URL (default: {ARCHIVE_URL}).",
    )
    args = parser.parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info(
            "Machine-DevBench already present at %s; pass --force to redownload.",
            args.cache_dir,
        )
        announce(args.cache_dir, "Machine-DevBench", env_var="MACHINE_DEVBENCH_DATA_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.archive is not None:
        if not args.archive.is_file():
            logger.error("--archive points at %s but no such file exists.", args.archive)
            sys.exit(1)
        _extract(args.archive, args.cache_dir)
    else:
        tmp_archive = args.cache_dir / "_machine_devbench.tar"
        try:
            _download(args.url, tmp_archive)
            _extract(tmp_archive, args.cache_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("Machine-DevBench download failed: %s", e)
            sys.exit(1)
        finally:
            if tmp_archive.exists():
                tmp_archive.unlink()

    if not _looks_like_extracted(args.cache_dir):
        # Tarball may be wrapped in a single top-level dir
        # (`tar -cf foo.tar MachineDevBench/`); strip it if so.
        flatten_single_wrapper_dir(args.cache_dir, ("Lexical", "Grammatical"))

    if not _looks_like_extracted(args.cache_dir):
        logger.error(
            "Extraction finished but %s does not contain the expected Lexical/ + Grammatical/ "
            "subdirectories. The archive layout may have changed.",
            args.cache_dir,
        )
        sys.exit(1)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "Machine-DevBench", env_var="MACHINE_DEVBENCH_DATA_ROOT")


if __name__ == "__main__":
    main()
