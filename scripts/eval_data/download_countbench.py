#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download CountBench (10-class object-counting) from HuggingFace into the cache."""

from __future__ import annotations

import logging

from datasets import load_dataset

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

#: Pinned dataset revision so future upstream changes don't silently shift
#: the eval score. Bump deliberately if the upstream dataset is corrected.
COUNTBENCH_REVISION = "e32ce2541299d12755894e1d487f6dd75bb0176c"


def main() -> None:
    args = cache_argparser("countbench").parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("CountBench already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "CountBench", env_var="COUNTBENCH_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading nielsr/countbench@%s to %s", COUNTBENCH_REVISION, args.cache_dir)
    load_dataset("nielsr/countbench", revision=COUNTBENCH_REVISION, cache_dir=str(args.cache_dir))

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "CountBench", env_var="COUNTBENCH_ROOT")


if __name__ == "__main__":
    main()
