#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download MNIST into the local eval-data cache via :mod:`torchvision`."""

from __future__ import annotations

import logging

from torchvision.datasets import MNIST

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)


def main() -> None:
    args = cache_argparser("mnist").parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("MNIST already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "MNIST", env_var="MNIST_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MNIST train + test splits to %s", args.cache_dir)
    MNIST(root=str(args.cache_dir), train=True, download=True)
    MNIST(root=str(args.cache_dir), train=False, download=True)

    mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "MNIST", env_var="MNIST_ROOT")


if __name__ == "__main__":
    main()
