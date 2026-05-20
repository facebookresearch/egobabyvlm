#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run every automatable eval-data download script.

Skips datasets that are already cached. Use the per-script ``--force`` flag if
you need to rebuild a single dataset; this orchestrator deliberately does not
expose ``--force`` to avoid accidentally re-downloading multi-GB COCO-Stuff.
"""

from __future__ import annotations

import logging
import sys

from scripts.eval_data import (
    download_cocostuff,
    download_countbench,
    download_devbench,
    download_ltswap,
    download_machine_devbench,
    download_mnist,
    download_zorro,
)
from scripts.eval_data._common import setup_logging

logger = logging.getLogger(__name__)


SCRIPTS = (
    ("MNIST", download_mnist),
    ("CountBench", download_countbench),
    ("Zorro", download_zorro),
    ("DevBench", download_devbench),
    ("Machine-DevBench", download_machine_devbench),
    ("LT-Swap", download_ltswap),
    ("COCO-Stuff", download_cocostuff),
)


def main() -> None:
    setup_logging()

    failures = []
    for name, module in SCRIPTS:
        logger.info("=== %s ===", name)
        # Reset argv so each script's argparse sees no extra args (and uses defaults).
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            module.main()
        except SystemExit as e:
            if e.code not in (0, None):
                failures.append(name)
        except Exception as e:  # noqa: BLE001
            logger.error("%s failed: %s", name, e)
            failures.append(name)
        finally:
            sys.argv = old_argv

    if failures:
        logger.error("Failed: %s", ", ".join(failures))
        sys.exit(1)
    logger.info("All automatable eval datasets ready.")
    logger.info(
        "Note: ImageNet and NYUv2 require manual setup; see docs/eval_data.md.",
    )


if __name__ == "__main__":
    main()
