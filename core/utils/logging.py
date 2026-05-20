# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Logging setup and config printing."""

import logging

from omegaconf import DictConfig, OmegaConf

from core.utils.yaml import to_yaml

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set the root logger to INFO with a consistent format on all handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        root.addHandler(ch)
    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s]  %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    for handler in root.handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(fmt)


def resolve_and_print_config(config: DictConfig) -> None:
    """Resolve OmegaConf interpolations and log the rendered config.

    Args:
        config: Hydra/OmegaConf config to resolve and print.
    """
    OmegaConf.resolve(config)
    logger.info("Config:\n%s\n%s\n%s", "-" * 50, to_yaml(config), "-" * 50)
