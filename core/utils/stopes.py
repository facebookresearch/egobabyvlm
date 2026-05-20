# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hydra config dataclasses for the Stopes launcher and file cache."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StopesCache:
    """Hydra-instantiable :class:`stopes.core.FileCache` config."""

    _target_: str = "stopes.core.FileCache"

    #: Directory to store cache files in.
    caching_dir: str = "cache"


@dataclass
class LauncherConfig:
    """Hydra-instantiable :class:`stopes.core.Launcher` config."""

    _target_: str = "stopes.core.Launcher"

    #: Cluster backend, e.g. ``"slurm"`` or ``"local"``.
    cluster: str = "local"

    #: Folder to store launcher logs in.
    log_folder: str = "executor_logs"

    #: Optional directory to dump per-task configuration files.
    config_dump_dir: str | None = None

    #: Maximum number of jobs to launch in parallel.
    max_jobarray_jobs: int = 128

    #: Per-cluster overrides such as ``slurm_partition`` or ``slurm_qos``.
    update_parameters: dict[str, Any] = field(default_factory=dict)

    #: Stopes file-cache config.
    cache: StopesCache = field(default_factory=StopesCache)
