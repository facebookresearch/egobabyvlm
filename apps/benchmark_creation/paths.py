# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Centralised path configuration for benchmark_creation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_PATHS_ENV_VAR = "BENCHMARK_CREATION_PATHS"
_STYLES_ENV_VAR = "BENCHMARK_CREATION_STYLES"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PACKAGE_ROOT / "configs" / "paths.yaml"
_STYLES_CONFIG = _PACKAGE_ROOT / "configs" / "styles.yaml"

_cached_paths: dict[str, Any] | None = None
_cached_styles: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Config file not found: {path}\nExpected it relative to the package root: {_PACKAGE_ROOT}"
        raise FileNotFoundError(msg)
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _apply_env_override(base: dict[str, Any], env_var: str) -> dict[str, Any]:
    """Merge values from an override YAML file if the env var is set."""
    override = os.environ.get(env_var)
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            base.update(_load_yaml(override_path))
    return base


def get_paths() -> dict[str, Any]:
    """Return the merged path configuration dict.

    The first call reads from disk and caches the result; subsequent calls
    return the cached copy. Override order:

    1. ``configs/paths.yaml`` (package default)
    2. File at ``$BENCHMARK_CREATION_PATHS`` (if set)
    """
    global _cached_paths  # noqa: PLW0603
    if _cached_paths is not None:
        return _cached_paths

    paths = _load_yaml(_DEFAULT_CONFIG)
    _apply_env_override(paths, _PATHS_ENV_VAR)

    _cached_paths = paths
    return _cached_paths


def reset_cache() -> None:
    """Clear the cached paths/styles so the next call re-reads from disk."""
    global _cached_paths, _cached_styles  # noqa: PLW0603
    _cached_paths = None
    _cached_styles = None


def get_styles() -> dict[str, Any]:
    """Return the styles configuration dict from ``configs/styles.yaml``.

    Supports override via ``$BENCHMARK_CREATION_STYLES``. Cached after first load.
    """
    global _cached_styles  # noqa: PLW0603
    if _cached_styles is not None:
        return _cached_styles

    styles = _load_yaml(_STYLES_CONFIG)
    _apply_env_override(styles, _STYLES_ENV_VAR)

    _cached_styles = styles
    return _cached_styles


def get_style_prefixes() -> dict[str, str]:
    """Return ``{style_name: prefix_string}`` for all configured styles."""
    return {name: info["prefix"] for name, info in get_styles().items()}


def _resolve_path(raw: str) -> Path:
    """Return an absolute Path, resolving relative paths against the package root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return _PACKAGE_ROOT / p


def get_outputs_root() -> Path:
    """Return ``outputs_root`` as a :class:`~pathlib.Path`."""
    return _resolve_path(get_paths()["outputs_root"])


def get_coco_captions() -> Path:
    """Return ``coco_captions`` as a :class:`~pathlib.Path`."""
    return _resolve_path(get_paths()["coco_captions"])


def get_howto100m_manifest() -> Path:
    """Return ``howto100m_manifest`` as a :class:`~pathlib.Path`."""
    return _resolve_path(get_paths()["howto100m_manifest"])


def get_ego4d_manifest() -> Path:
    """Return ``ego4d_manifest`` as a :class:`~pathlib.Path`."""
    return _resolve_path(get_paths()["ego4d_manifest"])


def get_babyview_manifest() -> Path:
    """Return ``babyview_manifest`` as a :class:`~pathlib.Path`."""
    return _resolve_path(get_paths()["babyview_manifest"])
