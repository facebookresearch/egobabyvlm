# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test that the in-tree DINOv2 fork is exposed as the bare ``dinov2`` package."""

from __future__ import annotations

from pathlib import Path


def test_bare_import_resolves_to_in_tree_copy() -> None:
    """``import dinov2`` must work directly, without first importing through
    ``apps.baselines.dinov2.third_party.dinov2`` to trigger a sys.modules
    side effect — that import-order trap is what the pyproject ``package-dir``
    mapping is meant to remove.
    """
    import dinov2

    pkg_path = Path(dinov2.__file__).resolve()
    assert pkg_path.parts[-3:] == ("third_party", "dinov2", "__init__.py"), (
        f"dinov2 should resolve into apps/baselines/dinov2/third_party/dinov2/, got {pkg_path}"
    )


def test_dinov2_subpackage_imports() -> None:
    """A representative subpackage (the dataset registry) imports without errors."""
    from dinov2.data.datasets import ImageNet

    assert hasattr(ImageNet, "Split"), "ImageNet should expose a Split enum"
