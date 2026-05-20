# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the DINOv2 SSL trainer entrypoint + dataset registry."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

_REQUIRES_FVCORE = pytest.mark.skipif(
    importlib.util.find_spec("fvcore") is None,
    reason="fvcore not installed; trainer-entry imports skipped (it's pinned in pixi.toml).",
)


@_REQUIRES_FVCORE
def test_training_entry_imports() -> None:
    """The OSS-facing entry shim re-exports the upstream trainer's public API."""
    train_mod = importlib.import_module("apps.baselines.dinov2.training.train")
    assert callable(train_mod.main)
    assert callable(train_mod.do_train)
    assert callable(train_mod.get_args_parser)


@_REQUIRES_FVCORE
def test_submit_entry_imports() -> None:
    """The Submitit driver shim is importable and exposes a ``main()``."""
    submit_mod = importlib.import_module("apps.baselines.dinov2.training.submit")
    assert callable(submit_mod.main)


@_REQUIRES_FVCORE
def test_argparse_smokes() -> None:
    """``get_args_parser()`` constructs without erroring and accepts the documented flags."""
    from apps.baselines.dinov2.training.train import get_args_parser

    parser = get_args_parser(add_help=False)
    args = parser.parse_args(["--config-file", "/tmp/dummy.yaml", "--no-wandb"])
    assert args.config_file == "/tmp/dummy.yaml"
    assert args.no_wandb is True


# ---- dataset registry -----------------------------------------------------


_EXPECTED_DATASETS = ("BabyView", "Ego4D", "HowToSubset", "ImageNet", "MSCOCO")


@pytest.mark.parametrize("name", _EXPECTED_DATASETS)
def test_dataset_imports(name: str) -> None:
    """Every dataset class shipped under ``data/datasets/`` is importable."""
    mod = importlib.import_module("dinov2.data.datasets")
    assert hasattr(mod, name), f"{name} missing from data.datasets exports"
    cls = getattr(mod, name)
    assert hasattr(cls, "Split"), f"{name} should expose a Split enum"


def test_make_dataset_resolves_known_names() -> None:
    """``make_dataset`` rejects unknown names and accepts each shipped one."""
    from dinov2.data.loaders import _parse_dataset_str

    # Each shipped dataset name should resolve without raising. We can't
    # actually instantiate without real data, so we just check the parser
    # branches.
    for name in ("ImageNet", "MSCOCO", "Ego4D", "HowTo", "BabyView"):
        cls, _ = _parse_dataset_str(f"{name};root=/tmp/dummy;extra=/tmp/dummy")
        assert cls is not None

    with pytest.raises(ValueError, match="Unsupported dataset"):
        _parse_dataset_str("NoSuchDataset;root=/tmp/dummy")
