# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper around :class:`torch.utils.data.DataLoader` for evaluation tasks."""

from torch.utils.data import DataLoader as TorchDataLoader


class EvalDataLoader(TorchDataLoader):
    """:class:`torch.utils.data.DataLoader` subclass reserved for eval-task customization."""
