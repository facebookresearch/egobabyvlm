# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for the benchmark_creation pipeline.

Submodules are imported on demand rather than re-exported here so that
lightweight entry points (e.g. ``vllm_server``) don't pull in heavy or
optional dependencies (nltk, transformers) just to expose the package
namespace.
"""
