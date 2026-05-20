# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LT-Swap (Long-Tail Swap) generation pipelines.

Hydra-driven runners (one per task) that drive the upstream LT-Swap
generators in ``apps.swapbench.third_party.lt_swap.generate_task``. The
upstream ``mp_main`` LLM-orchestrator is not used; instead the runners
call the async worker pool in ``apps.swapbench.utils.llm_runner``.
"""
