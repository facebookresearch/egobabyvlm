# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EgoVLP (Frozen-in-Time) baseline for the EgoBabyVLM MachineDevBench.

Vendors the EgoVLP dual encoder (SpaceTimeTransformer + DistilBERT) so the
released EgoClip checkpoint loads under the repo's torch/timm/transformers
stack, plus a contrastive fine-tuning entrypoint (BabyView and COCO-MC at five
alignment levels, images as 1-frame clips) and a DevBench feature extractor
that shares the trainer's model code. See ``README.md``.
"""
