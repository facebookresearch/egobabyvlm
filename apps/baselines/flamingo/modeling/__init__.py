# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Native Flamingo modeling package."""

from apps.baselines.flamingo.modeling.aux_heads import FlamingoAuxHeads, masked_mean
from apps.baselines.flamingo.modeling.decoder import (
    FlamingoDecoderOutput,
    FlamingoGPT2Decoder,
    build_gpt2_backbone,
)
from apps.baselines.flamingo.modeling.flamingo_model import (
    FlamingoConfig,
    FlamingoOutput,
    FlamingoVLM,
    build_flamingo_model_and_tokenizer,
    build_tokenizer,
    config_to_dict,
)
from apps.baselines.flamingo.modeling.gated_cross_attention import (
    GATE_MODES,
    GatedCrossAttentionBlock,
    MaskedCrossAttention,
)
from apps.baselines.flamingo.modeling.perceiver_resampler import (
    PerceiverAttention,
    PerceiverResampler,
)
from apps.baselines.flamingo.modeling.vision_tower import FrozenDINOv2VisionTower

__all__ = [
    "GATE_MODES",
    "FlamingoAuxHeads",
    "FlamingoConfig",
    "FlamingoDecoderOutput",
    "FlamingoGPT2Decoder",
    "FlamingoOutput",
    "FlamingoVLM",
    "FrozenDINOv2VisionTower",
    "GatedCrossAttentionBlock",
    "MaskedCrossAttention",
    "PerceiverAttention",
    "PerceiverResampler",
    "build_flamingo_model_and_tokenizer",
    "build_gpt2_backbone",
    "build_tokenizer",
    "config_to_dict",
    "masked_mean",
]
