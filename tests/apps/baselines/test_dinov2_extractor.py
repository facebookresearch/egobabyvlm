# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the DINOv2 feature extractor.

The instantiation test downloads weights from ``torch.hub`` on first use and is
marked ``integration``. The structural protocol test just constructs a stub class
that satisfies :class:`core.protocols.ImageFeatureExtractor` without any network
or heavy dependencies.
"""

from __future__ import annotations

import pytest
import torch

from core.protocols import ImageFeatureExtractor


def test_protocol_is_satisfied_structurally() -> None:
    """A class that defines the right surface satisfies :class:`ImageFeatureExtractor`."""

    class StubExtractor(torch.nn.Module):
        @property
        def feature_dim(self) -> int:
            return 8

        @property
        def input_size(self) -> tuple[int, int]:
            return (224, 224)

        @property
        def normalize_params(self) -> dict[str, list[float]]:
            return {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

        def extract_features(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0], self.feature_dim)

    assert isinstance(StubExtractor(), ImageFeatureExtractor)


@pytest.mark.integration
def test_dinov2_vitb14_extracts_cls_features() -> None:
    """``DINOv2FeatureExtractor`` produces ``(B, 768)`` CLS-pooled features for ViT-B/14."""
    from apps.baselines.dinov2.extractor import DINOv2FeatureExtractor

    extractor = DINOv2FeatureExtractor(pretrained_weights="dinov2_vitb14", pooling="cls")
    assert isinstance(extractor, ImageFeatureExtractor)
    assert extractor.feature_dim == 768
    assert extractor.input_size == (518, 518)

    images = torch.randn(2, 3, 518, 518)
    features = extractor.extract_features(images)
    assert features.shape == (2, 768)
