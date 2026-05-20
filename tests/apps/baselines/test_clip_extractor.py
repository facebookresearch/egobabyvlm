# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the CLIP feature extractors.

Instantiation tests download weights via ``open_clip`` (cached under ``~/.cache/clip``)
and are marked ``integration``. Structural protocol tests construct stubs that satisfy
the relevant protocols without any network or heavy dependencies.
"""

from __future__ import annotations

import pytest
import torch

from core.protocols import ImageFeatureExtractor, MultiModalFeatureExtractor, TextFeatureExtractor


def test_image_protocol_satisfied_structurally() -> None:
    """A minimal image stub satisfies :class:`ImageFeatureExtractor`."""

    class StubImage(torch.nn.Module):
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

    assert isinstance(StubImage(), ImageFeatureExtractor)


def test_text_protocol_satisfied_structurally() -> None:
    """A minimal text stub satisfies :class:`TextFeatureExtractor`."""

    class StubText(torch.nn.Module):
        @property
        def feature_dim(self) -> int:
            return 8

        def extract_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
            n = len(text) if isinstance(text, list) else text.shape[0]
            return torch.zeros(n, self.feature_dim)

        def tokenize(self, text: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:  # noqa: ARG002
            return {"input_ids": torch.zeros(len(text), 1, dtype=torch.long)}

    assert isinstance(StubText(), TextFeatureExtractor)


def test_multimodal_protocol_satisfied_structurally() -> None:
    """A minimal multimodal stub satisfies :class:`MultiModalFeatureExtractor`."""

    class StubMM(torch.nn.Module):
        @property
        def feature_dim(self) -> int:
            return 8

        def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0], self.feature_dim)

        def extract_text_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
            n = len(text) if isinstance(text, list) else text.shape[0]
            return torch.zeros(n, self.feature_dim)

        def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
            return torch.zeros(video.shape[0], self.feature_dim)

        def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.extract_image_features(inputs)

        def compute_similarity(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            normalize: bool | None = None,  # noqa: ARG002
        ) -> torch.Tensor:
            return a @ b.T

    assert isinstance(StubMM(), MultiModalFeatureExtractor)


@pytest.mark.integration
def test_clip_image_extractor_forward() -> None:
    """``CLIPImageFeatureExtractor`` produces image features of the expected shape."""
    from apps.baselines.clip.openclip_extractor import CLIPImageFeatureExtractor

    extractor = CLIPImageFeatureExtractor(model_name="ViT-B-16-quickgelu", pretrained="openai")
    assert isinstance(extractor, ImageFeatureExtractor)
    images = torch.randn(2, 3, *extractor.input_size)
    features = extractor.extract_features(images)
    assert features.shape == (2, extractor.feature_dim)


@pytest.mark.integration
def test_clip_text_extractor_forward() -> None:
    """``CLIPTextFeatureExtractor`` tokenizes + embeds text into a fixed-dim tensor."""
    from apps.baselines.clip.openclip_extractor import CLIPTextFeatureExtractor

    extractor = CLIPTextFeatureExtractor(model_name="ViT-B-16-quickgelu", pretrained="openai")
    assert isinstance(extractor, TextFeatureExtractor)
    text = ["a photo of a cat", "a photo of a dog"]
    features = extractor.extract_features(text)
    assert features.shape == (2, extractor.feature_dim)


@pytest.mark.integration
def test_clip_multimodal_extractor_forward() -> None:
    """``CLIPFeatureExtractor`` produces aligned image+text features."""
    from apps.baselines.clip.openclip_extractor import CLIPFeatureExtractor

    extractor = CLIPFeatureExtractor(model_name="ViT-B-16-quickgelu", pretrained="openai")
    assert isinstance(extractor, MultiModalFeatureExtractor)

    images = torch.randn(2, 3, 224, 224)
    image_features = extractor.extract_image_features(images)
    assert image_features.shape == (2, extractor.feature_dim)

    text = ["a photo of a cat", "a photo of a dog"]
    text_features = extractor.extract_text_features(text)
    assert text_features.shape == (2, extractor.feature_dim)

    sim = extractor.compute_similarity(image_features, text_features)
    assert sim.shape == (2, 2)
