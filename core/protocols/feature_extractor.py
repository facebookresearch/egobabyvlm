# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Feature extractor protocols."""

from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


class _ModuleLike(Protocol):
    """The :class:`nn.Module` surface the eval pipeline expects.

    Concrete extractors typically subclass ``nn.Module`` directly; this
    Protocol lets mypy verify the call sites without forcing the runtime
    ``isinstance(model, nn.Module)`` check.
    """

    def eval(self) -> "_ModuleLike": ...
    def train(self, mode: bool = True) -> "_ModuleLike": ...  # noqa: FBT001, FBT002
    def to(self, *args: Any, **kwargs: Any) -> "_ModuleLike": ...  # noqa: ANN401
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...  # noqa: FBT001, FBT002
    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...  # noqa: ANN401


@runtime_checkable
class FeatureExtractor(_ModuleLike, Protocol):
    """Maps inputs to a feature tensor."""

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""

    def extract_features(self, inputs: Any) -> torch.Tensor:  # noqa: ANN401
        """Extract features from inputs.

        Args:
            inputs: Model-specific input.

        Returns:
            Features of shape ``(B, D)`` or ``(B, T, D)``.
        """


@runtime_checkable
class ImageFeatureExtractor(FeatureExtractor, Protocol):
    """Image-only extractor."""

    @property
    def input_size(self) -> tuple[int, int]:
        """Expected ``(H, W)`` input size."""

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        """Input normalization parameters with ``mean`` and ``std`` keys."""

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Features of shape ``(B, D)``.
        """


@runtime_checkable
class TextFeatureExtractor(FeatureExtractor, Protocol):
    """Text-only extractor."""

    def extract_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
        """Extract features from text.

        Args:
            text: List of strings or pre-tokenized tensor.

        Returns:
            Features of shape ``(B, D)``.
        """

    def tokenize(self, text: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """Tokenize text inputs.

        Args:
            text: List of strings.
            device: Target device for the returned tensors.

        Returns:
            Tokenized inputs (e.g. ``input_ids``, ``attention_mask``).
        """


@runtime_checkable
class VideoFeatureExtractor(FeatureExtractor, Protocol):
    """Video extractor."""

    @property
    def num_frames(self) -> int:
        """Expected number of input frames."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """Expected ``(H, W)`` frame size."""

    @property
    def temporal_output(self) -> bool:
        """Whether :meth:`extract_features` returns per-frame features."""

    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from video.

        Args:
            video: Tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Features of shape ``(B, D)`` for video-level or ``(B, T, D)`` for
            frame-level outputs.
        """


@runtime_checkable
class MultiModalFeatureExtractor(_ModuleLike, Protocol):
    """Multi-modal extractor with explicit similarity computation.

    Implementations may also expose ``extract_features(inputs)`` that
    dispatches on input shape (tensor → image, list[str] → text,
    Mapping with ``"image"`` / ``"text"`` keys → both, returning a dict
    of feature tensors).
    """

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            images: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Features of shape ``(B, D)``.
        """

    def extract_text_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
        """Extract text features.

        Args:
            text: List of strings or pre-tokenized tensor.

        Returns:
            Features of shape ``(B, D)``.
        """

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract video features.

        Args:
            video: Tensor of shape ``(B, T, C, H, W)``.

        Returns:
            Features of shape ``(B, D)``.
        """

    def extract_features(
        self,
        inputs: torch.Tensor | list[str] | Mapping[str, Any],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Dispatch on input shape.

        * Image tensor or PIL list           → ``(B, D)`` image features.
        * Raw text list                      → ``(B, D)`` text features.
        * Mapping with ``"image"`` / ``"text"`` keys → dict of feature tensors
          keyed by ``image_features`` / ``text_features`` (the DevBench convention).
        """

    def compute_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Compute pairwise similarity.

        Args:
            a: Tensor of shape ``(N, D)``.
            b: Tensor of shape ``(M, D)``.
            normalize: If ``True``, L2-normalize before the dot product. If
                ``None``, the implementation decides (typically a no-op when
                features are already normalized at extraction time).

        Returns:
            Similarity matrix of shape ``(N, M)``.
        """
