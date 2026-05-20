# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLIP feature extractors via :mod:`open_clip`."""

import logging
import types
from enum import Enum
from typing import Any, cast

import open_clip
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


def _extract_normalize_params(preprocess: transforms.Compose) -> dict[str, list[float]]:
    """Extract mean/std normalization params from an open_clip preprocess pipeline."""
    # Default CLIP normalization (OpenAI)
    params: dict[str, list[float]] = {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }
    for t in preprocess.transforms:
        if isinstance(t, transforms.Normalize):
            params = {
                "mean": list(t.mean.tolist() if isinstance(t.mean, torch.Tensor) else t.mean),
                "std": list(t.std.tolist() if isinstance(t.std, torch.Tensor) else t.std),
            }
            break
    return params


def _resample_pos_embed(
    pos_embed: torch.Tensor,
    base_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    """Bicubically resample CLS+patch positional embeddings to a new grid size.

    Args:
        pos_embed: ``(1 + base_h * base_w, D)`` embedding from the loaded model.
        base_grid: ``(base_h, base_w)`` grid size the embedding was trained on.
        target_grid: ``(target_h, target_w)`` grid size to resample to.

    Returns:
        ``(1 + target_h * target_w, D)`` tensor preserving the CLS row.
    """
    if base_grid == target_grid:
        return pos_embed
    base_h, base_w = base_grid
    target_h, target_w = target_grid
    cls_pos, patch_pos = pos_embed[:1], pos_embed[1:]
    patch_pos = patch_pos.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=target_grid,
        mode="bicubic",
        align_corners=False,
    ).to(pos_embed.dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(target_h * target_w, -1)
    return torch.cat([cls_pos, patch_pos], dim=0)


def _enable_pos_embed_interpolation(visual_model: torch.nn.Module) -> None:
    """Allow an open_clip VisionTransformer to accept inputs at non-native resolutions.

    Wraps :py:meth:`_embeds` so the positional embedding is bicubically
    resampled when the input grid differs from the model's training grid
    (DeiT / MAE / DINOv2 approach).

    Resampled embeddings are cached per ``(H, W)`` grid size in eval mode, so
    the interpolation cost is paid once per unique input shape instead of every
    forward call. The cache is bypassed when autograd is enabled so that
    fine-tuning at non-native resolution still propagates gradients through
    ``positional_embedding``.
    """
    base_grid: tuple[int, int] = cast("tuple[int, int]", visual_model.grid_size)  # (H, W) at training resolution
    pos_embed_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _embeds_with_interpolation(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = cast("torch.nn.Module", self.conv1)(x)  # (B, width, grid_h, grid_w)
        target_grid = (x.shape[2], x.shape[3])
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, grid_h*grid_w, width)

        cls_embed = cast("torch.Tensor", self.class_embedding)
        cls = cls_embed.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)

        positional_embedding = cast("torch.Tensor", self.positional_embedding)
        if target_grid == base_grid:
            pos_embed = positional_embedding
        elif torch.is_grad_enabled():
            pos_embed = _resample_pos_embed(positional_embedding, base_grid, target_grid)
        else:
            if target_grid not in pos_embed_cache:
                pos_embed_cache[target_grid] = _resample_pos_embed(
                    positional_embedding,
                    base_grid,
                    target_grid,
                )
            pos_embed = pos_embed_cache[target_grid]

        x = x + pos_embed.to(device=x.device, dtype=x.dtype)

        x = cast("torch.nn.Module", self.patch_dropout)(x)
        x = cast("torch.nn.Module", self.ln_pre)(x)
        return x

    visual_model._embeds = types.MethodType(_embeds_with_interpolation, visual_model)  # type: ignore[assignment]
    logger.info(
        "Enabled positional embedding interpolation (base grid: %dx%d)",
        base_grid[0],
        base_grid[1],
    )


class CLIPPooling(str, Enum):
    """Pooling strategies for CLIP image feature extraction."""

    CLS = "cls"
    MEAN_PATCH = "mean_patch"
    CONCAT_CLS = "concat_cls"
    CONCAT_CLS_AVGPOOL = "concat_cls_avgpool"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class CLIPFeatureExtractor(torch.nn.Module):
    """
    Multi-modal (image + text) feature extractor for CLIP-family models via open_clip.

    Supports CLIP, SigLIP, and any model loadable by
    ``open_clip.create_model_and_transforms``.

    Args:
        model_name: open_clip model name (e.g., "ViT-B-32")
        pretrained: Pretrained weights tag (e.g., "openai", "meta")
        normalize: Whether to L2-normalize output features (default: True)
        device: Device to load model on
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        normalize: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model(model_name, pretrained)

    def _load_model(self, model_name: str, pretrained: str) -> None:
        logger.info("Loading CLIP model via open_clip: %s (pretrained=%s)", model_name, pretrained)

        self.model, _, val_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable positional embedding interpolation for non-native input sizes
        _enable_pos_embed_interpolation(self.model.visual)

        # Feature dimension is the shared projection space.
        self._feature_dim = self.model.visual.output_dim
        self._image_size = (
            self.model.visual.image_size[0]
            if isinstance(self.model.visual.image_size, tuple)
            else self.model.visual.image_size
        )
        self._normalize_params = _extract_normalize_params(val_preprocess)

        # Build image transforms for tensor and PIL inputs
        mean = self._normalize_params["mean"]
        std = self._normalize_params["std"]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self._image_size, self._image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(mean, std),
            ]
        )
        self.pil_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self._image_size, self._image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        logger.info(
            "Loaded CLIP model with feature_dim=%d, image_size=%d",
            self._feature_dim,
            self._image_size,
        )

    def extract_image_features(self, images: torch.Tensor | list) -> torch.Tensor:
        """
        Extract image features using CLIP vision encoder.

        Args:
            images: Tensor of shape (B, C, H, W) or list of PIL Images

        Returns:
            Features of shape (B, D)
        """
        if isinstance(images, list):
            images_tensor = torch.stack([self.pil_transform(img.convert("RGB")) for img in images])
        elif images.shape[-1] != self._image_size or images.shape[-2] != self._image_size:
            images_tensor = self.image_transform(images)
        else:
            images_tensor = images

        images_tensor = images_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_image(images_tensor, normalize=self.normalize)

        return outputs

    def extract_text_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
        """
        Extract text features using CLIP text encoder.

        Args:
            text: List of strings or pre-tokenized tensor

        Returns:
            Features of shape (B, D)
        """
        if isinstance(text, list):
            tokens = self.tokenizer(text).to(self.device)
        elif isinstance(text, dict):
            tokens = text["input_ids"].to(self.device)
        else:
            tokens = text.to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_text(tokens, normalize=self.normalize)

        return outputs

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract video features by averaging frame-level CLIP features.

        Args:
            video: Tensor of shape (B, T, C, H, W)

        Returns:
            Features of shape (B, D) averaged over frames
        """
        batch_size, num_frames = video.shape[:2]
        frames = video.view(-1, *video.shape[2:])
        frame_features = self.extract_image_features(frames)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        video_features = frame_features.mean(dim=1)

        if self.normalize:
            video_features = F.normalize(video_features, p=2, dim=-1)

        return video_features

    def extract_features(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Dispatch to the per-modality extractors based on the input dict's keys.

        Args:
            inputs: Mapping with optional ``image``, ``text``, and ``video`` keys.

        Returns:
            Dict with the matching ``image_features`` / ``text_features`` /
            ``video_features`` tensors.
        """
        outputs = {}
        if "image" in inputs:
            outputs["image_features"] = self.extract_image_features(inputs["image"])
        if "text" in inputs:
            outputs["text_features"] = self.extract_text_features(inputs["text"])
        if "video" in inputs:
            outputs["video_features"] = self.extract_video_features(inputs["video"])
        return outputs

    def tokenize(self, text: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """
        Tokenize text inputs.

        Args:
            text: List of strings
            device: Target device for tensors

        Returns:
            Dictionary with tokenized inputs
        """
        target_device = device or self.device
        tokens = self.tokenizer(text).to(target_device)
        return {"input_ids": tokens}

    def compute_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute CLIP-style similarity scores between two feature batches.

        Works for any combination of image, text, or video features as long as
        they live in the shared CLIP projection space (e.g. image vs. text,
        video vs. text).

        Args:
            a: Shape (N, D)
            b: Shape (M, D)
            normalize: If True, ensure features are L2-normalized before the dot product.
                When ``self.normalize`` is also True the encoders already returned
                normalized features and this is a no-op.

        Returns:
            Logits of shape (N, M) scaled by CLIP temperature
        """
        if normalize and not self.normalize:
            a = F.normalize(a, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        return logit_scale * a @ b.T

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def input_size(self) -> tuple[int, int]:
        return (self._image_size, self._image_size)

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        return self._normalize_params


class CLIPImageFeatureExtractor(torch.nn.Module):
    """
    Image-only feature extractor using CLIP's vision encoder via open_clip.

    Supports multiple pooling strategies including dense prediction modes
    for semantic segmentation and depth estimation (DPT).

    For depth estimation, use ``get_intermediate_layers()`` which returns
    multi-scale features from evenly-spaced transformer blocks.

    For semantic segmentation, use ``pooling="semantic_segmentation"`` which
    returns patch-level features of shape ``(B, N_patches, D)``.

    Args:
        model_name: open_clip model name (e.g., "ViT-B-32")
        pretrained: Pretrained weights tag (e.g., "openai", "meta")
        pooling: Pooling strategy ("cls", "mean_patch", "concat_cls", "concat_cls_avgpool", "semantic_segmentation")
        last_n_layers: Number of last layers to use for multi-layer pooling strategies
        normalize: Whether to L2-normalize output features (default: False)
        device: Device to load model on
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        pooling: str | CLIPPooling = CLIPPooling.CLS,
        last_n_layers: int = 1,
        *,
        normalize: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.pooling = CLIPPooling(pooling)
        self.last_n_layers = last_n_layers
        self._normalize = normalize
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model(model_name, pretrained)

    def _load_model(self, model_name: str, pretrained: str) -> None:
        logger.info("Loading CLIP visual encoder via open_clip: %s (pretrained=%s)", model_name, pretrained)

        model, _, val_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

        self.model = model.visual
        self.model.to(self.device)
        self.model.eval()

        # Enable positional embedding interpolation for non-native input sizes
        _enable_pos_embed_interpolation(self.model)

        # Extract architecture info from the visual encoder.
        self._embed_dim = self.model.transformer.width
        self._patch_size = (
            self.model.patch_size[0] if isinstance(self.model.patch_size, tuple) else self.model.patch_size
        )
        self._n_blocks = len(self.model.transformer.resblocks)
        self._image_size = (
            self.model.image_size[0] if isinstance(self.model.image_size, tuple) else self.model.image_size
        )
        self._normalize_params = _extract_normalize_params(val_preprocess)

        logger.info(
            "Loaded CLIP visual encoder: embed_dim=%d, patch_size=%d, n_blocks=%d, image_size=%d",
            self._embed_dim,
            self._patch_size,
            self._n_blocks,
            self._image_size,
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using the configured pooling strategy.

        Args:
            images: (B, C, H, W) normalized image tensor

        Returns:
            For cls/mean_patch pooling: (B, D) feature tensor
            For semantic_segmentation pooling: (B, N_patches, D * last_n_layers)
        """
        images = images.to(self.device)

        with torch.no_grad():
            match self.pooling:
                case CLIPPooling.CLS:
                    out = self.model.forward_intermediates(
                        images,
                        indices=[-1],
                        output_fmt="NLC",
                        normalize_intermediates=True,
                        intermediates_only=True,
                        output_extra_tokens=True,
                    )
                    cls_tokens = out["image_intermediates_prefix"][-1]  # (B, 1, D)
                    features = cls_tokens[:, 0, :]  # (B, D)

                case CLIPPooling.MEAN_PATCH:
                    out = self.model.forward_intermediates(
                        images,
                        indices=[-1],
                        output_fmt="NLC",
                        normalize_intermediates=True,
                        intermediates_only=True,
                    )
                    features = out["image_intermediates"][-1].mean(dim=1)  # (B, D)

                case CLIPPooling.CONCAT_CLS:
                    out = self.model.forward_intermediates(
                        images,
                        indices=self.last_n_layers,
                        output_fmt="NLC",
                        normalize_intermediates=True,
                        intermediates_only=True,
                        output_extra_tokens=True,
                    )
                    # Concatenate CLS tokens from the last N layers: (B, D * N)
                    features = torch.cat(
                        [prefix[:, 0, :] for prefix in out["image_intermediates_prefix"]],
                        dim=-1,
                    )

                case CLIPPooling.CONCAT_CLS_AVGPOOL:
                    out = self.model.forward_intermediates(
                        images,
                        indices=self.last_n_layers,
                        output_fmt="NLC",
                        normalize_intermediates=True,
                        intermediates_only=True,
                        output_extra_tokens=True,
                    )
                    # Concatenate CLS tokens from last N layers + avg-pooled patches from last layer
                    global_tokens = torch.cat(
                        [prefix[:, 0, :] for prefix in out["image_intermediates_prefix"]],
                        dim=-1,
                    )
                    last_patch_avg = out["image_intermediates"][-1].mean(dim=1)  # (B, D)
                    features = torch.cat([global_tokens, last_patch_avg], dim=-1)

                case CLIPPooling.SEMANTIC_SEGMENTATION:
                    out = self.model.forward_intermediates(
                        images,
                        indices=self.last_n_layers,
                        output_fmt="NLC",
                        normalize_intermediates=True,
                        intermediates_only=True,
                        output_extra_tokens=True,
                    )
                    # For each layer, concatenate a CLS token to every patch token
                    # before concatenating across layers.
                    layer_features = []
                    for i, patches in enumerate(out["image_intermediates"]):
                        global_token = out["image_intermediates_prefix"][i][:, 0:1, :]  # (B, 1, D)
                        global_expanded = global_token.expand_as(patches)  # (B, N_patches, D)
                        layer_features.append(torch.cat([patches, global_expanded], dim=-1))  # (B, N_patches, 2D)
                    features = torch.cat(layer_features, dim=-1)  # (B, N_patches, 2D*N)

        features = features.float()

        if self._normalize:
            features = F.normalize(features, dim=-1)

        return features

    def get_intermediate_layers(
        self,
        images: torch.Tensor,
        n: int | list[int] = 4,
        *,
        reshape: bool = True,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get intermediate layer features for dense prediction tasks (depth, segmentation).

        Provides the same interface as DINOv2FeatureExtractor.get_intermediate_layers(),
        enabling CLIP backbones to be used with the DPT depth head.

        Args:
            images: (B, C, H, W) normalized image tensor
            n: Number of layers to return (from end) or list of specific layer indices
            reshape: Whether to reshape patch tokens to spatial format (B, C, H, W)
            return_class_token: Whether to include CLS token with each layer
            norm: Whether to apply layer norm to intermediate outputs

        Returns:
            If return_class_token=False: List of feature tensors (B, C, H, W) or (B, N, D)
            If return_class_token=True: List of (patch_features, cls_token) tuples
        """
        images = images.to(self.device)
        output_fmt = "NCHW" if reshape else "NLC"

        with torch.no_grad():
            out = self.model.forward_intermediates(
                images,
                indices=n,
                output_fmt=output_fmt,
                normalize_intermediates=norm,
                intermediates_only=True,
                output_extra_tokens=return_class_token,
            )

        intermediates = out["image_intermediates"]

        if return_class_token:
            prefixes = out["image_intermediates_prefix"]
            return [
                (patch_feat.float(), prefix[:, 0, :].float())
                for patch_feat, prefix in zip(intermediates, prefixes, strict=False)
            ]

        return [f.float() for f in intermediates]

    @property
    def feature_dim(self) -> int:
        match self.pooling:
            case CLIPPooling.CLS | CLIPPooling.MEAN_PATCH:
                return self._embed_dim
            case CLIPPooling.CONCAT_CLS:
                return self._embed_dim * self.last_n_layers
            case CLIPPooling.CONCAT_CLS_AVGPOOL:
                return self._embed_dim * (self.last_n_layers + 1)
            case CLIPPooling.SEMANTIC_SEGMENTATION:
                # Each layer contributes patch tokens + CLS token concatenated: 2 * embed_dim
                return 2 * self._embed_dim * self.last_n_layers

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def input_size(self) -> tuple[int, int]:
        return (self._image_size, self._image_size)

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        return self._normalize_params


class CLIPTextFeatureExtractor(torch.nn.Module):
    """
    Text-only feature extractor using CLIP's text encoder via open_clip.

    Args:
        model_name: open_clip model name (e.g., "ViT-B-32")
        pretrained: Pretrained weights tag (e.g., "openai", "meta")
        normalize: Whether to L2-normalize output features (default: True)
        device: Device to load model on
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        normalize: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self._normalize = normalize
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model(model_name, pretrained)

    def _load_model(self, model_name: str, pretrained: str) -> None:
        logger.info("Loading CLIP text encoder via open_clip: %s (pretrained=%s)", model_name, pretrained)

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Keep the full model for encode_text
        self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)

        self._feature_dim = model.text_projection.shape[1]

        logger.info("Loaded CLIP text encoder with feature_dim=%d", self._feature_dim)

    def extract_features(self, text: list[str] | torch.Tensor) -> torch.Tensor:
        """
        Extract text features using CLIP text encoder.

        Args:
            text: List of strings or pre-tokenized tensor

        Returns:
            Features tensor of shape (B, D)
        """
        if isinstance(text, list):
            tokens = self.tokenizer(text).to(self.device)
        elif isinstance(text, dict):
            tokens = text["input_ids"].to(self.device)
        else:
            tokens = text.to(self.device)

        with torch.no_grad():
            outputs = self.model.encode_text(tokens, normalize=self._normalize)

        return outputs

    def tokenize(self, text: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        """
        Tokenize text inputs.

        Args:
            text: List of strings
            device: Target device for tensors

        Returns:
            Dictionary with tokenized inputs
        """
        target_device = device or self.device
        tokens = self.tokenizer(text).to(target_device)
        return {"input_ids": tokens}

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
