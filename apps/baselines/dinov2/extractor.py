# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DINOv2 feature extractor with flexible pooling strategies.

Three accepted ``pretrained_weights`` shapes, picked in this order:

1. **Hub name** (e.g. ``"dinov2_vitb14"``): loaded from ``torch.hub`` against
   the upstream ``facebookresearch/dinov2`` repo.
2. **Standalone DINOv2 SSL teacher checkpoint** (a ``.pth`` produced by the
   DINOv2 trainer): requires ``config_file`` pointing at the matching
   ``config.yaml``. Loaded via ``dinov2_utils.load_pretrained_weights``.
3. **Self-contained contrastive checkpoint** (a ``.pt`` produced by
   ``egobabyvlm-train-contrastive``): reads the embedded ``vision_encoder``
   config, builds the :class:`CustomDINOv2VisionEncoder` (which loads the
   SSL teacher weights from disk), then overlays the contrastive backbone
   state from ``model_state_dict["image_embed.backbone.*"]``.
"""

import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import ClassVar

# Pin ``sys.modules["dinov2"]`` to our in-tree fork before any
# ``torch.hub.load("facebookresearch/dinov2", ...)`` call below: torch.hub
# transiently puts its cache dir on ``sys.path`` while running upstream's
# ``hubconf.py`` and leaves the upstream ``dinov2.*`` submodules cached in
# ``sys.modules``, which would shadow our copy on every later import.
import dinov2  # noqa: F401
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class DINOv2Pooling(str, Enum):
    """Pooling strategies for DINOv2 feature extraction."""

    CLS = "cls"
    MEAN_PATCH = "mean_patch"
    CLS_MEAN_PATCH = "cls_mean_patch"
    CONCAT_CLS = "concat_cls"
    CONCAT_CLS_AVGPOOL = "concat_cls_avgpool"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor with flexible pooling strategies.

    See module docstring for the supported ``pretrained_weights`` shapes.
    """

    HUB_MODELS: ClassVar[list[str]] = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vitg14_reg",
    ]

    def __init__(
        self,
        pretrained_weights: Path | str = "dinov2_vitb14",
        config_file: Path | str | None = None,
        checkpoint_key: str = "teacher",
        pooling: str | DINOv2Pooling = DINOv2Pooling.CLS,
        last_n_layers: int = 4,
        *,
        dino_source: str = "ssl_teacher",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        if dino_source not in ("contrastive", "ssl_teacher"):
            msg = f"dino_source must be 'contrastive' or 'ssl_teacher', got {dino_source!r}"
            raise ValueError(msg)
        self.dino_source = dino_source
        self.pooling = DINOv2Pooling(pooling)
        self.last_n_layers = last_n_layers
        self.normalize = normalize

        self._load_model(pretrained_weights, config_file, checkpoint_key)

    def _load_model(
        self,
        pretrained_weights: Path | str,
        config_file: Path | str | None,
        checkpoint_key: str,
    ) -> None:
        """Resolve ``pretrained_weights`` to one of the three supported shapes."""
        path_str = str(pretrained_weights)

        # 1. Hub model name.
        if path_str in self.HUB_MODELS:
            logger.info("Loading Facebook DINOv2 %s from hub", path_str)
            self.model = torch.hub.load("facebookresearch/dinov2", path_str)
            self._autocast_dtype = torch.float16
        # 2. Self-contained contrastive checkpoint (``.pt``).
        elif path_str.endswith(".pt"):
            self._load_from_contrastive_pt(pretrained_weights)
        # 3. Standalone DINOv2 SSL teacher checkpoint (``.pth`` + sidecar config).
        else:
            self._load_custom_checkpoint(pretrained_weights, config_file, checkpoint_key)

        self._autocast_ctx = partial(torch.amp.autocast, device_type="cuda", enabled=True, dtype=self._autocast_dtype)
        logger.info("Using autocast with dtype: %s", self._autocast_dtype)
        self.model.eval()

    def _load_from_contrastive_pt(self, checkpoint_path: Path | str) -> None:
        """Load the DINOv2 backbone from a self-contained contrastive ``.pt``.

        The file is the format produced by ``egobabyvlm-train-contrastive``:
        a top-level ``model_state_dict`` + ``config`` dict, with the vision
        backbone weights under ``model_state_dict["image_embed.backbone.*"]``
        and a fully-specified ``vision_encoder`` Hydra config under
        ``config["model"]["vision_encoder"]``.

        We instantiate the embedded vision encoder (a
        :class:`apps.baselines.clip.modeling.CustomDINOv2VisionEncoder`,
        which builds the model from its own DINOv2 ``config_path`` and loads
        the SSL teacher weights), then overlay backbone weights from the
        checkpoint on top. The projection head is dropped — we serve the raw
        DINOv2 backbone.

        ``self.dino_source`` selects which set of weights to overlay:

        * ``"ssl_teacher"`` (default): the EMA teacher of the SSL branch of
          interleaved training, stored under
          ``ssl_state_dict["teacher"]["backbone.*"]``. This is the
          slow-moving, EMA-stabilized DINOv2 model that you'd deploy as the
          unimodal vision encoder coming out of a triple/interleaved run, and
          matches what the babyvlm paper evaluated. Fall back to
          ``"contrastive"`` only when the checkpoint has no SSL branch or
          you specifically want to measure the contrastive head.
        * ``"contrastive"``: the ``image_embed.backbone.*`` weights from the
          contrastive trainer's main model. This is a transient,
          mid-cycle state — it gets reset to the SSL teacher at every
          SSL→contrastive boundary and only accumulates ``clip_steps`` of
          contrastive updates before the next reset, so the "contrastive
          influence" it carries is just the most recent cycle's worth and
          is sensitive to where in the SSL/contrastive cycle the
          checkpoint was saved.
        """
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        path = Path(str(checkpoint_path))
        logger.info("Loading contrastive .pt checkpoint from %s (dino_source=%s)", path, self.dino_source)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in payload or "config" not in payload:
            msg = "Not a self-contained contrastive .pt: missing 'model_state_dict' or 'config'."
            raise ValueError(msg)

        cfg = OmegaConf.create(payload["config"])
        vision_encoder = instantiate(cfg.model.vision_encoder)

        if self.dino_source == "contrastive":
            backbone_state = {
                k.removeprefix("image_embed.backbone."): v
                for k, v in payload["model_state_dict"].items()
                if k.startswith("image_embed.backbone.")
            }
            if not backbone_state:
                msg = "No 'image_embed.backbone.*' keys in model_state_dict"
                raise ValueError(msg)
        else:  # "ssl_teacher"
            ssl_state = payload.get("ssl_state_dict") or {}
            teacher_state = ssl_state.get("teacher") if isinstance(ssl_state, dict) else None
            if not teacher_state:
                msg = (
                    "dino_source='ssl_teacher' requires ssl_state_dict['teacher'] in the .pt; "
                    "the checkpoint was saved without an SSL branch (mode != interleaved_dino/triple)."
                )
                raise ValueError(msg)
            backbone_state = {
                k.removeprefix("backbone."): v for k, v in teacher_state.items() if k.startswith("backbone.")
            }
            if not backbone_state:
                msg = "ssl_state_dict['teacher'] has no 'backbone.*' keys"
                raise ValueError(msg)

        msg = vision_encoder.backbone.load_state_dict(backbone_state, strict=False)
        logger.info("Loaded %s backbone: %s", self.dino_source, msg)

        self.model = vision_encoder.backbone
        self._autocast_dtype = torch.float16

    def _load_custom_checkpoint(
        self,
        pretrained_weights: Path | str,
        config_file: Path | str | None,
        checkpoint_key: str,
    ) -> None:
        """Load model from custom DINOv2 checkpoint with config."""
        try:
            import dinov2.utils.utils as dinov2_utils
            from dinov2.eval.setup import build_model_from_cfg, get_autocast_dtype
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "DINOv2 package is required for custom checkpoints. "
                "Install from: https://github.com/facebookresearch/dinov2"
            ) from e

        if config_file is None:
            logger.warning("No config file provided, trying to infer from weights path")
            config_file = Path(pretrained_weights).parent.parent.parent / "config.yaml"
            if not Path(str(config_file)).exists():
                raise ValueError(
                    f"Config file not found at inferred location: {config_file}. Please provide a valid config_file."
                )

        logger.info("Loading custom DINO model from %s", pretrained_weights)

        with Path(str(config_file)).open("r") as f:
            config = OmegaConf.load(f)

        self.model, _ = build_model_from_cfg(config, only_teacher=True)

        # Load pretrained weights from a local path.
        # ``dinov2_utils.load_pretrained_weights`` calls ``load_state_dict(strict=False)``
        # internally and only logs a message — if every key fails to match (e.g. when
        # someone points an interleaved-trainer ``.pth`` at this path, where the teacher
        # state is nested under ``dinov2_teacher`` rather than ``checkpoint_key``), the
        # model silently keeps its random-init weights and downstream eval scores are
        # garbage. We snapshot one parameter pre-load and assert it changed post-load
        # so this fails loudly instead.
        sentinel_name, sentinel_param = next(iter(self.model.named_parameters()))
        sentinel_pre = sentinel_param.detach().clone()
        pretrained_weights_str = str(pretrained_weights)
        dinov2_utils.load_pretrained_weights(self.model, pretrained_weights_str, checkpoint_key)
        if torch.equal(sentinel_pre, sentinel_param.detach()):
            msg = (
                f"_load_custom_checkpoint loaded zero matching weights from {pretrained_weights_str!r} "
                f"(checkpoint_key={checkpoint_key!r}); the model is still at random init. "
                "If this is a contrastive-trainer checkpoint, load it via the .pt path instead."
            )
            raise RuntimeError(msg)

        self._autocast_dtype = get_autocast_dtype(config)

    def _compute_feature_dim(self) -> int:
        """Compute output feature dimension based on pooling strategy."""
        embed_dim = self.model.embed_dim

        match self.pooling:
            case DINOv2Pooling.CLS | DINOv2Pooling.MEAN_PATCH:
                return embed_dim
            case DINOv2Pooling.CLS_MEAN_PATCH:
                return embed_dim * 2
            case DINOv2Pooling.CONCAT_CLS:
                return embed_dim * self.last_n_layers
            case DINOv2Pooling.CONCAT_CLS_AVGPOOL:
                return embed_dim * (self.last_n_layers + 1)
            case DINOv2Pooling.SEMANTIC_SEGMENTATION:
                # For semantic segmentation, concatenate patch tokens from last N layers
                return embed_dim * self.last_n_layers

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the configured pooling strategy.

        Args:
            images: (B, C, H, W) normalized image tensor

        Returns:
            Features of shape (B, D) for most pooling modes, or
            (B, N_patches, D) for semantic_segmentation mode
        """
        with self._autocast_ctx():
            match self.pooling:
                case DINOv2Pooling.CLS:
                    # forward() returns x_norm_clstoken when is_training=False
                    features = self.model(images)

                case DINOv2Pooling.MEAN_PATCH:
                    out = self.model.forward_features(images)
                    features = out["x_norm_patchtokens"].mean(dim=1)

                case DINOv2Pooling.CLS_MEAN_PATCH:
                    out = self.model.forward_features(images)
                    features = torch.cat(
                        [
                            out["x_norm_clstoken"],
                            out["x_norm_patchtokens"].mean(dim=1),
                        ],
                        dim=-1,
                    )

                case DINOv2Pooling.CONCAT_CLS:
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=True)
                    features = torch.cat([cls for _, cls in out], dim=-1)

                case DINOv2Pooling.CONCAT_CLS_AVGPOOL:
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=True)
                    cls_tokens = torch.cat([cls for _, cls in out], dim=-1)
                    last_patch_avg = out[-1][0].mean(dim=1)
                    features = torch.cat([cls_tokens, last_patch_avg], dim=-1)

                case DINOv2Pooling.SEMANTIC_SEGMENTATION:
                    # Extract patch tokens from last N layers without CLS token
                    # Returns (B, N_patches, D * last_n_layers)
                    out = self.model.get_intermediate_layers(images, n=self.last_n_layers, return_class_token=False)
                    features = torch.cat(out, dim=-1)

        features = features.float()

        if self.normalize:
            features = F.normalize(features, dim=-1)

        return features

    @property
    def feature_dim(self) -> int:
        return self._compute_feature_dim()

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def input_size(self) -> tuple[int, int]:
        patch_size = self.model.patch_size
        return (patch_size * 37, patch_size * 37)  # 518 for patch_size=14

    @property
    def normalize_params(self) -> dict[str, list[float]]:
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

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

        This method is needed for DPT-style heads that require multi-scale features.

        Args:
            images: (B, C, H, W) normalized image tensor
            n: Number of layers to return (from end) or list of specific layer indices
            reshape: Whether to reshape patch tokens to spatial format (B, C, H, W)
            return_class_token: Whether to include CLS token with each layer
            norm: Whether to apply LayerNorm to intermediate outputs.
                  The DINOv2 reference uses norm=False for dense prediction (depth, segmentation).

        Returns:
            If return_class_token=False: List of feature tensors (B, C, H, W) or (B, N, D)
            If return_class_token=True: List of (patch_features, cls_token) tuples
        """
        with self._autocast_ctx():
            features = self.model.get_intermediate_layers(
                images,
                n=n,
                reshape=reshape,
                return_class_token=return_class_token,
                norm=norm,
            )
        if return_class_token:
            return [(p.float(), c.float()) for p, c in features]
        return [f.float() for f in features]

    @property
    def n_blocks(self) -> int:
        """Return the number of transformer blocks in the model."""
        return self.model.n_blocks if hasattr(self.model, "n_blocks") else len(self.model.blocks)

    @property
    def patch_size(self) -> int:
        """Return the patch size of the ViT model."""
        return self.model.patch_size

    def set_pooling_strategy(
        self,
        pooling: str | DINOv2Pooling | None = None,
        last_n_layers: int | None = None,
        *,
        normalize: bool | None = None,
    ) -> "DINOv2FeatureExtractor":
        """
        Configure the pooling strategy for feature extraction.

        This allows reconfiguring the model after instantiation, which is useful
        when a shared backbone is injected into multiple tasks that need different
        pooling strategies.

        Args:
            pooling: Pooling strategy (cls, mean_patch, cls_mean_patch, concat_cls, etc.)
            last_n_layers: Number of layers to use for multi-layer pooling strategies
            normalize: Whether to L2-normalize output features

        Returns:
            Self, for method chaining
        """
        if pooling is not None:
            self.pooling = DINOv2Pooling(pooling)
        if last_n_layers is not None:
            self.last_n_layers = last_n_layers
        if normalize is not None:
            self.normalize = normalize
        return self
