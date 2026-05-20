# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Depth estimation eval module: DINOv3-style DPT head on a frozen backbone."""

import itertools
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements
from torch import nn
from tqdm import tqdm

from core.protocols import ImageFeatureExtractor
from core.utils import set_seed, setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.dataloader import EvalDataLoader
from evaluation.base.eval_module import EvalConfig, EvalModule
from evaluation.configs import EvalDatasetConfig
from evaluation.data.base import DepthSample
from evaluation.data.depth import NYUv2DepthEstimationDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DPT building blocks (inlined from the original DINOv2 DPT reference).
# ---------------------------------------------------------------------------


class Interpolate(nn.Module):
    """Wraps :func:`torch.nn.functional.interpolate` as an :class:`nn.Module`."""

    def __init__(self, scale_factor: float, mode: str = "bilinear", align_corners: bool = False) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )


class HeadDepth(nn.Module):
    """Final depth prediction head with one 2x upsampling step."""

    def __init__(self, features: int, init_bias: float = 1.0) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )
        # Bias the final conv positive so initial depth predictions don't trip a dead-ReLU downstream.
        last_conv = cast("nn.Conv2d", self.head[-1])
        assert last_conv.bias is not None  # nn.Conv2d defaults to bias=True
        nn.init.constant_(last_conv.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ReassembleBlocks(nn.Module):
    """Process ViT intermediate features into a multi-scale feature pyramid."""

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: list[int] | None = None,
        readout_type: str = "ignore",
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = [96, 192, 384, 768]

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=True) for out_channel in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        assert isinstance(inputs, list)
        out = []
        for i, sample in enumerate(inputs):
            assert len(sample) == 2, f"Expected (patches, cls_token) tuple, got {len(sample)} elements"
            x, cls_token = sample[0], sample[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class CenterPadding(nn.Module):
    """Center-pad the last two dimensions of ``x`` so they're a multiple of ``multiple``."""

    def __init__(self, multiple: int) -> None:
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> tuple[int, int]:
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:-3:-1]))
        return torch.nn.functional.pad(x, pads)


class BackboneLayersSet(Enum):
    """Predefined sets of ViT layer indices to extract."""

    LAST = "LAST"
    FOUR_LAST = "FOUR_LAST"
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"


def _get_backbone_out_indices(
    model: nn.Module,
    backbone_out_layers: list[int] | tuple[int, ...] | BackboneLayersSet = BackboneLayersSet.FOUR_EVEN_INTERVALS,
) -> list[int]:
    """Resolve a :class:`BackboneLayersSet` (or explicit list) to backbone block indices."""
    n_blocks = getattr(model, "n_blocks", 1)
    if isinstance(backbone_out_layers, (tuple, list)):
        out_indices = list(backbone_out_layers)
    elif backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = list(range(n_blocks - 4, n_blocks))
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        # ViT/L special case: keep the historically-incorrect indices for backward compatibility.
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    assert all(out_index < n_blocks for out_index in out_indices)
    return out_indices


class SigLoss(nn.Module):
    """Scale-invariant logarithmic loss used by the DINOv2 depth-eval reference."""

    def __init__(
        self,
        valid_mask: bool = True,
        loss_weight: float = 1.0,
        max_depth: float | None = None,
        warm_up: bool = True,
        warm_iter: int = 100,
    ) -> None:
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.eps = 0.001

        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up and self.warm_up_counter < self.warm_iter:
            g = torch.log(input + self.eps) - torch.log(target + self.eps)
            g = 0.15 * torch.pow(torch.mean(g), 2)
            self.warm_up_counter += 1
            return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(dg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute SigLoss.

        Args:
            pred: Predicted depth, ``(B, 1, H, W)`` or ``(B, H, W)``.
            target: Ground-truth depth with the same shape.
            mask: Unused; valid-pixel masking is handled internally.

        Returns:
            Scalar loss tensor.
        """
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)

        return self.loss_weight * self.sigloss(pred, target)


# ---------------------------------------------------------------------------
# DINOv3-style DPT head.
# ---------------------------------------------------------------------------


class PreActResidualConvUnitNoNorm(nn.Module):
    """Pre-activation residual conv unit without batch norm (toolbox-matching)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class FeatureFusionBlockNoNorm(nn.Module):
    """Feature fusion block matching the toolbox with ``norm_cfg=None``."""

    def __init__(self, in_channels: int, expand: bool = False, align_corners: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.align_corners = align_corners
        self.out_channels = in_channels // 2 if expand else in_channels

        self.project = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
        self.res_conv_unit1 = PreActResidualConvUnitNoNorm(self.in_channels)
        self.res_conv_unit2 = PreActResidualConvUnitNoNorm(self.in_channels)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = nn.functional.interpolate(
                    inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class ReassembleBlocksBN(nn.Module):
    """ReassembleBlocks with BatchNorm on backbone features (DINOv3 architecture)."""

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: list[int] | None = None,
        readout_type: str = "ignore",
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = [96, 192, 384, 768]

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channel, kernel_size=1, bias=True) for out_channel in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

        self.batchnorm_layers = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in out_channels])

    def forward(self, inputs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        assert isinstance(inputs, list)
        out = []
        for i, sample in enumerate(inputs):
            assert len(sample) == 2
            x, cls_token = sample[0], sample[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)

            x = self.batchnorm_layers[i](x)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class DPTHeadDINOv3(nn.Module):
    """DPT head matching the DINOv3 architecture (BN on backbone features only)."""

    def __init__(
        self,
        embed_dim: int,
        channels: int = 256,
        post_process_channels: list[int] | None = None,
        readout_type: str = "project",
        min_depth: float = 0.001,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()

        if post_process_channels is None:
            post_process_channels = [embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.reassemble_blocks = ReassembleBlocksBN(
            in_channels=embed_dim,
            out_channels=post_process_channels,
            readout_type=readout_type,
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channels, kernel_size=3, padding=1, bias=False) for channel in post_process_channels]
        )

        self.fusion_blocks = nn.ModuleList([FeatureFusionBlockNoNorm(channels) for _ in range(len(self.convs))])
        self.fusion_blocks[0].res_conv_unit1 = None  # type: ignore[assignment]

        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.conv_depth = HeadDepth(channels, init_bias=0.0)

    def forward(self, features: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        x = self.reassemble_blocks(features)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]

        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])

        out = self.project(out)
        out = self.conv_depth(out)

        return torch.relu(out) + self.min_depth


class DepthEstimationModel(nn.Module):
    """Frozen-backbone depth estimator with a DINOv3-style DPT head."""

    def __init__(self, backbone: nn.Module, n_output_layers: int = 4) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_output_layers = n_output_layers

        if hasattr(backbone, "embed_dim"):
            embed_dim = cast("int", backbone.embed_dim)
        elif hasattr(backbone, "feature_dim"):
            embed_dim = cast("int", backbone.feature_dim)
        else:
            raise ValueError("Backbone must have embed_dim or feature_dim attribute")

        patch_size = getattr(backbone, "patch_size", 14)
        n_blocks = getattr(backbone, "n_blocks", 12)

        logger.info(
            "DepthEstimationModel: embed_dim=%d, patch_size=%d, n_blocks=%d",
            embed_dim,
            patch_size,
            n_blocks,
        )

        self.backbone_out_indices = _get_backbone_out_indices(backbone, BackboneLayersSet.FOUR_EVEN_INTERVALS)
        logger.info("Using backbone output indices: %s", self.backbone_out_indices)

        self.patch_size_adapter = CenterPadding(patch_size)

        self.dpt_head = DPTHeadDINOv3(
            embed_dim=embed_dim,
            readout_type="project",
            min_depth=0.001,
            max_depth=10.0,
        )

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        logger.info("Freezing backbone parameters")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_size_adapter(x.float())

        get_intermediate_layers = cast("Any", self.backbone).get_intermediate_layers
        features = get_intermediate_layers(
            x,
            n=self.backbone_out_indices,
            reshape=True,
            return_class_token=True,
            norm=False,
        )
        features = [(p.float(), c.float()) for p, c in features]

        return self.dpt_head(features)


# ---------------------------------------------------------------------------
# Joint augmentations (image + depth, numpy-level).
# Reference: Monocular-Depth-Estimation-Toolbox/depth/datasets/pipelines/transforms.py
# ---------------------------------------------------------------------------


def _nyu_crop(img: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard NYU crop: ``[45:472, 43:608]`` → 427×565."""
    return img[45:472, 43:608].copy(), depth[45:472, 43:608].copy()


def _random_rotate(
    img: np.ndarray, depth: np.ndarray, *, prob: float = 0.5, degree: float = 2.5
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image (bilinear) and depth (nearest) by a random angle."""
    if np.random.rand() >= prob:
        return img, depth

    angle = np.random.uniform(-degree, degree)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    depth = cv2.warpAffine(depth, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, depth


def _random_flip_h(img: np.ndarray, depth: np.ndarray, *, prob: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip applied jointly to image and depth."""
    if np.random.rand() >= prob:
        return img, depth
    return np.fliplr(img).copy(), np.fliplr(depth).copy()


def _random_crop(
    img: np.ndarray, depth: np.ndarray, *, crop_size: tuple[int, int] = (416, 544)
) -> tuple[np.ndarray, np.ndarray]:
    """Random crop to ``(H, W) = crop_size``. Both arrays must be at least that size."""
    ch, cw = crop_size
    h, w = img.shape[:2]
    if h < ch or w < cw:
        raise ValueError(f"Image ({h}×{w}) smaller than crop ({ch}×{cw})")
    top = np.random.randint(0, h - ch + 1)
    left = np.random.randint(0, w - cw + 1)
    return img[top : top + ch, left : left + cw].copy(), depth[top : top + ch, left : left + cw].copy()


def _color_aug(
    img: np.ndarray,
    *,
    prob: float = 0.5,
    gamma_range: tuple[float, float] = (0.9, 1.1),
    brightness_range: tuple[float, float] = (0.75, 1.25),
    color_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Color augmentation (gamma + brightness + per-channel scale) on ``[0, 255]`` floats."""
    if np.random.rand() >= prob:
        return img

    img = img.astype(np.float32)

    gamma = np.random.uniform(*gamma_range)
    img = np.power(img / 255.0, gamma) * 255.0

    brightness = np.random.uniform(*brightness_range)
    img = img * brightness

    colors = np.random.uniform(*color_range, size=3)
    img = img * colors[np.newaxis, np.newaxis, :]

    return np.clip(img, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset wrapper with toolbox-matching augmentation.
# ---------------------------------------------------------------------------


class NYUv2DatasetWithAugmentations(NYUv2DepthEstimationDataset):
    """NYUv2 dataset with the toolbox augmentation pipeline."""

    def __init__(
        self,
        dataset_root: str,
        mode: str = "train",
        depth_image_size: tuple[int, int] = (480, 640),
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
        crop_size: tuple[int, int] = (416, 544),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            mode=mode,
            depth_image_size=depth_image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        self.crop_size = crop_size
        self.norm_mean = np.array(normalize_mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std = np.array(normalize_std or [0.229, 0.224, 0.225], dtype=np.float32)

    def __getitem__(self, index: int) -> DepthSample:
        rgb_rel_path, depth_rel_path, focal_length = self.samples[index]

        rgb_path = self.dataset_root / rgb_rel_path.removeprefix("/")
        depth_path = self.dataset_root / depth_rel_path.removeprefix("/")

        image_raw = cv2.imread(str(rgb_path))
        if image_raw is None:
            raise RuntimeError(f"cv2 failed to decode RGB image: {rgb_path}")
        image: np.ndarray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"cv2 failed to decode depth image: {depth_path}")
        if depth_raw.dtype == np.uint16:
            depth: np.ndarray = depth_raw.astype(np.float32) / 1000.0
        else:
            depth_arr = depth_raw[:, :, 0] if depth_raw.ndim == 3 else depth_raw
            depth = depth_arr.astype(np.float32)
            if depth.max() > 1000:
                depth = depth / 1000.0

        if self.mode == "train":
            image, depth = _nyu_crop(image, depth)
            image, depth = _random_rotate(image, depth, prob=0.5, degree=2.5)
            image, depth = _random_flip_h(image, depth, prob=0.5)
            image, depth = _random_crop(image, depth, crop_size=self.crop_size)
            image = _color_aug(image, prob=0.5, brightness_range=(0.75, 1.25))
        else:
            target_h, target_w = self.depth_image_size
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - self.norm_mean) / self.norm_std
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))

        depth_tensor = torch.from_numpy(depth)

        return DepthSample(media=image_tensor, depth=depth_tensor, focal_length=focal_length)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


_DELTA_THRESH = 1.25


def _compute_single_image_metrics(pred_np: np.ndarray, gt_np: np.ndarray) -> dict[str, float] | None:
    """Compute per-image depth metrics. Returns ``None`` if there are no valid pixels."""
    if pred_np.size == 0 or gt_np.size == 0:
        return None

    thresh = np.maximum(gt_np / pred_np, pred_np / gt_np)
    a1 = float((thresh < _DELTA_THRESH).mean())
    a2 = float((thresh < _DELTA_THRESH**2).mean())
    a3 = float((thresh < _DELTA_THRESH**3).mean())

    abs_rel = float(np.mean(np.abs(gt_np - pred_np) / gt_np))
    sq_rel = float(np.mean(((gt_np - pred_np) ** 2) / gt_np))
    rmse = float(np.sqrt(np.mean((gt_np - pred_np) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(gt_np) - np.log(pred_np)) ** 2)))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    eigen_crop: bool = True,
) -> list[dict[str, float]]:
    """Compute per-image depth metrics following the standard NYU protocol.

    Args:
        pred: Predicted depth, ``(B, 1, H, W)`` or ``(B, H, W)``.
        gt: Ground-truth depth, ``(B, 1, H, W)`` or ``(B, H, W)``.
        min_depth: Lower bound on valid GT depth (exclusive).
        max_depth: Upper bound on valid GT depth (exclusive).
        eigen_crop: Restrict evaluation to the standard center crop ``[45:471, 41:601]``.

    Returns:
        List of per-image metric dicts. Images with no valid pixels are skipped.
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    batch_size = pred.shape[0]
    results: list[dict[str, float]] = []

    for i in range(batch_size):
        pred_i = pred[i].detach().cpu().numpy()
        gt_i = gt[i].detach().cpu().numpy()

        valid = (gt_i > min_depth) & (gt_i < max_depth)
        if eigen_crop:
            crop_mask = np.zeros_like(valid)
            crop_mask[45:471, 41:601] = True
            valid = valid & crop_mask

        m = _compute_single_image_metrics(pred_i[valid], gt_i[valid])
        if m is not None:
            results.append(m)

    return results


def depth_collate_fn(samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate :class:`DepthSample` items into stacked ``(images, depths)`` tensors."""
    images = torch.stack([s.media for s in samples])
    depths = torch.stack([s.depth for s in samples])
    return images, depths


# ---------------------------------------------------------------------------
# Eval module config & class.
# ---------------------------------------------------------------------------


@dataclass
class DepthEstimationEvalModuleConfig(EvalConfig):
    """Configuration for :class:`DepthEstimationEvalModule`."""

    _target_: str = "evaluation.vision.depth_estimation.DepthEstimationEvalModule"

    name: str = "depth_estimation"

    train_dataset: EvalDatasetConfig = MISSING
    val_dataset: EvalDatasetConfig = MISSING
    test_dataset: EvalDatasetConfig | None = None

    backbone: dict[str, Any] = MISSING

    depth_image_size: tuple[int, int] = (480, 640)
    backbone_input_size: int | None = None

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    onecycle_final_div_factor: float = 1000.0
    onecycle_pct_start: float = 0.333
    batch_size: int = 16
    epochs: int = 25
    num_workers: int = 4
    max_depth: float = 10.0
    max_samples: int | None = None
    use_tta: bool = True
    seed: int = 42


class DepthEstimationEvalModule(EvalModule):
    """Train a DPT depth head on top of a frozen image backbone."""

    def __init__(self, config: DepthEstimationEvalModuleConfig) -> None:
        super().__init__(config, DepthEstimationEvalModuleConfig)

        self.output_dir = (
            to_path(self.config.output_dir)
            / self.config.name
            / self.config.train_dataset.name
            / self.config.backbone["name"]
            / self._hparam_str
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self.output_dir)

    @property
    def _hparam_str(self) -> str:
        bs = self.config.batch_size
        lr = self.config.learning_rate
        ep = self.config.epochs
        return f"bs{bs}_lr{lr}_ep{ep}_seed{self.config.seed}"

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=140,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        return f"{self.config.name}_{self.config.train_dataset.name}_{self._hparam_str}_seed{self.config.seed}"

    # ---- data ----

    def _get_dataloader(
        self,
        dataset_config: EvalDatasetConfig,
        backbone: ImageFeatureExtractor,
        *,
        shuffle: bool = False,
    ) -> EvalDataLoader:
        norm = backbone.normalize_params
        dataset = NYUv2DatasetWithAugmentations(
            dataset_root=dataset_config.kwargs["dataset_root"],
            mode=dataset_config.kwargs["mode"],
            depth_image_size=tuple(dataset_config.kwargs.get("depth_image_size", (480, 640))),
            normalize_mean=norm["mean"],
            normalize_std=norm["std"],
        )

        logger.info("Dataset %s loaded: %d samples", dataset_config.name, len(dataset))

        if self.config.max_samples is not None:
            indices = list(range(min(self.config.max_samples, len(dataset))))
            dataset = cast("NYUv2DatasetWithAugmentations", torch.utils.data.Subset(dataset, indices))
            logger.info("Using subset of %d samples", len(dataset))

        return EvalDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=depth_collate_fn,
        )

    # ---- model ----

    def _create_backbone(self, device: str = "cuda") -> ImageFeatureExtractor:
        cfg = self.config.backbone
        backbone: ImageFeatureExtractor = instantiate({"_target_": cfg["_target_"]}, **cfg.get("kwargs", {}))
        backbone.to(device)
        backbone.eval()
        return backbone

    def _create_model(self, backbone: ImageFeatureExtractor, device: str = "cuda") -> nn.Module:
        logger.info("Creating DepthEstimationModel (DINOv3-style: BN on backbone features)")
        model = DepthEstimationModel(backbone=cast("nn.Module", backbone))
        model.to(device)
        return model

    # ---- training ----

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: EvalDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        device: str,
        epoch: int,
    ) -> float:
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
            for images, depths in pbar:
                images = images.to(device)
                depths = depths.to(device)

                outputs = model(images)
                outputs = nn.functional.interpolate(
                    outputs, size=depths.shape[-2:], mode="bilinear", align_corners=True
                )
                outputs = torch.clamp(outputs, min=0.001, max=self.config.max_depth)

                loss = criterion(outputs, depths.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=35.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": epoch_loss / (pbar.n + 1), "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        return epoch_loss / len(train_loader)

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: EvalDataLoader,
        criterion: nn.Module,
        device: str,
        epoch: int | str,
    ) -> tuple[float, dict[str, float]]:
        model.eval()
        val_loss = 0.0
        all_metrics: list[dict[str, float]] = []

        with tqdm(val_loader, desc=f"Epoch {epoch} [Val]") as pbar:
            for images, depths in pbar:
                images = images.to(device)
                depths = depths.to(device)

                outputs = model(images)
                outputs = nn.functional.interpolate(
                    outputs, size=depths.shape[-2:], mode="bilinear", align_corners=True
                )

                if self.config.use_tta:
                    outputs_flip = model(images.flip(dims=[-1]))
                    outputs_flip = nn.functional.interpolate(
                        outputs_flip, size=depths.shape[-2:], mode="bilinear", align_corners=True
                    )
                    outputs = (outputs + outputs_flip.flip(dims=[-1])) / 2.0

                outputs = torch.clamp(outputs, min=0.001, max=self.config.max_depth)

                loss = criterion(outputs, depths.unsqueeze(1))
                val_loss += loss.item()

                per_image = compute_depth_metrics(
                    outputs,
                    depths,
                    min_depth=1e-3,
                    max_depth=self.config.max_depth,
                    eigen_crop=True,
                )
                all_metrics.extend(per_image)

                if all_metrics:
                    running = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                    pbar.set_postfix({"loss": val_loss / (pbar.n + 1), **{k: f"{v:.4f}" for k, v in running.items()}})

        if all_metrics:
            avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        else:
            logger.warning("No valid images found during validation")
            avg_metrics = dict.fromkeys(("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"), 0.0)

        return val_loss / max(len(val_loader), 1), avg_metrics

    def _plot_results(self, results: dict[str, Any], output_dir: Path) -> None:
        history = results.get("training_history", [])
        if not history:
            return

        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = [h["epoch"] for h in history]

        axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
        axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
        axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(visible=True)

        axes[0, 1].plot(epochs, [h["val_metrics"]["abs_rel"] for h in history])
        axes[0, 1].set(xlabel="Epoch", ylabel="Abs Rel", title="Absolute Relative Error")
        axes[0, 1].grid(visible=True)

        axes[1, 0].plot(epochs, [h["val_metrics"]["rmse"] for h in history])
        axes[1, 0].set(xlabel="Epoch", ylabel="RMSE", title="Root Mean Square Error")
        axes[1, 0].grid(visible=True)

        for key, label in [("a1", "δ<1.25"), ("a2", "δ<1.25²"), ("a3", "δ<1.25³")]:
            axes[1, 1].plot(epochs, [h["val_metrics"][key] for h in history], label=label)
        axes[1, 1].set(xlabel="Epoch", ylabel="Accuracy", title="Threshold Accuracies")
        axes[1, 1].legend()
        axes[1, 1].grid(visible=True)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved to %s", output_dir / "training_curves.png")

    def run(self, iteration_value: int = 0, iteration_index: int = 0) -> dict[str, Any]:
        setup_logging()
        set_seed(self.config.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        backbone = self._create_backbone(device)
        train_loader = self._get_dataloader(self.config.train_dataset, backbone, shuffle=True)
        val_loader = self._get_dataloader(self.config.val_dataset, backbone, shuffle=False)

        model = self._create_model(backbone, device)

        total = sum(p.numel() for p in model.parameters())
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info("Total parameters: %d, trainable: %d", total, sum(p.numel() for p in trainable))

        criterion = SigLoss(max_depth=self.config.max_depth, warm_up=True)

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=len(train_loader),
            div_factor=25,
            final_div_factor=self.config.onecycle_final_div_factor,
            pct_start=self.config.onecycle_pct_start,
        )

        best_a1 = 0.0
        training_history: list[dict[str, Any]] = []

        logger.info("Training for %d epochs, lr=%.1e, OneCycleLR", self.config.epochs, self.config.learning_rate)

        val_loss, val_metrics = self._validate(model, val_loader, criterion, device, 0)
        logger.info("Pre-training — RMSE: %.4f, a1: %.4f", val_metrics["rmse"], val_metrics["a1"])

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
            val_loss, val_metrics = self._validate(model, val_loader, criterion, device, epoch)

            logger.info("Epoch %d/%d", epoch, self.config.epochs)
            logger.info("Train Loss: %.4f, Val Loss: %.4f", train_loss, val_loss)
            logger.info("Abs Rel: %.4f, RMSE: %.4f", val_metrics["abs_rel"], val_metrics["rmse"])
            logger.info(
                "δ<1.25: %.4f, δ<1.25²: %.4f, δ<1.25³: %.4f",
                val_metrics["a1"],
                val_metrics["a2"],
                val_metrics["a3"],
            )

            training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if val_metrics["a1"] > best_a1:
                best_a1 = val_metrics["a1"]
                ckpt_path = self.output_dir / "best_model.pth"
                with ckpt_path.open("wb") as f:
                    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics}, f)
                logger.info("Saved best model — a1=%.4f", best_a1)

        with (self.output_dir / "best_model.pth").open("rb") as f:
            model.load_state_dict(torch.load(f, weights_only=False)["model_state_dict"])

        _, final_metrics = self._validate(model, val_loader, criterion, device, "Final")
        logger.info(
            "Final — Abs Rel: %.4f, RMSE: %.4f, a1: %.4f",
            final_metrics["abs_rel"],
            final_metrics["rmse"],
            final_metrics["a1"],
        )

        test_metrics = None
        if self.config.test_dataset is not None:
            test_loader = self._get_dataloader(self.config.test_dataset, backbone, shuffle=False)
            _, test_metrics = self._validate(model, test_loader, criterion, device, "Test")
            logger.info(
                "Test — Abs Rel: %.4f, RMSE: %.4f, a1: %.4f",
                test_metrics["abs_rel"],
                test_metrics["rmse"],
                test_metrics["a1"],
            )

        results = {
            "best_val_metrics": final_metrics,
            "test_metrics": test_metrics,
            "training_history": training_history,
            "config": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.max_depth,
                "use_tta": self.config.use_tta,
            },
        }

        with (self.output_dir / "results.yaml").open("w") as f:
            f.write(to_yaml(results))

        self._plot_results(results, self.output_dir)
        return results
