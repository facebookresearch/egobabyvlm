# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the vendored EgoVLP SpaceTimeTransformer + FrozenInTime.

Marked ``integration`` (not run by default CI) because they instantiate a
DistilBERT backbone and load the released EgoClip checkpoint, both of which need
locally-cached model files. Set ``$EGOVLP_CHECKPOINT`` to the released
``egovlp.pth`` to enable the checkpoint tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration

EGOVLP_CHECKPOINT = os.environ.get("EGOVLP_CHECKPOINT", "")


def _have_checkpoint() -> bool:
    return bool(EGOVLP_CHECKPOINT) and Path(EGOVLP_CHECKPOINT).exists()


def test_vendored_forward_shapes() -> None:
    """Random-init vendored FrozenInTime produces the right embedding shapes."""
    from apps.baselines.egovlp.model.video_transformer import FrozenInTime

    net = FrozenInTime(projection_dim=256, load_checkpoint=None, pretrained_text=False)
    net.eval()

    clip = torch.randn(2, 1, 3, 224, 224)  # (B, T=1, C, H, W)
    with torch.no_grad():
        vid = net.compute_video(clip)
    assert vid.shape == (2, 256)


def test_temporal_embed_matches_pretraining() -> None:
    """Vendored ViT is built at the EgoVLP pretraining temporal resolution."""
    from apps.baselines.egovlp.model.video_transformer import (
        EGOVLP_NUM_FRAMES,
        FrozenInTime,
    )

    net = FrozenInTime(projection_dim=256, load_checkpoint=None, pretrained_text=False)
    assert net.video_model.temporal_embed.shape == (1, EGOVLP_NUM_FRAMES, 768)


@pytest.mark.skipif(not _have_checkpoint(), reason="released EgoVLP checkpoint not present")
def test_strict_load_released_checkpoint() -> None:
    """The released egovlp.pth loads with strict=True (no missing/unexpected keys)."""
    from apps.baselines.egovlp.model.video_transformer import (
        FrozenInTime,
        _load_egovlp_state_dict,
    )

    net = FrozenInTime(projection_dim=256, load_checkpoint=None, pretrained_text=True)
    state_dict = _load_egovlp_state_dict(EGOVLP_CHECKPOINT)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    assert missing == [], f"missing keys: {missing[:5]}"
    assert unexpected == [], f"unexpected keys: {unexpected[:5]}"

    # And strict=True must not raise.
    net.load_state_dict(state_dict, strict=True)


@pytest.mark.skipif(not _have_checkpoint(), reason="released EgoVLP checkpoint not present")
def test_real_weights_cosine_matrix() -> None:
    """Real weights produce finite image/text embeddings + a valid cosine matrix."""
    import torch.nn.functional as F

    from apps.baselines.egovlp.model import EgoVLPConfig, build_egovlp_model

    config = EgoVLPConfig(pretrained_checkpoint=EGOVLP_CHECKPOINT)
    model = build_egovlp_model(config)
    model.eval()

    clip = torch.randn(3, 1, 3, 224, 224)
    texts = ["a child playing with toys", "a person cooking food", "a dog running"]
    with torch.no_grad():
        img = F.normalize(model.encode_image(clip), dim=-1)
        txt = F.normalize(model.encode_text(texts), dim=-1)

    assert img.shape == (3, 256)
    assert txt.shape == (3, 256)
    assert torch.isfinite(img).all()
    assert torch.isfinite(txt).all()

    sim = img @ txt.T
    assert sim.shape == (3, 3)
    assert (sim <= 1.0 + 1e-4).all()
    assert (sim >= -1.0 - 1e-4).all()


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """save -> load reproduces the same embeddings and drops unknown config keys."""
    from apps.baselines.egovlp.model import (
        EgoVLPConfig,
        build_egovlp_model,
        load_egovlp_model,
        save_egovlp_checkpoint,
    )

    torch.manual_seed(0)
    config = EgoVLPConfig(pretrained_text=False)
    model = build_egovlp_model(config).eval()
    path = tmp_path / "best.pth"
    save_egovlp_checkpoint(model, config, path, epoch=3, step=7)

    # Paper-era checkpoints carry extra config keys; loading must ignore them.
    payload = torch.load(path, weights_only=False)
    payload["egovlp_config"]["use_official"] = True
    torch.save(payload, path)

    loaded, loaded_config = load_egovlp_model(path, device="cpu")
    assert loaded_config.embed_dim == config.embed_dim

    clip = torch.randn(2, 3, 224, 224)
    texts = ["a ball", "a cup"]
    with torch.no_grad():
        torch.testing.assert_close(loaded.encode_image(clip), model.encode_image(clip))
        torch.testing.assert_close(loaded.encode_text(texts), model.encode_text(texts))
