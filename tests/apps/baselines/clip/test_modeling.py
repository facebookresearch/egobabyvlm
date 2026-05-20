# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the modeling components.

Marked ``integration`` (not run by default CI) because every test instantiates
a HuggingFace BERT backbone — that requires the model files to be cached
locally or network access to download them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration


def test_text_encoder_shapes() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    proj, hidden = te(["hello world", "a cat sits on a mat"])
    assert proj.shape == (2, 128)
    assert hidden.shape[0] == 2
    assert hidden.shape[2] == 768  # bert-base hidden size


def test_text_encoder_pooling_modes() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    cls = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0, pooling="cls")
    mean = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0, pooling="mean")
    proj_cls, _ = cls(["hello world"])
    proj_mean, _ = mean(["hello world"])
    assert proj_cls.shape == proj_mean.shape == (1, 128)


def test_text_encoder_invalid_pooling_raises() -> None:
    from apps.baselines.clip.modeling import TextEncoder

    with pytest.raises(ValueError, match="pooling must be"):
        TextEncoder("bert-base-uncased", pooling="bogus")  # type: ignore[arg-type]


def test_random_vit_vision_encoder_shapes() -> None:
    from apps.baselines.clip.modeling import RandomViTVisionEncoder

    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    out = ve(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 128)
    assert ve.output_dim == 768
    assert ve.arch == "vitb14"
    assert ve.image_size == 224


def test_multimodal_model_contrastive_loss() -> None:
    from apps.baselines.clip.modeling import MultiModalModel, RandomViTVisionEncoder, TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    ve = RandomViTVisionEncoder("vitb14", embedding_dim=128)
    mm = MultiModalModel(ve, te, normalize_features=True, temperature=0.07)

    out = mm.compute_contrastive_loss(torch.randn(4, 3, 224, 224), [f"caption {i}" for i in range(4)])
    assert torch.isfinite(out.loss)
    assert out.logits_per_image.shape == (4, 4)
    assert out.logits_per_text.shape == (4, 4)
    assert 0.0 <= out.image_accuracy.item() <= 1.0


def test_mlm_head_shapes_and_loss() -> None:
    from apps.baselines.clip.modeling import MLMHead, TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    mlm = MLMHead(te)

    _, hidden = te(["hello world", "another caption"])
    preds = mlm(hidden)
    assert preds.shape[:2] == hidden.shape[:2]
    assert preds.shape[2] == te.config.vocab_size

    labels = torch.full(hidden.shape[:2], -100, dtype=torch.long)
    labels[0, 1] = 1234
    labels[1, 2] = 5678
    loss = MLMHead.loss(preds, labels)
    accuracy = MLMHead.accuracy(preds, labels)
    assert torch.isfinite(loss)
    assert 0.0 <= accuracy.item() <= 1.0


def test_mlm_head_ties_decoder_to_word_embeddings() -> None:
    """Decoder weight must share storage with the encoder's word embedding.

    HF's ``BertForMaskedLM`` ties these by default (so the on-disk checkpoint
    omits ``cls.predictions.decoder.weight``). Without the tie, MLM gradients
    don't flow into the embedding and a random decoder + missing pretrained
    weights yield ``loss ~ ln(vocab_size) ~ 10`` instead of the expected ~2-3
    on a fresh BERT-MLM checkpoint.
    """
    from apps.baselines.clip.modeling import MLMHead, TextEncoder

    te = TextEncoder("bert-base-uncased", embedding_dim=128, dropout=0.0)
    mlm = MLMHead(te)
    assert mlm.head.predictions.decoder.weight is te.model.embeddings.word_embeddings.weight


def test_mlm_head_loads_pretrained_cls_from_local_dir(tmp_path: Path) -> None:
    """When ``hf_model_name`` is a local dir with ``cls.*`` weights, load them."""
    import shutil

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors

    from apps.baselines.clip.modeling import MLMHead, TextEncoder

    src = Path(snapshot_download("bert-base-uncased"))
    ckpt_dir = tmp_path / "bert-mlm"
    ckpt_dir.mkdir()
    for fname in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json"):
        if (src / fname).exists():
            shutil.copy(src / fname, ckpt_dir / fname)
    state = load_safetensors(str(src / "model.safetensors"))
    state["cls.predictions.bias"] = torch.full((state["bert.embeddings.word_embeddings.weight"].shape[0],), 1.5)
    state["cls.predictions.transform.LayerNorm.bias"] = torch.full(
        (state["bert.embeddings.word_embeddings.weight"].shape[1],), 0.25
    )
    state["cls.predictions.transform.LayerNorm.weight"] = torch.full(
        (state["bert.embeddings.word_embeddings.weight"].shape[1],), 0.75
    )
    state["cls.predictions.transform.dense.bias"] = torch.full(
        (state["bert.embeddings.word_embeddings.weight"].shape[1],), 0.5
    )
    state["cls.predictions.transform.dense.weight"] = torch.eye(
        state["bert.embeddings.word_embeddings.weight"].shape[1]
    )
    save_safetensors(state, str(ckpt_dir / "model.safetensors"))

    te = TextEncoder(str(ckpt_dir), embedding_dim=128, dropout=0.0)
    mlm = MLMHead(te)

    assert torch.equal(mlm.head.predictions.bias, torch.full_like(mlm.head.predictions.bias, 1.5))
    assert torch.equal(
        mlm.head.predictions.transform.LayerNorm.bias,
        torch.full_like(mlm.head.predictions.transform.LayerNorm.bias, 0.25),
    )
    assert torch.equal(
        mlm.head.predictions.transform.dense.weight,
        torch.eye(mlm.head.predictions.transform.dense.weight.shape[0]),
    )
    assert mlm.head.predictions.decoder.weight is te.model.embeddings.word_embeddings.weight
