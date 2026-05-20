# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from core.modeling.similarity import cosine_pairwise, cosine_similarity


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = torch.tensor([[1.0, 0.0, 0.0]])
    sim = cosine_similarity(a, a)
    assert torch.allclose(sim, torch.tensor([[1.0]]))


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    sim = cosine_similarity(a, b)
    assert torch.allclose(sim, torch.tensor([[0.0]]))


def test_cosine_similarity_shape() -> None:
    a = torch.randn(3, 5)
    b = torch.randn(4, 5)
    assert cosine_similarity(a, b).shape == (3, 4)


def test_cosine_similarity_no_normalize_matches_matmul() -> None:
    a = torch.tensor([[2.0, 0.0]])
    b = torch.tensor([[3.0, 0.0]])
    assert torch.allclose(cosine_similarity(a, b, normalize=False), a @ b.T)


def test_cosine_pairwise_shape_and_values() -> None:
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = cosine_pairwise(a, b)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([1.0, 0.0]))
