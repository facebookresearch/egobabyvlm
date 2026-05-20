# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for DevBench metrics computation."""

import numpy as np
from scipy.optimize import direct
from scipy.special import softmax
from scipy.stats import entropy, spearmanr


def softmax_with_beta(logits: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature parameter beta to logits.

    Args:
        logits: Array of shape (n_trials, n_dimensions) containing logits
        beta: Temperature parameter for softmax

    Returns:
        Array of shape (n_trials, n_dimensions) containing probabilities
    """
    return softmax(beta * logits, axis=1)


def compute_kl_divergence(human_probs: np.ndarray, model_probs: np.ndarray) -> float:
    """Compute KL divergence between human and model probabilities.

    Args:
        human_probs: Array of shape (n_trials, n_dimensions) containing human probabilities
        model_probs: Array of shape (n_trials, n_dimensions) containing model probabilities

    Returns:
        Mean KL divergence across trials
    """
    epsilon = 1e-10
    human_probs = np.maximum(human_probs, epsilon)
    model_probs = np.maximum(model_probs, epsilon)

    human_probs = human_probs / np.sum(human_probs, axis=1, keepdims=True)
    model_probs = model_probs / np.sum(model_probs, axis=1, keepdims=True)

    kl_divs = np.array([entropy(h, m) for h, m in zip(human_probs, model_probs, strict=False)])

    return float(np.mean(kl_divs))


def objective_function(beta: float, human_probs: np.ndarray, model_logits: np.ndarray) -> float:
    """Objective function to minimize: KL divergence with temperature scaling.

    Args:
        beta: Temperature parameter for softmax
        human_probs: Array of shape (n_trials, n_dimensions) containing human probabilities
        model_logits: Array of shape (n_trials, n_dimensions) containing model logits

    Returns:
        Mean KL divergence across trials
    """
    model_probs = softmax_with_beta(model_logits, beta)
    return compute_kl_divergence(human_probs, model_probs)


def get_optimal_kl_divergence(
    human_probs: np.ndarray,
    model_logits: np.ndarray,
    beta_min: float = 0.025,
    beta_max: float = 40.0,
) -> dict:
    """Compute optimal KL divergence by finding the best temperature parameter.

    Args:
        human_probs: Array of shape (n_trials, n_dimensions) containing human probabilities
        model_logits: Array of shape (n_trials, n_dimensions) containing model logits
        beta_min: Minimum value for beta
        beta_max: Maximum value for beta

    Returns:
        Dictionary containing:
            - 'kl_divergence': Minimal KL divergence
            - 'optimal_beta': Optimal beta value
            - 'success': Whether optimization succeeded
    """
    human_probs = np.asarray(human_probs)
    model_logits = np.asarray(model_logits)

    if human_probs.shape != model_logits.shape:
        raise ValueError(f"Shape mismatch: human_probs {human_probs.shape} vs model_logits {model_logits.shape}")

    if not np.allclose(np.sum(human_probs, axis=1), 1.0, rtol=1e-5):
        human_probs = human_probs / np.sum(human_probs, axis=1, keepdims=True)

    def opt_func(beta):
        return objective_function(beta[0], human_probs, model_logits)

    result = direct(
        opt_func,
        bounds=[(beta_min, beta_max)],
        f_min_rtol=1e-4,
        maxfun=200,
        locally_biased=True,
    )

    return {
        "kl_divergence": float(result.fun),
        "optimal_beta": float(result.x[0]),
        "success": bool(result.success),
    }


def compute_accuracy(scores: np.ndarray, correct_index: int = 0) -> float:
    """Compute accuracy assuming correct answer is at a specific index.

    Args:
        scores: Array of shape (n_trials, n_options) containing scores
        correct_index: Index of the correct answer (default: 0)

    Returns:
        Accuracy as a fraction
    """
    predictions = np.argmax(scores, axis=1)
    num_correct = np.sum(predictions == correct_index)
    return float(num_correct / len(scores))


def compute_rsm_correlation(
    model_embeddings: np.ndarray,
    human_similarity: np.ndarray,
) -> dict:
    """Compute Spearman correlation between model and human RSMs.

    Args:
        model_embeddings: Array of shape (n_items, embedding_dim)
        human_similarity: Array of shape (n_items, n_items) with human similarity ratings

    Returns:
        Dictionary with correlation and p-value
    """
    normalized = model_embeddings / np.linalg.norm(model_embeddings, axis=1)[:, np.newaxis]

    model_rsm = np.dot(normalized, normalized.T)

    lower_indices = np.tril_indices(model_rsm.shape[0], k=-1)

    result = spearmanr(model_rsm[lower_indices], human_similarity[lower_indices])

    return {
        "spearman_correlation": float(result.statistic),
        "p_value": float(result.pvalue),
    }
