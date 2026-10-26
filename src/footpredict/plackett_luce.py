"""Plackett-Luce ranking model, used as the model of a single voter.

A voter with log-strengths ``u`` picks 1st place with probability
softmax(u), removes that candidate, picks 2nd from the rest, and so on.
Sampling uses the Gumbel-top-k trick: add Gumbel noise to ``u`` and sort.
"""

from __future__ import annotations

import numpy as np


def sample_top_k(logits: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample top-k rankings. ``logits`` has shape (..., n); returns (..., k) indices."""
    g = logits + rng.gumbel(size=logits.shape)
    top = np.argpartition(-g, kth=min(k, g.shape[-1]) - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(g, top, axis=-1), axis=-1)
    return np.take_along_axis(top, order, axis=-1)


def log_likelihood(logits: np.ndarray, ranking: np.ndarray) -> float:
    """Log-probability of a (possibly partial) ranking under Plackett-Luce."""
    remaining = np.ones(logits.shape[0], dtype=bool)
    ll = 0.0
    for i in ranking:
        u = logits[remaining]
        m = u.max()
        ll += logits[i] - (m + np.log(np.exp(u - m).sum()))
        remaining[i] = False
    return float(ll)


def _grad(logits: np.ndarray, ranking: np.ndarray) -> np.ndarray:
    """Gradient of log_likelihood with respect to the logits."""
    remaining = np.ones(logits.shape[0], dtype=bool)
    g = np.zeros_like(logits)
    for i in ranking:
        u = np.where(remaining, logits, -np.inf)
        p = np.exp(u - u.max())
        p /= p.sum()
        g -= p
        g[i] += 1.0
        remaining[i] = False
    return g


def fit_linear(
    features: list[np.ndarray],
    rankings: list[np.ndarray],
    l2: float = 0.1,
    lr: float = 0.05,
    steps: int = 2000,
) -> np.ndarray:
    """Fit weights ``w`` so that logits = X @ w explain the observed rankings.

    ``features[j]`` is an (n_j, d) matrix for edition j and ``rankings[j]``
    the observed order (indices into its rows, best first, may be partial).
    Uses plain gradient ascent on the L2-penalised log-likelihood.
    """
    d = features[0].shape[1]
    w = np.zeros(d)
    for _ in range(steps):
        grad = -l2 * w
        for X, r in zip(features, rankings):
            grad += X.T @ _grad(X @ w, r)
        w += lr * grad / len(features)
    return w
