"""Probabilistic scoring metrics for three-class match outcomes.

Every function takes predictions as an ``(n, 3)`` array of probabilities whose
columns are ordered ``[win_a, draw, win_b]`` and the truth as an integer array
of class indices in ``{0, 1, 2}`` with the same ordering. Lower is better for
all three metrics.
"""

from __future__ import annotations

import numpy as np

# Column order shared by every metric and by the prediction engine.
CLASS_ORDER = ("win_a", "draw", "win_b")
_EPS = 1e-15


def _as_arrays(probs: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(truth, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probs must have shape (n, 3) ordered [win_a, draw, win_b]")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and truth must have the same number of rows")
    return p, y


def log_loss(probs: np.ndarray, truth: np.ndarray) -> float:
    """Multiclass cross-entropy (the primary metric)."""
    p, y = _as_arrays(probs, truth)
    p = np.clip(p, _EPS, 1.0)
    picked = p[np.arange(len(y)), y]
    return float(-np.mean(np.log(picked)))


def brier_score(probs: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error between probability vectors and one-hot truth."""
    p, y = _as_arrays(probs, truth)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ranked_probability_score(probs: np.ndarray, truth: np.ndarray) -> float:
    """Ranked Probability Score for the ordered outcome scale win_a<draw<win_b.

    RPS penalises probability mass placed far (in rank) from the true outcome,
    which suits ordered three-way football results.
    """
    p, y = _as_arrays(probs, truth)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(onehot, axis=1)
    # Divide by (categories - 1) so a perfect prediction scores 0 and the
    # worst scores 1.
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / (p.shape[1] - 1)))


def outcome_index(goals_a: int, goals_b: int) -> int:
    """Map a scoreline to a class index: win_a=0, draw=1, win_b=2."""
    if goals_a > goals_b:
        return 0
    if goals_a == goals_b:
        return 1
    return 2
