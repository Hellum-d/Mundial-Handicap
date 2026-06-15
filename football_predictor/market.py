"""Bookmaker-odds utilities: de-vigging and using the market as a prior.

The single highest-ROI accuracy lever for a football model is usually not an
exotic feature but the betting market itself: closing odds are near-efficient,
so blending a model with de-vigged implied probabilities reliably lowers log
loss. This module provides the two pieces needed to do that:

  1. **De-vig** raw decimal odds into a fair probability distribution (the
     bookmaker's margin / "overround" removed), and
  2. **Blend** those market probabilities into the model output as a prior.

No live odds feed is wired into the MVP (the default `martj42` dataset is
results-only). These functions are the integration point: pass per-match
decimal odds for ``(home, draw, away)`` and call
:func:`PredictionEngine`-side blending, or use :func:`blend_with_market`
directly on any ``{win_a, draw, win_b}`` mapping.
"""

from __future__ import annotations

import math

import numpy as np

OUTCOME_KEYS = ("win_a", "draw", "win_b")


def _to_probs(values: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(OUTCOME_KEYS, values)}


def _validate_odds(*odds: float) -> None:
    for o in odds:
        if not math.isfinite(o) or o <= 1.0:
            raise ValueError("Decimal odds must be finite and > 1.")


def remove_vig_proportional(
    odds_home: float, odds_draw: float, odds_away: float
) -> dict[str, float]:
    """De-vig decimal odds by proportional normalisation (the basic method).

    Implied probability of each outcome is ``1 / odds``; their sum is the
    overround (> 1). Dividing through by the overround removes the margin
    uniformly across outcomes.

    Args:
        odds_home, odds_draw, odds_away: Decimal odds (> 1) for A win, draw,
            B win respectively.

    Returns:
        Fair ``{win_a, draw, win_b}`` probabilities summing to 1.
    """
    _validate_odds(odds_home, odds_draw, odds_away)
    implied = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return _to_probs(implied / implied.sum())


def remove_vig_shin(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, float]:
    """De-vig decimal odds with Shin's method (favourite-longshot aware).

    Shin's model attributes the bookmaker margin to a proportion ``z`` of
    insider money and recovers fair probabilities that, unlike proportional
    de-vigging, shrink the margin more from longshots than from favourites —
    usually a better-calibrated estimate of true probability. The method
    reduces to proportional de-vigging as ``z -> 0``.

    Args:
        odds_home, odds_draw, odds_away: Decimal odds (> 1).
        max_iter, tol: Fixed-point solver controls for the insider fraction z.
    """
    _validate_odds(odds_home, odds_draw, odds_away)
    q = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    booksum = q.sum()
    q_norm = q / booksum

    z = 0.0
    for _ in range(max_iter):
        # p_i(z) = (sqrt(z^2 + 4(1-z) q_i^2 / booksum) - z) / (2(1-z))
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * q_norm * q)
        p = (root - z) / (2.0 * (1.0 - z))
        p_sum = p.sum()
        # Update z so that probabilities sum to 1.
        new_z = (p_sum - 1.0) / (booksum - 1.0) if booksum != 1.0 else 0.0
        new_z = float(np.clip(new_z, 0.0, 0.99))
        if abs(new_z - z) < tol:
            z = new_z
            break
        z = new_z

    root = np.sqrt(z * z + 4.0 * (1.0 - z) * q_norm * q)
    p = (root - z) / (2.0 * (1.0 - z))
    return _to_probs(p / p.sum())


def blend_with_market(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    market_weight: float = 0.5,
) -> dict[str, float]:
    """Convex blend of model probabilities with a de-vigged market prior.

    Args:
        model_probs: ``{win_a, draw, win_b}`` from the model.
        market_probs: De-vigged market ``{win_a, draw, win_b}`` (e.g. from
            :func:`remove_vig_shin`).
        market_weight: Weight ``w`` on the market in ``[0, 1]``; the result is
            ``w * market + (1 - w) * model``, renormalised. The market is a
            strong prior, so weights around 0.4-0.6 are typical.

    Returns:
        Blended ``{win_a, draw, win_b}`` probabilities summing to 1.
    """
    if not 0.0 <= market_weight <= 1.0:
        raise ValueError("market_weight must be in [0, 1].")
    w = market_weight
    blended = np.array(
        [w * market_probs[k] + (1.0 - w) * model_probs[k] for k in OUTCOME_KEYS]
    )
    total = blended.sum()
    if total <= 0:
        return {k: 1 / 3 for k in OUTCOME_KEYS}
    return _to_probs(blended / total)
