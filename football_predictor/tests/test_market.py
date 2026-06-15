"""Tests for the odds de-vig and market-prior utilities."""

from __future__ import annotations

import math

import pytest

from football_predictor import market


def _sums_to_one(p: dict[str, float]) -> bool:
    return abs(sum(p.values()) - 1.0) < 1e-9


def test_proportional_devig_sums_to_one_and_removes_margin():
    # Odds with an obvious overround.
    p = market.remove_vig_proportional(2.0, 3.5, 4.0)
    assert _sums_to_one(p)
    # Favourite (lowest odds) keeps the highest probability.
    assert p["win_a"] > p["draw"] and p["win_a"] > p["win_b"]


def test_proportional_devig_fair_book_is_identity():
    # A 2.0/2.0 two-way book with no margin -> 0.5/0.5 (use a tiny draw odds
    # to keep three outcomes but dominate elsewhere is awkward; test 3-way fair)
    p = market.remove_vig_proportional(3.0, 3.0, 3.0)
    assert _sums_to_one(p)
    assert math.isclose(p["win_a"], 1 / 3, abs_tol=1e-9)
    assert math.isclose(p["draw"], 1 / 3, abs_tol=1e-9)


def test_shin_devig_sums_to_one():
    p = market.remove_vig_shin(2.0, 3.5, 4.0)
    assert _sums_to_one(p)
    assert all(0.0 < v < 1.0 for v in p.values())


def test_shin_reduces_to_proportional_for_low_margin():
    # Near-fair book -> Shin and proportional should nearly agree.
    prop = market.remove_vig_proportional(3.02, 3.0, 2.99)
    shin = market.remove_vig_shin(3.02, 3.0, 2.99)
    for k in market.OUTCOME_KEYS:
        assert abs(prop[k] - shin[k]) < 0.02


def test_invalid_odds_raise():
    with pytest.raises(ValueError):
        market.remove_vig_proportional(0.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        market.remove_vig_shin(2.0, -1.0, 4.0)


def test_blend_with_market_convex():
    model = {"win_a": 0.5, "draw": 0.3, "win_b": 0.2}
    mkt = {"win_a": 0.3, "draw": 0.3, "win_b": 0.4}
    out = market.blend_with_market(model, mkt, market_weight=0.5)
    assert _sums_to_one(out)
    # Each component lies between the two inputs (convexity).
    for k in market.OUTCOME_KEYS:
        assert min(model[k], mkt[k]) - 1e-9 <= out[k] <= max(model[k], mkt[k]) + 1e-9


def test_blend_weight_endpoints():
    model = {"win_a": 0.5, "draw": 0.3, "win_b": 0.2}
    mkt = {"win_a": 0.3, "draw": 0.3, "win_b": 0.4}
    assert market.blend_with_market(model, mkt, 0.0) == pytest.approx(model)
    assert market.blend_with_market(model, mkt, 1.0) == pytest.approx(mkt)


def test_blend_weight_out_of_range_raises():
    model = {"win_a": 0.5, "draw": 0.3, "win_b": 0.2}
    with pytest.raises(ValueError):
        market.blend_with_market(model, model, market_weight=1.5)
