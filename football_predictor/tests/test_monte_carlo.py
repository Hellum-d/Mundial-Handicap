"""Tests for the vectorised Monte-Carlo engine."""

from __future__ import annotations

import numpy as np

from football_predictor.simulation.monte_carlo import MonteCarloEngine


def test_outcome_probabilities_sum_to_one_group():
    res = MonteCarloEngine(n_sims=50_000).run(1.5, 1.2, stage="group")
    assert abs(res.win_a + res.draw + res.win_b - 1.0) < 1e-9


def test_knockout_resolves_all_draws():
    res = MonteCarloEngine(n_sims=50_000).run(1.4, 1.4, stage="final")
    assert res.draw == 0.0
    assert abs(res.win_a + res.win_b - 1.0) < 1e-9


def test_expected_goals_match_lambda():
    """Mean simulated goals should approximate the Poisson rate."""
    res = MonteCarloEngine(n_sims=100_000).run(1.8, 0.9, stage="group")
    assert abs(res.xg_a - 1.8) < 0.05
    assert abs(res.xg_b - 0.9) < 0.05


def test_symmetry_equal_lambdas():
    res = MonteCarloEngine(n_sims=200_000).run(1.3, 1.3, stage="group")
    assert abs(res.win_a - res.win_b) < 0.01


def test_higher_lambda_team_wins_more():
    res = MonteCarloEngine(n_sims=100_000).run(2.2, 0.8, stage="group")
    assert res.win_a > res.win_b


def test_markets_in_range():
    res = MonteCarloEngine(n_sims=50_000).run(1.5, 1.5, stage="group")
    for v in (res.over_2_5, res.btts, res.clean_sheet_a, res.clean_sheet_b,
              res.p_team_a_scores, res.p_team_b_scores):
        assert 0.0 <= v <= 1.0


def test_scoreline_probabilities_valid():
    res = MonteCarloEngine(n_sims=50_000).run(1.5, 1.1, stage="group")
    probs = [s["probability"] for s in res.top_scorelines]
    assert probs == sorted(probs, reverse=True)  # descending
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_reproducible_with_seed():
    a = MonteCarloEngine(n_sims=10_000, seed=7).run(1.5, 1.2)
    b = MonteCarloEngine(n_sims=10_000, seed=7).run(1.5, 1.2)
    assert a.win_a == b.win_a and a.draw == b.draw


def test_zero_lambda_b_means_team_b_never_scores():
    res = MonteCarloEngine(n_sims=10_000).run(1.5, 0.0, stage="group")
    assert res.p_team_b_scores == 0.0
    assert res.clean_sheet_a == 1.0
