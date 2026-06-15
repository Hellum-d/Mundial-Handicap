"""Tests for the probabilistic scoring metrics and the back-tester."""

from __future__ import annotations

import math

import numpy as np
import pytest

from football_predictor.evaluation import metrics
from football_predictor.evaluation.backtester import Backtester


def test_log_loss_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    truth = np.array([0, 2])
    assert metrics.log_loss(probs, truth) < 1e-10


def test_log_loss_uniform_matches_analytic():
    probs = np.full((10, 3), 1.0 / 3.0)
    truth = np.array([0, 1, 2] * 3 + [0])
    assert abs(metrics.log_loss(probs, truth) - math.log(3)) < 1e-9


def test_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    truth = np.array([0])
    assert metrics.brier_score(probs, truth) == 0.0


def test_rps_orders_better_than_worse():
    truth = np.array([0])  # win_a is the truth
    near = np.array([[0.6, 0.3, 0.1]])   # mass near the truth
    far = np.array([[0.1, 0.3, 0.6]])    # mass far from the truth
    assert metrics.ranked_probability_score(near, truth) < (
        metrics.ranked_probability_score(far, truth)
    )


def test_outcome_index():
    assert metrics.outcome_index(2, 0) == 0
    assert metrics.outcome_index(1, 1) == 1
    assert metrics.outcome_index(0, 3) == 2


def test_log_loss_per_match_mean_matches_log_loss():
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    truth = np.array([0, 2, 1])
    per = metrics.log_loss_per_match(probs, truth)
    assert per.shape == (3,)
    assert abs(per.mean() - metrics.log_loss(probs, truth)) < 1e-12


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = np.array([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])
    mean, lo, hi = metrics.bootstrap_ci(vals, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    # Reproducible with the same seed.
    assert (mean, lo, hi) == metrics.bootstrap_ci(vals, n_boot=2000, seed=1)


def test_bootstrap_ci_constant_array_has_zero_width():
    vals = np.full(20, 0.7)
    mean, lo, hi = metrics.bootstrap_ci(vals, n_boot=500, seed=3)
    assert mean == pytest.approx(0.7) and (hi - lo) == pytest.approx(0.0)


def test_bootstrap_diff_ci_detects_consistent_winner():
    # `a` is always smaller than `b` -> a - b is significantly negative.
    a = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
    b = np.array([0.9, 0.8, 0.95, 0.85, 0.88])
    diff, lo, hi = metrics.bootstrap_diff_ci(a, b, n_boot=2000, seed=2)
    assert diff < 0 and hi < 0  # wholly-negative interval => significant


def test_bootstrap_diff_ci_requires_equal_shapes():
    with pytest.raises(ValueError):
        metrics.bootstrap_diff_ci(np.zeros(3), np.zeros(4))


def test_split_fold_is_leak_free(sample_matches):
    """train < validation < test by date, with no row overlap (no leakage)."""
    bt = Backtester(sample_matches, train_window_years=12)
    train, validation, test = bt.split_fold(2018)

    assert len(validation) > 0 and len(test) > 0
    # Strict temporal ordering between the three partitions.
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()
    # Disjoint row sets.
    idx = set(train.index) | set(validation.index) | set(test.index)
    assert len(idx) == len(train) + len(validation) + len(test)
    # The test fold is exactly the 2018 World Cup.
    assert (test["competition"] == "world_cup").all()
    assert (test["date"].dt.year == 2018).all()


def test_backtester_ensemble_beats_uniform(sample_matches):
    """The fitted ensemble should beat the uniform baseline on every fold."""
    folds = Backtester(sample_matches).run()
    assert len(folds) == 3  # 2014 / 2018 / 2022
    for fold in folds:
        assert (
            fold.scores["ensemble"]["log_loss"]
            < fold.scores["uniform"]["log_loss"]
        )
