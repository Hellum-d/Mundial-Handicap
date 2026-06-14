"""Tests for the probabilistic scoring metrics and the back-tester."""

from __future__ import annotations

import math

import numpy as np

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


def test_backtester_ensemble_beats_uniform(sample_matches):
    """The fitted ensemble should beat the uniform baseline on every fold."""
    folds = Backtester(sample_matches).run()
    assert len(folds) == 3  # 2014 / 2018 / 2022
    for fold in folds:
        assert (
            fold.scores["ensemble"]["log_loss"]
            < fold.scores["uniform"]["log_loss"]
        )
